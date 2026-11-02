"""Independent analytic IMU inputs and a recording wrench backend; no physics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.model.description import Description
from humanoid_sim.physics.isaaclab import IsaacLabRobot
from humanoid_sim.replay.exc import ImuTrace
from humanoid_sim.replay.inertial import AccelerationCfg, prepare_acceleration, source_identity
from humanoid_sim.sim import SimCore

G = np.array([0.0, 0.0, -9.81])


def analytic_trace(desc, *, acceleration=(0, 0, 0), spin=0.0, angular_accel=0.0,
                   yaw0=0.0, tilt=0.0, stamps=None):
    """Closed-form rotation about z plus static y tilt, independently built matrices."""
    stamps = (1_730_000_000_000_000_000 + np.arange(121, dtype=np.int64) * 2_500_000
              if stamps is None else stamps)
    t = (stamps - stamps[0]) * 1e-9
    angle = yaw0 + spin * t + 0.5 * angular_accel * t * t
    cy, sy, cp, sp = np.cos(angle), np.sin(angle), np.cos(tilt), np.sin(tilt)
    rotation = np.zeros((len(t), 3, 3))
    rotation[:, 0, 0], rotation[:, 0, 1], rotation[:, 0, 2] = cy * cp, -sy, cy * sp
    rotation[:, 1, 0], rotation[:, 1, 1], rotation[:, 1, 2] = sy * cp, cy, sy * sp
    rotation[:, 2, 0], rotation[:, 2, 2] = -sp, cp
    # q_z(angle) * q_y(tilt), xyzw.
    q = np.stack([-np.sin(angle / 2) * np.sin(tilt / 2),
                  np.cos(angle / 2) * np.sin(tilt / 2),
                  np.sin(angle / 2) * np.cos(tilt / 2),
                  np.cos(angle / 2) * np.cos(tilt / 2)], axis=1)
    axis = np.array([-sp, 0, cp])
    omega = (spin + angular_accel * t)[:, None] * axis
    alpha = np.broadcast_to(angular_accel * axis, omega.shape)
    r = np.asarray(desc.imu_mount()[1])
    rot = np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
    a = np.broadcast_to(np.asarray(acceleration), (len(t), 3))
    specific = np.einsum("nji,nj->ni", rotation, a - G) + rot
    return ImuTrace(stamps, np.arange(len(t), dtype=np.int64), q, omega, specific)


@pytest.mark.parametrize("window", [0, 20_000_000])
@pytest.mark.parametrize("tilt,spin,alpha", [(0.4, 0, 0), (0, 1.8, 0), (0.25, 0.4, 2.0)])
def test_rotation_at_offset_does_not_become_translation(window, tilt, spin, alpha):
    desc = Description.load()
    trace = analytic_trace(desc, tilt=tilt, spin=spin, angular_accel=alpha)
    result = prepare_acceleration(trace, desc, G, AccelerationCfg(window))
    np.testing.assert_allclose(result.acceleration_w, 0, atol=1e-9)
    assert result.t_ns[0] > trace.t_ns[0] and result.t_ns[-1] < trace.t_ns[-1]


@pytest.mark.parametrize("a", [(0, 0, 1.962), (1.2, -0.4, 0.3)])
def test_translation_and_initial_heading_are_recovered(a):
    desc = Description.load()
    heading = 0.7
    tr = analytic_trace(desc, acceleration=a, yaw0=heading, spin=0.3, tilt=0.2)
    sig = prepare_acceleration(tr, desc, G, AccelerationCfg(0))
    c, s = np.cos(heading), np.sin(heading)
    expected = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ a
    np.testing.assert_allclose(sig.acceleration_w, np.broadcast_to(expected, sig.acceleration_w.shape), atol=1e-9)


def test_irregular_epoch_timestamps_and_interval_impulse():
    desc = Description.load()
    stamps = 1_730_000_000_000_000_000 + np.cumsum(np.resize([2_400_000, 2_700_000, 2_500_000], 121))
    tr = analytic_trace(desc, stamps=stamps.astype(np.int64), spin=0.7, angular_accel=1.2)
    # Add a linear world-z acceleration; trapezoidal interval integral is exact.
    t = (stamps - stamps[0]) * 1e-9
    tr.specific_force[:, 2] += 2 * t
    sig = prepare_acceleration(tr, desc, G, AccelerationCfg(0))
    start, end = sig.t_ns[2:50:3], sig.t_ns[3:51:3] + 300_000
    avg = sig.intervals(start, end)
    np.testing.assert_allclose(avg[:, :2], 0, atol=1e-9)
    np.testing.assert_allclose(avg[:, 2], (start + (end - start) // 2 - stamps[0]) * 2e-9, atol=1e-9)
    with pytest.raises(ValueError, match="covered"):
        sig.intervals(np.array([stamps[0]], dtype=np.int64), np.array([stamps[-1]], dtype=np.int64))


@pytest.mark.parametrize("bad", ["nan", "quaternion", "sequence", "gap", "short"])
def test_invalid_motion_is_rejected(bad):
    desc = Description.load()
    tr = analytic_trace(desc)
    if bad == "nan":
        tr.specific_force[5, 0] = np.nan
    elif bad == "quaternion":
        tr.orientation_xyzw[5] = 0
    elif bad == "sequence":
        tr.seq[5] += 1
    elif bad == "gap":
        tr.t_ns[50:] += 70_000_000
    else:
        tr = replace(tr, t_ns=tr.t_ns[:3])
    with pytest.raises(ValueError):
        prepare_acceleration(tr, desc, G, AccelerationCfg(0))


def test_modes_require_full_orientation():
    for mode in (None, "fixed", "yaw"):
        with pytest.raises(ValueError, match="orientation"):
            AccelerationCfg(0).validate_mode(mode)
    AccelerationCfg(0).validate_mode("orientation")


def test_wrench_uses_current_masses_and_centres_without_overwriting_other_loads():
    calls = []

    class Composer:
        def add_forces_and_torques_index(self, **kw):
            calls.append(kw)

    backend = IsaacLabRobot.__new__(IsaacLabRobot)
    backend._inertial_bodies = [1, 2]  # index 0 represents a synthetic support link
    masses = torch.tensor([[1000, 2, 3], [1000, 4, 6], [1000, 1, 2], [1000, 3, 1.]])
    backend.robot = SimpleNamespace(data=SimpleNamespace(body_mass=SimpleNamespace(torch=masses)),
                                    instantaneous_wrench_composer=Composer())
    a = torch.tensor([[0, 0, 2.], [1., 0, 0], [0, -2., 0], [0, 0, 0]])
    backend.apply_base_acceleration(a)
    call = calls[0]
    torch.testing.assert_close(call["forces"], -masses[:, 1:, None] * a[:, None])
    assert call["is_global"] and "positions" not in call  # API's explicit CoM default
    assert not call["torques"].any() and call["body_ids"] == [1, 2]
    # Incremental API is used, never the composer's set/reset API.
    backend.apply_base_acceleration(a * 0)
    assert len(calls) == 2 and not calls[1]["forces"].any()


def test_step_cleanup_after_callback_failure(monkeypatch):
    from humanoid_sim.replay import replay as module

    sim = SimpleNamespace(value=torch.ones(4, 3))
    sim.set_base_acceleration = lambda value: setattr(sim, "value", value)

    def failing(*_):
        sim.set_base_acceleration(torch.ones(4, 3))
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(module, "_run_steps", failing)
    with pytest.raises(RuntimeError, match="injected"):
        module._run(None, sim, None, None)
    assert sim.value is None


def test_setter_requires_prescribed_translation_and_inactive_gantry():
    sim = SimCore.__new__(SimCore)
    sim.plant = SimpleNamespace(num_envs=4, cfg=SimpleNamespace(device="cpu"))
    sim.backend = SimpleNamespace(cfg=SimpleNamespace(fixed_base=True, base_joints=False))
    sim.gantry = SimpleNamespace(active=False)
    sim.set_base_acceleration(torch.zeros(4, 3))
    sim.gantry.active = True
    with pytest.raises(ValueError, match="gantry"):
        sim.set_base_acceleration(torch.zeros(4, 3))
    sim.set_base_acceleration(None)
    assert sim._base_acceleration is None


def test_missing_container_git_metadata_is_explicit(tmp_path):
    (tmp_path / ".git").write_text("gitdir: /nonexistent/external/worktree\n")
    code = tmp_path / "module.py"
    code.write_text("# source still available for hashing\n")
    identity = source_identity(code)
    assert identity["revision"] is None and identity["dirty"] is None
    assert not identity["available"] and "inaccessible" in identity["reason"]


@pytest.mark.parametrize("entrypoint", ["replay", "replay_position_rms"])
def test_preparation_failure_also_clears_external_setting(entrypoint, monkeypatch):
    from humanoid_sim.replay import replay as module

    sim = SimpleNamespace(value=torch.ones(4, 3), plant=SimpleNamespace(num_envs=4))
    sim.set_base_acceleration = lambda value: setattr(sim, "value", value)

    def invalid(*_, **__):
        raise ValueError("invalid IMU fixture")

    monkeypatch.setattr(module, "_prepare", invalid)
    with pytest.raises(ValueError, match="invalid IMU"):
        getattr(module, entrypoint)(None, sim)
    assert sim.value is None


@pytest.mark.parametrize("scale", [0.9999, 1.0001])
def test_rotation_and_acceleration_use_same_normalized_orientation(scale):
    from humanoid_sim.replay.exc import DriveTrace
    from humanoid_sim.replay.replay import _prepare

    desc = Description.load()
    tr = analytic_trace(desc, tilt=0.4)
    tr.orientation_xyzw *= scale
    pdo = {"fixture": DriveTrace("fixture", tr.t_ns, {n: np.zeros(len(tr.t_ns)) for n in
            ("native_position_rad", "velocity_rad_s", "torque_nm", "position_demand_rad")})}
    cap = SimpleNamespace(dataset_id="quaternion-fixture", imu=lambda: tr, pdo=lambda: pdo,
                          time_span_ns=lambda: (int(tr.t_ns[0]), int(tr.t_ns[-1])))
    sim = SimpleNamespace(desc=desc, period_ns=4_000_000, gantry=SimpleNamespace(active=False),
                          plant=SimpleNamespace(desc=desc, map=SimpleNamespace(drive_names=["fixture"])),
                          backend=SimpleNamespace(gravity_w=G, contacts=object()))
    run = _prepare(cap, sim, None, "orientation", base_acceleration=AccelerationCfg(0))
    np.testing.assert_allclose(run.acceleration_w, 0, atol=1e-9)
    np.testing.assert_allclose(run.base.angles, np.tile([0, 0.4, 0], (len(run.t_grid), 1)), atol=1e-9)
