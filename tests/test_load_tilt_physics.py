"""Load tilt on the solvers: a standing V1 whose given leg joints tilt by the law of their sole's vertical load.

Each check stands the V1 on the ground in the Train-Time scene, holds its drives at the standing pose with the
baseline PD, and loads the robot with a known extra vertical force on its base (the total on the soles is then the
weight plus that force). The tilt is the tilt joint's angle; the link's angle is also read from the bodies' poses,
independently of the joint states, as the drive's reading plus the tilt.
"""
import math

import pytest
import torch
from conftest import GRAVITY, built_usd

from humanoid_sim.devices.demand import DriveDemand

STEPS = 1000  # 1 s at the V1's 1 ms: settled well before, and a robot holding the home pose at the baseline gains
# topples slowly from about 1.5 s on (the rigid asset too), so no check runs longer than this standing
WINDOW = 100  # the last 0.1 s are averaged: contact forces are not steady step to step
# Test inputs: the joint carrying the tilt, by pattern (the knee: the shank's tilt), the angles drawn per joint and
# the saturation loads (as multiples of the robot's weight: one a leg saturates below, one far above).
TILT = ("knee_flexion_.*",)
ANGLE = 0.03
SATURATION = (0.15, 1.5)
# Loads on the base (multiples of the weight): none, then some.
EXTRA = (0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3)


@pytest.fixture(autouse=True)
def clean_solver_output(capfd):
    yield
    output = "".join(capfd.readouterr())
    assert "possibly invalid inertia tensor" not in output
    assert "exceeded MJWarp limit" not in output
    assert "Contact buffer overflowed" not in output


def _cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("physics checks run on CUDA")


def _names():
    from humanoid_sim.model.asset import load_profile
    from humanoid_sim.model.description import Description
    from humanoid_sim.transmissions.compliance import resolve_tilt

    return resolve_tilt(Description.load("ethrc_v1"), load_profile("ethrc_v1"), TILT)


def _link_angle(desc, robot, names):
    """Each joint's link angle ``[E, T]`` from the bodies' poses: the parent's rotation turned by the angle about the
    joint's axis is the link's."""
    from humanoid_sim.physics.crank_stops import _matrix

    pose = robot.data.body_link_pose_w.torch
    bodies = list(robot.body_names)
    out = []
    for name in names:
        parent, link = desc.usd_model.joint_bodies(name)
        frame0, frame1 = (torch.tensor(f[:3, :3], device=pose.device, dtype=torch.float32)
                          for f in desc.usd_model.joint_frames(name))
        axis = torch.tensor(desc.usd_model.joint_axis(name), device=pose.device, dtype=torch.float32)
        m = frame0.T @ _matrix(pose[:, bodies.index(parent), 3:]).transpose(-1, -2) @ _matrix(
            pose[:, bodies.index(link), 3:]) @ frame1
        sin = axis @ torch.stack([m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]]) / 2
        out.append(torch.atan2(sin, (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2))
    return torch.stack(out, -1)


def _run(solver, *, tilt: bool, specs=(), overrides=None, extra=EXTRA, steps=STEPS, shake_hz: float | None = None,
         lift_m: float = 0.0, window=WINDOW):
    """Stand the robot (``tilt``: the asset with load tilt on its knees, else the rigid one), load its base and step.
    Returns the averages over the last ``window`` steps and the extremes over the run; the simulation is closed
    (one Isaac Lab simulation per process at a time). ``extra``: one base load per env, multiples of the weight."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.model.asset import load_profile
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg
    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg
    from humanoid_sim.transmissions.compliance import tilt_legs

    names = _names()
    envs, device = len(extra), "cuda:0"
    overrides = {} if overrides is None else overrides
    usd = built_usd("ethrc_v1", load_tilt=names if tilt else ())
    sim = sim_utils.SimulationContext(robot_sim_cfg("ethrc_v1", device=device, solver=solver,
                                                    physics_overrides=overrides))
    try:
        with robot_build("ethrc_v1", solver=solver, physics_overrides=overrides):
            scene = instantiate(robot_scene_cfg(envs, usd, physics_overrides=overrides))
            sim.reset()
        art = scene["robot"]
        plant = RobotPlant(art, TrainPlantCfg(reporting=False, feedback_sampling=False, randomization=list(specs),
                                              physics_overrides=overrides),
                           imu=scene["imu"], contacts=scene["contacts"])
        core = plant.core
        q = art.data.default_joint_pos.torch[:, [list(art.joint_names).index(n) for n in core.map.joint_names]]
        drive_q, _, _ = core.map.drive_state(q, torch.zeros_like(q))
        hold = DriveDemand.pd(drive_q, core.drive_stiffness.expand(envs, -1), core.drive_damping.expand(envs, -1))
        stand = art.data.default_root_pose.torch[0, 2].item()
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, stand + lift_m], device=device),
                          torch.tensor([[0.0, 0, 0, 1]] * envs, device=device)], 1)
        plant.reset(torch.arange(envs, device=device), root_pose=root, root_vel=torch.zeros(envs, 6, device=device),
                    demand=hold)
        weight = float(art.data.body_mass.torch[0].sum()) * GRAVITY
        base = torch.tensor([0], device=device)
        everyone = torch.arange(envs, device=device)
        load = torch.tensor(extra, device=device, dtype=torch.float32) * weight
        sole = [n.rsplit("/", 1)[-1] for n in scene["contacts"].sensor_names]
        legs = tilt_legs(plant.desc, load_profile("ethrc_v1"))
        sole_ids = torch.tensor([sole.index(legs[n]) for n in plant.tilt_joints] if tilt else [], device=device)
        keys = ("force", "tilt", "q", "link", "drive", "contact", "height", "target")
        total = dict.fromkeys(keys)
        worst = {"tilt": 0.0, "speed": 0.0, "finite": True}
        for step in range(steps):
            plant.pre_physics()
            phase = 2 * math.pi * (shake_hz or 0.0) * step * plant.period_ns * 1e-9
            f = load * (1.0 if shake_hz is None else math.sin(phase))
            force = torch.zeros(envs, 1, 3, device=device)
            force[:, 0, 2] = -f
            art.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=base, env_ids=everyone, is_global=True)
            scene.write_data_to_sim()
            sim.step()
            scene.update(plant.period_ns / 1e9)
            if tilt and step % 10 == 0:
                worst["tilt"] = max(worst["tilt"], float(art.data.joint_pos.torch[:, plant.tilt_ids].abs().max()))
                worst["speed"] = max(worst["speed"], float(art.data.joint_vel.torch[:, plant.tilt_ids].abs().max()))
            if step % 100 == 0:
                worst["finite"] &= bool(torch.isfinite(art.data.body_link_pose_w.torch).all())
            if step >= steps - window:
                truth = plant.truth()
                now = {"force": truth.contact_forces_w[..., 2],
                       "tilt": truth.load_tilt_rad if tilt else torch.zeros(envs, 0, device=device),
                       "q": truth.joint_pos, "drive": plant.measured().drive.position_rad,
                       "contact": truth.contact_forces_w[..., 2].clamp(min=0) > 0,
                       "height": truth.root_pose_w[:, 2],
                       "link": _link_angle(plant.desc, art, names) if tilt else torch.zeros(envs, 0, device=device),
                       "target": plant._targets[:, plant.tilt_ids] if tilt else torch.zeros(envs, 0, device=device)}
                for key in keys:
                    total[key] = now[key].float().clone() / window if total[key] is None else (
                        total[key] + now[key].float() / window)
        out = dict(total)
        out.update(worst=worst, weight=weight, names=names, joints=list(plant.desc.joints), desc=plant.desc,
                   params={k: v.clone() for k, v in plant.params().items()}, extra=load,
                   bodies=[n.rsplit("/", 1)[-1] for n in scene["contacts"].sensor_names], soles=sole_ids, tilt_joints=list(plant.tilt_joints) if tilt else [],
                   drive_cols=plant._tilt_drives, drive_sign=plant._tilt_sign,
                   solver_tilt_gains=(art.data.joint_stiffness.torch[:, plant.tilt_ids].clone(),
                                      art.data.joint_damping.torch[:, plant.tilt_ids].clone()) if tilt else None)
        return out
    finally:
        sim_utils.SimulationContext.clear_instance()


def _specs():
    """Per env: one angle drawn from ±ANGLE for both legs and a small independent difference per leg. (A large
    left-right difference topples a robot holding its pose at the baseline gains: one sole would sit on its edge.)"""
    return [{"param": "load_tilt_angle", "dist": "uniform", "range": [-ANGLE, ANGLE], "op": "set", "per_target": False},
            {"param": "load_tilt_angle", "dist": "uniform", "range": [-0.2 * ANGLE, 0.2 * ANGLE], "op": "add"}]


def _overrides(saturation_n):
    return {"load_tilt": {"angle_rad": 0.0, "saturation_load_n": saturation_n}}


def _weight(solver):
    """The robot's weight (N), from a run without tilt."""
    return _run(solver, tilt=False, extra=(0.0,), steps=10, window=1)["weight"]


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_the_tilt_follows_the_law_of_the_soles_load_and_the_reading_stays_the_encoders(solver):
    _cuda()
    from humanoid_sim.transmissions.compliance import load_tilt

    weight = _weight(solver)
    # Saturation load per env: one of two multiples of the weight; the nominal is the profile's.
    specs = [*_specs(),
             {"param": "load_tilt_saturation_load", "dist": "discrete_uniform",
              "range": [SATURATION[0] * weight, SATURATION[1] * weight], "op": "set",
              "quantum": (SATURATION[1] - SATURATION[0]) * weight, "per_target": False}]
    state = _run(solver, tilt=True, specs=specs, overrides=_overrides(SATURATION[0] * weight))
    angle, sat = state["params"]["load_tilt_angle"], state["params"]["load_tilt_saturation_load"]
    assert len({round(float(v), 3) for v in sat.flatten()}) == 2 and (angle[:, 0] != angle[:, 1]).all()
    assert (state["height"] > 0.5 * state["height"].max()).all(), "the robot fell"
    # The robot stands on both soles, carrying its weight and the extra load.
    force = state["force"]
    on = force[:, state["soles"]]
    expected_total = state["weight"] + state["extra"]
    torch.testing.assert_close(on.sum(1), expected_total, rtol=0.1, atol=0)
    assert (on > 0.1 * state["weight"]).all()
    # The law at the (averaged) load on each sole.
    law = load_tilt(on, angle, sat)
    tilt = state["tilt"]
    assert law.abs().max() > 0.01  # not a vacuous check: both the saturated and the linear regimes appear
    assert (on / sat > 1).any() and (on / sat < 0.6).any()
    torch.testing.assert_close(tilt, law, rtol=0.1, atol=1.5e-3)
    # The link is where the bodies' poses say: the drive's reading plus the tilt, and the reading is the joint's own.
    q = state["q"][:, [state["joints"].index(n) for n in state["names"]]]
    torch.testing.assert_close(state["link"], q + tilt, rtol=0, atol=1e-3)
    # The knee drives report the encoders' side: the drive's reading is the joint's angle, not the link's.
    torch.testing.assert_close(state["drive"][:, state["drive_cols"]], q * state["drive_sign"], rtol=0, atol=1e-3)
    assert (state["link"] - q).abs().max() > 5e-3
    # The solver holds the tilt joint's gains from the profile.
    k, c = state["solver_tilt_gains"]
    assert (k > 0).all() and (c > 0).all()


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_tilt_off_and_an_unloaded_leg_are_the_rigid_leg(solver):
    _cuda()
    weight = _weight(solver)
    over = _overrides(0.5 * weight)
    rigid = _run(solver, tilt=False, extra=(0.0, 0.5))
    again = _run(solver, tilt=False, extra=(0.0, 0.5))
    off = _run(solver, tilt=True, overrides=over, extra=(0.0, 0.5))  # asset with tilt, angle 0 (the profile's default)
    # The asset with tilt has the tilt body more per knee: compare the bodies both have.
    common = [n for n in rigid["bodies"] if n in off["bodies"]]
    assert len(common) == len(rigid["bodies"])
    for run in (rigid, again, off):
        run["force"] = run["force"][:, [run["bodies"].index(n) for n in common]]
    keys = ("q", "height", "force")
    noise = {k: float((again[k] - rigid[k]).abs().max()) for k in keys}
    assert off["tilt"].abs().max() < 1e-3  # settled: the drive's torque through the joint is fed forward
    # Contacts and self-collision are the rigid asset's: the same bodies carry the same forces, the robot the same
    # pose, within the solver's run to run noise.
    for key, unit in (("q", 5e-3), ("height", 5e-3), ("force", 0.02 * rigid["weight"])):
        assert float((off[key] - rigid[key]).abs().max()) <= max(3 * noise[key], unit), key
    assert ((off["force"] > 0) == (rigid["force"] > 0)).float().mean() > 0.9
    # An unloaded leg does not tilt, whatever the angle: in free fall for the first 0.2 s the sole's force is zero.
    free = _run(solver, tilt=True, overrides={"load_tilt": {"angle_rad": 0.05, "saturation_load_n": 0.5 * weight}},
                extra=(0.0, 0.0), lift_m=0.5, steps=150, window=50)
    assert free["force"].abs().max() < 1e-3 * weight
    assert free["tilt"].abs().max() < 1e-3 and free["worst"]["tilt"] < 1e-3


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_tilt_stays_bounded_under_a_shaking_load(solver):
    """Gravity on, an 8 Hz load of up to 0.8 weights on the base, the largest angle a fit could ask for and a
    saturation load that the shaking load crosses: the tilt stays within its angle and the solver stays finite."""
    _cuda()
    weight = _weight(solver)
    angle = 0.09
    over = {"load_tilt": {"angle_rad": angle, "saturation_load_n": 0.4 * weight}}
    state = _run(solver, tilt=True, overrides=over, extra=(0.8,) * 4, steps=1500, shake_hz=8.0, window=100)
    assert state["worst"]["finite"]
    assert state["worst"]["tilt"] < 1.15 * angle  # tracking overshoot of the stiff joint, at most a few percent
    assert state["worst"]["speed"] < 100.0  # a tilt joint speed (rad/s): a hunting tilt would be far beyond it
    assert torch.isfinite(state["tilt"]).all()


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_the_knobs_are_checked(solver):
    _cuda()
    weight = _weight(solver)
    # Tilt knobs on the rigid asset, a missing or non-positive saturation load: refused.
    with pytest.raises(ValueError, match="load_tilt"):
        _run(solver, tilt=False, specs=_specs(), steps=1, window=1)
    with pytest.raises(ValueError, match="saturation_load_n"):
        _run(solver, tilt=True, steps=1, window=1)  # the profile's null
    with pytest.raises(ValueError, match="saturation_load"):
        _run(solver, tilt=True, overrides=_overrides(0.5 * weight), steps=1, window=1,
             specs=[{"param": "load_tilt_saturation_load", "dist": "uniform", "range": [-1.0, 1.0], "op": "set"}])
