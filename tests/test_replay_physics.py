"""Drive-level replay/parameter mechanism, independent of any fitting package."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import built_usd


@pytest.mark.physics
def test_parameterized_drive_replay_roundtrip(request):
    """Four fixed-base plants record a drive stream and replay the same physical inputs."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.replay.exc import DriveGains, DriveTrace
    from humanoid_sim.replay.parameters import apply_parameters
    from humanoid_sim.replay.replay import DriveCommands, initial_joint_state, replay_position_rms
    from humanoid_sim.sim import SimCore

    sim = SimCore(
        built_usd("ethrc_v1"), device=request.config.getoption("--replay-device"), num_envs=4, fixed_base=True
    )
    request.addfinalizer(sim.close)
    plant = sim.plant
    names = list(plant.map.joint_names)
    damping = (
        np.array([[sim.desc.joints[n].damping for n in names]]) * np.array([0.8, 1.0, 1.2, 1.4])[:, None]
    )
    apply_parameters(sim, names, damping=damping)
    q = torch.tensor([[sim.desc.joints[n].standing_offset_rad for n in names]], device=plant.cfg.device)
    targets = plant.map.drive_state(q, torch.zeros_like(q))[0][0]
    drive_names = tuple(plant.map.drive_names)
    dev = plant.cfg.device
    # Only env 0 is recorded; the others demonstrate that parameters are batched.
    physical = initial_joint_state(plant, targets).expand(4, -1)
    sim.write_joint_state(physical, torch.zeros_like(physical))
    kp, kd = plant.drive_stiffness.expand(4, -1), plant.drive_damping.expand(4, -1)
    demand = DriveDemand.pd(targets.expand(4, -1), kp, kd)
    plant.reset(torch.arange(4, device=dev), demand)
    sim.t_ns = 0
    t = [0]
    positions = [targets.double().cpu().numpy()]
    velocities = [np.zeros(len(drive_names))]
    torques = [np.zeros(len(drive_names))]
    for _ in range(60):
        fb = sim.step()
        t.append(sim.t_ns)
        positions.append(fb.position_rad[0].double().cpu().numpy())
        velocities.append(fb.velocity_rad_s[0].double().cpu().numpy())
        torques.append(fb.applied_torque_nm[0].double().cpu().numpy())
    stamps = np.asarray(t, dtype=np.int64)
    pdo = {
        n: DriveTrace(
            n,
            stamps,
            {
                "native_position_rad": np.asarray(positions)[:, i],
                "velocity_rad_s": np.asarray(velocities)[:, i],
                "torque_nm": np.asarray(torques)[:, i],
                "position_demand_rad": np.full(len(stamps), float(targets[i])),
            },
        )
        for i, n in enumerate(drive_names)
    }
    gains = {n: DriveGains(float(kp[0, i]), float(kd[0, i]), 0, 0) for i, n in enumerate(drive_names)}
    capture = SimpleNamespace(
        pdo=lambda: pdo, gains=lambda: gains, time_span_ns=lambda: (int(stamps[0]), int(stamps[-1]))
    )
    commands = DriveCommands(
        drive_names, np.array([0], dtype=np.int64), targets.double().cpu().numpy()[None], "applied"
    )
    rms = replay_position_rms(capture, sim, source="command", commands=commands, base_motion=None, skip_s=0.0)
    assert tuple(rms.shape) == (4, len(drive_names)) and torch.isfinite(rms).all()
    assert float(rms[0].max()) < 1e-4
