"""Benchmark of the Train-Time API scene: throughput and host syncs with sensors and randomization on.

Workload: the training scene (robot, IMU, contacts, flat ground) with :class:`RobotPlant` and a randomization
like a locomotion policy's (masses and payload, armature, friction, gain and torque scales, latency, encoder
offsets, IMU bias). Every env holds the standing pose with a new demand every 20 ms (the policy period);
reads ``measured()`` and ``truth()`` once per policy period; resets 1/50 of the envs per policy period (as
episodes ending). The steady-state loop runs under ``torch.cuda.set_sync_debug_mode("error")`` on CUDA.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.timing.clock import NS_PER_MS, NS_PER_S

POLICY_PERIOD_NS = 20 * NS_PER_MS

RANDOMIZATION = [
    {"param": "body_mass", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
    {"param": "body_mass", "bodies": "Component171_1", "dist": "uniform", "range": [0.0, 4.5], "op": "add"},
    {"param": "joint_armature", "dist": "uniform", "range": [0.8, 1.2], "op": "scale"},
    {"param": "friction", "dist": "uniform", "range": [0.3, 1.0], "op": "set"},
    {"param": "drive_stiffness", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
    {"param": "drive_damping", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
    {"param": "torque_limit", "dist": "uniform", "range": [0.85, 1.0], "op": "scale"},
    {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
    {"param": "latency", "drives": ".*ANK.*", "dist": "uniform", "range": [0.010, 0.020], "op": "add",
     "per_target": False},
    {"param": "encoder_offset", "dist": "uniform", "range": [-0.026, 0.026], "op": "add"},
    {"param": "encoder_offset", "drives": "maxonHEJ50L_ANK.*", "dist": "uniform", "range": [-0.14, 0.14], "op": "add"},
    {"param": "imu_gyro_bias", "dist": "normal", "range": [0.0, 0.0015], "op": "add"},
]


def run_train_bench(args, usd: Path) -> dict:
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg
    from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg

    n = args.num_envs
    sim = sim_utils.SimulationContext(robot_sim_cfg(args.robot, device=args.device))
    scene = instantiate(robot_scene_cfg(n, usd, robot=args.robot, ground_friction=0.0,
                                        self_collision=False if args.no_self_collision else None))
    sim.reset()
    plant = RobotPlant(scene["robot"], TrainPlantCfg(robot=args.robot, randomization=RANDOMIZATION),
                       imu=scene["imu"], contacts=scene["contacts"], physics_manager=sim.physics_manager)
    q, qd = plant.joint_state()
    stand, _, _ = plant.core.map.drive_state(q, qd)
    stiff, damp = plant.core.drive_stiffness.expand(n, -1), plant.core.drive_damping.expand(n, -1)
    every = POLICY_PERIOD_NS // plant.period_ns
    period_s = plant.period_ns / NS_PER_S
    gen = torch.Generator(device=q.device).manual_seed(0)
    resets = max(1, n // 50)
    sink = torch.zeros((), device=q.device)

    def policy_step(k):
        nonlocal sink
        m, t = plant.measured(), plant.truth()
        sink = sink + m.drive.position_rad.sum() + m.imu.gyro[:, 0].sum() + t.contact_forces_w[..., 2].sum()
        ids = torch.randint(0, n, (resets,), generator=gen, device=q.device)
        plant.reset(ids)
        sent = torch.full((n,), plant.t_ns, dtype=torch.int64, device=q.device)
        plant.submit(DriveDemand.jvpt(stand, stiff, damp), sent)

    def run(steps):
        for k in range(steps):
            if k % every == 0:
                policy_step(k)
            plant.pre_physics()
            scene.write_data_to_sim()
            sim.step()
            scene.update(period_s)

    run(50)
    cuda = str(q.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
    t = time.perf_counter()
    error = None
    try:
        run(args.steps)
    except RuntimeError as e:
        import traceback

        frames = [f"{f.filename.rsplit('/', 2)[-1]}:{f.lineno} {f.name}" for f in traceback.extract_tb(e.__traceback__)]
        error = f"{str(e).splitlines()[0]} at {' <- '.join(reversed(frames[-6:]))}"
    finally:
        if cuda:
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
    wall = time.perf_counter() - t
    sim_utils.SimulationContext.clear_instance()
    if error:
        return {"error": "host sync in steady state", "detail": error}
    sim_s = args.steps * period_s
    return {"workload": "train", "device": args.device, "num_envs": n, "steps": args.steps, "wall_s": wall,
            "env_steps_per_s": n * args.steps / wall, "sim_seconds_per_wall_second": n * sim_s / wall,
            "self_collision": not args.no_self_collision, "host_syncs": "none (torch sync debug mode)" if cuda else "n/a",
            "checksum_finite": bool(torch.isfinite(sink).item())}
