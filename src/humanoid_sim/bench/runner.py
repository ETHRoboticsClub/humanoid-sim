"""Benchmarks on Isaac Lab / MJWarp: single-robot real-time factor and batched throughput.

``run_runtime_bench``: the real-time host's whole step (commands, demands, the graphed plant and physics,
publishing feedback and truth to the segment), unpaced, as fast as it goes: its real-time factor is the
headroom the host has when paced. A client in the same process writes new demands before every step (a bus
cycle shorter than the physics period gives the host a new record every step), holding the start pose at the
baseline gains.

Workload: free base, standing on flat ground, every drive holding the standing pose at baseline gains,
with a new demand every 20 ms (the policy period) carrying a small random target perturbation. Timing
covers the full step (plant + solver), after a warm-up. On CUDA the steady-state loop runs under
``torch.cuda.set_sync_debug_mode("error")``, so any torch-side host sync fails the benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.sim import SimCore
from humanoid_sim.timing.clock import NS_PER_MS, NS_PER_S

POLICY_PERIOD_NS = 20 * NS_PER_MS


def run_bench(args, usd: Path) -> dict:
    device = args.device
    sim = SimCore(usd, device=device, robot=args.robot, num_envs=args.num_envs, base_height_m=0.93,
                  integrator=args.integrator,
                  self_collision=False if getattr(args, "no_self_collision", False) else None)
    plant = sim.plant
    q, qd = sim.joint_state()
    q0 = q.clone()
    stiff = plant.drive_stiffness.expand(args.num_envs, -1)
    damp = plant.drive_damping.expand(args.num_envs, -1)
    gen = torch.Generator(device=q.device).manual_seed(0)
    every = POLICY_PERIOD_NS // sim.period_ns

    def submit():
        qdrv, _, _ = plant.map.drive_state(q0, torch.zeros_like(q0))
        noise = 0.02 * torch.randn(qdrv.shape, generator=gen, device=qdrv.device, dtype=qdrv.dtype)
        arrival = torch.full((args.num_envs,), sim.t_ns, dtype=torch.long, device=qdrv.device)
        plant.submit(DriveDemand.jvpt(qdrv + noise, stiff, damp), arrival)

    def run(n):
        for k in range(n):
            if k % every == 0:
                submit()
            sim.step()

    run(50)  # warm-up: kernel compilation, graph capture
    cuda = str(q.device).startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
    t = time.perf_counter()
    syncs_ok = True
    try:
        run(args.steps)
    except RuntimeError as e:  # a host sync in steady state
        syncs_ok = False
        sync_error = str(e).splitlines()[0]
    finally:
        if cuda:
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
    wall = time.perf_counter() - t
    if not syncs_ok:
        return {"error": "host sync in steady state", "detail": sync_error}
    q, _ = sim.joint_state()
    sim_s = args.steps * sim.period_ns / NS_PER_S
    upright = (q - q0).abs().amax(dim=1).lt(0.5).float().mean().item()
    return {
        "physics": "isaaclab/mjwarp", "device": device, "num_envs": args.num_envs, "period_ns": sim.period_ns,
        "steps": args.steps, "wall_s": wall, "rtf_per_env": sim_s / wall,
        "env_steps_per_s": args.num_envs * args.steps / wall,
        "sim_seconds_per_wall_second": args.num_envs * sim_s / wall,
        "fraction_envs_near_standing": upright,
        "self_collision": not getattr(args, "no_self_collision", False), "host_syncs": "none (torch sync debug mode)" if cuda else "n/a",
    }


def run_runtime_bench(args, usd: Path) -> dict:
    import tempfile

    import numpy as np

    from humanoid_sim.model.description import Description
    from humanoid_sim.runtime.abi import plant_abi as abi
    from humanoid_sim.runtime.host import RuntimeHost

    gains = Description.load(args.robot).table["baseline_position_gains"]
    with tempfile.TemporaryDirectory() as tmp:
        host = RuntimeHost(Path(tmp) / "plant", usd, device=args.device, robot=args.robot, support="free")
        client = abi.Segment.open(Path(tmp) / "plant")
        try:
            n = len(host.drives)
            hold = client.regions["feedback"].read()[1]["drive"]["position_rad"][:n].copy()
            demand = {name: np.zeros(n) for name in abi.DRIVE_DEMAND.names if not name.startswith("_")}
            demand |= {"torque_enabled": np.ones(n), "mode": np.full(n, -64), "target_position_rad": hold,
                       "p_gain_raw": np.array([round(gains[d]["stiffness"] * 1000) for d in host.drives]),
                       "d_gain_raw": np.array([round(gains[d]["damping"] * 1000) for d in host.drives])}
            cycle = 0

            def step():
                nonlocal cycle
                cycle += 1
                client.regions["demands"].write(abi.demands_record(demand, cycle=cycle, session_id=host.session_id))
                host.step()

            for _ in range(50):
                step()
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            walls = np.empty(args.steps)
            t = time.perf_counter()
            for k in range(args.steps):
                step()
                walls[k] = time.perf_counter()
            wall = walls[-1] - t
        finally:
            client.close()
            host.close()
    step_s = wall / args.steps
    each_ms = np.diff(walls, prepend=t) * 1e3
    return {"workload": "runtime host, a new demand every step", "device": args.device, "steps": args.steps,
            "ms_per_step": step_s * 1e3, "ms_p99": float(np.percentile(each_ms, 99)), "ms_max": float(each_ms.max()),
            "steps_over_period": int((each_ms > host.period_ns / 1e6).sum()),
            "real_time_factor": host.period_ns / NS_PER_S / step_s, "cuda_graph": host.sim._graphs is not None}

