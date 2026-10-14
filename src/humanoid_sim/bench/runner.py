"""Benchmarks: single-robot real-time factor (CPU) and batched throughput (GPU).

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
    sim = SimCore(args.backend, usd, robot=args.robot, num_envs=args.num_envs, base_height_m=0.93,
                  integrator=args.integrator, device=device,
                  use_mujoco_cpu=(args.backend == "newton" and device == "cpu"),
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
        "backend": args.backend, "device": device, "num_envs": args.num_envs, "period_ns": sim.period_ns,
        "steps": args.steps, "wall_s": wall, "rtf_per_env": sim_s / wall,
        "env_steps_per_s": args.num_envs * args.steps / wall,
        "sim_seconds_per_wall_second": args.num_envs * sim_s / wall,
        "fraction_envs_near_standing": upright,
        "self_collision": not getattr(args, "no_self_collision", False), "host_syncs": "none (torch sync debug mode)" if cuda else "n/a",
    }
