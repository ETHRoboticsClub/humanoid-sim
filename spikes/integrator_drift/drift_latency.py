"""Angular momentum drift and step time for one robot, solver, integrator and step.

The motion of tests/test_angular_momentum.py: every robot falls freely from rest, every 1:1 joint swinging fast, so
its angular momentum about its centre of mass must stay zero; whatever appears is the solver's. One configuration
per process (Isaac Lab allows one simulation per process). Prints one JSON line. Needs pytest (the test's helpers).

    python spikes/integrator_drift/drift_latency.py --device cuda:0 --robot ethrc_v1 --integrator rk4 --period-ms 4
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import torch
from conftest import built_usd
from test_angular_momentum import (
    DURATION_NS,
    QUEUE_CAPACITY,
    RELEASE_HEIGHT_M,
    angular_momentum,
    swing_amplitudes,
    swing_demand,
)

from humanoid_sim.sim import SimCore
from humanoid_sim.timing.clock import NS_PER_S

p = argparse.ArgumentParser()
p.add_argument("--robot", default="ethrc_v1")
p.add_argument("--device", default="cuda:0")
p.add_argument("--num-envs", type=int, default=1)
p.add_argument("--solver", default="mjwarp")
p.add_argument("--integrator", default=None, help="default: the profile's for the solver")
p.add_argument("--period-ms", type=float, default=None, help="default: the profile's")
p.add_argument("--self-collision", action="store_true")
p.add_argument("--warmup", type=int, default=20)
args = p.parse_args()

cuda = args.device.startswith("cuda")
period_ns = None if args.period_ms is None else round(args.period_ms * 1_000_000)
sim = SimCore(built_usd(args.robot), device=args.device, num_envs=args.num_envs, robot=args.robot,
              period_ns=period_ns, solver=args.solver, integrator=args.integrator, base_height_m=RELEASE_HEIGHT_M,
              self_collision=args.self_collision, queue_capacity=QUEUE_CAPACITY)
try:
    data = sim.backend.robot.data
    q0, _ = sim.joint_state()
    amplitude = swing_amplitudes(sim)
    n_envs = q0.shape[0]
    step_s = []
    worst = torch.zeros(n_envs, dtype=torch.float64, device=q0.device)
    scale = torch.zeros_like(worst)
    while sim.t_ns < DURATION_NS:
        sim.plant.submit(swing_demand(sim, q0, amplitude),
                         torch.full((n_envs,), sim.t_ns, dtype=torch.long, device=q0.device))
        if cuda:
            torch.cuda.synchronize()  # time the step alone, not the work queued before it
        t0 = time.perf_counter()
        sim.step()
        if cuda:
            torch.cuda.synchronize()
        step_s.append(time.perf_counter() - t0)
        net, carried = angular_momentum(data)
        worst, scale = torch.maximum(worst, net.norm(dim=-1)), torch.maximum(scale, carried)
    timed = sorted(step_s[args.warmup:])
    median = statistics.median(timed)
    ratio = worst / scale
    print(json.dumps({
        "robot": args.robot, "solver": sim.solver, "integrator": sim.integrator, "period_ms": sim.period_ns / 1e6,
        "device": args.device, "num_envs": n_envs, "self_collision": args.self_collision,
        "drift_pct_max": 100 * float(ratio.nan_to_num(float("inf")).max()),
        "diverged_envs": int((~torch.isfinite(ratio)).sum()),
        "step_ms_median": 1e3 * median, "step_ms_p90": 1e3 * timed[int(0.9 * len(timed))],
        "wall_s_per_sim_s": median * NS_PER_S / sim.period_ns,
        "env_sim_s_per_wall_s": n_envs * sim.period_ns / NS_PER_S / median,
    }), flush=True)
finally:
    sim.close()
