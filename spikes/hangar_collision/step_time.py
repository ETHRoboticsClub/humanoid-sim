"""Spike (issue #57): physics step time of a room's collision, boxes against convex pieces, at a number of envs.

One configuration per process (Isaac Lab keeps one simulation per process). Every env's V1 starts at the spawn point
in its home pose and holds it; prints one JSON line.

    python spikes/hangar_collision/step_time.py 1024 BOXES.npz|- HULLS.npz|- --spawn -4.42 3.48
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.poses import STANDING_CLEARANCE_M, initial_pose
from humanoid_sim.sim import SimCore
from humanoid_sim.world.boxes import StaticBoxes
from humanoid_sim.world.hulls import StaticHulls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("envs", type=int)
    ap.add_argument("boxes", help="the room's boxes (.npz) or -")
    ap.add_argument("hulls", help="the room's convex pieces (.npz) or -")
    ap.add_argument("--spawn", nargs=2, type=float, default=(0.0, 0.0))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--assets", default="outputs/assets")
    args = ap.parse_args()
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), Path(args.assets))
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    _, height = initial_pose(usd, "ethrc_v1", "home")
    boxes = None if args.boxes == "-" else StaticBoxes.load(args.boxes, args.spawn)
    hulls = None if args.hulls == "-" else StaticHulls.load(args.hulls, args.spawn)
    sim = SimCore(usd, device="cuda:0", num_envs=args.envs, base_height_m=height + STANDING_CLEARANCE_M, scene=boxes,
                  collision=hulls)
    try:
        for _ in range(50):
            sim.step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.steps):
            sim.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / args.steps * 1e3
        z = sim.backend.robot.data.root_link_pose_w.torch[:, 2]
        print(json.dumps({"envs": args.envs, "boxes": boxes is not None, "pieces": None if hulls is None else len(hulls),
                          "ms_per_step": round(ms, 3), "fraction_up": z.gt(0.75 * height).float().mean().item(),
                          "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
