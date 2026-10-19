"""A robot standing among a scene's static boxes, recorded to Rerun: for looking at a robot in a room such as Controls'
hangar.

    python examples/scene_stand.py --scene <controls>/ros2_ws/src/sim/ethrc_sim/environments/hangar/hangar_proxy.npz \\
        --spawn -4.42 3.48 --seconds 4 --rrd outputs/hangar_v1.rrd
    python examples/scene_stand.py --robot g1_29dof_with_hand_rev_1_0 --scene ... --spawn -4.42 3.48 \\
        --rrd outputs/hangar_g1.rrd

The robot starts at the spawn point (the world origin) in its home pose on a free base (or a welded one with
`--fixed-base`) and holds it at the baseline gains; the recording shows the boxes, the floor and the robot. `-4.42 3.48` is Controls' `HANGAR_SPAWN`
(`ethrc_sim/environments.py`), open floor 1.35 m from the nearest box. Runs in the Isaac Lab image (Warp's CPU
device works, slowly).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--robot", default="ethrc_v1")
    p.add_argument("--scene", required=True, help="static boxes, an .npz as src/humanoid_sim/world/boxes.py describes")
    p.add_argument("--spawn", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Y"))
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--fixed-base", action="store_true", help="weld the base where it starts, feet just above the floor")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rrd", required=True)
    p.add_argument("--assets", default=str(REPO / "outputs" / "assets"))
    args = p.parse_args()

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.model.asset import asset_dir, load_profile
    from humanoid_sim.model.description import Description
    from humanoid_sim.model.poses import STANDING_CLEARANCE_M, initial_pose
    from humanoid_sim.signals.scene import SceneRecorder
    from humanoid_sim.sim import SimCore
    from humanoid_sim.world.boxes import StaticBoxes

    out = asset_dir(Description.load(args.robot), load_profile(args.robot), Path(args.assets))
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    scene = StaticBoxes.load(args.scene, args.spawn)
    print(json.dumps(scene.describe()))
    pose, height = initial_pose(usd, args.robot, "home")
    sim = SimCore(usd, device=args.device, robot=args.robot, base_height_m=height + STANDING_CLEARANCE_M,
                  fixed_base=args.fixed_base, contacts=True, scene=scene)
    recorder = SceneRecorder(sim.backend, usd, rrd=args.rrd, app_id="humanoid-sim scene")
    try:
        plant = sim.plant
        q = pose.to(device=plant.cfg.device, dtype=sim.joint_state()[0].dtype).unsqueeze(0)
        sim.write_joint_state(q, torch.zeros_like(q))
        hold, _, _ = plant.map.drive_state(q, torch.zeros_like(q))
        plant.submit(DriveDemand.pd(hold, plant.drive_stiffness.expand(1, -1), plant.drive_damping.expand(1, -1)),
                     torch.zeros(1, dtype=torch.long, device=hold.device))
        every = max(1, 20_000_000 // sim.period_ns)  # a frame every 20 ms
        for k in range(int(args.seconds * 1e9) // sim.period_ns):
            if k % every == 0:
                recorder.frame(sim.t_ns)
            sim.step()
        root = sim.backend.robot.data.root_link_pose_w.torch[0].tolist()
        print(json.dumps({"rrd": args.rrd, "seconds": sim.t_ns / 1e9, "root_pose_w": root}))
    finally:
        recorder.close()
        sim.close()


if __name__ == "__main__":
    main()
