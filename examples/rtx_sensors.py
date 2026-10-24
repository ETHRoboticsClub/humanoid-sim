"""A robot with its rig's RTX sensors in a scene: on the G1 the Mid-360 of its rig, and with ``--sensors
g1-cameras`` the ZED Mini and D435i cameras that extra rig adds (``sensors/rigs.py``), rendered by Isaac Sim while Newton runs the
physics, published as Controls reads them and recorded to Rerun. Runs in the Isaac Sim image (docker/Dockerfile.kit)
on an RTX GPU.

    H=/data/hangar   # Controls' ros2_ws/src/sim/ethrc_sim/environments/hangar
    humanoid-sim splat-usd $H/hangar_splat.ply outputs/rtx/hangar_splat.usdc --spawn -4.42 3.48 --half
    python examples/rtx_sensors.py --sensors g1-cameras --scene $H/hangar_proxy.npz --spawn -4.42 3.48 \\
        --splat outputs/rtx/hangar_splat.usdc --scan-mesh $H/hangar_visual.obj \\
        --seconds 10 --feeds /dev/shm/ethrc-camera-feed --lidar-domain 0 \\
        --rrd outputs/rtx/g1_hangar_sensors.rrd --images outputs/rtx/frames

The robot hangs at the spawn point on a welded base, feet just above the floor, holding its home pose (a G1 cannot
stand on a fixed-gain hold: it topples within a second). Each camera stream renders at its rate and goes to the
``ethrc-camera-sim-feed/v1`` rings under ``--feeds``; each lidar sweep goes to Unitree's DDS topic on
``--lidar-domain``. The physics runs as fast as it can, so the summary's real-time factor is what the sensors cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROBOT = "g1_29dof_with_hand_rev_1_0"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--robot", default=ROBOT)
    p.add_argument("--sensors", default=None,
                   help="an extra rig whose sensors join the robot's (humanoid_sim.sensors.rigs.EXTRAS, e.g. g1-cameras)")
    p.add_argument("--scene", default=None, help="static boxes (.npz, world/boxes.py)")
    p.add_argument("--spawn", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Y"))
    p.add_argument("--splat", default=None, help="the room's Gaussian splat as USD (humanoid-sim splat-usd)")
    p.add_argument("--splat-intensity", type=float, default=1.0,
                   help="scale the splat's emission (RTX shows it darker than the scan's photographs)")
    p.add_argument("--scan-mesh", default=None,
                   help="the room's scan (OBJ): the lidar's surfaces with the boxes; added only with a lidar, so "
                        "without --splat --no-lidar changes the cameras' depth")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--no-cameras", action="store_true")
    p.add_argument("--no-lidar", action="store_true")
    p.add_argument("--feeds", default=None, help="camera feed root (ethrc-camera-sim-feed/v1)")
    p.add_argument("--lidar-domain", type=int, default=None, help="publish the Mid-360 on Unitree's DDS domain")
    p.add_argument("--rrd", default=None)
    p.add_argument("--images", default=None, help="save every stream's first and last frame here (PNG)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--assets", default=str(REPO / "outputs" / "assets"))
    p.add_argument("--summary", default=None, help="write the timing summary here (JSON)")
    args = p.parse_args()

    from humanoid_sim.sensors.rtx import start_kit

    kit = start_kit(args.device)
    from humanoid_sim.sensors.session import SensorSession, SessionCfg

    session = SensorSession(SessionCfg(
        robot=args.robot, sensors=args.sensors, scene=args.scene, spawn_m=tuple(args.spawn), splat=args.splat,
        splat_intensity=args.splat_intensity, scan_mesh=args.scan_mesh, cameras=not args.no_cameras,
        lidar=not args.no_lidar, feeds=args.feeds, lidar_domain=args.lidar_domain, rrd=args.rrd, images=args.images,
        device=args.device, assets=args.assets))
    try:
        start = time.perf_counter()
        session.run(args.seconds)
        wall = time.perf_counter() - start
        summary = session.summary(wall)
        print(json.dumps(summary, indent=1))
        if args.summary:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(json.dumps(summary, indent=1))
    finally:
        session.close()
        kit.close()


if __name__ == "__main__":
    main()
