"""One render of the G1 with the ``g1-cameras-d435`` rig for tests/test_rtx_lights.py, in a process of its own (Kit
starts once a process): the sensor session under a lighting, run past the arms' settling, then every camera stream's
last image, the torso's world pose and the session's scene summary saved to ``--out`` (.npz, and .json beside it).
Runs in the Isaac Sim image (docker/Dockerfile.kit) on an RTX GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rtx_oracles import CAMERAS, G1, SETTLE_S, quat_matrix_xyzw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--scene", default=None)
    p.add_argument("--spawn", nargs=2, type=float, default=(0.0, 0.0))
    p.add_argument("--splat", default=None)
    p.add_argument("--lights", default=None)
    args = p.parse_args()

    from humanoid_sim.sensors.rtx import start_kit

    kit = start_kit("cuda:0")
    from humanoid_sim.sensors.session import SensorSession, SessionCfg

    session = SensorSession(SessionCfg(robot=G1, sensors=CAMERAS, scene=args.scene, spawn_m=tuple(args.spawn),
                                       splat=args.splat, lights=args.lights, lidar=False))
    try:
        session.run(SETTLE_S + 0.4)
        robot = session.sim.backend.robot
        i = robot.body_names.index("torso_link")
        out = Path(args.out)
        np.savez(out, torso_R=quat_matrix_xyzw(robot.data.body_quat_w[0, i].cpu().numpy().astype(float)),
                 torso_t=robot.data.body_pos_w[0, i].cpu().numpy().astype(float),
                 **{f"{camera}/{stream}": image for (camera, stream), image in session.last_images.items()})
        summary = session.summary(1.0)
        out.with_suffix(".json").write_text(json.dumps({k: summary[k] for k in ("scene", "rtx_post")}))
    finally:
        session.close()
        kit.close()


if __name__ == "__main__":
    main()
