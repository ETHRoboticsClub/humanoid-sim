"""One render of the G1 with the ``g1-cameras-d435`` rig for tests/test_rtx_camera_imaging.py, in a process of its own
(Kit starts once a process): the sensor session with camera effects, every imaged camera at one set exposure or at
the device's auto exposure, run past the arms' settling; then every camera stream's last image, each frame's
exposure per sync group and the session's summary saved to ``--out`` (.npz, and .json beside it). Runs in the Isaac
Sim image (docker/Dockerfile.kit) on an RTX GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rtx_oracles import CAMERAS, G1, SETTLE_S


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--effects", default="")
    p.add_argument("--exposure", nargs=2, type=float, default=None, metavar=("NS", "GAIN"),
                   help="every imaged stream at this exposure (default: the instance's, auto)")
    p.add_argument("--light", type=float, default=None, help="the dome light (nits)")
    p.add_argument("--seconds", type=float, default=SETTLE_S + 0.4)
    args = p.parse_args()

    from humanoid_sim.sensors.rtx import start_kit

    kit = start_kit("cuda:0")
    from humanoid_sim.sensors import imaging, rigs
    from humanoid_sim.sensors.session import SensorSession, SessionCfg

    load = rigs.load

    def load_set(*a, **kw):
        rig = load(*a, **kw)
        for sensor in rig.sensors.values():
            if args.exposure and sensor.kind == "camera":
                sensor.exposure = {s: (int(args.exposure[0]), float(args.exposure[1])) for s in sensor.exposure}
        return rig

    rigs.load = load_set
    history: dict[str, list] = {}
    end_frame = imaging.ImagingChain.end_frame

    def end_frame_noted(chain, now_ns):
        end_frame(chain, now_ns)
        for key, e in chain.exposures.items():
            history.setdefault(f"{chain.camera.name}/{key}", []).append((now_ns, e.exposure_ns, e.gain))

    imaging.ImagingChain.end_frame = end_frame_noted
    extra = {} if args.light is None else {"light": args.light}
    session = SensorSession(SessionCfg(robot=G1, sensors=CAMERAS, lidar=False,
                                       camera_effects=imaging.parse_effects(args.effects), **extra))
    try:
        session.run(args.seconds)
        out = Path(args.out)
        np.savez(out, **{f"{camera}/{stream}": image for (camera, stream), image in session.last_images.items()},
                 **{f"exposure/{key}": np.asarray(rows, dtype=np.float64) for key, rows in history.items()})
        summary = session.summary(1.0)
        out.with_suffix(".json").write_text(json.dumps({k: summary[k] for k in ("camera_effects", "imaging")}))
    finally:
        session.close()
        kit.close()


if __name__ == "__main__":
    main()
