"""Measure a live host's sensor delivery after warmup, without stepping physics in Kit.

Run one combined renderer or two processes with --no-cameras / --no-lidar against
the same plant. The JSON keeps sweep capture times and local observation times;
it measures the renderer boundary, not DDS or ROS delivery.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def stats(values) -> dict | None:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {"count": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment")
    parser.add_argument("--robot", default="g1_29dof_with_hand_rev_1_0")
    parser.add_argument("--sensors")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--splat")
    parser.add_argument("--lights")
    parser.add_argument("--feeds")
    parser.add_argument("--lidar-domain", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--wait", type=float, default=600.0)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup < 0:
        parser.error("seconds must be positive and warmup nonnegative")

    from humanoid_sim.runtime.render import RuntimeRender
    from humanoid_sim.sensors.rtx import start_kit
    from humanoid_sim.sensors.session import SessionCfg

    cfg = SessionCfg(robot=args.robot, sensors=args.sensors, cameras=not args.no_cameras,
                     lidar=not args.no_lidar, splat=args.splat, lights=args.lights,
                     feeds=args.feeds, lidar_domain=args.lidar_domain, device=args.device,
                     keep_sweeps=8)
    kit = start_kit(args.device)
    renderer = None
    try:
        renderer = RuntimeRender(args.segment, cfg, wait_s=args.wait)
        session = renderer.session
        started = time.monotonic()
        print("BENCH_READY", flush=True)
        while time.monotonic() - started < args.warmup:
            if not renderer.frame():
                raise RuntimeError("plant session ended during warmup")
            time.sleep(0.001)

        before = dict(session.counts)
        indices = {name: len(samples) for name, samples in session.ms.items()}
        lag_index, host_index = len(renderer.lag_ms), len(renderer.host_keeping_up)
        last_sweep = max((s.start_ns for s in session.sweeps or []), default=-1)
        sweeps = []
        render_wall_ms = []
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            count = session.counts["renders"]
            frame_start = time.monotonic_ns()
            if not renderer.frame():
                raise RuntimeError("plant session ended during measurement")
            received = time.monotonic_ns()
            if session.counts["renders"] != count:
                render_wall_ms.append((received - frame_start) / 1e6)
            for sweep in session.sweeps or []:
                if sweep.start_ns > last_sweep:
                    sweeps.append({"start_ns": sweep.start_ns, "received_mono_ns": received,
                                   "rays": len(sweep.hit), "returns": int(sweep.hit.sum())})
                    last_sweep = sweep.start_ns
            time.sleep(0.001)
        wall_s = time.monotonic() - started
        counts = {name: total - before[name] for name, total in session.counts.items()}
        if len(sweeps) != counts["lidar_sweeps"]:
            raise RuntimeError("benchmark missed a sweep: increase keep_sweeps")
        streams = sum(len(cam.streams) for cam in session.camera_specs)
        keeping_up = renderer.host_keeping_up[host_index:]
        result = {
            "source_revision": args.source_revision, "session_id": renderer.session_id,
            "configuration": vars(args), "wall_s": wall_s, "counts": counts,
            "lidar_sweep_rate_hz": counts["lidar_sweeps"] / wall_s,
            "camera_stream_rate_hz": counts["camera_frames"] / max(streams, 1) / wall_s,
            "render_rate_hz": counts["renders"] / wall_s,
            "render_wall_ms": stats(render_wall_ms),
            "sensor_ms": {name: stats(samples[indices[name]:]) for name, samples in session.ms.items()},
            "lidar_capture_gap_ms": stats(np.diff([s["start_ns"] for s in sweeps]) / 1e6),
            "lidar_observed_gap_ms": stats(np.diff([s["received_mono_ns"] for s in sweeps]) / 1e6),
            "lag_ms": stats(renderer.lag_ms[lag_index:]),
            "host_keeping_up": sum(keeping_up) / len(keeping_up) if keeping_up else None,
            "camera_mean": {"/".join(key): float(image.mean()) for key, image in session.last_images.items()},
            "sweeps": sweeps,
            "session_summary": renderer.summary(wall_s + args.warmup),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("source_revision", "counts", "wall_s",
              "lidar_sweep_rate_hz", "camera_stream_rate_hz", "lidar_observed_gap_ms")}), flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        kit.close()


if __name__ == "__main__":
    main()
