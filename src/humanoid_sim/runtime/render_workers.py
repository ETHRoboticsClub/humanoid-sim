"""Independent camera and LiDAR Kit processes reading one live plant segment."""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import signal
import tempfile
import time
from pathlib import Path

STOP_TIMEOUT_S = 20.0  # Let Kit close before terminating an unresponsive worker.


def render_configs(cfg, rig) -> dict:
    ready = [sensor for sensor in rig.sensors.values() if sensor.rtx_ready]
    cameras = cfg.cameras and any(sensor.kind == "camera" for sensor in ready)
    lidar = cfg.lidar and any(sensor.kind == "lidar" for sensor in ready)
    if not cameras or not lidar:
        return {"sensors": cfg}
    return {
        "cameras": dataclasses.replace(cfg, lidar=False, lidar_domain=None),
        "lidar": dataclasses.replace(cfg, cameras=False, splat=None, lights=None,
                                     feeds=None, images=None,
                                     rrd=None if cfg.rrd is None else str(Path(cfg.rrd).with_suffix(".lidar.rrd"))),
    }


def _render(segment, cfg, kwargs):
    from humanoid_sim.runtime.render import render_runtime

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    render_runtime(segment, cfg, **kwargs)


def render_runtime_workers(segment, cfg, *, seconds=None, wait_s=60.0, summary=None, window=False):
    from humanoid_sim.model.description import Description
    from humanoid_sim.runtime.render import render_runtime
    from humanoid_sim.sensors import rigs

    configurations = render_configs(cfg, rigs.load(cfg.robot, Description.load(cfg.robot), cfg.sensors))
    if len(configurations) == 1:
        return render_runtime(segment, cfg, seconds=seconds, wait_s=wait_s, summary=summary, window=window)

    context = multiprocessing.get_context("spawn")  # Kit/CUDA must never inherit a forked runtime.
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    processes = {}
    failure = None
    began = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="hsim-render-") as directory:
            outputs = {name: Path(directory) / f"{name}.json" for name in configurations}
            try:
                for name, worker_cfg in configurations.items():
                    kwargs = dict(seconds=seconds, wait_s=wait_s, summary=str(outputs[name]),
                                  window=window and name == "cameras")
                    process = context.Process(target=_render, name=f"hsim-render-{name}",
                                              args=(str(segment), worker_cfg, kwargs))
                    process.start()
                    processes[name] = process
                while not stopped and any(process.is_alive() for process in processes.values()):
                    for name, process in processes.items():
                        if process.exitcode is not None and (process.exitcode != 0 or not outputs[name].is_file()):
                            failure = f"{name} renderer ended with exit code {process.exitcode}"
                            break
                    if failure:
                        break
                    time.sleep(0.05)
            finally:
                for process in processes.values():
                    if process.is_alive():
                        try:
                            os.kill(process.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass  # The worker finished between the liveness check and signal.
                deadline = time.monotonic() + STOP_TIMEOUT_S
                for process in processes.values():
                    process.join(max(0.0, deadline - time.monotonic()))
                for process in processes.values():
                    if process.is_alive():
                        process.kill()
                        process.join()
                        failure = failure or f"{process.name} did not stop within {STOP_TIMEOUT_S:g} s"
            out = {"mode": "separate", "wall_s": time.monotonic() - began, "interrupted": stopped,
                   "exit_codes": {name: process.exitcode for name, process in processes.items()},
                   "workers": {name: json.loads(path.read_text()) for name, path in outputs.items() if path.is_file()}}
            if summary:
                Path(summary).write_text(json.dumps(out, indent=1))
            print(json.dumps(out, indent=1), flush=True)
            if failure or (not stopped and (len(out["workers"]) != len(configurations)
                                           or any(process.exitcode != 0 for process in processes.values()))):
                raise RuntimeError(failure or "a renderer failed")
            return out
    finally:
        for process in processes.values():
            process.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
