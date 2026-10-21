"""The rig's RTX sensors on a running runtime segment: Kit renders the robot where the real-time host (``serve``) has
it, and the frames and sweeps go out as Controls reads them (its camera feed, Unitree's lidar topic).

The renderer never steps physics and never writes to the segment. Each render it takes the latest truth (base pose
and joint positions), poses the robot there by forward kinematics (``SensorSession.pose``) and renders at that plant
time. It renders as fast as Kit allows, at most once per camera period (the lidar's sweep period without cameras);
the host keeps real time on its own, and a render that falls behind skips to the latest truth. Stamps map plant time
onto the realtime clock through the host's own mapping (``plant_abi``: the feedback region's ``written_mono_ns``
at ``sim_time_ns``).

The scene is the host's: its static boxes and scan (from the descriptor, when the files are readable here with the
same digests), and the room's splat for the cameras when one is given. Needs the Isaac Sim image
(``docker/Dockerfile.kit``) on an RTX GPU, like ``sensors/session.py``.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from humanoid_sim.runtime.abi import plant_abi as abi

FOLLOW_S = 10.0  # how long to wait for the next session of a `serve --persist` host


def plant_now_ns(feedback, mono_now_ns: int) -> int:
    """Where the host's plant is now, from its last feedback write (``plant_abi``: plant time ``t`` is at
    ``written_mono_ns - (sim_time_ns - t)``)."""
    return int(feedback["sim_time_ns"]) + (mono_now_ns - int(feedback["written_mono_ns"]))


def host_scene(record: dict | None) -> tuple[str | None, tuple[float, float], str | None]:
    """The host's boxes file, spawn point and scan mesh, as far as they are readable here."""
    if not record or not Path(record["file"]).is_file():
        return None, (0.0, 0.0), None
    mesh = ((record.get("scan") or {}).get("mesh") or {}).get("file")
    return record["file"], tuple(record["spawn_m"]), mesh if mesh and Path(mesh).is_file() else None


class RuntimeRender:
    """Renders one segment's robot; :meth:`frame` renders once if a render is due."""

    def __init__(self, segment: str | Path, cfg, *, wait_s: float = 60.0):
        from humanoid_sim.model.description import Description
        from humanoid_sim.sensors.session import SensorSession

        self.path = Path(segment)
        desc = Description.load(cfg.robot)
        self._identity = {"robot_id": desc.robot, "description_digest": desc.digest}
        self.seg = abi.Segment.open(self.path, wait_s=wait_s, **self._identity)
        record = self.seg.descriptor()
        scene, spawn, scan_mesh = host_scene(record.get("scene"))
        cfg = dataclasses.replace(cfg, physics=False, scene=cfg.scene or scene,
                                  spawn_m=cfg.spawn_m if cfg.scene else spawn,
                                  scan_mesh=cfg.scan_mesh or scan_mesh)
        self.session = SensorSession(cfg)
        joints = record["joints"]
        self._order = [joints.index(n) for n in desc.joints]  # the description's order, from the host's
        self.session_id = self.seg.session_id
        self._last_t = None
        self.lag_ms: list[float] = []  # plant time now minus the render's, when its frames are out
        self.host_keeping_up: list[bool] = []

    def _read(self, region: str):
        try:
            return self.seg.regions[region].read()
        except abi.StaleRead:
            return None

    def frame(self) -> bool:
        """Render the latest truth if a render is due; False once the session has ended."""
        import torch

        if self.seg.state == abi.STATE_ENDED or self.seg.session_id != self.session_id:
            return False
        truth, fb = self._read("truth"), self._read("feedback")
        if truth is None or fb is None:
            return True
        t_ns = int(truth[1]["sim_time_ns"])
        session = self.session
        if self._last_t is not None and t_ns - self._last_t < session.render_ns:
            return True
        tr = truth[1]
        device = session.sim.plant.cfg.device
        q = np.asarray(tr["joint_pos_rad"], dtype=np.float64)[self._order]
        base = np.concatenate([tr["base_position_m"], tr["base_quat_xyzw"]])
        session.pose(t_ns, torch.as_tensor(q, device=device)[None], torch.as_tensor(base, device=device)[None])
        session.clock_ns = plant_now_ns(fb[1], abi.mono_ns())
        dt = session.render_ns if self._last_t is None else t_ns - self._last_t
        session.render_at(t_ns, dt)
        self._last_t = t_ns
        self.lag_ms.append((plant_now_ns(fb[1], abi.mono_ns()) - t_ns) / 1e6)
        self.host_keeping_up.append(bool(fb[1]["keeping_up"]))
        return True

    def reopen(self, wait_s: float) -> None:
        """Follow the next session at the same path (``serve --persist``)."""
        self.seg.close()
        self.seg = abi.Segment.open(self.path, wait_s=wait_s, **self._identity)
        self.session_id = self.seg.session_id
        self._last_t = None

    def summary(self, wall_s: float) -> dict:
        out = self.session.summary(wall_s)
        lag = np.asarray(self.lag_ms) if self.lag_ms else None
        out["render_rate_hz"] = self.session.counts["renders"] / wall_s
        out["lag_ms"] = None if lag is None else {"median": float(np.median(lag)), "p95": float(np.percentile(lag, 95)),
                                                   "max": float(lag.max())}
        out["host_keeping_up"] = (sum(self.host_keeping_up) / len(self.host_keeping_up)
                                  if self.host_keeping_up else None)
        return out

    def close(self) -> None:
        self.session.close()
        if self.seg.buf is not None:
            self.seg.close()


def render_runtime(segment: str | Path, cfg, *, seconds: float | None = None, wait_s: float = 60.0,
                   summary: str | None = None) -> dict:
    """Render ``segment``'s robot (waiting up to ``wait_s`` for it) until its host is gone or for ``seconds``,
    following a ``serve --persist`` host's next session. The summary is printed (and written to ``summary``) before
    Kit closes, which may end the process."""
    import json

    from humanoid_sim.sensors.rtx import start_kit

    kit = start_kit(cfg.device)
    render = RuntimeRender(segment, cfg, wait_s=wait_s)
    start = time.monotonic()
    end = None if seconds is None else start + seconds
    try:
        while end is None or time.monotonic() < end:
            if not render.frame():
                try:
                    render.reopen(FOLLOW_S)
                except (abi.AbiError, FileNotFoundError):
                    break  # the host is gone
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        out = render.summary(time.monotonic() - start)
        text = json.dumps(out, indent=1)
        if summary:
            Path(summary).write_text(text)
        print(text, flush=True)
        render.close()
        kit.close()
    return out


__all__ = ["RuntimeRender", "host_scene", "plant_now_ns", "render_runtime"]
