"""Render the rig's sensors along a recorded run (``serve --trajectory``), in lockstep with plant time.

The live run has no render in its loop: the host keeps real time and writes its truth every step to a trajectory file.
This module then poses the robot at each render time (the trajectory interpolated to it) and renders the cameras and
lidars at exactly their nominal rates, as slowly as the GPU needs, into a dataset (:mod:`humanoid_sim.sensors.dataset`)
stamped in plant time. Nothing here steps physics or reads a segment, so a render is repeatable and its speed does not
change what the sensors see. Needs the Isaac Sim image (``docker/Dockerfile.kit``) on an RTX GPU.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from humanoid_sim.runtime.trajectory import Trajectory


def render_times(start_ns: int, end_ns: int, period_ns: int) -> np.ndarray:
    """Plant times ``start, start + period, ...`` up to ``end``."""
    if period_ns <= 0:
        raise ValueError("the rig has no camera or lidar to render")
    return start_ns + period_ns * np.arange((end_ns - start_ns) // period_ns + 1, dtype=np.int64)


def check_trajectory(traj: Trajectory, desc) -> list[int]:
    """The index of each of the description's joints in the trajectory's, refusing a trajectory of another robot."""
    if traj.robot != desc.robot or traj.header["description_digest"] != desc.digest:
        raise ValueError(f"the trajectory is of {traj.robot} (description {traj.header['description_digest'][:12]}), "
                         f"not of {desc.robot} ({desc.digest[:12]}): render it with the description it was recorded on")
    missing = [j for j in desc.joints if j not in traj.joints]
    if missing:
        raise ValueError(f"the trajectory lacks joints {missing}")
    return [traj.joints.index(j) for j in desc.joints]


def render_trajectory(trajectory: str | Path, cfg, out: str | Path, *, start_s: float | None = None,
                      end_s: float | None = None, window: bool = False, log=print) -> dict:
    """Render ``trajectory`` with the sensors of ``cfg`` (a :class:`~humanoid_sim.sensors.session.SessionCfg`) into the
    dataset directory ``out``. ``start_s`` / ``end_s`` bound the plant time (default: the whole run). The room's boxes
    and scan and the crane are the host's, as the trajectory's header recorded them, unless ``cfg`` names its own.
    Returns the summary. A render that produces nothing is an error.

    Kit runs inside a ``with``: Isaac Lab picks the process's exit status from the exception its context is left with,
    which ``ExitStack.close()`` would hide, ending the process with status 0 whatever went wrong."""
    from humanoid_sim.model.description import Description
    from humanoid_sim.runtime.render import host_scene
    from humanoid_sim.sensors.rtx import start_kit
    from humanoid_sim.world.trolley import GantryRig

    traj = Trajectory.load(trajectory)
    desc = Description.load(cfg.robot)
    order = check_trajectory(traj, desc)
    scene, spawn, scan_mesh = host_scene(traj.header.get("scene"))
    cfg = dataclasses.replace(cfg, scene=cfg.scene or scene, spawn_m=cfg.spawn_m if cfg.scene else spawn,
                              scan_mesh=cfg.scan_mesh or scan_mesh)
    host_gantry = traj.header.get("gantry")
    gantry = GantryRig.from_dict(host_gantry) if host_gantry else None
    with start_kit(cfg.device, window=window):
        return _render(traj, Path(trajectory).name, desc, order, cfg, out, start_s, end_s, gantry, log)


def visual_links(urdf: str | Path) -> list[str]:
    """The description's links with a visual mesh: what a ray can hit of the robot, as the asset keeps them."""
    import xml.etree.ElementTree as ET

    return sorted(link.get("name") for link in ET.parse(urdf).getroot().findall("link")
                  if link.find("visual") is not None)


def describe_sensors(writer, session, robot_links: list[str] | None = None) -> None:
    """The manifest's cameras (calibration, mount) and lidars (frame, period, mount), each with what its rays can hit
    besides the room: whether the splat is on the stage, and the robot's links (a sensor sits among the robot's own
    meshes and traces them: the Mid-360's head and shoulders occlude it, as the meshes are)."""
    from humanoid_sim.sensors import feeds

    stage = {"splat": bool(session.cfg.splat), "robot_links_traced": robot_links or []}

    for spec in session.camera_specs:
        name = spec.camera_id.removeprefix("sim-")  # the session's mounts are by the rig's sensor name
        if name not in session.mounts:
            raise KeyError(f"no mount for camera {spec.camera_id!r} (mounts: {sorted(session.mounts)})")
        record = feeds.descriptor(spec, "dataset")
        record.pop("session", None)
        writer.describe_camera(spec.camera_id, {**record, "mount": _mount(session, name), "stage": stage})
    for lidar in session.lidars:
        writer.describe_lidar(lidar.sensor.name, {
            "model": lidar.sensor.model, "product": lidar.sensor.product, "frame": lidar.sensor.frame,
            "period_ns": int(lidar.period_ns), "range_min_m": lidar.sensor.range_min_m,
            "mount": _mount(session, lidar.sensor.name), "stage": stage})


def missing_data(session, counts: dict[str, int]) -> list[str]:
    """The sensors (``<camera_id>/<stream>`` or a lidar's name) of ``session`` that wrote nothing."""
    expected = [f"{c.camera_id}/{s.name}" for c in session.camera_specs for s in c.streams]
    expected += [lidar.sensor.name for lidar in session.lidars]
    return [name for name in expected if not counts.get(name)]


def _render(traj, name, desc, order, cfg, out, start_s, end_s, gantry, log) -> dict:
    import torch

    from humanoid_sim.runtime.render import crane_parts, move_crane
    from humanoid_sim.sensors import feeds
    from humanoid_sim.sensors.dataset import DatasetWriter
    from humanoid_sim.sensors.session import SensorSession

    session = SensorSession(dataclasses.replace(cfg, physics=False, feeds=None, lidar_domain=None, keep_sweeps=16))
    writer = None
    try:
        if not session.camera_specs and not session.lidars:
            raise ValueError(f"nothing to render: {desc.robot}'s rig with sensors={cfg.sensors!r} has no camera or "
                             f"lidar the RTX sensors simulate (unsimulated: {session.unsimulated})")
        if len(session.lidars) > 1:
            raise ValueError("a dataset holds one lidar's sweeps: the session has " + str(len(session.lidars)))
        crane = crane_parts(gantry) if cfg.splat else {}
        t0 = traj.start_ns if start_s is None else max(traj.start_ns, int(start_s * 1e9))
        t1 = traj.end_ns if end_s is None else min(traj.end_ns, int(end_s * 1e9))
        if t1 <= t0:
            raise ValueError(f"nothing to render between {t0} and {t1} ns of the trajectory "
                             f"{traj.start_ns}..{traj.end_ns} ns")
        times = render_times(t0, t1, session.render_ns)
        writer = DatasetWriter(out, {
            "robot": desc.robot, "description_digest": desc.digest, "trajectory": name, "start_ns": int(t0), "end_ns": int(t1), "render_period_ns": int(session.render_ns),
            "unsimulated": session.unsimulated, "stamps": "plant time, ns"})
        describe_sensors(writer, session, visual_links(desc.urdf_path))
        device = session.sim.plant.cfg.device
        due = {(c.camera_id, s.name): t0 for c in session.camera_specs for s in c.streams}
        seen = 0
        wall = time.perf_counter()
        for i, t in enumerate(times):
            tr = traj.at(int(t))
            q = tr["joint_pos_rad"][0][order]
            base = np.concatenate([tr["base_position_m"][0], tr["base_quat_xyzw"][0]])
            session.pose(int(t), torch.as_tensor(q, device=device)[None], torch.as_tensor(base, device=device)[None])
            move_crane(crane, tr["gantry_position_m"][0])
            session.clock_ns = int(t)
            session.render_at(int(t), int(session.render_ns))
            for spec in session.camera_specs:
                for s in spec.streams:
                    if t < due[(spec.camera_id, s.name)]:
                        continue
                    due[(spec.camera_id, s.name)] += int(round(1e9 / s.rate_hz))
                    image = session.last_images[(spec.camera_id, s.name)]
                    writer.camera_frame(spec.camera_id, s.name, int(t),
                                        feeds.depth_mm(image, s.range_m) if s.kind == "depth" else image)
            total = session.counts["lidar_sweeps"]
            if total > seen:
                for sweep in list(session.sweeps)[max(0, len(session.sweeps) - (total - seen)):]:
                    writer.lidar_sweep(session.lidars[0].sensor.name, sweep)
            seen = total
            if i % 100 == 0:
                log(f"rendered {i + 1}/{len(times)} (plant {(t - t0) / 1e9:.2f} s)")
        wall = time.perf_counter() - wall
        counts = writer.counts()
        empty = missing_data(session, counts)
        if empty:
            raise RuntimeError(f"the render produced no data for {empty} (frames and sweeps written: {counts})")
        summary = {"renders": len(times), "plant_s": float((t1 - t0) / 1e9), "wall_s": wall, "counts": counts,
                   "render_ms": _median(session.ms["render"])}
        # Written and printed before Kit closes, which may end the process.
        (Path(out) / "summary.json").write_text(json.dumps(summary, indent=1))
        log(json.dumps(summary, indent=1))
        return summary
    finally:
        if writer is not None:
            writer.close()
        session.close()


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mount(session, name: str) -> dict:
    m = session.mounts[name]
    return {"body": m.link, "sensor_link": m.sensor_link, "translation_m": list(m.translation_m),
            "rotation": [list(row) for row in m.rotation]}
