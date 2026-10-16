"""Rerun recording of a replay: the simulated robot with the real robot as a ghost, plus every drive signal.

3D: ``/world/sim`` (opaque) and ``/world/real`` (translucent orange) on the same base (fixed, or turning as the
replay prescribed it), so any disagreement shows as the ghost separating from the robot. Signals: ``/sim/0/drives/<drive>/{position,velocity,torque}``,
``/real/drives/<drive>/{position,velocity,torque}``, ``/demand/drives/<drive>/position`` and
``/residual/0/drives/<drive>/position`` (sim − real). Written to an .rrd file (or streamed to a viewer the
caller started); nothing here spawns a viewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.signals.log import SignalLog
from humanoid_sim.signals.robot import RobotMeshes
from humanoid_sim.timing.clock import NS_PER_S

SIM_COLOR = (190, 190, 200, 255)
REAL_COLOR = (255, 140, 0, 90)
# Plot colors: sim grey-blue, real orange (as in 3D), demand green.
SERIES = {"sim": ("sim", (90, 110, 160)), "real": ("real", (255, 140, 0)), "demand": ("demand", (60, 170, 90))}


def blueprint(drives: list[str], joint_of: dict[str, str]):
    """Overlay view, sim-only and real-only views, one plot tab per drive, all residuals, timeline open."""
    import rerun.blueprint as rrb

    plots = [rrb.TimeSeriesView(name=f"{joint_of.get(d, d)}", origin="/",
                                contents=[f"/sim/0/drives/{d}/position", f"/real/drives/{d}/position",
                                          f"/demand/drives/{d}/position"]) for d in drives]
    plots.append(rrb.TimeSeriesView(name="All residuals (sim - real)", origin="/residual"))
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="Overlay: SIM (grey) + REAL (orange ghost)", origin="/world"),
                rrb.Horizontal(
                    rrb.Spatial3DView(name="SIM only", origin="/world", contents="/world/sim/**"),
                    rrb.Spatial3DView(name="REAL only", origin="/world", contents="/world/real/**"),
                ),
                row_shares=[2, 1],
            ),
            rrb.Tabs(*plots),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=False,
    )


def label(root: str, text: str, color, offset_y: float) -> None:
    """A text label riding on the robot's base body, above its head."""
    import rerun as rr

    rr.log(f"{root}/base_link/label", rr.Points3D([[0.0, offset_y, 0.95]], labels=[text], colors=[color[:3]],
                                                  radii=[0.02], show_labels=True), static=True)


def turn_base(poses: np.ndarray, angles: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Body poses ``[F, B, 7]`` (xyz, quaternion xyzw) of a base at ``origin`` turned by yaw, pitch, roll ``[F, 3]``."""
    from scipy.spatial.transform import Rotation

    out = poses.copy()
    for f, a in enumerate(angles):
        r = Rotation.from_euler("ZYX", a)
        out[f, :, :3] = r.apply(poses[f, :, :3] - origin) + origin
        out[f, :, 3:] = (r * Rotation.from_quat(poses[f, :, 3:])).as_quat()
    return out


def view_replay(trace_dir: Path, usd: Path, rrd: str | None, *, fps: float = 50.0, robot: str = "ethrc_v1",
                connect: str | None = None) -> None:
    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.description import Description
    from humanoid_sim.model.kinematics import RobotKinematics
    from humanoid_sim.replay.replay import ReplayResult

    tr = np.load(trace_dir / "trace.npz")
    summary = json.loads((trace_dir / "summary.json").read_text())
    drives, t_ns = summary["drives"], tr["t_ns"]
    desc = Description.load(robot)
    joints = list(desc.joints)
    plant = PlantCore(desc, joints, 1, PlantCfg(robot=robot, dtype=torch.float64))
    kin = RobotKinematics(usd, joints)

    stride = max(1, int(round(NS_PER_S / fps / summary["period_ns"])))
    frames = np.arange(0, len(t_ns), stride)
    res = ReplayResult(summary["dataset_id"], summary["period_ns"], drives, t_ns[frames], {},
                       {"position": tr["real_position"][frames]}, tr["demand"][frames])
    real_q = res.real_joint_q(plant)

    base = tr["sim_base_angles"][frames] if "sim_base_angles" in tr else np.zeros((len(frames), 3))

    def body_poses(qs):
        poses = np.stack([kin.body_poses(torch.as_tensor(q)[None]) for q in qs])
        return turn_base(poses, base, kin.base_origin) if base.any() else poses

    joint_of = {d.name: (d.joints[0] if d.crank is None else f"{d.joints[0]}/{d.crank} crank")
                for d in desc.drives.values()}
    bp = blueprint(drives, joint_of)
    log = SignalLog(f"humanoid-sim replay {summary['name']}", rrd=None, init=True)
    if rrd:
        log.rr.save(rrd, default_blueprint=bp)
    if connect:
        log.rr.connect_grpc(connect, default_blueprint=bp)
    rel = t_ns - t_ns[0]
    meshes = RobotMeshes(kin.model)
    meshes.log_static("/world/sim", SIM_COLOR)
    meshes.log_static("/world/real", REAL_COLOR)
    label("/world/sim", "SIM", SIM_COLOR, -0.15)
    label("/world/real", "REAL", (255, 140, 0), 0.15)
    meshes.log_frames("/world/sim", rel[frames], body_poses(tr["sim_joint_q"][frames]))
    meshes.log_frames("/world/real", rel[frames], body_poses(real_q))
    for i, d in enumerate(drives):
        for src_name, path in (("sim", f"/sim/0/drives/{d}/position"), ("real", f"/real/drives/{d}/position"),
                               ("demand", f"/demand/drives/{d}/position")):
            name, color = SERIES[src_name]
            log.rr.log(path, log.rr.SeriesLines(names=[name], colors=[color]), static=True)
        for key, src in (("position", "position"), ("velocity", "velocity"), ("applied", "torque")):
            log.series(f"/sim/0/drives/{d}/{src}", rel, tr[f"sim_{key}"][:, i])
        for key in ("position", "velocity", "torque"):
            log.series(f"/real/drives/{d}/{key}", rel, tr[f"real_{key}"][:, i])
        log.series(f"/demand/drives/{d}/position", rel, tr["demand"][:, i])
        log.series(f"/residual/0/drives/{d}/position", rel, tr["sim_position"][:, i] - tr["real_position"][:, i])
