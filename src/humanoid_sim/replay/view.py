"""Rerun recording of a replay: the simulated robot with the real robot as a ghost, plus every drive signal.

3D: ``/world/sim`` (opaque) and ``/world/real`` (translucent orange), same fixed base, so any disagreement
shows as the ghost separating from the robot. Signals: ``/sim/0/drives/<drive>/{position,velocity,torque}``,
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


def view_replay(trace_dir: Path, usd: Path, rrd: str | None, *, fps: float = 50.0, robot: str = "ethrc_v1",
                connect: str | None = None) -> None:
    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.description import Description
    from humanoid_sim.model.joint_properties import joint_properties
    from humanoid_sim.physics.newton_direct import NewtonCfg, NewtonRobot
    from humanoid_sim.replay.replay import ReplayResult
    from humanoid_sim.sim import SimCore

    tr = np.load(trace_dir / "trace.npz")
    summary = json.loads((trace_dir / "summary.json").read_text())
    drives, t_ns = summary["drives"], tr["t_ns"]
    desc = Description.load(robot)
    joints = list(desc.joints)
    plant = PlantCore(desc, joints, 1, PlantCfg(robot=robot, dtype=torch.float64))
    kin = NewtonRobot(usd, joint_properties(plant, joints),
                      NewtonCfg(num_envs=1, fixed_base=True, ground=False, kinematics_only=True))

    stride = max(1, int(round(NS_PER_S / fps / summary["period_ns"])))
    frames = np.arange(0, len(t_ns), stride)
    sim1 = SimCore("newton", usd, robot=robot, fixed_base=True)
    res = ReplayResult(summary["dataset_id"], summary["period_ns"], drives, t_ns[frames], {},
                       {"position": tr["real_position"][frames]}, tr["demand"][frames])
    real_q = res.real_joint_q(sim1)

    def body_poses(qs):
        out = []
        for q in qs:
            kin.write_state(torch.as_tensor(q)[None], torch.zeros(1, len(q), dtype=torch.float64))
            out.append(kin.state0.body_q.numpy().copy())
        return np.stack(out)

    log = SignalLog(f"humanoid-sim replay {summary['name']}", rrd=rrd, init=True)
    if connect:
        log.rr.connect_grpc(connect)
    rel = t_ns - t_ns[0]
    meshes = RobotMeshes(kin.model)
    meshes.log_static("/world/sim", SIM_COLOR)
    meshes.log_static("/world/real", REAL_COLOR)
    meshes.log_frames("/world/sim", rel[frames], body_poses(tr["sim_joint_q"][frames]))
    meshes.log_frames("/world/real", rel[frames], body_poses(real_q))
    for i, d in enumerate(drives):
        for key, src in (("position", "position"), ("velocity", "velocity"), ("applied", "torque")):
            log.series(f"/sim/0/drives/{d}/{src}", rel, tr[f"sim_{key}"][:, i])
        for key in ("position", "velocity", "torque"):
            log.series(f"/real/drives/{d}/{key}", rel, tr[f"real_{key}"][:, i])
        log.series(f"/demand/drives/{d}/position", rel, tr["demand"][:, i])
        log.series(f"/residual/0/drives/{d}/position", rel, tr["sim_position"][:, i] - tr["real_position"][:, i])
