"""Robot kinematics on the host: the asset USD in a Newton model, forward kinematics only.

Used by viewers (body poses from joint positions) and model checks. There is no solver here: physics runs
only through Isaac Lab on MJWarp (``physics/isaaclab.py``). MuJoCo never steps the robot in this repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class RobotKinematics:
    def __init__(self, usd_path: Path, joint_names: list[str], *, num_envs: int = 1,
                 spacing_m: tuple[float, float, float] = (3.0, 3.0, 0.0), base_height_m: float = 1.0):
        import newton
        import warp as wp

        robot = newton.ModelBuilder()
        robot.add_usd(str(usd_path), xform=wp.transform((0.0, 0.0, base_height_m), wp.quat_identity()),
                      floating=False)
        labels = [lbl.rsplit("/", 1)[-1] for lbl in robot.joint_label]
        missing = [n for n in joint_names if n not in labels]
        if missing:
            raise RuntimeError(f"USD lacks joints {missing}")
        self.joint_names = list(joint_names)
        self.base_origin = np.array([0.0, 0.0, base_height_m])  # the robot root's position in world 0
        qstart = list(robot.joint_q_start)
        scene = newton.ModelBuilder()
        scene.replicate(robot, num_envs, spacing=spacing_m)
        self.model = scene.finalize(device="cpu")
        self.state = self.model.state()
        per_q = len(robot.joint_q)
        e = np.arange(num_envs)[:, None]
        self._qcols = torch.as_tensor(e * per_q + np.asarray([qstart[labels.index(n)] for n in joint_names])[None])
        self._newton, self._wp = newton, wp

    def body_poses(self, q: torch.Tensor) -> np.ndarray:
        """World poses ``[E * B, 7]`` (position, quaternion xyzw) for joint positions ``q`` ``[E, J]``."""
        jq = self._wp.to_torch(self.model.joint_q).clone()
        jq[self._qcols] = q.to(jq.dtype)
        joint_q = self._wp.from_torch(jq)
        self._newton.eval_fk(self.model, joint_q, self.model.joint_qd, self.state)
        return self.state.body_q.numpy().copy()


def standing_base_height(usd_path: Path, desc, soles: list[dict]) -> float:
    """Height of the upright base at which the standing pose touches the ground with its lowest sole-box corner
    (``asset.json`` ``sole_boxes``). With the description's standing offsets the soles are not level under an
    upright base (they pitch toes-down by 10° and 17°), so the robot stands on their edges; as a rigid pose it
    is not balanced and tips over within about a second."""
    joints = list(desc.joints)
    kin = RobotKinematics(usd_path, joints, base_height_m=0.0)
    q = torch.tensor([[desc.joints[n].standing_offset_rad for n in joints]], dtype=torch.float64)
    poses = kin.body_poses(q)
    labels = [lbl.rsplit("/", 1)[-1] for lbl in kin.model.body_label]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    lowest = np.inf
    for sole in soles:
        pose = poses[labels.index(sole["body"])]
        local = np.asarray(sole["center"]) + (signs * np.asarray(sole["half_extents"])) @ np.asarray(sole["rotation"]).T
        lowest = min(lowest, float((_rotate(pose[3:], local) + pose[:3])[:, 2].min()))
    return -lowest


def _rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (``[3]`` or ``[N, 3]``) by the quaternion."""
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))
