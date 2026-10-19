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


SOLE_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


def _sole_corners(pose: np.ndarray, sole: dict) -> np.ndarray:
    """World corners ``[8, 3]`` of a sole box (``asset.json`` ``sole_boxes`` entry) on a body at ``pose``."""
    local = np.asarray(sole["center"]) + (SOLE_CORNERS * np.asarray(sole["half_extents"])) @ np.asarray(sole["rotation"]).T
    return _rotate(pose[3:], local) + pose[:3]


def _sole_normal(pose: np.ndarray, sole: dict) -> np.ndarray:
    """World direction of a sole box's normal (its local z axis)."""
    return _rotate(pose[3:], np.asarray(sole["rotation"])[:, 2])


def standing_base_height(usd_path: Path, desc, soles: list[dict], q: torch.Tensor | None = None) -> float:
    """Height of the upright base at which joint positions ``q`` ``[J]`` (default: the standing pose) touch the
    ground with their lowest sole-box corner (``asset.json`` ``sole_boxes``). With the description's standing
    offsets the soles are not level under an upright base (they pitch toes-down by 10° and 17°), so the robot
    stands on their edges; as a rigid pose it is not balanced and tips over within about a second.
    :func:`flat_foot_pose` levels them."""
    joints = list(desc.joints)
    kin = RobotKinematics(usd_path, joints, base_height_m=0.0)
    if q is None:
        q = torch.tensor([desc.joints[n].standing_offset_rad for n in joints], dtype=torch.float64)
    poses = kin.body_poses(q.reshape(1, -1))
    labels = [lbl.rsplit("/", 1)[-1] for lbl in kin.model.body_label]
    return -min(float(_sole_corners(poses[labels.index(sole["body"])], sole)[:, 2].min()) for sole in soles)


def _newton_solve(kin: RobotKinematics, q: torch.Tensor, cols: list[int], residual, *, tol: float,
                  eps: float = 1e-3, iterations: int = 50) -> torch.Tensor:
    """Solve ``residual(poses) = 0`` (one value per column) for the joint positions at ``cols``, by Newton's
    method with a forward-difference Jacobian; ``kin`` must hold ``len(cols) + 1`` envs, all at the origin.
    ``eps`` is large because Newton's forward kinematics is float32."""
    n = len(cols)
    bodies = kin.model.body_count // (n + 1)
    for _ in range(iterations):
        batch = q.repeat(n + 1, 1)
        for k, c in enumerate(cols):
            batch[k + 1, c] += eps
        poses = kin.body_poses(batch)
        r = [np.asarray(residual(poses[e * bodies:(e + 1) * bodies])) for e in range(n + 1)]
        if float(np.abs(r[0]).max()) < tol:
            return q
        jac = np.stack([(r[k + 1] - r[0]) / eps for k in range(n)], axis=1)
        q[cols] -= torch.as_tensor(np.linalg.solve(jac, r[0]), dtype=q.dtype)
    raise RuntimeError("the flat-foot pose did not converge")


def flat_foot_pose(usd_path: Path, desc, soles: list[dict], *, tol: float = 1e-5) -> torch.Tensor:
    """A pose ``[J]`` (description joint order) that stands on flat ground under an upright base: the standing
    pose with each ankle's pitch and roll changed so that its sole box lies level, then the leg whose sole hangs
    lower bent at hip and knee (its sole kept level and at the same place fore and aft) until both soles are
    at the same height. Every other joint keeps its standing offset.

    On the current description the right ankle pitches by 0.30 rad; the left leg shortens by about 15 mm (hip
    +0.085 rad, knee -0.167 rad) and its ankle pitches by 0.26 rad in all, levelled again after the bend. Use it as the default pose of a task that starts on the ground; the description's
    standing pose stays the reference for the drives, the gains and the mirror map."""
    joints = list(desc.joints)
    q = torch.tensor([desc.joints[n].standing_offset_rad for n in joints], dtype=torch.float64)
    labels = [lbl.rsplit("/", 1)[-1] for lbl in RobotKinematics(usd_path, joints, base_height_m=0.0).model.body_label]

    def pose_of(poses, sole):
        return poses[labels.index(sole["body"])]

    kin3 = RobotKinematics(usd_path, joints, num_envs=3, spacing_m=(0.0, 0.0, 0.0), base_height_m=0.0)
    for sole in soles:  # level each sole with its ankle
        cols = [joints.index(f"foot_pitch_{sole['side']}"), joints.index(f"foot_roll_{sole['side']}")]
        q = _newton_solve(kin3, q, cols, lambda poses, sole=sole: _sole_normal(pose_of(poses, sole), sole)[:2],
                          tol=tol)

    kin = RobotKinematics(usd_path, joints, base_height_m=0.0)
    poses = kin.body_poses(q.reshape(1, -1))
    bottom = {s["side"]: float(_sole_corners(pose_of(poses, s), s)[:, 2].min()) for s in soles}
    low = min(soles, key=lambda s: bottom[s["side"]])
    high = max(soles, key=lambda s: bottom[s["side"]])
    if bottom[high["side"]] - bottom[low["side"]] > 1e-4:
        side = low["side"]
        start = pose_of(poses, low)
        along = _rotate(start[3:], np.asarray(low["rotation"])[:, 0])  # the sole's long axis: fore and aft
        along[2] = 0.0
        along /= np.linalg.norm(along)
        center0 = float(_sole_corners(start, low).mean(0) @ along)
        target = bottom[high["side"]]

        def residual(poses):
            pose = pose_of(poses, low)
            corners = _sole_corners(pose, low)
            return [*_sole_normal(pose, low)[:2], corners[:, 2].min() - target, corners.mean(0) @ along - center0]

        kin5 = RobotKinematics(usd_path, joints, num_envs=5, spacing_m=(0.0, 0.0, 0.0), base_height_m=0.0)
        cols = [joints.index(n) for n in (f"leg_flexion_{side}", f"knee_{side}", f"foot_pitch_{side}",
                                          f"foot_roll_{side}")]
        q = _newton_solve(kin5, q, cols, residual, tol=tol)

    for c, name in enumerate(joints):
        lo, hi = desc.calibration_travel(name)
        if not lo <= float(q[c]) <= hi:
            raise ValueError(f"the flat-foot pose needs {name} at {float(q[c]):.3f}, outside [{lo:.3f}, {hi:.3f}]")
    return q


def _rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (``[3]`` or ``[N, 3]``) by the quaternion."""
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))
