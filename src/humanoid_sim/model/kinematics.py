"""Robot kinematics on the host: the asset USD in a Newton model, forward kinematics only.

Used by viewers (body poses from joint positions) and model checks. There is no solver here: physics runs
only through Isaac Lab on Newton's solvers (``physics/isaaclab.py``). MuJoCo never steps the robot in this repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class RobotKinematics:
    def __init__(self, usd_path: Path, joint_names: list[str], *, num_envs: int = 1,
                 spacing_m: tuple[float, float, float] = (3.0, 3.0, 0.0), base_height_m: float = 1.0,
                 self_collision: bool | None = None):
        """``self_collision``: whether the robot's own shapes collide in :attr:`model`, over what the USD authors
        (the scene sets it at spawn, ``physics.self_collision``); None: as the USD authors it."""
        import newton
        import warp as wp

        source = str(usd_path)
        if self_collision is not None:
            from pxr import Usd

            source = Usd.Stage.Open(source)
            source.SetEditTarget(source.GetSessionLayer())
            for prim in source.Traverse():
                if prim.HasAttribute("newton:selfCollisionEnabled"):
                    prim.GetAttribute("newton:selfCollisionEnabled").Set(self_collision)
        robot = newton.ModelBuilder()
        robot.add_usd(source, xform=wp.transform((0.0, 0.0, base_height_m), wp.quat_identity()),
                      floating=False, **({} if self_collision is None else {"enable_self_collisions": self_collision}))
        labels = [lbl.rsplit("/", 1)[-1] for lbl in robot.joint_label]
        missing = [n for n in joint_names if n not in labels]
        if missing:
            raise RuntimeError(f"USD lacks joints {missing}")
        self.joint_names = list(joint_names)
        self.num_envs = num_envs
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
        self._native = None
        self._native_columns = {}
        from humanoid_sim.model.asset import asset_meta

        try:
            meta = asset_meta(usd_path)
        except FileNotFoundError:
            meta = {}
        if meta.get("ankle_representation") == "native":
            from humanoid_sim.model.description import Description
            from humanoid_sim.model.native_state import NativeState

            self._native = NativeState(Description.load(meta["robot"]))
            for i, name in enumerate(labels):
                width = 4 if robot.joint_type[i] == newton.JointType.BALL else 1
                self._native_columns[name] = torch.as_tensor(
                    np.arange(num_envs)[:, None] * per_q + qstart[i] + np.arange(width)[None])

    def body_poses(self, q: torch.Tensor) -> np.ndarray:
        """World poses ``[E * B, 7]`` (position, quaternion xyzw) for joint positions ``q`` ``[E, J]``."""
        jq = self._wp.to_torch(self.model.joint_q).clone()
        jq[self._qcols] = q.to(jq.dtype)
        if self._native is not None:
            ordered = q[:, [self.joint_names.index(n) for n in self._native.names]]
            values, balls = self._native.coordinates(ordered)
            for name, value in values.items():
                jq[self._native_columns[name]] = value[:, None].to(jq.dtype)
            for name, value in balls.items():
                jq[self._native_columns[name]] = value.to(jq.dtype)
        joint_q = self._wp.from_torch(jq)
        self._newton.eval_fk(self.model, joint_q, self.model.joint_qd, self.state)
        return self.state.body_q.numpy().copy()


SOLE_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


def _sole_corners(pose: np.ndarray, sole: dict) -> np.ndarray:
    """World corners ``[8, 3]`` of a sole box (``asset.json`` ``sole_boxes`` entry) on a body at ``pose``."""
    local = np.asarray(sole["center"]) + (SOLE_CORNERS * np.asarray(sole["half_extents"])) @ np.asarray(sole["rotation"]).T
    return _rotate(pose[3:], local) + pose[:3]


def standing_base_height(usd_path: Path, desc, q: torch.Tensor | None = None) -> float:
    """Height of the upright base at which joint positions ``q`` ``[J]`` (default: the standing pose) touch the
    ground with their lowest collision point: a sole box's corner, a foot sphere's bottom, a hull's vertex."""
    joints = list(desc.joints)
    kin = RobotKinematics(usd_path, joints, base_height_m=0.0)
    if q is None:
        q = torch.tensor([desc.joints[n].standing_offset_rad for n in joints], dtype=torch.float64)
    return -lowest_collision_z(kin, kin.body_poses(q.reshape(1, -1)))


def lowest_collision_z(kin: RobotKinematics, poses: np.ndarray) -> float:
    """World height of the lowest point of env 0's collision shapes, with its bodies at ``poses`` (``body_poses``)."""
    body, local, radius = _collision_points(kin)
    world = np.stack([_rotate(poses[b, 3:], p) for b, p in zip(body, local, strict=True)]) + poses[body, :3]
    return float((world[:, 2] - radius).min())


def _collision_points(kin: RobotKinematics):
    """Points bounding env 0's collision shapes from below: body index ``[N]``, the point in the body frame
    ``[N, 3]``, and the radius around it ``[N]`` (spheres and capsules). Cached on ``kin``."""
    if getattr(kin, "_points", None) is not None:
        return kin._points
    import newton

    m = kin.model
    bodies = m.body_count // kin.num_envs
    kind, flags, scale = m.shape_type.numpy(), m.shape_flags.numpy(), m.shape_scale.numpy()
    xform, owner = m.shape_transform.numpy(), m.shape_body.numpy()
    body_ids, points, radii = [], [], []
    for i in range(m.shape_count):
        b = int(owner[i])
        if not flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES) or b < 0 or b >= bodies:
            continue
        sx, sy, sz = scale[i]
        if kind[i] == newton.GeoType.SPHERE:
            pts, r = np.zeros((1, 3)), sx
        elif kind[i] == newton.GeoType.BOX:
            pts, r = SOLE_CORNERS * np.array([sx, sy, sz]), 0.0
        elif kind[i] in (newton.GeoType.CAPSULE, newton.GeoType.CYLINDER):
            pts, r = np.array([[0.0, 0.0, -sy], [0.0, 0.0, sy]]), sx  # radius, half height along local z
        elif kind[i] in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            pts, r = np.asarray(m.shape_source[i].vertices, dtype=np.float64) * scale[i], 0.0
        else:
            continue
        pos, quat = xform[i][:3], xform[i][3:]
        local = _rotate(quat, pts) + pos
        body_ids += [b] * len(local)
        points.append(local)
        radii += [r] * len(local)
    kin._points = (np.asarray(body_ids), np.concatenate(points), np.asarray(radii))
    return kin._points


def _rotate(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (``[3]`` or ``[N, 3]``) by the quaternion."""
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))
