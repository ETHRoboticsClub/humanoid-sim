"""Complete a semantic joint pose with the native USD linkage coordinates.

The four crank angles come from the shared description model. Each rod's open
joint closes its other ball onto the foot: a ball joint by the shortest rotation, a
universal joint (profile ``model.rod_joints: universal``) by its two angles.
This is reset/kinematics work only; the solver owns every coordinate during physics.
"""

from __future__ import annotations

import numpy as np
import torch


def _rotation(axis, angle):
    x, y, z = axis.unbind()
    zero = x * 0
    cross = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
    c, s = torch.cos(angle)[..., None, None], torch.sin(angle)[..., None, None]
    return c * torch.eye(3, device=axis.device, dtype=axis.dtype) + (1 - c) * torch.outer(axis, axis) + s * cross


def _quat_between(a, b):
    a, b = torch.nn.functional.normalize(a, dim=-1), torch.nn.functional.normalize(b, dim=-1)
    xyz = torch.linalg.cross(a.expand_as(b), b)
    w = 1 + (a * b).sum(-1, keepdim=True)
    fallback_axis = torch.nn.functional.one_hot(a.abs().argmin(-1), 3).to(a)
    fallback = torch.linalg.cross(a, fallback_axis).expand_as(b)
    xyz = torch.where(w < 1e-10, fallback, xyz)
    return torch.nn.functional.normalize(torch.cat([xyz, w], dim=-1), dim=-1)


def universal_rod_frame(rod_vector) -> np.ndarray:
    """Rotation of a rod's universal-joint frame within its ball-joint frame: Z along the rod (ball to ball), X
    and Y across it, the shortest rotation from the ball frame. Shared by the asset build and the reset."""
    z = np.asarray(rod_vector, dtype=np.float64)
    z = z / np.linalg.norm(z)
    axis, c = np.cross([0.0, 0.0, 1.0], z), z[2]
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _rotvec(quat):
    quat = torch.where(quat[..., 3:] < 0, -quat, quat)
    length = quat[..., :3].norm(dim=-1, keepdim=True)
    scale = torch.where(length > 1e-10, 2 * torch.atan2(length, quat[..., 3:]) / length.clamp(min=1e-10),
                        torch.full_like(length, 2.0))
    return quat[..., :3] * scale


def _relative_quat(a, b):
    """a * inverse(b), xyzw."""
    av, aw, bv, bw = a[..., :3], a[..., 3:], -b[..., :3], b[..., 3:]
    return torch.cat([aw * bv + bw * av + torch.linalg.cross(av, bv), aw * bw - (av * bv).sum(-1, keepdim=True)], -1)


class NativeState:
    def __init__(self, desc, *, device="cpu", dtype=torch.float64, compile_batch_size: int | None = None):
        from pxr import UsdPhysics

        self.names = list(desc.joints)
        self.device, self.dtype = torch.device(device), dtype
        source = desc.usd_model
        if source is None:
            raise ValueError("native linkage state requires a USD description")

        def tensor(x):
            return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

        self.ankles = []
        for lk in desc.linkages:
            spec = desc.ankle_model.payload["sides"][lk.side]
            self.ankles.append(([self.names.index(n) for n in lk.joints],
                                [spec["usd_joints"][r] for r in ("upper", "lower")],
                                desc.ankle_transmission(lk.side, device=device, dtype=dtype)))
        pending, balls, closing = [], {}, []
        for name, joint in source.joints.items():
            parent, child = source.joint_bodies(name)
            f0, f1 = source.joint_frames(name)
            if joint.GetExcludeFromArticulationAttr().Get():
                closing.append((name, parent, child, f0, f1))
            elif joint.GetPrim().IsA(UsdPhysics.SphericalJoint):
                balls[child] = (name, parent, f0, f1)
            else:
                axis = source.joint_axis(name) if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint) else None
                pending.append((name, parent, child, tensor(f0), tensor(np.linalg.inv(f1)),
                                None if axis is None else tensor(axis)))
        self.root = source.root_link
        self.tree = []
        known = {self.root}
        while pending:
            ready = [row for row in pending if row[1] in known]
            if not ready:
                raise ValueError(f"USD has unsupported branches behind spherical joints: {[r[0] for r in pending]}")
            for row in ready:
                self.tree.append(row)
                known.add(row[2])
                pending.remove(row)
        self.rods = []
        self.rod_frames = {}
        for name, a, b, ca, cb in closing:
            if a in balls and b in known:
                rod, foot, rod_frame, foot_frame = a, b, ca, cb
            elif b in balls and a in known:
                rod, foot, rod_frame, foot_frame = b, a, cb, ca
            else:
                raise ValueError(f"{name}: unsupported loop closure, expected a rod and foot")
            ball, parent, f0, f1 = balls[rod]
            self.rod_frames[ball] = (rod, tensor(np.linalg.inv(f1)))
            vector = np.linalg.inv(f1) @ np.r_[rod_frame[:3, 3], 1.0]
            self.rods.append((ball, parent, foot, tensor(f0), tensor(vector[:3]), tensor(foot_frame[:3, 3])))

        # A rod's orientation depends only on the paths below the common
        # ancestor of its shaft and foot. Upstream rigid motion cancels. Build
        # those small trees once, sharing branches between rods in each tree.
        parents = {row[2]: row[1] for row in self.tree}

        def ancestry(body):
            path = [body]
            while body in parents:
                body = parents[body]
                path.append(body)
            return path

        groups = {}
        for rod in self.rods:
            _, parent, foot, *_ = rod
            parent_path, foot_path = ancestry(parent), ancestry(foot)
            root = next(body for body in parent_path if body in foot_path)
            children, rods = groups.setdefault(root, (set(), []))
            children.update(parent_path[:parent_path.index(root)])
            children.update(foot_path[:foot_path.index(root)])
            rods.append(rod)
        self.closure_trees = [(root, [row for row in self.tree if row[2] in children], rods)
                              for root, (children, rods) in groups.items()]
        from humanoid_sim.model.asset import load_profile

        kind = load_profile(desc.robot)["model"].get("rod_joints", "spherical")
        if kind not in ("spherical", "universal"):
            raise ValueError(f"rod_joints {kind!r}: spherical or universal")
        # ball name -> its universal frame within the ball frame; balls not listed stay spherical
        self.universal = ({name: tensor(universal_rod_frame(vector.cpu().numpy()))
                           for name, _, _, _, vector, _ in self.rods} if kind == "universal" else {})
        if compile_batch_size is not None and compile_batch_size <= 0:
            raise ValueError("compile_batch_size must be positive")
        self.compile_batch_size = compile_batch_size
        self._dofs_run = (torch.compile(self._dofs, fullgraph=True, dynamic=False)
                          if compile_batch_size is not None else self._dofs)

    def _values(self, q):
        values = {name: q[:, i] for i, name in enumerate(self.names)}
        for cols, names, tr in self.ankles:
            cranks = tr.shafts(q[:, cols])
            values.update({name: cranks[:, i] for i, name in enumerate(names)})
        return values

    def _poses(self, q, values, root, tree):
        eye = torch.eye(4, dtype=q.dtype, device=q.device).expand(q.shape[0], -1, -1)
        poses = {root: eye}
        for name, parent, child, f0, inv_f1, axis in tree:
            motion = eye.clone()
            if name in values and axis is not None:
                motion[:, :3, :3] = _rotation(axis, values[name])
            poses[child] = poses[parent] @ f0 @ motion @ inv_f1
        return poses

    def forward(self, q: torch.Tensor) -> tuple[dict, dict]:
        """Scalar joint positions and body transforms along the non-rod branches."""
        q = q.to(device=self.device, dtype=self.dtype)
        values = self._values(q)
        return values, self._poses(q, values, self.root, self.tree)

    def closing(self, name: str, rod_vector: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
        """The open rod joint's coordinates that turn ``rod_vector`` onto ``wanted`` (both in its parent frame):
        a ball's quaternion (xyzw), or a universal joint's angles about its X, then its rotated Y."""
        frame = self.universal.get(name)
        if frame is None:
            return _quat_between(rod_vector, wanted)
        w = torch.nn.functional.normalize(wanted, dim=-1) @ frame  # in the universal frame, rest along Z
        return torch.stack([torch.atan2(-w[:, 1], w[:, 2]), torch.asin(w[:, 0].clamp(-1, 1))], -1)

    def closing_dofs(self, name: str, coordinates: torch.Tensor, rates: torch.Tensor):
        """Per-DOF positions and velocities ``[(position, velocity), ...]`` of an open rod joint."""
        if name in self.universal:
            return [(coordinates[:, i], rates[:, i]) for i in range(2)]
        position = _rotvec(coordinates)
        velocity = 2 * _relative_quat(rates, coordinates)[..., :3]
        return [(position[:, i], velocity[:, i]) for i in range(3)]

    def coordinates(self, q: torch.Tensor) -> tuple[dict, dict]:
        """Scalar joint positions and each open rod joint's coordinates (:meth:`closing`), for a batch of poses."""
        q = q.to(device=self.device, dtype=self.dtype)
        values = self._values(q)
        balls = {}
        for root, tree, rods in self.closure_trees:
            poses = self._poses(q, values, root, tree)
            for name, parent, foot, f0, rod_vector, foot_point in rods:
                joint = poses[parent] @ f0
                target = (poses[foot][:, :3, :3] @ foot_point) + poses[foot][:, :3, 3]
                wanted = (joint[:, :3, :3].transpose(-1, -2) @ (target - joint[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
                balls[name] = self.closing(name, rod_vector, wanted)
        return values, balls

    def body_transforms(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """All rigid-body transforms in the root frame, with rods closed."""
        _, poses = self.forward(q)
        _, balls = self.coordinates(q)
        for name, parent, _, f0, _, _ in self.rods:
            motion = torch.eye(4, device=q.device, dtype=self.dtype).expand(q.shape[0], -1, -1).clone()
            frame = self.universal.get(name)
            if frame is None:
                quat = balls[name]
                # Batched quaternion rotation matrix; unlike Rodrigues, no division at zero.
                xyz, w = quat[:, :3], quat[:, 3:]
                x, y, z = xyz.unbind(-1)
                zeros = torch.zeros_like(x)
                cross = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], -1).reshape(-1, 3, 3)
                eye = torch.eye(3, device=q.device, dtype=self.dtype)
                motion[:, :3, :3] = eye + 2 * w[:, :, None] * cross + 2 * cross @ cross
            else:
                ex, ey = torch.eye(3, device=q.device, dtype=self.dtype)[:2]
                about_x, about_y = balls[name].unbind(-1)
                motion[:, :3, :3] = frame @ _rotation(ex, about_x) @ _rotation(ey, about_y) @ frame.T
            rod, inv_f1 = self.rod_frames[name]
            poses[rod] = poses[parent] @ f0 @ motion @ inv_f1
        return poses

    def dofs(self, q: torch.Tensor, qd: torch.Tensor) -> tuple[dict, dict]:
        """Isaac Lab DOF-space positions and velocities, including each open rod joint's axes."""
        if self.compile_batch_size is None:
            return self._dofs_run(q, qd)
        rows = q.shape[0]
        if rows > self.compile_batch_size:
            raise ValueError("reset batch exceeds compile_batch_size")
        # JVP specializes on batch size even with dynamic compilation. Pad to the
        # plant's fixed capacity so changing reset subsets reuse the same graph.
        padding = (0, 0, 0, self.compile_batch_size - rows)
        values, rates = self._dofs_run(torch.nn.functional.pad(q, padding),
                                       torch.nn.functional.pad(qd, padding))
        return ({name: value[:rows] for name, value in values.items()},
                {name: rate[:rows] for name, rate in rates.items()})

    def _dofs(self, q: torch.Tensor, qd: torch.Tensor) -> tuple[dict, dict]:
        q = q.to(device=self.device, dtype=self.dtype)
        qd = qd.to(device=self.device, dtype=self.dtype)
        # JVP supplies the primal coordinates as well as every scalar/ball rate;
        # avoid repeating the position solve and a separate crank Jacobian.
        (values, balls), (rates, derivatives) = torch.func.jvp(self.coordinates, (q,), (qd,))
        for name, coordinates in balls.items():
            for i, (position, velocity) in enumerate(self.closing_dofs(name, coordinates, derivatives[name])):
                values[f"{name}:{i}"], rates[f"{name}:{i}"] = position, velocity
        return values, rates
