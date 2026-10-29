"""Complete a semantic joint pose with the native USD linkage coordinates.

The four crank angles come from the shared description model. Each rod's open
ball joint takes the shortest rotation that closes its other ball onto the foot.
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
    def __init__(self, desc, *, device="cpu", dtype=torch.float64):
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

    def _values(self, q):
        values = {name: q[:, i] for i, name in enumerate(self.names)}
        for cols, names, tr in self.ankles:
            cranks = tr.cranks(q[:, cols])
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

    def coordinates(self, q: torch.Tensor) -> tuple[dict, dict]:
        """Scalar joint positions and open ball quaternions, for a batch of body poses."""
        q = q.to(device=self.device, dtype=self.dtype)
        values = self._values(q)
        balls = {}
        for root, tree, rods in self.closure_trees:
            poses = self._poses(q, values, root, tree)
            for name, parent, foot, f0, rod_vector, foot_point in rods:
                joint = poses[parent] @ f0
                target = (poses[foot][:, :3, :3] @ foot_point) + poses[foot][:, :3, 3]
                wanted = (joint[:, :3, :3].transpose(-1, -2) @ (target - joint[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
                balls[name] = _quat_between(rod_vector, wanted)
        return values, balls

    def body_transforms(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """All rigid-body transforms in the root frame, with rods closed."""
        _, poses = self.forward(q)
        _, balls = self.coordinates(q)
        for name, parent, _, f0, _, _ in self.rods:
            quat = balls[name]
            # Batched quaternion rotation matrix; unlike Rodrigues, no division at zero.
            xyz, w = quat[:, :3], quat[:, 3:]
            x, y, z = xyz.unbind(-1)
            zeros = torch.zeros_like(x)
            cross = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], -1).reshape(-1, 3, 3)
            motion = torch.eye(4, device=q.device, dtype=self.dtype).expand(q.shape[0], -1, -1).clone()
            motion[:, :3, :3] = torch.eye(3, device=q.device, dtype=self.dtype) + 2 * w[:, :, None] * cross + 2 * cross @ cross
            rod, inv_f1 = self.rod_frames[name]
            poses[rod] = poses[parent] @ f0 @ motion @ inv_f1
        return poses

    def dofs(self, q: torch.Tensor, qd: torch.Tensor) -> tuple[dict, dict]:
        """Isaac Lab DOF-space positions and velocities, including each ball's three axes."""
        q = q.to(device=self.device, dtype=self.dtype)
        qd = qd.to(device=self.device, dtype=self.dtype)
        # JVP supplies the primal coordinates as well as every scalar/ball rate;
        # avoid repeating the position solve and a separate crank Jacobian.
        (values, balls), (rates, derivatives) = torch.func.jvp(self.coordinates, (q,), (qd,))
        for name, quat in balls.items():
            position = _rotvec(quat)
            velocity = 2 * _relative_quat(derivatives[name], quat)[..., :3]
            for i in range(3):
                values[f"{name}:{i}"], rates[f"{name}:{i}"] = position[:, i], velocity[:, i]
        return values, rates
