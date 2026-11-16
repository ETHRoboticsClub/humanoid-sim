"""Disturbances a training turns: force pushes on a body and torque perturbations at joints.

A push of ``magnitude_n`` (uniform) on ``body`` lasts ``duration_s`` (uniform); the next one starts ``interval_s``
(uniform) after it ends. Forces go through the articulation's permanent wrench composer (graph-safe, no host
syncs).
Call :meth:`ForcePush.step` once per physics or policy step with the plant time; :meth:`ForcePush.reset` for the
envs being reset. A level in [0, 1] scales the magnitude (curricula).

:class:`JointTorquePerturbation` holds a random torque on each selected joint for a random time, as an equal and
opposite pair on the joint's child and parent links. Both go through :class:`WrenchMixer`, which sums the
disturbances of one articulation into its permanent wrench composer, so they combine on shared bodies.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from humanoid_sim.timing.clock import NS_PER_S, ns_from_s


class _Slot:
    """One disturbance's wrench on its bodies, ``[E, K, 3]``, written in place before :meth:`WrenchMixer.flush`."""

    def __init__(self, bodies: torch.Tensor, num_envs: int):
        self.bodies = bodies
        self.force = torch.zeros(num_envs, len(bodies), 3, device=bodies.device)
        self.torque = torch.zeros_like(self.force)


class WrenchMixer:
    """The sum of every disturbance's wrench on one articulation, written to its permanent wrench composer.

    The composer's ``set_forces_and_torques_index`` first zeroes every body of the envs it is given, then writes, so
    two disturbances setting their own wrenches erase each other. Each disturbance holds a slot instead; a flush
    writes the sum over slots on every body any of them touches (other bodies stay as they are) for all envs.
    :func:`wrench_mixer` gives the articulation's mixer. Anything else that sets the permanent composer on these
    bodies still replaces the sum, and the next flush restores it.
    """

    def __init__(self, robot):
        self.robot = robot
        self.num_envs = robot.num_instances
        self.device = torch.device(robot.device)
        e, b = self.num_envs, len(robot.body_names)
        self._force = torch.zeros(e, b, 3, device=self.device)
        self._torque = torch.zeros_like(self._force)
        self._all = torch.arange(e, device=self.device)
        self._slots: list[_Slot] = []
        self._touched = torch.zeros(0, dtype=torch.long, device=self.device)

    def slot(self, bodies: Sequence[int]) -> _Slot:
        """A new slot on ``bodies`` (distinct body indices)."""
        ids = torch.as_tensor(list(bodies), dtype=torch.long, device=self.device)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"a wrench slot needs distinct bodies, got {ids.tolist()}")
        slot = _Slot(ids, self.num_envs)
        self._slots.append(slot)
        union = sorted({b for s in self._slots for b in s.bodies.tolist()})
        self._touched = torch.tensor(union, dtype=torch.long, device=self.device)
        return slot

    def flush(self) -> None:
        self._force.zero_()
        self._torque.zero_()
        for slot in self._slots:
            self._force.index_add_(1, slot.bodies, slot.force)
            self._torque.index_add_(1, slot.bodies, slot.torque)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._force[:, self._touched], torques=self._torque[:, self._touched], body_ids=self._touched,
            env_ids=self._all, is_global=True)


def wrench_mixer(robot) -> WrenchMixer:
    """The articulation's :class:`WrenchMixer`, created on first use."""
    mixer = getattr(robot, "_wrench_mixer", None)
    if mixer is None:
        mixer = robot._wrench_mixer = WrenchMixer(robot)
    return mixer


class VelocityPush:
    """Periodically set root horizontal velocity, preserving vertical and angular velocity.

    The caller chooses the interval and uniform velocity range. Times are ns;
    masks and writes are batched without extracting active indices to the host.
    """

    def __init__(self, robot, *, interval_ns: int, velocity_range_m_s: tuple[float, float], seed: int = 0):
        if interval_ns <= 0 or velocity_range_m_s[0] > velocity_range_m_s[1]:
            raise ValueError("velocity push needs a positive interval and ordered velocity range")
        self.robot, self.interval_ns, self.velocity_range = robot, interval_ns, velocity_range_m_s
        self.next_ns = torch.full((robot.num_instances,), interval_ns, device=robot.device, dtype=torch.long)
        self.generator = torch.Generator(device=robot.device).manual_seed(seed)

    def reset(self, ids: torch.Tensor, t_ns: int) -> None:
        self.next_ns[ids] = t_ns + self.interval_ns

    def step(self, t_ns: int) -> None:
        active = self.next_ns <= t_ns
        velocity = self.robot.data.root_com_vel_w.torch.clone()
        lo, hi = self.velocity_range
        drawn = lo + (hi - lo) * torch.rand(velocity.shape[0], 2, device=velocity.device, generator=self.generator)
        velocity[:, :2] = torch.where(active[:, None], drawn, velocity[:, :2])
        # A state write clears body accelerations and invalidates solver caches.
        # Preserve both for environments with no push due this tick.
        self.robot.write_root_velocity_to_sim_mask(root_velocity=velocity, env_mask=active)
        self.next_ns = torch.where(active, t_ns + self.interval_ns, self.next_ns)


class ForcePush:
    def __init__(self, robot, body: str, *, interval_s: tuple[float, float], duration_s: tuple[float, float],
                 magnitude_n: tuple[float, float], level: float = 1.0, seed: int = 0):
        self.robot = robot
        self.body_id = robot.body_names.index(body)
        self.interval, self.duration, self.magnitude = interval_s, duration_s, magnitude_n
        self.level = level
        e, dev = robot.num_instances, torch.device(robot.device)
        self.num_envs, self.device = e, dev
        self.generator = torch.Generator(device=dev).manual_seed(seed)
        self.next_ns = torch.zeros(e, dtype=torch.int64, device=dev)
        self.end_ns = torch.zeros(e, dtype=torch.int64, device=dev)
        self.force = torch.zeros(e, 3, device=dev)
        self._mixer = wrench_mixer(robot)
        self._slot = self._mixer.slot([self.body_id])

    def _uniform(self, n: int, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(n, generator=self.generator, device=self.device)

    def reset(self, env_ids: torch.Tensor, t_ns: int) -> None:
        self.next_ns[env_ids] = t_ns + (self._uniform(len(env_ids), self.interval) * NS_PER_S).to(torch.int64)
        self.end_ns[env_ids] = t_ns
        self.force.index_fill_(0, env_ids, 0.0)

    def step(self, t_ns: int) -> None:
        start = self.next_ns <= t_ns
        heading = self._uniform(self.num_envs, (0.0, 2 * math.pi))
        size = self._uniform(self.num_envs, self.magnitude) * self.level
        push = torch.stack([size * torch.cos(heading), size * torch.sin(heading), torch.zeros_like(size)], dim=1)
        self.force = torch.where(start.unsqueeze(1), push, self.force)
        end = t_ns + (self._uniform(self.num_envs, self.duration) * NS_PER_S).to(torch.int64)
        self.end_ns = torch.where(start, end, self.end_ns)
        gap = (self._uniform(self.num_envs, self.interval) * NS_PER_S).to(torch.int64)
        self.next_ns = torch.where(start, end + gap, self.next_ns)
        self.force = torch.where((self.end_ns <= t_ns).unsqueeze(1), torch.zeros_like(self.force), self.force)
        self._slot.force[:, 0] = self.force
        self._mixer.flush()


@dataclass(frozen=True)
class JointGeometry:
    """Where revolute joints act: parent and child body indices (into ``robot.body_names``) and each joint's axis in
    the child body's frame."""

    parent: torch.Tensor  # [J] long
    child: torch.Tensor  # [J] long
    axis: torch.Tensor  # [J, 3]


def _rotate(quat_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` ``[..., 3]`` by unit quaternions ``[..., 4]`` (x, y, z, w)."""
    xyz, w = quat_xyzw[..., :3], quat_xyzw[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def newton_joint_geometry(robot, names: Sequence[str]) -> JointGeometry:
    """The running Newton model's parent body, child body and axis of the articulation's joints ``names``.

    Newton's joint axis is in the joint frame, which sits at ``joint_X_c`` in the child body, so the child-frame axis
    is the axis turned by that frame's rotation. Raises for a joint that is not revolute or hangs from the world.
    """
    from isaaclab_newton.physics import NewtonManager
    from newton import JointType

    model, view = NewtonManager.get_model(), robot.root_view
    ids = [model.joint_label.index(view.joint_labels[view.joint_names.index(n)]) for n in names]
    kind = model.joint_type.numpy()[ids]
    bad = [n for n, k in zip(names, kind, strict=True) if int(k) != int(JointType.REVOLUTE)]
    if bad:
        raise ValueError(f"joint torque perturbation needs revolute joints; not revolute: {bad}")
    parent = model.joint_parent.numpy()[ids]
    child = model.joint_child.numpy()[ids]
    if (parent < 0).any():
        raise ValueError(f"joints {[n for n, p in zip(names, parent, strict=True) if p < 0]} hang from the world, "
                         "which takes no reaction torque")
    axis = model.joint_axis.numpy()[model.joint_qd_start.numpy()[ids]]
    frame = model.joint_X_c.numpy()[ids]  # [J, 7]: position, quaternion (x, y, z, w)
    dev = torch.device(robot.device)
    axis = _rotate(torch.as_tensor(frame[:, 3:], dtype=torch.float32), torch.as_tensor(axis, dtype=torch.float32))

    def body(i: int) -> int:
        return list(robot.body_names).index(model.body_label[i].rsplit("/", 1)[-1])

    return JointGeometry(torch.tensor([body(i) for i in parent], device=dev),
                         torch.tensor([body(i) for i in child], device=dev), axis.to(dev))


class JointTorquePerturbation:
    """Random torque held on revolute joints, as a pure joint torque: no net force or torque on the robot.

    Per env and joint, the torque is piecewise constant: uniform in ``[-level * max_nm, +level * max_nm]`` (zero with
    probability ``1 - active_share``) for a time uniform in ``hold_s`` (low, high; integer ns), then a new draw.
    ``joints`` are articulation joint names or regexes (full match), in the order given; ``max_nm`` is one number or
    one per selected joint, in the order of :attr:`joint_names`. The torque ``+tau`` turns the child link about the
    joint's axis in the world frame and ``-tau`` turns the parent, through :class:`WrenchMixer`, so it adds to
    :class:`ForcePush` and any other disturbance on the same articulation. The drives do not see it. ``level``
    (float or ``[E]`` tensor) scales draws made from then on.

    :meth:`step` once per physics or policy step with the plant time; :meth:`reset` for envs being reset (zero torque,
    a new draw at the next step). :attr:`torque_nm` ``[E, J]`` is the torque being applied.
    """

    def __init__(self, robot, joints: str | Sequence[str], *, max_nm: float | Sequence[float] | torch.Tensor,
                 hold_s: tuple[float, float], active_share: float, level: float | torch.Tensor = 1.0, seed: int = 0):
        self.robot = robot
        names = self._select(list(robot.joint_names), [joints] if isinstance(joints, str) else list(joints))
        self.joint_names = names
        e, j, dev = robot.num_instances, len(names), torch.device(robot.device)
        self.num_envs, self.device = e, dev
        peak = torch.as_tensor(max_nm, dtype=torch.float32, device=dev)
        peak = peak.expand(j).clone() if peak.ndim == 0 else peak
        if peak.shape != (j,) or not bool((torch.isfinite(peak) & (peak >= 0)).all()):
            raise ValueError(f"max_nm needs one non-negative finite number or one per joint ({j}), got {max_nm!r}")
        lo, hi = ns_from_s(hold_s[0]), ns_from_s(hold_s[1])
        if not 0 < lo <= hi:
            raise ValueError(f"hold_s needs 0 < low <= high (at ns resolution), got {hold_s!r}")
        if not 0.0 <= active_share <= 1.0:
            raise ValueError(f"active_share is a probability, got {active_share!r}")
        self.max_nm, self.hold_ns, self.active_share, self.level = peak, (lo, hi), float(active_share), level
        self.generator = torch.Generator(device=dev).manual_seed(seed)
        geometry = newton_joint_geometry(robot, names)
        self._axis = geometry.axis.to(dev, torch.float32)
        bodies = sorted({*geometry.parent.tolist(), *geometry.child.tolist()})
        local = {b: i for i, b in enumerate(bodies)}
        self._child = torch.tensor([local[b] for b in geometry.child.tolist()], device=dev)
        self._parent = torch.tensor([local[b] for b in geometry.parent.tolist()], device=dev)
        self._child_body = geometry.child.to(dev)
        self.torque_nm = torch.zeros(e, j, device=dev)
        self.next_ns = torch.zeros(e, j, dtype=torch.int64, device=dev)
        self._mixer = wrench_mixer(robot)
        self._slot = self._mixer.slot(bodies)

    @staticmethod
    def _select(available: list[str], patterns: list[str]) -> list[str]:
        chosen: list[str] = []
        for pattern in patterns:
            found = [n for n in available if re.fullmatch(pattern, n)]
            if not found:
                raise ValueError(f"no joint matches {pattern!r}; the articulation has {available}")
            chosen += [n for n in found if n not in chosen]
        return chosen

    def reset(self, env_ids: torch.Tensor, t_ns: int) -> None:
        self.torque_nm.index_fill_(0, env_ids, 0.0)
        self.next_ns.index_fill_(0, env_ids, t_ns)

    def step(self, t_ns: int) -> None:
        shape = self.torque_nm.shape
        due = self.next_ns <= t_ns
        level = torch.as_tensor(self.level, dtype=torch.float32, device=self.device)
        level = level.reshape(-1, 1) if level.ndim else level
        size = (2.0 * torch.rand(shape, generator=self.generator, device=self.device) - 1.0) * self.max_nm * level
        on = torch.rand(shape, generator=self.generator, device=self.device) < self.active_share
        hold = torch.randint(self.hold_ns[0], self.hold_ns[1] + 1, shape, generator=self.generator,
                             device=self.device)
        self.torque_nm = torch.where(due, torch.where(on, size, torch.zeros_like(size)), self.torque_nm)
        self.next_ns = torch.where(due, t_ns + hold, self.next_ns)
        quat = self.robot.data.body_link_pose_w.torch[:, self._child_body, 3:]  # [E, J, 4]
        pair = self.torque_nm.unsqueeze(-1) * _rotate(quat, self._axis.expand_as(quat[..., :3]))  # [E, J, 3]
        self._slot.torque.zero_()
        self._slot.torque.index_add_(1, self._child, pair)
        self._slot.torque.index_add_(1, self._parent, -pair)
        self._mixer.flush()
