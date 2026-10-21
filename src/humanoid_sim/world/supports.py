"""Gantry supports: a rope that holds the robot up, attached to its base or to another of its bodies.

Mirrors Controls' test gantry (``ethrc_sim/support.py`` @ 447b4acb), batched over envs:

* **tension program**: a vertical force, linear in time between breakpoints and held beyond both ends, split
  evenly over points fixed in the harness body's frame (one harness, or a rope per shoulder);
* **slack cable**: one rope from a harness point in the harness body's frame to a world anchor ``overhead_m``
  above where that point was when the cable was hung. It pulls along the rope only beyond its unstretched length
  (also a program in time): ``T = max(0, k · extension + c · extension rate)``.

The harness body is the base unless a rope names another (Controls' ``attachment_body``: the lab holds the G1 by
its torso, at the shoulder straps, so the waist carries the hanging legs).

Programs start when they are hung (times relative to then). Each env has its own rope or none. Forces go
through the articulation's instantaneous wrench composer, which Isaac Lab clears after every physics step, so
call :meth:`Gantry.apply` before every physics step; it adds to the other forces on the base. No host syncs.

The other supports are not ropes: *suspended* is the fixed-base articulation (the base welded to the world, as
in Controls' suspended profile), chosen when the scene is built; *free* is no support. Writing the base pose
back every step does not suspend the robot: the whole robot falls freely within each step, so the joints never
carry its weight.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from humanoid_sim.timing.clock import NS_PER_S

NONE, TENSION, CABLE = 0, 1, 2


def interp_program(t_ns: torch.Tensor, times_ns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear program per env, held beyond both ends. ``times_ns`` ``[E, K]`` nondecreasing (K ≥ 2),
    ``values`` ``[E, K]``, ``t_ns`` ``[E]`` → ``[E]``."""
    t = t_ns.to(torch.int64).unsqueeze(1)
    i = torch.searchsorted(times_ns, t, right=True).clamp(1, times_ns.shape[1] - 1)
    t0, t1 = times_ns.gather(1, i - 1), times_ns.gather(1, i)
    v0, v1 = values.gather(1, i - 1), values.gather(1, i)
    w = ((t - t0).to(values.dtype) / (t1 - t0).clamp(min=1).to(values.dtype)).clamp(0.0, 1.0)
    return (v0 + w * (v1 - v0)).squeeze(1)


def _quat_rotate(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` ``[..., 3]`` by unit quaternions ``[..., 4]`` (x, y, z, w)."""
    xyz, w = q_xyzw[..., :3], q_xyzw[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_rotate_inverse(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(torch.cat([-q_xyzw[..., :3], q_xyzw[..., 3:]], dim=-1), v)


class Gantry:
    """Per-env gantry rope on one of the robot's bodies (``body``, default the articulation root, unless a rope
    names its own)."""

    def __init__(self, robot, *, body: str | None = None, max_points: int = 2, max_breakpoints: int = 8):
        self.robot = robot
        e, dev = robot.num_instances, torch.device(robot.device)
        self.num_envs, self.device = e, dev
        self.body_id = 0 if body is None else robot.body_names.index(body)
        # Each env's harness body, and every body a rope has been hung from (host side, so apply() knows which
        # bodies to push without reading the per-env tensor back).
        self.body_ids = torch.full((e,), self.body_id, dtype=torch.int64, device=dev)
        self._harness_bodies = {self.body_id: torch.tensor([self.body_id], device=dev)}
        self._all = torch.arange(e, device=dev)
        self.mode = torch.zeros(e, dtype=torch.int64, device=dev)
        # Harness-frame rope points [E, P, 3]; `points` of them used per env (tension), point 0 (cable).
        self.points_b = torch.zeros(e, max_points, 3, device=dev)
        self.num_points = torch.ones(e, dtype=torch.int64, device=dev)
        self.anchor_w = torch.zeros(e, 3, device=dev)
        self.stiffness = torch.zeros(e, device=dev)
        self.damping = torch.zeros(e, device=dev)
        # One program per env: tension (N) or unstretched cable length (m) over time.
        k = max_breakpoints
        self.program_ns = torch.arange(k, dtype=torch.int64, device=dev).expand(e, k).clone()
        self.program = torch.zeros(e, k, device=dev)
        self.tension_n = torch.zeros(e, device=dev)  # applied on the last step, total
        self.distance_m = torch.zeros(e, device=dev)  # cable: harness point to anchor
        self.active = False  # host-side; False only when no env has a rope, so apply() can skip the work
        # Host side as well: the most points any rope uses, so apply() pushes only those. Each step's work reuses
        # these buffers: the rope runs on the runtime's real-time path, where every kernel launch counts.
        self._points_used = 0
        self._point_index = torch.arange(max_points, device=dev)
        self._t_ns = torch.zeros(e, dtype=torch.int64, device=dev)
        self._force = torch.zeros(e, max_points, 3, device=dev)

    # -- configuration (host side, at reset or on request) ----------------------------------------

    def _set_program(self, env_ids: torch.Tensor, t_ns: int, breakpoints: Sequence[tuple[float, float]]) -> None:
        k = self.program_ns.shape[1]
        if not 1 <= len(breakpoints) <= k:
            raise ValueError(f"a program needs 1 to {k} breakpoints, got {len(breakpoints)}")
        times = [int(round(float(t) * NS_PER_S)) for t, _ in breakpoints]
        values = [float(v) for _, v in breakpoints]
        if any(t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("program times must be non-negative and increasing")
        # Pad with the last value, 1 ns apart, so every env's program has k breakpoints.
        times += [times[-1] + i + 1 for i in range(k - len(times))]
        values += [values[-1]] * (k - len(values))
        self.program_ns[env_ids] = t_ns + torch.tensor(times, dtype=torch.int64, device=self.device)
        self.program[env_ids] = torch.tensor(values, device=self.device)

    def _harness(self, env_ids: torch.Tensor, body: str | None) -> int:
        """The harness body's index for a rope on ``env_ids``: ``body``, else the default; recorded per env."""
        if body is None:
            index = self.body_id
        elif body in self.robot.body_names:
            index = self.robot.body_names.index(body)
        else:
            raise ValueError(f"no body {body!r} to hang from")
        if index not in self._harness_bodies:
            self._harness_bodies[index] = torch.tensor([index], device=self.device)
        return index

    def _points(self, env_ids: torch.Tensor, body_id: int, attachment_b=None,
                attachment_bodies: Sequence[str] | None = None):
        if attachment_bodies is not None:
            pose = self.robot.data.body_link_pose_w.torch
            root = pose[env_ids, body_id]
            unknown = [b for b in attachment_bodies if b not in self.robot.body_names]
            if unknown:
                raise ValueError(f"no bodies {unknown} to attach to")
            ids = [self.robot.body_names.index(b) for b in attachment_bodies]
            rel = pose[env_ids][:, ids, :3] - root[:, None, :3]
            return _quat_rotate_inverse(root[:, None, 3:].expand(-1, len(ids), -1), rel)
        if attachment_b is None:
            raise ValueError("a rope needs attachment points: attachment_b or attachment_bodies")
        p = torch.as_tensor(attachment_b, dtype=torch.float32, device=self.device).reshape(-1, 3)
        return p.expand(len(env_ids), -1, -1)

    def hang_tension(self, env_ids: torch.Tensor, t_ns: int, breakpoints_s_n: Sequence[tuple[float, float]], *,
                     attachment_b=None, attachment_bodies: Sequence[str] | None = None,
                     body: str | None = None) -> None:
        """Vertical tension over time (s, N) from ``t_ns``, split over ``attachment_b`` points in the harness
        body's frame or over the current positions of ``attachment_bodies`` fixed in that frame (e.g. both
        shoulders). The harness body is ``body``, else the default."""
        if any(float(n) < 0 for _, n in breakpoints_s_n):
            raise ValueError("tension must be non-negative")
        body_id = self._harness(env_ids, body)
        pts = self._points(env_ids, body_id, attachment_b, attachment_bodies)
        if pts.shape[1] > self.points_b.shape[1]:
            raise ValueError(f"at most {self.points_b.shape[1]} attachment points")
        self._set_program(env_ids, t_ns, breakpoints_s_n)
        self.body_ids[env_ids] = body_id
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, : pts.shape[1]] = pts
        self.num_points[env_ids] = pts.shape[1]
        self.mode[env_ids] = TENSION
        self.active = True
        self._points_used = max(self._points_used, pts.shape[1])

    def hang_cable(self, env_ids: torch.Tensor, t_ns: int, *, attachment_b, overhead_m: float,
                   length_breakpoints_s_m: Sequence[tuple[float, float]], stiffness_n_m: float,
                   damping_n_s_m: float, body: str | None = None) -> None:
        """One cable from the harness point ``attachment_b`` (in the frame of ``body``, else the default) to an
        anchor ``overhead_m`` above where that point is now; unstretched length over time (s, m) from ``t_ns``."""
        if overhead_m <= 0 or stiffness_n_m <= 0 or damping_n_s_m < 0:
            raise ValueError("cable overhead distance and stiffness must be positive, damping non-negative")
        if any(float(length) <= 0 for _, length in length_breakpoints_s_m):
            raise ValueError("cable lengths must be positive")
        body_id = self._harness(env_ids, body)
        a = torch.as_tensor(attachment_b, dtype=torch.float32, device=self.device).reshape(3)
        root = self.robot.data.body_link_pose_w.torch[env_ids, body_id]
        point = root[:, :3] + _quat_rotate(root[:, 3:], a.expand(len(env_ids), 3))
        self._set_program(env_ids, t_ns, length_breakpoints_s_m)
        self.body_ids[env_ids] = body_id
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, 0] = a
        self.num_points[env_ids] = 1
        self.anchor_w[env_ids] = point + torch.tensor([0.0, 0.0, overhead_m], device=self.device)
        self.stiffness[env_ids] = stiffness_n_m
        self.damping[env_ids] = damping_n_s_m
        self.mode[env_ids] = CABLE
        self.active = True
        self._points_used = max(self._points_used, 1)

    def release(self, env_ids: torch.Tensor) -> None:
        self.mode.index_fill_(0, env_ids, NONE)
        self.tension_n.index_fill_(0, env_ids, 0.0)
        self.distance_m.index_fill_(0, env_ids, 0.0)
        if len(env_ids) == self.num_envs:
            self.active = False
            self._points_used = 0

    # -- every physics step -----------------------------------------------------------------------

    def apply(self, t_ns: int) -> None:
        """Add this step's rope forces (``t_ns``: the step start, plant time)."""
        if not self.active:
            return
        root = self.robot.data.body_link_pose_w.torch[self._all, self.body_ids]
        vel = self.robot.data.body_link_vel_w.torch[self._all, self.body_ids]
        p = self._points_used
        points_b = self.points_b[:, :p]
        pts_w = root[:, None, :3] + _quat_rotate(root[:, None, 3:].expand(-1, p, -1), points_b)
        value = interp_program(self._t_ns.fill_(t_ns), self.program_ns, self.program)

        # Tension program: vertical, split evenly over the env's points.
        share = torch.where(self.mode == TENSION, value, 0.0) / self.num_points.to(value.dtype)
        force = self._force[:, :p].zero_()
        force[..., 2] = share[:, None] * (self._point_index[None, :p] < self.num_points[:, None])

        # Cable: along the rope from point 0 to the anchor, beyond the unstretched length.
        line = self.anchor_w - pts_w[:, 0]
        dist = line.norm(dim=1)
        direction = line / dist.clamp(min=1e-9)[:, None]
        v_point = vel[:, :3] + torch.cross(vel[:, 3:], pts_w[:, 0] - root[:, :3], dim=1)
        extension = dist - value
        rate = -(direction * v_point).sum(1)
        cable = (self.stiffness * extension + self.damping * rate).clamp(min=0.0)
        cable = torch.where((self.mode == CABLE) & (extension > 0), cable, 0.0)
        force[:, 0] += direction * cable[:, None]

        self.tension_n = torch.where(self.mode == CABLE, cable, share * self.num_points.to(value.dtype))
        self.distance_m = torch.where(self.mode == CABLE, dist, 0.0)
        composer = self.robot.instantaneous_wrench_composer
        one_body = len(self._harness_bodies) == 1
        for b, body in self._harness_bodies.items():
            on_b = force if one_body else force * (self.body_ids == b).to(force.dtype)[:, None, None]
            for i in range(p):
                composer.add_forces_and_torques_index(forces=on_b[:, i:i + 1], positions=pts_w[:, i:i + 1],
                                                      body_ids=body, env_ids=self._all, is_global=True)
