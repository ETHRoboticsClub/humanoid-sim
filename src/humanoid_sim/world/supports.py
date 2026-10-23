"""Gantry supports: a rope that holds the robot up, attached to its base.

Mirrors Controls' test gantry (``ethrc_sim/support.py`` @ e3bce71e2), batched over envs:

* **tension program**: a vertical force, linear in time between breakpoints and held beyond both ends, split
  evenly over points fixed in the base frame (one harness, or a rope per shoulder);
* **slack cable**: one rope from a base-frame harness point to a world anchor ``overhead_m`` above where that
  point was when the cable was hung (or a given anchor). It pulls along the rope only beyond its unstretched
  length (also a program in time): ``T = max(0, k · extension + c · extension rate)``; while taut, a swing
  damping resists the harness point's motion across the rope (the wire's friction at its hook).

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
    """Per-env gantry rope on the robot's base (``body``, default the articulation root)."""

    def __init__(self, robot, *, body: str | None = None, max_points: int = 2, max_breakpoints: int = 8):
        self.robot = robot
        e, dev = robot.num_instances, torch.device(robot.device)
        self.num_envs, self.device = e, dev
        self.body_id = 0 if body is None else robot.body_names.index(body)
        self._body = torch.tensor([self.body_id], device=dev)
        self._all = torch.arange(e, device=dev)
        self.mode = torch.zeros(e, dtype=torch.int64, device=dev)
        # Base-frame rope points [E, P, 3]; `points` of them used per env (tension), point 0 (cable).
        self.points_b = torch.zeros(e, max_points, 3, device=dev)
        self.num_points = torch.ones(e, dtype=torch.int64, device=dev)
        self.anchor_w = torch.zeros(e, 3, device=dev)
        self.stiffness = torch.zeros(e, device=dev)
        self.damping = torch.zeros(e, device=dev)
        self.swing_damping = torch.zeros(e, device=dev)
        # One program per env: tension (N) or unstretched cable length (m) over time.
        k = max_breakpoints
        self.program_ns = torch.arange(k, dtype=torch.int64, device=dev).expand(e, k).clone()
        self.program = torch.zeros(e, k, device=dev)
        self.tension_n = torch.zeros(e, device=dev)  # applied on the last step, total
        self.distance_m = torch.zeros(e, device=dev)  # cable: harness point to anchor
        self.active = False  # host-side; False only when no env has a rope, so apply() can skip the work

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

    def _points(self, env_ids: torch.Tensor, attachment_b=None, attachment_bodies: Sequence[str] | None = None):
        if attachment_bodies is not None:
            pose = self.robot.data.body_link_pose_w.torch
            root = pose[env_ids, self.body_id]
            ids = [self.robot.body_names.index(b) for b in attachment_bodies]
            rel = pose[env_ids][:, ids, :3] - root[:, None, :3]
            return _quat_rotate_inverse(root[:, None, 3:].expand(-1, len(ids), -1), rel)
        if attachment_b is None:
            raise ValueError("a rope needs attachment points: attachment_b or attachment_bodies")
        p = torch.as_tensor(attachment_b, dtype=torch.float32, device=self.device).reshape(-1, 3)
        return p.expand(len(env_ids), -1, -1)

    def hang_tension(self, env_ids: torch.Tensor, t_ns: int, breakpoints_s_n: Sequence[tuple[float, float]], *,
                     attachment_b=None, attachment_bodies: Sequence[str] | None = None) -> None:
        """Vertical tension over time (s, N) from ``t_ns``, split over base-frame ``attachment_b`` points or over
        the current positions of ``attachment_bodies`` fixed in the base frame (e.g. both shoulders)."""
        if any(float(n) < 0 for _, n in breakpoints_s_n):
            raise ValueError("tension must be non-negative")
        pts = self._points(env_ids, attachment_b, attachment_bodies)
        if pts.shape[1] > self.points_b.shape[1]:
            raise ValueError(f"at most {self.points_b.shape[1]} attachment points")
        self._set_program(env_ids, t_ns, breakpoints_s_n)
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, : pts.shape[1]] = pts
        self.num_points[env_ids] = pts.shape[1]
        self.mode[env_ids] = TENSION
        self.active = True

    def hang_cable(self, env_ids: torch.Tensor, t_ns: int, *, attachment_b, overhead_m: float,
                   length_breakpoints_s_m: Sequence[tuple[float, float]], stiffness_n_m: float,
                   damping_n_s_m: float, swing_damping_n_s_m: float = 0.0) -> None:
        """One cable from the base-frame harness point ``attachment_b`` to an anchor ``overhead_m`` above where
        that point is now; unstretched length over time (s, m) from ``t_ns``."""
        if overhead_m <= 0 or stiffness_n_m <= 0 or damping_n_s_m < 0 or swing_damping_n_s_m < 0:
            raise ValueError("cable overhead distance and stiffness must be positive, dampings non-negative")
        if any(float(length) <= 0 for _, length in length_breakpoints_s_m):
            raise ValueError("cable lengths must be positive")
        a = torch.as_tensor(attachment_b, dtype=torch.float32, device=self.device).reshape(3)
        root = self.robot.data.body_link_pose_w.torch[env_ids, self.body_id]
        point = root[:, :3] + _quat_rotate(root[:, 3:], a.expand(len(env_ids), 3))
        self._set_program(env_ids, t_ns, length_breakpoints_s_m)
        self._hang_cable(env_ids, a, point + torch.tensor([0.0, 0.0, overhead_m], device=self.device),
                         stiffness_n_m, damping_n_s_m, swing_damping_n_s_m)

    def hang_cables(self, env_ids: torch.Tensor, t_ns: int, *, attachment_b, anchor_w: torch.Tensor,
                    times_ns: torch.Tensor, lengths_m: torch.Tensor, stiffness_n_m: torch.Tensor,
                    damping_n_s_m: torch.Tensor, swing_damping_n_s_m: torch.Tensor) -> None:
        """One cable per env in ``env_ids`` ``[n]``, all from the base-frame harness point ``attachment_b``: world
        anchors ``[n, 3]``, unstretched length programs ``times_ns`` (int64, from ``t_ns``, nondecreasing) and
        ``lengths_m`` ``[n, K]`` with K up to ``max_breakpoints``, and per-env stiffness and dampings ``[n]``.
        Batched for resets: nothing is checked on the host, so pass positive lengths and stiffness."""
        k = self.program_ns.shape[1]
        if times_ns.shape != lengths_m.shape or not 1 <= times_ns.shape[1] <= k:
            raise ValueError(f"cable programs need matching [n, K] times and lengths, K from 1 to {k}")
        pad = k - times_ns.shape[1]  # hold the last length, 1 ns apart
        steps = torch.arange(1, pad + 1, dtype=torch.int64, device=self.device)
        self.program_ns[env_ids] = t_ns + torch.cat([times_ns, times_ns[:, -1:] + steps], dim=1)
        self.program[env_ids] = torch.cat([lengths_m, lengths_m[:, -1:].expand(-1, pad)], dim=1)
        a = torch.as_tensor(attachment_b, dtype=torch.float32, device=self.device).reshape(3)
        self._hang_cable(env_ids, a, anchor_w, stiffness_n_m, damping_n_s_m, swing_damping_n_s_m)

    def _hang_cable(self, env_ids, attachment, anchor_w, stiffness, damping, swing_damping) -> None:
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, 0] = attachment
        self.num_points[env_ids] = 1
        self.anchor_w[env_ids] = anchor_w
        self.stiffness[env_ids] = stiffness
        self.damping[env_ids] = damping
        self.swing_damping[env_ids] = swing_damping
        self.mode[env_ids] = CABLE
        self.active = True

    def release(self, env_ids: torch.Tensor) -> None:
        self.mode.index_fill_(0, env_ids, NONE)
        self.tension_n.index_fill_(0, env_ids, 0.0)
        self.distance_m.index_fill_(0, env_ids, 0.0)
        if len(env_ids) == self.num_envs:
            self.active = False

    # -- every physics step -----------------------------------------------------------------------

    def apply(self, t_ns: int) -> None:
        """Add this step's rope forces (``t_ns``: the step start, plant time)."""
        if not self.active:
            return
        root = self.robot.data.body_link_pose_w.torch[:, self.body_id]
        vel = self.robot.data.body_link_vel_w.torch[:, self.body_id]
        e, p = self.points_b.shape[:2]
        pts_w = root[:, None, :3] + _quat_rotate(root[:, None, 3:].expand(-1, p, -1), self.points_b)
        value = interp_program(torch.full((e,), t_ns, dtype=torch.int64, device=self.device), self.program_ns,
                               self.program)

        # Tension program: vertical, split evenly over the env's points.
        share = torch.where(self.mode == TENSION, value, 0.0) / self.num_points.to(value.dtype)
        used = torch.arange(p, device=self.device)[None, :] < self.num_points[:, None]
        force = torch.zeros(e, p, 3, device=self.device)
        force[..., 2] = share[:, None] * used

        # Cable: along the rope from point 0 to the anchor, beyond the unstretched length.
        line = self.anchor_w - pts_w[:, 0]
        dist = line.norm(dim=1)
        direction = line / dist.clamp(min=1e-9)[:, None]
        v_point = vel[:, :3] + torch.cross(vel[:, 3:], pts_w[:, 0] - root[:, :3], dim=1)
        extension = dist - value
        rate = -(direction * v_point).sum(1)
        cable = (self.stiffness * extension + self.damping * rate).clamp(min=0.0)
        cable = torch.where((self.mode == CABLE) & (extension > 0), cable, 0.0)
        across = v_point + rate[:, None] * direction
        swing = torch.where(cable > 0, self.swing_damping, 0.0)
        force[:, 0] += direction * cable[:, None] - swing[:, None] * across

        self.tension_n = torch.where(self.mode == CABLE, cable, share * self.num_points.to(value.dtype))
        self.distance_m = torch.where(self.mode == CABLE, dist, 0.0)
        composer = self.robot.instantaneous_wrench_composer
        for i in range(p):
            composer.add_forces_and_torques_index(forces=force[:, i:i + 1], torques=torch.zeros_like(force[:, i:i + 1]),
                                                  positions=pts_w[:, i:i + 1], body_ids=self._body, env_ids=self._all,
                                                  is_global=True)
