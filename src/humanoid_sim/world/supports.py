"""Gantry supports: a rope that holds the robot up.

After Controls' test gantry (``ethrc_sim/support.py`` @ 447b4acb), batched over envs; unlike it, ropes named by
attachment bodies pull on the harness body (below):

* **tension program**: a vertical force, linear in time between breakpoints and held beyond both ends, split
  evenly over attachment points (one harness, or a rope per shoulder);
* **slack cable**: one rope from a harness point to a world anchor ``overhead_m`` above where that point was
  when the cable was hung. It pulls along the rope only beyond its unstretched length (also a program in time):
  ``T = max(0, k · extension + c · extension rate)``.

A rope pulls on the body that carries it. Points given in the base frame hang on the base; points named by
``attachment_bodies`` (the shoulders) hang on the profile's harness body at those bodies' positions, as the lab's
straps hold the torso: on a G1 the base is the pelvis, and a rope there lets the torso fold over the waist.

Programs start when they are hung (times relative to then). Each env has its own rope or none, all on one body.
Forces go through the articulation's instantaneous wrench composer, which Isaac Lab clears after every physics
step, so call :meth:`Gantry.apply` before every physics step; it adds to the other forces on that body. No host
syncs.

*Suspended* sessions hang from a harness cable (``runtime.host``); *free* is no support. Writing the base pose
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
    """Per-env gantry rope on the robot's base, or on ``harness`` (default the base) for named attachment bodies."""

    def __init__(self, robot, *, harness: str | None = None, max_points: int = 2, max_breakpoints: int = 8):
        self.robot = robot
        e, dev = robot.num_instances, torch.device(robot.device)
        self.num_envs, self.device = e, dev
        self.harness_id = 0 if harness is None else robot.body_names.index(harness)
        self.body_id = 0  # the body the ropes pull on
        self._body = torch.tensor([self.body_id], device=dev)
        self._all = torch.arange(e, device=dev)
        self.mode = torch.zeros(e, dtype=torch.int64, device=dev)
        # Rope points in the rope body's frame [E, P, 3]; `points` of them used per env (tension), point 0 (cable).
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

    # -- configuration (host side, at reset or on request) ----------------------------------------
    # A hang checks everything before it changes anything: a rejected rope leaves the previous one hanging. With
    # ``dry_run`` it only checks, without reading the body poses (the host hangs once they are current).

    def _program(self, t_ns: int, breakpoints: Sequence[tuple[float, float]]) -> tuple[torch.Tensor, torch.Tensor]:
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
        return (t_ns + torch.tensor(times, dtype=torch.int64, device=self.device),
                torch.tensor(values, device=self.device))

    def _target(self, attachment_b, attachment_bodies: Sequence[str] | None) -> tuple[int, list[int] | torch.Tensor]:
        """The rope body and what goes on it: the ids of ``attachment_bodies`` (on the harness body), else the
        base-frame points ``attachment_b`` (on the base)."""
        if attachment_bodies is not None:
            unknown = [b for b in attachment_bodies if b not in self.robot.body_names]
            if unknown:
                raise ValueError(f"unknown attachment bodies {unknown}")
            body_id, spec = self.harness_id, [self.robot.body_names.index(b) for b in attachment_bodies]
        elif attachment_b is None:
            raise ValueError("a rope needs attachment points: attachment_b or attachment_bodies")
        else:
            spec = torch.as_tensor(attachment_b, dtype=torch.float32, device=self.device).reshape(-1, 3)
            if not bool(torch.isfinite(spec).all()):
                raise ValueError("attachment points must be finite")
            body_id = 0
        if not 1 <= len(spec) <= self.points_b.shape[1]:
            raise ValueError(f"a rope needs 1 to {self.points_b.shape[1]} attachment points")
        return body_id, spec

    def _check_body(self, env_ids: torch.Tensor, body_id: int) -> None:
        """Ropes of every env pull on one body: refuse another body while some other env has a rope."""
        others = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        others[env_ids] = False
        if body_id != self.body_id and bool((others & (self.mode != NONE)).any()):
            raise ValueError("every env's rope pulls on one body; release the others first")

    def _points(self, env_ids: torch.Tensor, body_id: int, spec) -> torch.Tensor:
        """The rope points ``[E, P, 3]`` in the rope body's frame."""
        if isinstance(spec, list):
            pose = self.robot.data.body_link_pose_w.torch
            carrier = pose[env_ids, body_id]
            rel = pose[env_ids][:, spec, :3] - carrier[:, None, :3]
            return _quat_rotate_inverse(carrier[:, None, 3:].expand(-1, len(spec), -1), rel)
        return spec.expand(len(env_ids), -1, -1)

    def _commit(self, env_ids: torch.Tensor, body_id: int, program: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.body_id = body_id
        self._body = torch.tensor([body_id], device=self.device)
        self.program_ns[env_ids], self.program[env_ids] = program

    def hang_tension(self, env_ids: torch.Tensor, t_ns: int, breakpoints_s_n: Sequence[tuple[float, float]], *,
                     attachment_b=None, attachment_bodies: Sequence[str] | None = None, dry_run: bool = False) -> None:
        """Vertical tension over time (s, N) from ``t_ns``, split over base-frame ``attachment_b`` points or over
        the current positions of ``attachment_bodies`` (e.g. both shoulders) on the harness body."""
        if any(float(n) < 0 for _, n in breakpoints_s_n):
            raise ValueError("tension must be non-negative")
        program = self._program(t_ns, breakpoints_s_n)
        body_id, spec = self._target(attachment_b, attachment_bodies)
        self._check_body(env_ids, body_id)
        if dry_run:
            return
        pts = self._points(env_ids, body_id, spec)
        self._commit(env_ids, body_id, program)
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, : pts.shape[1]] = pts
        self.num_points[env_ids] = pts.shape[1]
        self.mode[env_ids] = TENSION
        self.active = True

    def hang_cable(self, env_ids: torch.Tensor, t_ns: int, *, overhead_m: float,
                   length_breakpoints_s_m: Sequence[tuple[float, float]], stiffness_n_m: float,
                   damping_n_s_m: float, attachment_b=None, attachment_bodies: Sequence[str] | None = None,
                   dry_run: bool = False) -> None:
        """One cable from a harness point (base-frame ``attachment_b``, or the midpoint of ``attachment_bodies``
        on the harness body: the hook of both shoulder straps) to an anchor ``overhead_m`` above where that point
        is now; unstretched length over time (s, m) from ``t_ns``."""
        if overhead_m <= 0 or stiffness_n_m <= 0 or damping_n_s_m < 0:
            raise ValueError("cable overhead distance and stiffness must be positive, damping non-negative")
        if any(float(length) <= 0 for _, length in length_breakpoints_s_m):
            raise ValueError("cable lengths must be positive")
        program = self._program(t_ns, length_breakpoints_s_m)
        body_id, spec = self._target(attachment_b, attachment_bodies)
        self._check_body(env_ids, body_id)
        if dry_run:
            return
        a = self._points(env_ids, body_id, spec).mean(dim=1)
        body = self.robot.data.body_link_pose_w.torch[env_ids, body_id]
        point = body[:, :3] + _quat_rotate(body[:, 3:], a)
        self._commit(env_ids, body_id, program)
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, 0] = a
        self.num_points[env_ids] = 1
        self.anchor_w[env_ids] = point + torch.tensor([0.0, 0.0, overhead_m], device=self.device)
        self.stiffness[env_ids] = stiffness_n_m
        self.damping[env_ids] = damping_n_s_m
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
        force[:, 0] += direction * cable[:, None]

        self.tension_n = torch.where(self.mode == CABLE, cable, share * self.num_points.to(value.dtype))
        self.distance_m = torch.where(self.mode == CABLE, dist, 0.0)
        composer = self.robot.instantaneous_wrench_composer
        for i in range(p):
            composer.add_forces_and_torques_index(forces=force[:, i:i + 1], torques=torch.zeros_like(force[:, i:i + 1]),
                                                  positions=pts_w[:, i:i + 1], body_ids=self._body, env_ids=self._all,
                                                  is_global=True)
