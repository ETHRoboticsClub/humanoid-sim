"""Gantry supports: a rope that holds the robot up.

After Controls' test gantry (``ethrc_sim/support.py`` @ 447b4acb), batched over envs; unlike it, ropes named by
attachment bodies pull on the harness body (below):

* **tension program**: a vertical force, linear in time between breakpoints and held beyond both ends, split
  evenly over attachment points (one harness, or a rope per shoulder);
* **slack cable**: a rope from the harness to a world anchor (the hook) ``overhead_m`` above where the harness
  point was when the cable was hung. It pulls along the rope only beyond its unstretched length (also a program in
  time): ``T = max(0, k · extension + c · extension rate)``; while taut, ``swing_damping`` resists the harness
  point's motion across the rope (the wire's friction at its hook), so a swing dies away. With two attachment
  points (both shoulders) the hook carries a spreader bar and each point hangs on its own strap from the bar,
  the bar's ends as far apart as the points were when hung (or ``spreader_width_m``); the bar keeps its heading,
  so the straps resist twisting as a bifilar pendulum does. Each strap takes ``1 / P`` of the stiffness and both
  dampings, so the rope's vertical stiffness and its swing are those of one rope;
* **hoist**: the same cable hung from a crane's hook (:class:`~humanoid_sim.world.trolley.Trolley`, mounted with
  :meth:`Gantry.mount`): its top moves with the bridge and trolley and its unstretched length with the hoist, and
  the cable's damping acts on its stretch rate, relative to the moving top. Modeled, not in Controls' gantry.

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

import math
from collections.abc import Sequence

import torch

from humanoid_sim.timing.clock import NS_PER_S

NONE, TENSION, CABLE, HOIST = 0, 1, 2, 3
GRAVITY_M_S2 = 9.81  # Isaac Lab's SimulationCfg default; humanoid-sim never sets another


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


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a b`` of (x, y, z, w) quaternions."""
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def level_under(hook: torch.Tensor, com: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    """The root pose ``[x, y, z, qx, qy, qz, qw]`` that turns the rigid robot about ``hook`` until its centre of mass
    ``com`` hangs straight below it: the attitude a robot on a cable from the hook comes to rest in."""
    down = (com - hook) / (com - hook).norm()
    axis = torch.linalg.cross(down, torch.tensor([0.0, 0.0, -1.0], dtype=down.dtype, device=down.device))
    sine = float(axis.norm())
    if sine == 0.0:
        return root.clone()
    half = math.atan2(sine, float(-down[2])) / 2
    turn = torch.cat([axis / sine * math.sin(half), axis.new_tensor([math.cos(half)])])
    return torch.cat([hook + _quat_rotate(turn, root[:3] - hook), _quat_multiply(turn, root[3:])])


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
        # Rope points in the rope body's frame [E, P, 3], `num_points` of them used per env: one strap each on a
        # cable, its top `top_offset_w` from the hook (the spreader bar's ends, world frame, horizontal).
        self.points_b = torch.zeros(e, max_points, 3, device=dev)
        self.num_points = torch.ones(e, dtype=torch.int64, device=dev)
        self.top_offset_w = torch.zeros(e, max_points, 3, device=dev)
        self.anchor_w = torch.zeros(e, 3, device=dev)  # cable: the hook, the spreader bar's middle
        self.stiffness = torch.zeros(e, device=dev)  # cable gains, all straps together
        self.damping = torch.zeros(e, device=dev)
        self.swing_damping = torch.zeros(e, device=dev)
        # One program per env: tension (N) or unstretched cable length (m) over time.
        k = max_breakpoints
        self.program_ns = torch.arange(k, dtype=torch.int64, device=dev).expand(e, k).clone()
        self.program = torch.zeros(e, k, device=dev)
        self.tension_n = torch.zeros(e, device=dev)  # applied on the last step, total over the straps
        self.distance_m = torch.zeros(e, device=dev)  # cable: rope point to its top, mean over the straps
        # On the last step: the hook and the harness point (the straps' points' mean) in the world, and each
        # strap's top (its bar end) and bottom (its point) [E, P, 3]; NaN without a cable or for an unused point.
        self.top_w = torch.full((e, 3), float("nan"), device=dev)
        self.harness_w = torch.full((e, 3), float("nan"), device=dev)
        self.strap_top_w = torch.full((e, max_points, 3), float("nan"), device=dev)
        self.strap_bottom_w = torch.full((e, max_points, 3), float("nan"), device=dev)
        self.trolley = None  # a crane to hang hoist ropes from (mount)
        self.active = False  # host-side; False only when no env has a rope, so apply() can skip the work

    def mount(self, rig, origin_w: torch.Tensor | None = None):
        """Put a crane (:class:`~humanoid_sim.world.trolley.GantryRig`, in world coordinates) over every env,
        moved by ``origin_w`` ``[E, 3]`` per env; returns its :class:`~humanoid_sim.world.trolley.Trolley`, at
        home. :meth:`apply` advances it."""
        from humanoid_sim.world.trolley import Trolley

        self.trolley = Trolley(rig, self.num_envs, self.device, origin_w)
        return self.trolley

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
        if body_id != self.body_id:  # a host copy, so not at every reset
            self.body_id = body_id
            self._body = torch.tensor([body_id], device=self.device)
        self.program_ns[env_ids], self.program[env_ids] = program

    @staticmethod
    def _check_spreader(spreader_width_m: float | None, num_points: int) -> None:
        if spreader_width_m is not None and not (num_points == 2 and math.isfinite(spreader_width_m)
                                                 and spreader_width_m > 0):
            raise ValueError("a spreader bar's width must be positive and finite, for two attachment points")

    def _spread(self, env_ids: torch.Tensor, body_id: int, pts: torch.Tensor,
                spreader_width_m: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        """The hook (the points' mean, world ``[n, 3]``) and each strap's top from it ``[n, P, 3]``: the points'
        horizontal offsets from the hook, stretched to ``spreader_width_m`` apart when given."""
        body = self.robot.data.body_link_pose_w.torch[env_ids, body_id]
        pts_w = body[:, None, :3] + _quat_rotate(body[:, None, 3:].expand(-1, pts.shape[1], -1), pts)
        hook = pts_w.mean(dim=1)
        offset = pts_w - hook[:, None]
        offset[..., 2] = 0.0
        if spreader_width_m is not None:
            offset = offset / offset.norm(dim=2, keepdim=True).clamp(min=1e-9) * (spreader_width_m / 2)
        return hook, offset

    def _set_cable(self, env_ids: torch.Tensor, mode: int, pts: torch.Tensor, offsets: torch.Tensor | None,
                   anchor_w, stiffness_n_m, damping_n_s_m, swing_damping_n_s_m) -> None:
        """A cable with one strap per rope point ``pts`` ``[n, P, 3]``, its top ``offsets`` from the hook."""
        p = pts.shape[1]
        self.points_b[env_ids] = 0.0
        self.points_b[env_ids, :p] = pts
        self.top_offset_w[env_ids] = 0.0
        if offsets is not None:
            self.top_offset_w[env_ids, :p] = offsets
        self.num_points[env_ids] = p
        self.anchor_w[env_ids] = anchor_w
        self.stiffness[env_ids] = stiffness_n_m
        self.damping[env_ids] = damping_n_s_m
        self.swing_damping[env_ids] = swing_damping_n_s_m
        self.mode[env_ids] = mode
        self.active = True

    def _rope_points(self, env_ids: torch.Tensor, point_b) -> torch.Tensor:
        """``point_b`` ``[3]``, ``[n, 3]`` (one strap) or ``[n, P, 3]`` as ``[n, P, 3]``."""
        pts = torch.as_tensor(point_b, dtype=torch.float32, device=self.device)
        pts = pts.reshape(-1, 1, 3) if pts.dim() < 3 else pts
        if not 1 <= pts.shape[1] <= self.points_b.shape[1]:
            raise ValueError(f"a rope needs 1 to {self.points_b.shape[1]} attachment points")
        return pts.expand(len(env_ids), -1, -1)

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
                   swing_damping_n_s_m: float = 0.0, spreader_width_m: float | None = None,
                   dry_run: bool = False) -> None:
        """One cable from the harness (base-frame ``attachment_b``, or ``attachment_bodies`` on the harness body,
        such as both shoulders) to an anchor ``overhead_m`` above the points' midpoint now, one strap per point
        from a spreader bar (``spreader_width_m`` wide, else as wide as the points are apart now); unstretched
        length over time (s, m) from ``t_ns``."""
        if overhead_m <= 0 or stiffness_n_m <= 0 or damping_n_s_m < 0 or swing_damping_n_s_m < 0:
            raise ValueError("cable overhead distance and stiffness must be positive, damping non-negative")
        if any(float(length) <= 0 for _, length in length_breakpoints_s_m):
            raise ValueError("cable lengths must be positive")
        program = self._program(t_ns, length_breakpoints_s_m)
        body_id, spec = self._target(attachment_b, attachment_bodies)
        self._check_spreader(spreader_width_m, len(spec))
        self._check_body(env_ids, body_id)
        if dry_run:
            return
        pts = self._points(env_ids, body_id, spec)
        hook, offsets = self._spread(env_ids, body_id, pts, spreader_width_m)
        self._commit(env_ids, body_id, program)
        self._set_cable(env_ids, CABLE, pts, offsets, hook + torch.tensor([0.0, 0.0, overhead_m], device=self.device),
                        stiffness_n_m, damping_n_s_m, swing_damping_n_s_m)

    def hang_cables(self, env_ids: torch.Tensor, t_ns: int, *, point_b: torch.Tensor, anchor_w: torch.Tensor,
                    times_ns: torch.Tensor, lengths_m: torch.Tensor, stiffness_n_m: torch.Tensor,
                    damping_n_s_m: torch.Tensor, swing_damping_n_s_m: torch.Tensor, harness: bool = False,
                    top_offset_w: torch.Tensor | None = None) -> None:
        """One cable per env in ``env_ids`` ``[n]``, for resets: from ``point_b`` (``[3]`` or ``[n, 3]`` for one
        strap, ``[n, P, 3]`` for one strap per point; in the harness body's frame with ``harness``, else the base's)
        to world anchors (hooks) ``[n, 3]``, each strap's top ``top_offset_w`` ``[n, P, 3]`` from its hook (a
        spreader bar's ends, world frame; default at the hook), with unstretched length
        programs ``times_ns`` (int64, from ``t_ns``, nondecreasing) and ``lengths_m`` ``[n, K]``, K up to
        ``max_breakpoints``, and per-env stiffness and dampings ``[n]``. Nothing is checked on the host, so pass
        positive lengths and stiffness; the anchors are given because body poses are stale right after a reset."""
        k = self.program_ns.shape[1]
        if times_ns.shape != lengths_m.shape or not 1 <= times_ns.shape[1] <= k:
            raise ValueError(f"cable programs need matching [n, K] times and lengths, K from 1 to {k}")
        body_id = self.harness_id if harness else 0
        pts = self._rope_points(env_ids, point_b)
        self._check_body(env_ids, body_id)
        pad = k - times_ns.shape[1]  # hold the last length, 1 ns apart
        steps = torch.arange(1, pad + 1, dtype=torch.int64, device=self.device)
        self._commit(env_ids, body_id, (t_ns + torch.cat([times_ns, times_ns[:, -1:] + steps], dim=1),
                                        torch.cat([lengths_m, lengths_m[:, -1:].expand(-1, pad)], dim=1)))
        self._set_cable(env_ids, CABLE, pts, top_offset_w, anchor_w, stiffness_n_m, damping_n_s_m,
                        swing_damping_n_s_m)

    def hang_hoist(self, env_ids: torch.Tensor, *, stiffness_n_m, damping_n_s_m, swing_damping_n_s_m=0.0,
                   attachment_b=None, attachment_bodies: Sequence[str] | None = None, point_b=None,
                   harness: bool = False, spreader_width_m: float | None = None,
                   top_offset_w: torch.Tensor | None = None, dry_run: bool = False) -> None:
        """One cable from the harness to the mounted crane's hook: its top at the trolley, its unstretched
        length the hoist's. The points and the spreader bar are as for :meth:`hang_cable` (base-frame
        ``attachment_b``, or ``attachment_bodies`` on the harness body), or, for resets with stale poses, as for
        :meth:`hang_cables`: ``point_b`` in the harness body's frame with ``harness``, else the base's, and
        ``top_offset_w``. Stiffness and dampings are floats or ``[n]`` tensors; floats are checked."""
        if self.trolley is None:
            raise ValueError("no crane to hang from: mount a gantry rig first")
        gains = (stiffness_n_m, damping_n_s_m, swing_damping_n_s_m)
        if all(isinstance(g, (int, float)) for g in gains) and not (
                math.isfinite(sum(gains)) and stiffness_n_m > 0 and damping_n_s_m >= 0 and swing_damping_n_s_m >= 0):
            raise ValueError("hoist stiffness must be positive, dampings non-negative, all finite")
        if point_b is None:
            body_id, spec = self._target(attachment_b, attachment_bodies)
            self._check_spreader(spreader_width_m, len(spec))
        else:
            body_id, pts = self.harness_id if harness else 0, self._rope_points(env_ids, point_b)
        self._check_body(env_ids, body_id)
        if dry_run:
            return
        if point_b is None:
            pts = self._points(env_ids, body_id, spec)
            _, top_offset_w = self._spread(env_ids, body_id, pts, spreader_width_m)
        program = self.program_ns[env_ids], self.program[env_ids]  # unused: the hoist sets the length
        self._commit(env_ids, body_id, program)
        self._set_cable(env_ids, HOIST, pts, top_offset_w, self.trolley.anchor_w[env_ids], stiffness_n_m,
                        damping_n_s_m, swing_damping_n_s_m)

    def rest_length_m(self, t_ns: int) -> torch.Tensor:
        """The cables' unstretched length at ``t_ns`` ``[E]`` (the hoist's for a hoist rope); NaN without a cable.
        For viewers: it reads the program and the crane, and :meth:`apply` does not call it."""
        length = interp_program(torch.full((self.num_envs,), t_ns, dtype=torch.int64, device=self.device),
                                self.program_ns, self.program)
        if self.trolley is not None:
            length = torch.where(self.mode == HOIST, self.trolley.length_m, length)
        return torch.where((self.mode == CABLE) | (self.mode == HOIST), length, float("nan"))

    def release(self, env_ids: torch.Tensor) -> None:
        self.mode.index_fill_(0, env_ids, NONE)
        self.tension_n.index_fill_(0, env_ids, 0.0)
        self.distance_m.index_fill_(0, env_ids, 0.0)
        self.top_w[env_ids] = float("nan")
        self.harness_w[env_ids] = float("nan")
        self.strap_top_w[env_ids] = float("nan")
        self.strap_bottom_w[env_ids] = float("nan")
        if len(env_ids) == self.num_envs:
            self.active = False

    # -- every physics step -----------------------------------------------------------------------

    def apply(self, t_ns: int) -> None:
        """Advance the crane, if one is mounted, to ``t_ns`` (the step start, plant time) and add this step's
        rope forces."""
        if self.trolley is not None:
            self.trolley.advance(t_ns)
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

        # Cable: along each strap from its point to its top on the bar, beyond the unstretched length, each strap
        # with 1 / P of the gains. A hoist rope's hook and length are the crane's; its stretch rate is relative to
        # the moving hook, less the hoist's pay-out.
        top, v_top, pay_out = self.anchor_w, torch.zeros_like(self.anchor_w), torch.zeros_like(value)
        if self.trolley is not None:
            hoist = self.mode == HOIST
            top = torch.where(hoist[:, None], self.trolley.anchor_w, top)
            v_top = torch.where(hoist[:, None], self.trolley.anchor_velocity_w, v_top)
            value = torch.where(hoist, self.trolley.length_m, value)
            pay_out = torch.where(hoist, self.trolley.length_rate_m_s, pay_out)
        roped = (self.mode == CABLE) | (self.mode == HOIST)
        n = self.num_points.to(value.dtype)[:, None]
        line = top[:, None] + self.top_offset_w - pts_w
        dist = line.norm(dim=2)
        direction = line / dist.clamp(min=1e-9)[..., None]
        v_point = vel[:, None, :3] + torch.cross(vel[:, None, 3:].expand(-1, p, -1), pts_w - root[:, None, :3], dim=2)
        v_rel = v_point - v_top[:, None]
        extension = dist - value[:, None]
        rate = -(direction * v_rel).sum(2) - pay_out[:, None]
        cable = ((self.stiffness[:, None] * extension + self.damping[:, None] * rate) / n).clamp(min=0.0)
        taut = roped[:, None] & used & (extension > 0)
        cable = torch.where(taut, cable, 0.0)
        across = v_rel - (direction * v_rel).sum(2, keepdim=True) * direction
        swing = torch.where(cable > 0, self.swing_damping[:, None] / n, 0.0)  # a strap without tension pushes nothing
        force += direction * cable[..., None] - swing[..., None] * across

        self.tension_n = torch.where(roped, cable.sum(1), share * self.num_points.to(value.dtype))
        self.distance_m = torch.where(roped, (dist * used).sum(1) / n[:, 0], 0.0)
        self.top_w = torch.where(roped[:, None], top, float("nan"))
        self.harness_w = torch.where(roped[:, None], (pts_w * used[..., None]).sum(1) / n, float("nan"))
        strap = (roped[:, None] & used)[..., None]
        self.strap_top_w = torch.where(strap, top[:, None] + self.top_offset_w, float("nan"))
        self.strap_bottom_w = torch.where(strap, pts_w, float("nan"))
        composer = self.robot.instantaneous_wrench_composer
        for i in range(p):
            composer.add_forces_and_torques_index(forces=force[:, i:i + 1], torques=torch.zeros_like(force[:, i:i + 1]),
                                                  positions=pts_w[:, i:i + 1], body_ids=self._body, env_ids=self._all,
                                                  is_global=True)
