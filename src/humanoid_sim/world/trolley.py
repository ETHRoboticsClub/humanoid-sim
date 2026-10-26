"""A gantry crane's moving parts: the bridge and trolley that carry the rope's top in x and y, and the hoist that
pays the rope out in z, batched over envs.

:class:`GantryRig` is the crane's geometry and drive limits, in its own axes: ``x`` along the runway rails (the
bridge's travel), ``y`` along the bridge (the trolley's), from the rig's ``center_m``, turned by ``yaw_rad`` about
the vertical; ``z`` is the hook's height above the floor. A position ``(x, y, z)`` puts the rope's top (the hoist,
``anchor_z_m`` above the floor) over rig point ``(x, y)`` with the hook hanging down to ``z``, so the rope's
unstretched length is ``anchor_z_m - z``.

:class:`Trolley` keeps, per env, the commanded position and the actual one, which follows it axis by axis at most
``max_speed_m_s`` fast and ``max_accel_m_s2`` hard, braking to stop on the command. A position outside the travel is
clamped to it. :class:`~humanoid_sim.world.supports.Gantry` hangs a rope from it (``hang_hoist``) and advances it
before every physics step.

Controls' test gantry has a fixed anchor and a length program only (``ethrc_sim/support.py`` @ e3bce71e2); the
moving bridge, trolley and hoist are modeled here, not mirrored.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.timing.clock import NS_PER_S

Pair = tuple[float, float]
Triple = tuple[float, float, float]


@dataclass(frozen=True)
class GantryRig:
    """A gantry's geometry and limits (m, rad, s). ``center_m`` is in the scene file's frame until :meth:`placed`
    moves it by the scene's spawn point, as the room's boxes are. ``scan_m`` is where the crane stood (rig axes)
    when the room was scanned, for moving the crane's part of a splat (``world/splat.py``)."""

    center_m: Pair
    yaw_rad: float
    anchor_z_m: float
    travel_x_m: Pair
    travel_y_m: Pair
    hook_z_m: Pair
    max_speed_m_s: Triple
    max_accel_m_s2: Triple
    home_m: Triple
    scan_m: Triple | None = None

    def __post_init__(self):
        for name in ("travel_x_m", "travel_y_m", "hook_z_m"):
            lo, hi = getattr(self, name)
            if not lo <= hi:
                raise ValueError(f"{name} {lo, hi}: the low end above the high end")
        if not self.hook_z_m[1] < self.anchor_z_m:
            raise ValueError(f"the hook's highest {self.hook_z_m[1]} m must stay below the hoist at "
                             f"{self.anchor_z_m} m")
        if min(self.max_speed_m_s) <= 0 or min(self.max_accel_m_s2) <= 0:
            raise ValueError("speed and acceleration limits must be positive")
        if self.clamp(self.home_m) != tuple(map(float, self.home_m)):
            raise ValueError(f"home {self.home_m} is outside the travel")

    @classmethod
    def from_dict(cls, d: dict) -> GantryRig:
        known = set(cls.__dataclass_fields__)
        if unknown := set(d) - known:
            raise ValueError(f"unknown gantry keys {sorted(unknown)}")
        tup = {k: tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else v for k, v in d.items()}
        return cls(**tup)

    @classmethod
    def load(cls, path: str | Path) -> GantryRig:
        """A rig from a JSON object with this class's fields."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def describe(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    def placed(self, spawn_m: Pair) -> GantryRig:
        """The rig with its centre moved by ``-spawn_m``: scene-file coordinates to world coordinates."""
        return replace(self, center_m=(self.center_m[0] - spawn_m[0], self.center_m[1] - spawn_m[1]))

    def lower(self) -> Triple:
        return self.travel_x_m[0], self.travel_y_m[0], self.hook_z_m[0]

    def upper(self) -> Triple:
        return self.travel_x_m[1], self.travel_y_m[1], self.hook_z_m[1]

    def clamp(self, xyz) -> Triple:
        return tuple(min(max(float(v), lo), hi) for v, lo, hi in zip(xyz, self.lower(), self.upper(), strict=True))

    def contains(self, xyz) -> bool:
        return all(lo <= float(v) <= hi for v, lo, hi in zip(xyz, self.lower(), self.upper(), strict=True))

    def to_world(self, xyz) -> np.ndarray:
        """Rig position ``(x, y, z)`` to the world point of the rope's top over it (host side)."""
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        x, y = float(xyz[0]), float(xyz[1])
        return np.array([self.center_m[0] + c * x - s * y, self.center_m[1] + s * x + c * y, self.anchor_z_m])

    def travel_outline_w(self) -> np.ndarray:
        """The corners ``[4, 3]`` of the area the rope's top can reach, at the hoist's height (host side)."""
        (x0, x1), (y0, y1) = self.travel_x_m, self.travel_y_m
        return np.stack([self.to_world((x, y, 0.0)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])

    def front_view(self, distance_m: float = 5.0, height_m: float = 2.2) -> tuple[np.ndarray, np.ndarray]:
        """An eye and target for viewing the crane's long side (along the runway, ``x``) from ``-y``, aimed at
        its middle at hip height (host side)."""
        target = self.to_world((0.0, 0.0, 0.0))
        target[2] = 0.9
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return target + np.array([s * distance_m, -c * distance_m, height_m - target[2]]), target

    def part_offsets(self, xyz) -> dict[str, np.ndarray]:
        """How far the crane's parts have moved from where they were scanned (``scan_m``), in the world: the bridge
        with x, the trolley with x and y, the hook with x, y and z. For moving them in a splat (host side)."""
        if self.scan_m is None:
            raise ValueError("the rig has no scan_m: where the crane was when the room was scanned")
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        dx, dy, dz = (float(a) - float(b) for a, b in zip(xyz, self.scan_m, strict=True))
        along_x, along_y = np.array([c * dx, s * dx, 0.0]), np.array([-s * dy, c * dy, 0.0])
        return {"bridge": along_x, "trolley": along_x + along_y, "hook": along_x + along_y + [0.0, 0.0, dz]}

    def over(self, hook_w, stretch_m: float) -> Triple | None:
        """The crane position that hangs a hook at world ``hook_w`` on a rope stretched ``stretch_m``: the trolley
        over it, the hoist that much short of it. None when out of reach."""
        xyz = self.from_world([float(hook_w[0]), float(hook_w[1]), float(hook_w[2]) + stretch_m])
        return xyz if self.contains(xyz) else None

    def from_world(self, xyz_w) -> Triple:
        """A world point's rig ``(x, y)``, with its ``z`` as the hook height (host side)."""
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        dx, dy = float(xyz_w[0]) - self.center_m[0], float(xyz_w[1]) - self.center_m[1]
        return c * dx + s * dy, -s * dx + c * dy, float(xyz_w[2])


class Trolley:
    """Commanded and actual crane positions ``[E, 3]`` (rig axes, hook height), on ``device``. ``origin_w``
    ``[E, 3]`` moves each env's rig (training envs on a grid: their env origins); zero by default."""

    def __init__(self, rig: GantryRig, num_envs: int, device, origin_w: torch.Tensor | None = None):
        dev = torch.device(device)
        self.rig, self.num_envs, self.device = rig, num_envs, dev
        self.lo = torch.tensor(rig.lower(), device=dev)
        self.hi = torch.tensor(rig.upper(), device=dev)
        self.v_max = torch.tensor(rig.max_speed_m_s, device=dev)
        self.a_max = torch.tensor(rig.max_accel_m_s2, device=dev)
        c, s = math.cos(rig.yaw_rad), math.sin(rig.yaw_rad)
        self._axes = torch.tensor([[c, s, 0.0], [-s, c, 0.0]], device=dev)  # rig x and y in the world
        top = torch.tensor([rig.center_m[0], rig.center_m[1], rig.anchor_z_m], device=dev)
        self.origin_w = top.expand(num_envs, 3).clone()
        if origin_w is not None:
            self.origin_w += origin_w.to(dev, torch.float32)
        home = torch.tensor(rig.home_m, device=dev).expand(num_envs, 3)
        self.command = home.clone()
        self.position = home.clone()
        self.velocity = torch.zeros(num_envs, 3, device=dev)
        self.t_ns: int | None = None  # plant time of `position`; set by the first advance
        self._dt = torch.zeros((), device=dev)  # the step advance moves by (s)
        self.cuda_graph = False  # advance's motion replays as a captured CUDA graph
        self._graph = None

    def set_command(self, env_ids: torch.Tensor, xyz: torch.Tensor) -> None:
        """Command ``xyz`` ``[n, 3]`` (or ``[3]``) for ``env_ids``, clamped to the travel."""
        self.command[env_ids] = torch.clamp(torch.as_tensor(xyz, dtype=torch.float32, device=self.device),
                                            self.lo, self.hi).expand(len(env_ids), 3)

    def place(self, env_ids: torch.Tensor, xyz: torch.Tensor) -> None:
        """Put the crane at ``xyz`` (clamped) at rest, its command there too: for resets."""
        self.set_command(env_ids, xyz)
        self.position[env_ids] = self.command[env_ids]
        self.velocity[env_ids] = 0.0

    def advance(self, t_ns: int) -> None:
        """Move from the last plant time to ``t_ns``: each axis accelerates toward its command at most ``a_max``,
        no faster than ``v_max``, and brakes so that it stops there. The first call only sets the time."""
        if self.t_ns is None or t_ns <= self.t_ns:
            self.t_ns = t_ns if self.t_ns is None else max(self.t_ns, t_ns)
            return
        self._dt.fill_((t_ns - self.t_ns) / NS_PER_S)
        self.t_ns = t_ns
        if not self.cuda_graph:
            self._move()
            return
        if self._graph is None:
            self._graph = self._capture()
        self._graph.replay()

    def _capture(self) -> torch.cuda.CUDAGraph:
        """:meth:`_move` as a CUDA graph, after a warm-up run on a side stream that is undone."""
        saved = self.position.clone(), self.velocity.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._move()
        torch.cuda.current_stream().wait_stream(side)
        self.position.copy_(saved[0])
        self.velocity.copy_(saved[1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._move()
        return graph

    def _move(self) -> None:
        """One advance by ``_dt`` (s, positive), in place."""
        dt = self._dt
        a_dt = self.a_max * dt
        err = self.command - self.position
        # The fastest speed from which the axis still stops on the command, braking a_dt per step.
        v_brake = a_dt * (torch.sqrt(0.25 + 2.0 * err.abs() / (a_dt * dt)) - 0.5)
        v_want = torch.sign(err) * torch.minimum(self.v_max, v_brake)
        v = self.velocity + torch.clamp(v_want - self.velocity, -a_dt, a_dt)
        p = self.position + 0.5 * (self.velocity + v) * dt
        # Crossing the command slowly enough to stop within the step: stop on it (else overshoot, by a few mm).
        land = ((self.command - p) * err <= 0) & (self.velocity.abs() <= a_dt)
        p = torch.where(land, self.command, p)
        v = torch.where(land, 0.0, v)
        # End stops.
        stopped = (p < self.lo) | (p > self.hi)
        self.position.copy_(torch.clamp(p, self.lo, self.hi))
        self.velocity.copy_(torch.where(stopped, 0.0, v))

    @property
    def anchor_w(self) -> torch.Tensor:
        """The rope's top in the world ``[E, 3]``."""
        return self.origin_w + self.position[:, :2] @ self._axes

    @property
    def anchor_velocity_w(self) -> torch.Tensor:
        return self.velocity[:, :2] @ self._axes

    @property
    def length_m(self) -> torch.Tensor:
        """The rope's unstretched length ``[E]``: from the hoist down to the hook height."""
        return self.rig.anchor_z_m - self.position[:, 2]

    @property
    def length_rate_m_s(self) -> torch.Tensor:
        return -self.velocity[:, 2]
