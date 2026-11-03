"""Moving objects: bodies on scripted paths that the robot can bump into, and free bodies it can push.

Two kinds of primitive body (``box``, ``capsule``, ``cylinder``) stand in the scene besides the static boxes:

* :class:`MovingObject`: a kinematic body, driven along a caller-supplied :class:`ObjectPath` (waypoints with times,
  linear or cubic Hermite interpolation, one-shot or looping). The solver treats it as immovable: it pushes the
  robot and nothing pushes it back. People, a tug and a cart are such bodies.
* :class:`FreeObject`: a dynamic rigid body with a mass and a friction, at a start pose. The robot pushes it.

Both collide with the robot and the ground, and are USD prims on the stage, so the RTX cameras and lidars see them
where the physics has them. A scene without objects is the default.

This module is the mechanism only. Which objects a task spawns, along which paths, and how they are swept or
randomized belong to humanoid-learning; the knobs are the per-env :attr:`ObjectsDriver.phase_ns` and
:attr:`ObjectsDriver.offset_m`.

**Frames.** A path and a free body's start pose are in the env's frame: the world frame shifted by the env origin
(``scene.env_origins``; the world origin in a scene with static boxes, where every env's origin is 0). Positions are
the body's centre, ``yaw`` turns it about the vertical. Box sizes are full extents ``(x, y, z)``; a capsule or
cylinder is ``(radius, height)`` along z, a capsule's height being its straight part.

**Time.** Integer nanoseconds. A path's waypoint times are in the clock the caller passes to
:meth:`ObjectsDriver.update` (the plant's ``t_ns`` or a per-env episode clock); the first waypoint's time is the
path's start. A one-shot path holds its first pose before its start and its last pose after its end; a looping path
repeats with its duration as period, so it must end where it starts.

**Batched.** :meth:`ObjectsDriver.update` evaluates every env's every kinematic body at once (torch over
``[envs, objects]``) and writes the poses and velocities into the solver state with no Python per env and no host
sync, so it is safe inside a CUDA graph when ``t_ns`` is a device tensor. Isaac Lab's own pose writers flag the
written worlds for a solver reset and a forward-kinematics pass before the next step, which for every env and every
step would reset the robots' solver warm start; the driver writes the arrays directly and flags nothing.

Isaac Lab and Newton are imported lazily; the paths, the specs and :meth:`ObjectPaths.evaluate` need torch only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.timing.clock import NS_PER_S

SHAPES = ("box", "capsule", "cylinder")
INTERPOLATIONS = ("linear", "spline")
KINEMATIC_ENTITY = "moving_objects"  # the scene entity holding every moving object
FREE_ENTITY = "free_objects"  # the scene entity holding every free object
OBJECTS_PRIM = "{ENV_REGEX_NS}/Objects"
DEFAULT_RGB = (0.85, 0.45, 0.2)
_TOL = 1e-9


def _check_shape(name: str, shape: str, size_m: Sequence[float]) -> tuple[float, ...]:
    if shape not in SHAPES:
        raise ValueError(f"object {name!r}: shape {shape!r}, one of {', '.join(SHAPES)}")
    size = tuple(float(v) for v in size_m)
    expected = 3 if shape == "box" else 2
    if len(size) != expected or not all(math.isfinite(v) and v > 0 for v in size):
        what = "(x, y, z) full extents" if shape == "box" else "(radius, height)"
        raise ValueError(f"object {name!r}: a {shape}'s size is {what} in metres, all positive; got {tuple(size_m)}")
    return size


def _check_common(name: str, friction: float, rgb: Sequence[float]) -> tuple[float, tuple[float, ...]]:
    if not name or not math.isfinite(friction) or friction < 0:
        raise ValueError(f"object {name!r}: needs a name and a friction of 0 or more, got {friction}")
    rgb = tuple(float(v) for v in rgb)
    if len(rgb) != 3 or not all(0.0 <= v <= 1.0 for v in rgb):
        raise ValueError(f"object {name!r}: rgb is three values in [0, 1], got {rgb}")
    return float(friction), rgb


@dataclass(frozen=True)
class ObjectPath:
    """Where a kinematic body is at what time: ``K`` waypoints of centre position and yaw.

    ``times_ns`` are strictly increasing; ``interpolation`` is ``"linear"`` (constant velocity between waypoints) or
    ``"spline"`` (cubic Hermite with Catmull-Rom tangents, so the velocity is continuous; the end tangents of a
    one-shot path continue the end segments' slope). ``loop`` repeats the path with period ``times_ns[-1] -
    times_ns[0]``; the last waypoint must then equal the first (positions) and turn the body a whole number of
    revolutions (yaw)."""

    times_ns: np.ndarray  # [K] int64
    position_m: np.ndarray  # [K, 3]
    yaw_rad: np.ndarray  # [K], continuous (not wrapped)
    interpolation: str = "linear"
    loop: bool = False

    @classmethod
    def of(cls, times_s: Sequence[float], position_m: Sequence[Sequence[float]], yaw_rad: Sequence[float] | None = None,
           *, interpolation: str = "linear", loop: bool = False) -> ObjectPath:
        """A path from waypoint times in seconds (rounded to nanoseconds), positions ``[K, 3]`` and optional yaws."""
        times = np.rint(np.asarray(times_s, dtype=np.float64) * NS_PER_S).astype(np.int64)
        pos = np.asarray(position_m, dtype=np.float64)
        yaw = np.zeros(len(times)) if yaw_rad is None else np.asarray(yaw_rad, dtype=np.float64)
        return cls(times, pos, yaw, interpolation, bool(loop))

    def __post_init__(self):
        k = len(self.times_ns)
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(f"path interpolation {self.interpolation!r}: one of {', '.join(INTERPOLATIONS)}")
        if k < 2 or self.position_m.shape != (k, 3) or self.yaw_rad.shape != (k,):
            raise ValueError("a path needs at least two waypoints: times [K], positions [K, 3], yaws [K]")
        if not (np.isfinite(self.position_m).all() and np.isfinite(self.yaw_rad).all()):
            raise ValueError("path waypoints must be finite")
        if (np.diff(self.times_ns) <= 0).any():
            raise ValueError("path waypoint times must be strictly increasing (after rounding to nanoseconds)")
        if self.loop:
            if np.abs(self.position_m[-1] - self.position_m[0]).max() > _TOL:
                raise ValueError("a looping path must end where it starts: last position != first")
            turns = (self.yaw_rad[-1] - self.yaw_rad[0]) / (2 * math.pi)
            if abs(turns - round(turns)) > _TOL:
                raise ValueError("a looping path must end at its start yaw (modulo whole turns)")

    @property
    def duration_ns(self) -> int:
        return int(self.times_ns[-1] - self.times_ns[0])

    def values(self) -> np.ndarray:
        """``[K, 4]`` waypoints: x, y, z, yaw."""
        return np.concatenate([self.position_m, self.yaw_rad[:, None]], axis=1)

    def describe(self) -> dict:
        return {"waypoints": len(self.times_ns), "start_ns": int(self.times_ns[0]), "duration_ns": self.duration_ns,
                "interpolation": self.interpolation, "loop": self.loop}


@dataclass(frozen=True)
class MovingObject:
    """A kinematic body on ``path``. ``friction`` is its contact friction (static and dynamic): the solver takes the
    larger of the two shapes' in a contact, so a robot whose friction is randomized keeps its range against an
    object of lower friction only."""

    name: str
    shape: str
    size_m: tuple[float, ...]
    path: ObjectPath
    friction: float
    rgb: tuple[float, float, float] = DEFAULT_RGB

    def __post_init__(self):
        object.__setattr__(self, "size_m", _check_shape(self.name, self.shape, self.size_m))
        friction, rgb = _check_common(self.name, self.friction, self.rgb)
        object.__setattr__(self, "friction", friction)
        object.__setattr__(self, "rgb", rgb)


@dataclass(frozen=True)
class FreeObject:
    """A dynamic rigid body of ``mass_kg`` with ``friction``, at ``position_m`` (its centre) turned by ``yaw_rad``
    when an env resets."""

    name: str
    shape: str
    size_m: tuple[float, ...]
    mass_kg: float
    friction: float
    position_m: tuple[float, float, float]
    yaw_rad: float = 0.0
    rgb: tuple[float, float, float] = DEFAULT_RGB

    def __post_init__(self):
        object.__setattr__(self, "size_m", _check_shape(self.name, self.shape, self.size_m))
        friction, rgb = _check_common(self.name, self.friction, self.rgb)
        object.__setattr__(self, "friction", friction)
        object.__setattr__(self, "rgb", rgb)
        pos = tuple(float(v) for v in self.position_m)
        if len(pos) != 3 or not all(math.isfinite(v) for v in pos) or not math.isfinite(self.yaw_rad):
            raise ValueError(f"object {self.name!r}: position_m is (x, y, z) in metres and yaw_rad finite")
        if not (math.isfinite(self.mass_kg) and self.mass_kg > 0):
            raise ValueError(f"object {self.name!r}: mass_kg must be positive, got {self.mass_kg}")
        object.__setattr__(self, "position_m", pos)


@dataclass(frozen=True)
class WorldObjects:
    """The moving and free objects of a scene (none by default). Names are unique across both kinds."""

    moving: tuple[MovingObject, ...] = ()
    free: tuple[FreeObject, ...] = ()
    file: str | None = None
    sha256: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "moving", tuple(self.moving))
        object.__setattr__(self, "free", tuple(self.free))
        names = [o.name for o in (*self.moving, *self.free)]
        if len(set(names)) != len(names):
            raise ValueError(f"object names must be unique, got {names}")

    def __len__(self) -> int:
        return len(self.moving) + len(self.free)

    @classmethod
    def load(cls, path: str | Path) -> WorldObjects:
        """Objects from a JSON file::

            {"moving": [{"name": "person", "shape": "capsule", "size": [0.2, 1.3], "friction": 0.8,
                         "rgb": [0.2, 0.4, 0.8],
                         "path": {"times_s": [0, 4, 8], "positions_m": [[3, -2, 0.85], [3, 2, 0.85], [3, -2, 0.85]],
                                  "yaw_rad": [1.57, -1.57, 1.57], "interpolation": "spline", "loop": true}}],
             "free": [{"name": "crate", "shape": "box", "size": [0.4, 0.4, 0.4], "mass_kg": 6.0, "friction": 0.6,
                       "position_m": [1.2, 0, 0.2], "yaw_rad": 0}]}

        ``rgb`` and ``yaw_rad`` are optional."""
        path = Path(path)
        raw = path.read_bytes()
        data = json.loads(raw)
        unknown = set(data) - {"moving", "free"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}; expected moving and free")
        moving = []
        for m in data.get("moving", []):
            p = m["path"]
            moving.append(MovingObject(
                name=m["name"], shape=m["shape"], size_m=m["size"], friction=m["friction"],
                rgb=m.get("rgb", DEFAULT_RGB),
                path=ObjectPath.of(p["times_s"], p["positions_m"], p.get("yaw_rad"),
                                   interpolation=p.get("interpolation", "linear"), loop=p.get("loop", False))))
        free = [FreeObject(name=f["name"], shape=f["shape"], size_m=f["size"], mass_kg=f["mass_kg"],
                           friction=f["friction"], position_m=f["position_m"], yaw_rad=f.get("yaw_rad", 0.0),
                           rgb=f.get("rgb", DEFAULT_RGB)) for f in data.get("free", [])]
        return cls(tuple(moving), tuple(free), file=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def describe(self) -> dict:
        """What a client needs to tell which objects are in the scene."""
        return {"file": self.file, "sha256": self.sha256,
                "moving": [{"name": o.name, "shape": o.shape, "size_m": list(o.size_m), "friction": o.friction,
                            **o.path.describe()} for o in self.moving],
                "free": [{"name": o.name, "shape": o.shape, "size_m": list(o.size_m), "mass_kg": o.mass_kg,
                          "friction": o.friction, "position_m": list(o.position_m), "yaw_rad": o.yaw_rad}
                         for o in self.free]}


# -- batched path evaluation -------------------------------------------------------------------------------------


def _tangents(path: ObjectPath) -> tuple[np.ndarray, np.ndarray]:
    """Segment start and end tangents ``[K-1, 4]`` (units per second) of ``path``."""
    v, t = path.values(), path.times_ns.astype(np.float64) / NS_PER_S
    h = np.diff(t)[:, None]
    slope = np.diff(v, axis=0) / h
    if path.interpolation == "linear" or len(t) == 2 and not path.loop:
        return slope, slope
    m = np.empty_like(v)
    m[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    if path.loop:
        # The ends are one waypoint: its neighbours are the second and the one before the last, the latter shifted by
        # the net change over a period (a whole number of yaw turns).
        net = v[-1] - v[0]
        span = (t[1] - t[0]) + (t[-1] - t[-2])
        m[0] = m[-1] = (v[1] - (v[-2] - net)) / span
    else:
        m[0], m[-1] = slope[0], slope[-1]
    return m[:-1], m[1:]


class ObjectPaths:
    """The paths of ``N`` kinematic bodies as padded segment tables on ``device``, evaluated for ``[E, N]`` at once."""

    def __init__(self, paths: Sequence[ObjectPath], device: str | torch.device = "cpu"):
        n = len(paths)
        s = max(len(p.times_ns) for p in paths) - 1
        self.n, self.device = n, torch.device(device)
        t0 = np.full((n, s), np.iinfo(np.int64).max, dtype=np.int64)  # padding is never selected
        h_ns = np.ones((n, s), dtype=np.int64)
        v0 = np.zeros((n, s, 4))
        v1 = np.zeros((n, s, 4))
        m0 = np.zeros((n, s, 4))
        m1 = np.zeros((n, s, 4))
        for i, p in enumerate(paths):
            k = len(p.times_ns) - 1
            vals = p.values()
            a, b = _tangents(p)
            t0[i, :k], h_ns[i, :k] = p.times_ns[:-1], np.diff(p.times_ns)
            v0[i, :k], v1[i, :k], m0[i, :k], m1[i, :k] = vals[:-1], vals[1:], a, b
        self.n_segments = torch.tensor([len(p.times_ns) - 1 for p in paths], device=self.device)
        self.start_ns = torch.tensor([int(p.times_ns[0]) for p in paths], dtype=torch.int64, device=self.device)
        self.duration_ns = torch.tensor([p.duration_ns for p in paths], dtype=torch.int64, device=self.device)
        self.loop = torch.tensor([p.loop for p in paths], device=self.device)
        self.t0_ns = torch.from_numpy(t0).to(self.device)
        self.h_ns = torch.from_numpy(h_ns).to(self.device)
        self.v0, self.v1, self.m0, self.m1 = (torch.from_numpy(a).to(self.device) for a in (v0, v1, m0, m1))

    def evaluate(self, t_ns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Waypoint values and their rates at ``t_ns`` ``[E, N]`` (int64, the path clock): ``[E, N, 4]`` (x, y, z, yaw
        in m and rad) and ``[E, N, 4]`` (per second). Float64. No host sync."""
        x = t_ns.transpose(0, 1).contiguous()  # [N, E]
        rel = x - self.start_ns[:, None]
        dur = self.duration_ns[:, None]
        looped = self.loop[:, None]
        rel = torch.where(looped, torch.remainder(rel, dur), rel)
        inside = looped | ((rel >= 0) & (rel < dur))  # a one-shot path rests before its start and after its end
        rel = torch.minimum(torch.clamp(rel, min=0), dur)
        at = rel + self.start_ns[:, None]
        seg = torch.searchsorted(self.t0_ns, at.contiguous(), right=True) - 1
        seg = torch.minimum(torch.clamp(seg, min=0), (self.n_segments - 1)[:, None])
        idx = seg[..., None]
        v0, v1 = (torch.take_along_dim(a, idx, dim=1) for a in (self.v0, self.v1))
        m0, m1 = (torch.take_along_dim(a, idx, dim=1) for a in (self.m0, self.m1))
        h_ns = torch.take_along_dim(self.h_ns, seg, dim=1)
        h = (h_ns.to(torch.float64) / NS_PER_S)[..., None]
        u = ((at - torch.take_along_dim(self.t0_ns, seg, dim=1)).to(torch.float64)
             / h_ns.to(torch.float64))[..., None]
        u2, u3 = u * u, u * u * u
        value = ((2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * h * m0
                 + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * h * m1)
        rate = ((6 * u2 - 6 * u) * (v0 - v1) / h + (3 * u2 - 4 * u + 1) * m0 + (3 * u2 - 2 * u) * m1)
        rate = torch.where(inside[..., None], rate, torch.zeros_like(rate))
        return value.transpose(0, 1), rate.transpose(0, 1)


def yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    """``[..., 4]`` quaternions (x, y, z, w) of rotations ``yaw`` about the vertical."""
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


# -- the scene ---------------------------------------------------------------------------------------------------


def _spawn_cfg(obj: MovingObject | FreeObject):
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

    kinematic = isinstance(obj, MovingObject)
    common = dict(
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=kinematic),
        mass_props=sim_utils.MassCfg(mass=1.0 if kinematic else obj.mass_kg),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        physics_material=RigidBodyMaterialBaseCfg(static_friction=obj.friction, dynamic_friction=obj.friction,
                                                  restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=obj.rgb))
    if obj.shape == "box":
        return sim_utils.CuboidCfg(size=obj.size_m, **common)
    radius, height = obj.size_m
    cls = sim_utils.CapsuleCfg if obj.shape == "capsule" else sim_utils.CylinderCfg
    return cls(radius=radius, height=height, axis="Z", **common)


def objects_scene_cfg(objects: WorldObjects) -> dict:
    """The scene entities of ``objects``, by name: :data:`KINEMATIC_ENTITY` and :data:`FREE_ENTITY`, each an Isaac
    Lab rigid object collection with a body per object, in order, under ``{ENV_REGEX_NS}/Objects`` (one set per
    env). Empty without objects."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.assets.rigid_object_collection import RigidObjectCollectionCfg

    def body(prefix: str, i: int, obj, position, yaw):
        quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        return RigidObjectCfg(prim_path=f"{OBJECTS_PRIM}/{prefix}_{i:04d}", spawn=_spawn_cfg(obj),
                              init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in position), rot=quat))

    entities = {}
    if objects.moving:
        start = ObjectPaths([o.path for o in objects.moving]).evaluate(
            torch.tensor([[o.path.times_ns[0] for o in objects.moving]], dtype=torch.int64))[0][0].numpy()
        entities[KINEMATIC_ENTITY] = RigidObjectCollectionCfg(rigid_objects={
            o.name: body("moving", i, o, start[i, :3], float(start[i, 3])) for i, o in enumerate(objects.moving)})
    if objects.free:
        entities[FREE_ENTITY] = RigidObjectCollectionCfg(rigid_objects={
            o.name: body("free", i, o, o.position_m, o.yaw_rad) for i, o in enumerate(objects.free)})
    return entities


def add_objects(scene_cfg, objects: WorldObjects | None) -> None:
    """Put ``objects``' entities into an ``InteractiveSceneCfg``."""
    for name, entity in ({} if objects is None else objects_scene_cfg(objects)).items():
        setattr(scene_cfg, name, entity)


@dataclass
class ObjectsState:
    """Where the objects are, per env: world-frame centres ``[E, N, 3]`` and linear velocities ``[E, N, 3]``."""

    moving_position_m: torch.Tensor
    moving_velocity_m_s: torch.Tensor
    free_position_m: torch.Tensor
    free_velocity_m_s: torch.Tensor
    free_quat_xyzw: torch.Tensor


class ObjectsDriver:
    """Drives and reads a built scene's objects (``scene``: Isaac Lab's ``InteractiveScene``).

    Call :meth:`update` once before every physics step with the path clock; the kinematic bodies are then where
    their paths put them and move at their paths' velocity through the step. :meth:`reset` puts the free bodies back
    at their start poses, at rest. ``phase_ns`` ``[E, N]`` shifts each env's clock per moving object (a path
    started earlier or later) and ``offset_m`` ``[E, N, 3]`` shifts its positions, both zero at first and yours to
    randomize."""

    def __init__(self, scene, objects: WorldObjects):
        self.objects = objects
        self.device = torch.device(scene.device)
        self.num_envs = scene.num_envs
        self.origins = scene.env_origins.to(torch.float32)
        e = self.num_envs
        self.moving = scene[KINEMATIC_ENTITY] if objects.moving else None
        self.free = scene[FREE_ENTITY] if objects.free else None
        self.paths = ObjectPaths([o.path for o in objects.moving], self.device) if objects.moving else None
        n = len(objects.moving)
        self._t = torch.zeros((), dtype=torch.int64, device=self.device)  # a python int clock, without a copy
        self.phase_ns = torch.zeros(e, n, dtype=torch.int64, device=self.device)
        self.offset_m = torch.zeros(e, n, 3, dtype=torch.float32, device=self.device)
        self._free_start = None
        if objects.free:
            pos = torch.tensor([o.position_m for o in objects.free], dtype=torch.float32, device=self.device)
            yaw = torch.tensor([o.yaw_rad for o in objects.free], dtype=torch.float32, device=self.device)
            self._free_start = torch.cat([pos, yaw_quat_xyzw(yaw)], dim=-1)  # [M, 7]
        for collection, objs in ((self.moving, objects.moving), (self.free, objects.free)):
            if collection is not None and list(collection.body_names) != [o.name for o in objs]:
                raise RuntimeError(f"the scene's objects are {collection.body_names}, not {[o.name for o in objs]}")

    def pose(self, t_ns: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame poses ``[E, N, 7]`` (position, quaternion x y z w) and velocities ``[E, N, 6]`` (linear,
        angular) of the moving objects at the path clock ``t_ns`` (an int, or a device int64 scalar or ``[E]`` tensor)."""
        if isinstance(t_ns, int):
            t = self._t.fill_(t_ns)
        else:
            t = t_ns.to(self.device)
        t = (t.expand(self.num_envs) if t.ndim == 0 else t)[:, None] + self.phase_ns  # [E, N]
        value, rate = self.paths.evaluate(t)
        value, rate = value.to(torch.float32), rate.to(torch.float32)
        position = self.origins[:, None, :] + value[..., :3] + self.offset_m
        velocity = torch.cat([rate[..., :3], torch.zeros_like(rate[..., :2]), rate[..., 3:4]], dim=-1)
        return torch.cat([position, yaw_quat_xyzw(value[..., 3])], dim=-1), velocity

    def update(self, t_ns: torch.Tensor | int) -> None:
        """Put the moving objects where their paths are at ``t_ns``, moving at the paths' velocity. No-op without
        moving objects."""
        if self.moving is None:
            return
        pose, velocity = self.pose(t_ns)
        self._write(self.moving, pose, velocity)

    def _write(self, collection, pose: torch.Tensor, velocity: torch.Tensor) -> None:
        """Write ``[E, N, 7]`` poses and ``[E, N, 6]`` velocities into the solver state, as joint coordinates (what
        MJWarp reads) and as body state (what the collision pipeline and Kamino read), with no reset flagged."""
        data = collection.data
        data.body_link_pose_w.torch.copy_(pose)
        data.body_com_vel_w.torch.copy_(velocity)
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state_0()
        for name, value in (("body_q", pose), ("body_qd", velocity)):
            view = wp.to_torch(collection.root_view.get_attribute(name, state))
            (view[:, :, 0] if view.ndim == 4 else view).copy_(value)  # [E, N, links = 1, 7 or 6]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Free bodies back to their start poses, at rest, in ``env_ids`` (all envs by default). For an env reset
        that Isaac Lab's scene reset does not already do."""
        if self.free is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        pose = self._free_start[None].expand(len(ids), -1, -1).clone()
        pose[..., :3] += self.origins[ids][:, None, :]
        self.free.write_body_link_pose_to_sim_index(body_poses=pose, env_ids=ids)
        self.free.write_body_com_velocity_to_sim_index(
            body_velocities=torch.zeros(len(ids), pose.shape[1], 6, device=self.device), env_ids=ids)

    def state(self) -> ObjectsState:
        """The objects' world-frame centres and velocities now, from the solver."""
        empty = torch.zeros(self.num_envs, 0, 3, device=self.device)
        mp, mv, fp, fv, fq = empty, empty, empty, empty, torch.zeros(self.num_envs, 0, 4, device=self.device)
        if self.moving is not None:
            mp = self.moving.data.body_link_pose_w.torch[..., :3]
            mv = self.moving.data.body_com_vel_w.torch[..., :3]
        if self.free is not None:
            fp = self.free.data.body_link_pose_w.torch[..., :3]
            fq = self.free.data.body_link_pose_w.torch[..., 3:]
            fv = self.free.data.body_com_vel_w.torch[..., :3]
        return ObjectsState(mp, mv, fp, fv, fq)
