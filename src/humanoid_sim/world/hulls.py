"""Static convex pieces around the robot: a scanned room's collision geometry as convex hulls, read from a file.

A scan's triangle mesh does not collide on MJWarp (docs/world.md, Triangle-mesh terrain), but convex shapes do, on
both solvers, as the static boxes (``world/boxes.py``) and the robot's own hulls do. So a room's collision can come
as convex pieces fitted to its scan, each a set of points whose convex hull is the piece. They stand on the scene's
flat ground, which stays the walking surface, as the boxes do.

The file is an ``.npz``, in metres, z up, the floor at z = 0, in the scan's frame:

* ``points`` ``[P, 3]``: every piece's points, piece after piece;
* ``counts`` ``[N]``: how many points each piece has (at least 4, not all in a plane);
* ``rgb`` ``[N, 3]`` (optional): colours in [0, 1], for drawing.

Other arrays are ignored. :meth:`StaticHulls.load` places the pieces as :meth:`StaticBoxes.load` places boxes:
``spawn`` is the file point put at the world origin, and every piece moves by ``-spawn``.

In an Isaac Lab scene each piece is a USD ``Mesh`` with the collision API and the ``convexHull`` approximation
under :data:`STATIC_HULLS_PRIM`, outside the env namespace, so every env's solver world collides with one set of
static shapes at the same world coordinates (and every env's robot starts at the world origin). Isaac Lab, pxr and
SciPy are imported lazily.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.world.boxes import StaticBoxes

STATIC_HULLS_PRIM = "/World/StaticHulls"
DEFAULT_RGB = (0.55, 0.62, 0.7)  # drawing only
MIN_THICKNESS_M = 0.01  # a thinner piece is a sheet, which MuJoCo refuses to compile ("mesh volume is too small")


@dataclass(frozen=True)
class StaticHulls:
    """Convex pieces placed in the world frame (``world = scene file frame - spawn``)."""

    points: np.ndarray  # [P, 3] world frame (m)
    counts: np.ndarray  # [N] points per piece
    rgb: np.ndarray  # [N, 3] drawing colour in [0, 1]
    file: str
    sha256: str
    spawn_m: tuple[float, float]

    @classmethod
    def load(cls, path: str | Path, spawn_m: Sequence[float] = (0.0, 0.0)) -> StaticHulls:
        """The pieces of the file at ``path``, with the scene point ``spawn_m`` (x, y) at the world origin."""
        path = Path(path)
        raw = path.read_bytes()
        if len(raw) < 1024 and raw.startswith(b"version https://git-lfs"):
            raise ValueError(f"{path} is a git-lfs pointer; run 'git lfs pull' where it comes from")
        if len(spawn_m) != 2:
            raise ValueError(f"spawn_m {tuple(spawn_m)}: the scene point (x, y) placed at the world origin")
        with np.load(path) as data:
            missing = {"points", "counts"} - set(data.files)
            if missing:
                raise ValueError(f"{path}: convex pieces need {sorted(missing)}")
            points = np.asarray(data["points"], dtype=np.float64)
            counts = np.asarray(data["counts"])
            n = len(counts)
            rgb = (np.asarray(data["rgb"], dtype=np.float64) if "rgb" in data.files
                   else np.tile(DEFAULT_RGB, (n, 1)))
        if counts.ndim != 1 or n == 0 or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError(f"{path}: counts must be a non-empty [N] integer array")
        if points.ndim != 2 or points.shape[1] != 3 or counts.sum() != len(points) or rgb.shape != (n, 3):
            raise ValueError(f"{path}: points [P, 3] with P = sum(counts) and rgb [N, 3] (N = {n})")
        if (counts < 4).any():
            raise ValueError(f"{path}: pieces {np.flatnonzero(counts < 4).tolist()} have fewer than 4 points")
        if not (np.isfinite(points).all() and np.isfinite(rgb).all()):
            raise ValueError(f"{path}: non-finite values")
        hulls = cls(points=points - np.array([float(spawn_m[0]), float(spawn_m[1]), 0.0]), counts=counts.astype(np.int64),
                    rgb=np.clip(rgb, 0.0, 1.0), file=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                    spawn_m=(float(spawn_m[0]), float(spawn_m[1])))
        thin = [i for i, p in enumerate(hulls.pieces()) if thickness(p) < MIN_THICKNESS_M]
        if thin:
            raise ValueError(f"{path}: pieces {thin[:10]} are thinner than {MIN_THICKNESS_M} m")
        return hulls

    def __len__(self) -> int:
        return len(self.counts)

    def pieces(self) -> list[np.ndarray]:
        """Each piece's points ``[k, 3]``, world frame."""
        return np.split(self.points, np.cumsum(self.counts)[:-1])

    def clearance_m(self, xy: Sequence[float] = (0.0, 0.0)) -> float:
        """Horizontal distance from the world point ``xy`` to the nearest piece's footprint, 0 inside one."""
        p = np.array([float(xy[0]), float(xy[1])])
        return float(min(polygon_distance(convex_polygon(piece[:, :2]), p) for piece in self.pieces()))

    def describe(self) -> dict:
        """What a client needs to tell which collision geometry it is in and where."""
        return {"file": self.file, "sha256": self.sha256, "pieces": len(self), "spawn_m": list(self.spawn_m),
                "spawn_clearance_m": self.clearance_m()}


def thickness(points: np.ndarray) -> float:
    """Smallest extent of a point set along its principal axes (m)."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ axes.T
    return float((proj.max(axis=0) - proj.min(axis=0)).min())


def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def convex_polygon(xy: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull ``[k, 2]`` of 2D points (Andrew's monotone chain)."""
    pts = np.unique(np.asarray(xy, dtype=np.float64), axis=0)  # sorted by x, then y
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def polygon_distance(polygon: np.ndarray, p: np.ndarray) -> float:
    """Distance from ``p`` to a counter-clockwise convex polygon, 0 inside it."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edge = b - a
    if len(polygon) >= 3 and (_cross2(edge, p - a) >= 0).all():
        return 0.0
    t = np.clip(np.einsum("ij,ij->i", p - a, edge) / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-18), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * edge - p, axis=1).min())


def hull_triangles(points: np.ndarray) -> np.ndarray:
    """Outward-wound triangles ``[F, 3]`` of the convex hull of ``points`` (indices into ``points``). SciPy, lazily."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    tris = hull.simplices.copy()
    centre = points[hull.vertices].mean(axis=0)
    normal = np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
    inward = np.einsum("ij,ij->i", normal, points[tris[:, 0]] - centre) < 0
    tris[inward] = tris[inward][:, ::-1]
    return tris


def hull_prim_path(index: int, root: str = STATIC_HULLS_PRIM) -> str:
    """The ``Mesh`` prim of the file's piece ``index``."""
    return f"{root}/hull_{index:04d}"


def spawn_static_hulls(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab spawner of :func:`static_hulls_cfg`: an Xform at ``prim_path`` holding one physics material (the
    ground's friction, no restitution) and one ``Mesh`` prim per piece, its convex hull, with the collision API and
    the ``convexHull`` approximation, its display colour the piece's."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    if not (translation is None or np.allclose(translation, 0.0)) or not (
            orientation is None or np.allclose(orientation, (0.0, 0.0, 0.0, 1.0))):
        raise ValueError("static hulls are placed by their file and spawn point, not by an entity transform")
    hulls = StaticHulls.load(cfg.file, cfg.spawn_m)
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    material = f"{prim_path}/PhysicsMaterial"
    sim_utils.spawn_rigid_body_material(material, RigidBodyMaterialBaseCfg(
        static_friction=cfg.friction, dynamic_friction=cfg.friction, restitution=0.0))
    for i, piece in enumerate(hulls.pieces()):
        tris = hull_triangles(piece)
        used = np.unique(tris)
        remap = np.full(len(piece), -1)
        remap[used] = np.arange(len(used))
        mesh = UsdGeom.Mesh.Define(stage, hull_prim_path(i, prim_path))
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(piece[used].astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remap[tris].astype(np.int32).ravel()))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, hulls.rgb[i]))])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        sim_utils.bind_physics_material(mesh.GetPath(), material, stage=stage)
    return root


def add_room(scene_cfg, boxes: StaticBoxes | None, hulls: StaticHulls | None, *, friction: float) -> None:
    """Add a room to an Isaac Lab scene config: its ``boxes`` as ``static_boxes`` and its convex pieces as
    ``static_hulls``. With pieces the pieces collide and the boxes stay on the stage for rays and drawing only."""
    if boxes is not None and hulls is not None and tuple(boxes.spawn_m) != tuple(hulls.spawn_m):
        raise ValueError(f"the boxes' spawn {boxes.spawn_m} and the pieces' {hulls.spawn_m} differ")
    if boxes is not None:
        from humanoid_sim.world.boxes import static_boxes_cfg

        scene_cfg.static_boxes = static_boxes_cfg(boxes.file, boxes.spawn_m, friction=friction, collide=hulls is None)
    if hulls is not None:
        scene_cfg.static_hulls = static_hulls_cfg(hulls.file, hulls.spawn_m, friction=friction)


def static_hulls_cfg(file: str | Path, spawn_m: Sequence[float] = (0.0, 0.0), *, friction: float):
    """The pieces in ``file`` (``spawn_m`` at the world origin) as an Isaac Lab scene entity at
    :data:`STATIC_HULLS_PRIM`, every piece with the ground's ``friction``."""
    from dataclasses import MISSING

    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import SpawnerCfg
    from isaaclab.utils import configclass

    @configclass
    class StaticHullsCfg(SpawnerCfg):
        func: Callable = spawn_static_hulls
        file: str = MISSING
        spawn_m: tuple[float, float] = (0.0, 0.0)
        friction: float = MISSING

    spawn = StaticHullsCfg(file=str(file), spawn_m=(float(spawn_m[0]), float(spawn_m[1])), friction=float(friction))
    return AssetBaseCfg(prim_path=STATIC_HULLS_PRIM, spawn=spawn)
