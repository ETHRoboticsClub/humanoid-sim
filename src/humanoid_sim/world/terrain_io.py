"""Heightfield terrain for replays and the runtime: a terrain from a file, and the ground height under the robot.

Training builds its terrain with :func:`~humanoid_sim.world.terrain.heightfield_terrain_cfg`, an Isaac Lab
``TerrainImporterCfg`` that a task lays out in Python. Replays (:class:`~humanoid_sim.sim.SimCore`) and the runtime
host take that same config object (``terrain=``), so the scene they build is the training scene's terrain: the same
generator, the same collision geometry, the same solid base on MJWarp. This module adds what a command line needs on
top, without a second terrain code path:

* :func:`load_terrain` reads a terrain file into the same config, through ``heightfield_terrain_cfg``:

  * ``.npz``, one tile given as heights: ``heights`` ``[n, n]`` (m, the surface at the grid points; ``n`` at least 2)
    and ``horizontal_scale_m`` (the grid spacing). The tile is ``(n - 1) * horizontal_scale_m`` on a side, flat at
    z = 0 for ``border_m`` around it, and the robot starts at its centre (:func:`write_heightfield_file` writes one).
  * ``.json``, the tiles of a training terrain: the arguments of ``heightfield_terrain_cfg`` (``size_m``, ``rows``,
    ``cols``, ``border_m``, ``curriculum``, and optionally ``friction``, ``horizontal_scale_m``,
    ``vertical_scale_m``, ``seed``) and ``tiles``, ``{name: {"type": "HfRandomUniformTerrainCfg", ...fields}}`` of
    Isaac Lab's heightfield sub-terrain configs, or of its triangle-mesh ones (``"type":
    "MeshPyramidStairsTerrainCfg"``, from ``isaaclab.terrains.trimesh``), which make the terrain a
    ``mesh_terrain_cfg``. Set a ``seed`` for the same ground every time.

  Terrain generators and curricula of a task stay in humanoid-learning: it passes its own config in Python, or
  writes a heights file for the command line.
* :class:`Terrain` is a config with where it came from (file and SHA-256) for descriptors and recordings.
* :class:`Ground` reads the height of the generated surface back from the scene (:meth:`Ground.from_stage`), so a
  robot is put down on the ground that is there, whatever generated it.

Every env starts at its tile's origin, the highest point within 1 m of the tile centre (docs/world.md). Isaac Lab and
pxr are imported lazily; :class:`Ground` is NumPy only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TERRAIN_PRIM = "/World/ground/terrain"  # the importer's prim for the generated mesh (heightfield_terrain_cfg)
FILE_TILE = "file"
# Heights are kept to a millimetre in a heights file's tile (the training default step is 5 mm).
FILE_VERTICAL_SCALE_M = 0.001
FILE_BORDER_M = 4.0


# -- the terrain record ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Terrain:
    """A terrain config (an Isaac Lab ``TerrainImporterCfg``) and where it came from."""

    cfg: object
    source: str | None = None  # the file it was read from
    sha256: str | None = None

    def describe(self) -> dict:
        """What a client or recording needs to tell which terrain it is on: the source file with its digest, and the
        layout the config states."""
        gen = getattr(self.cfg, "terrain_generator", None)
        out = {"source": self.source, "sha256": self.sha256, "type": "heightfield"}
        if gen is None:
            return out
        if not all(getattr(c, "convert_to_heightfield", True) for c in gen.sub_terrains.values()):
            out["type"] = "mesh"  # one collision mesh, not a heightfield
        material = getattr(self.cfg, "physics_material", None)
        return out | {
            "size_m": gen.size[0], "rows": gen.num_rows, "cols": gen.num_cols, "border_m": gen.border_width,
            "curriculum": gen.curriculum, "horizontal_scale_m": gen.horizontal_scale,
            "vertical_scale_m": gen.vertical_scale, "seed": gen.seed, "tiles": sorted(gen.sub_terrains),
            "friction": None if material is None else material.static_friction}


def as_terrain(terrain) -> Terrain | None:
    """``terrain`` as a :class:`Terrain`: a bare ``TerrainImporterCfg`` has no source."""
    return terrain if terrain is None or isinstance(terrain, Terrain) else Terrain(terrain)


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# -- terrain files -----------------------------------------------------------------------------------------------


def write_heightfield_file(path: str | Path, heights_m: np.ndarray, *, horizontal_scale_m: float) -> None:
    """Write a heights file: ``heights_m`` ``[n, n]`` (the surface at the grid points, row ``i`` at x = ``i *
    horizontal_scale_m``, column ``j`` at y = ``j * horizontal_scale_m``)."""
    _check_heights(np.asarray(heights_m), horizontal_scale_m)
    np.savez_compressed(path, heights=np.asarray(heights_m, dtype=np.float32),
                        horizontal_scale_m=np.float64(horizontal_scale_m))


def read_heightfield_file(path: str | Path) -> tuple[np.ndarray, float]:
    """A heights file's ``heights`` (float64 ``[n, n]``) and ``horizontal_scale_m``."""
    with np.load(path) as f:
        for key in ("heights", "horizontal_scale_m"):
            if key not in f:
                raise ValueError(f"{path}: no {key!r} array (a heights file has heights and horizontal_scale_m)")
        heights, scale = np.asarray(f["heights"], dtype=np.float64), float(f["horizontal_scale_m"])
    _check_heights(heights, scale, path)
    return heights, scale


def _check_heights(heights: np.ndarray, scale: float, path=None) -> None:
    where = f"{path}: " if path else ""
    if heights.ndim != 2 or heights.shape[0] != heights.shape[1] or heights.shape[0] < 2:
        raise ValueError(f"{where}heights are a square [n, n] grid with n >= 2, got {heights.shape}")
    if not np.isfinite(heights).all():
        raise ValueError(f"{where}heights are not all finite")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"{where}horizontal_scale_m must be positive, got {scale}")


def load_terrain(path: str | Path, *, friction: float | None = None, border_m: float = FILE_BORDER_M) -> Terrain:
    """The terrain of a ``.npz`` heights file or a ``.json`` tile spec (see the module docstring), as the config
    training would build. ``friction`` (static and dynamic) defaults to the ground's 1.0 for a heights file and to
    the spec's; ``border_m`` is the flat apron around a heights file's tile."""
    path = Path(path)
    if path.suffix == ".npz":
        cfg = _heights_file_cfg(path, 1.0 if friction is None else friction, border_m)
    elif path.suffix == ".json":
        cfg = _tile_spec_cfg(path, friction)
    else:
        raise ValueError(f"{path}: a terrain file is a .npz of heights or a .json tile spec")
    return Terrain(cfg, source=str(path), sha256=sha256_of(path))


def _heights_file_cfg(path: Path, friction: float, border_m: float):
    from humanoid_sim.world.terrain import heightfield_terrain_cfg

    heights, scale = read_heightfield_file(path)
    side = (heights.shape[0] - 1) * scale
    tile = _file_tile_class()(proportion=1.0, path=str(path))
    return heightfield_terrain_cfg({FILE_TILE: tile}, size_m=side, rows=1, cols=1, border_m=border_m,
                                   curriculum=False, friction=friction, horizontal_scale_m=scale,
                                   vertical_scale_m=FILE_VERTICAL_SCALE_M)


_SPEC_KEYS = {"size_m", "rows", "cols", "border_m", "curriculum", "friction", "horizontal_scale_m",
              "vertical_scale_m", "seed", "tiles"}


def _tile_spec_cfg(path: Path, friction: float | None):
    import isaaclab.terrains.height_field as height_field
    import isaaclab.terrains.trimesh as trimesh
    from isaaclab.terrains import SubTerrainBaseCfg

    from humanoid_sim.world.terrain import heightfield_terrain_cfg, mesh_terrain_cfg

    spec = json.loads(path.read_text())
    unknown = sorted(set(spec) - _SPEC_KEYS)
    missing = sorted({"size_m", "rows", "cols", "border_m", "curriculum", "tiles"} - set(spec))
    if unknown or missing or not isinstance(spec.get("tiles"), dict) or not spec["tiles"]:
        raise ValueError(f"{path}: a tile spec has {sorted(_SPEC_KEYS)} (tiles: a non-empty object); "
                         f"unknown {unknown}, missing {missing}")
    tiles = {}
    for name, fields in spec["tiles"].items():
        fields = dict(fields)
        kind = fields.pop("type", None)
        cls = getattr(height_field, str(kind), None) or getattr(trimesh, str(kind), None)
        if not (isinstance(cls, type) and issubclass(cls, SubTerrainBaseCfg)):
            raise ValueError(f"{path}: tile {name!r}: type {kind!r} is not a sub-terrain of "
                             "isaaclab.terrains.height_field or isaaclab.terrains.trimesh")
        tiles[name] = cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in fields.items()})
    args = {k: v for k, v in spec.items() if k != "tiles"}
    if friction is not None:
        args["friction"] = friction
    if all(isinstance(c, height_field.HfTerrainBaseCfg) and c.convert_to_heightfield for c in tiles.values()):
        return heightfield_terrain_cfg(tiles, **args)
    for key in ("horizontal_scale_m", "vertical_scale_m"):
        if key in args:
            raise ValueError(f"{path}: {key} is for heightfield tiles; a spec with mesh tiles has none")
    return mesh_terrain_cfg(tiles, **args)


_FILE_TILE_CLASS = None


def _file_tile_class():
    """The heightfield sub-terrain config that reads its heights from a file (built once; Isaac Lab is imported
    here)."""
    global _FILE_TILE_CLASS
    if _FILE_TILE_CLASS is None:
        from dataclasses import MISSING

        from isaaclab.terrains.height_field import HfTerrainBaseCfg
        from isaaclab.utils import configclass

        @configclass
        class HfFileTerrainCfg(HfTerrainBaseCfg):
            function: str = "humanoid_sim.world.terrain_io:heights_file_tile"
            path: str = MISSING
            """The heights file."""

        _FILE_TILE_CLASS = HfFileTerrainCfg
    return _FILE_TILE_CLASS


def heights_file_tile(difficulty: float, cfg):
    """Isaac Lab's sub-terrain function for a heights file: the same height-field-to-mesh conversion the stock
    ``Hf*`` terrains use, with the origin they use (the tile centre at the highest point within 1 m)."""
    import trimesh
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

    heights, scale = read_heightfield_file(cfg.path)
    if abs(scale - cfg.horizontal_scale) > 1e-9 or abs((heights.shape[0] - 1) * scale - cfg.size[0]) > 1e-6:
        raise ValueError(f"{cfg.path}: its grid ({heights.shape[0]} points at {scale} m) does not make the "
                         f"{cfg.size[0]} m tile at {cfg.horizontal_scale} m the generator asks for")
    raw = np.rint(heights / cfg.vertical_scale)
    if np.abs(raw).max() > np.iinfo(np.int16).max:
        raise ValueError(f"{cfg.path}: heights beyond {np.iinfo(np.int16).max * cfg.vertical_scale:.1f} m")
    vertices, triangles = convert_height_field_to_mesh(raw.astype(np.int16), cfg.horizontal_scale,
                                                       cfg.vertical_scale, cfg.slope_threshold)
    centre = (heights.shape[0] - 1) / 2
    reach = int(round(1.0 / cfg.horizontal_scale))
    lo, hi = int(centre) - reach, int(np.ceil(centre)) + reach + 1
    top = raw[max(lo, 0):hi, max(lo, 0):hi].max() * cfg.vertical_scale
    return [trimesh.Trimesh(vertices=vertices, faces=triangles)], np.array([centre * scale, centre * scale, top])


# -- the ground height -------------------------------------------------------------------------------------------


class Ground:
    """The height of a triangle-mesh surface at (x, y), by a vertical ray through the mesh. NumPy only."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if not len(self.faces):
            raise ValueError("a ground mesh needs triangles")
        self.lower, self.upper = self.vertices.min(axis=0), self.vertices.max(axis=0)

    @classmethod
    def from_stage(cls, prim_path: str = TERRAIN_PRIM) -> Ground:
        """The meshes under ``prim_path`` in the current USD stage, in world coordinates: the surface the scene was
        built with."""
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdGeom

        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            raise RuntimeError(f"no terrain prim at {prim_path}: the scene was built without a terrain")
        vertices, faces, offset = [], [], 0
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64).reshape(-1, 3)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if not (counts == 3).all():
                raise RuntimeError(f"{prim.GetPath()}: the terrain mesh has faces that are not triangles")
            world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            vertices.append(points @ world[:3, :3] + world[3, :3])  # USD matrices act on row vectors
            faces.append(indices.reshape(-1, 3) + offset)
            offset += len(points)
        if not vertices:
            raise RuntimeError(f"no mesh under {prim_path}")
        return cls(np.concatenate(vertices), np.concatenate(faces))

    def height(self, xy) -> np.ndarray:
        """The surface height (m) at each (x, y) ``[N, 2]``; NaN where there is no surface. Where triangles overlap
        in plan, the highest."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        if not len(xy):
            return out
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        keep = np.ones(len(self.faces), dtype=bool)
        for axis in (0, 1):
            coords = self.vertices[self.faces[keep], axis]
            keep[keep] = (coords.max(axis=1) >= lo[axis]) & (coords.min(axis=1) <= hi[axis])
        tri = self.vertices[self.faces[keep]]  # [F, 3, 3]
        if not len(tri):
            return out
        a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        ok = np.abs(det) > 1e-12
        a, e1, e2, det = a[ok], e1[ok], e2[ok], det[ok]
        eps = 1e-9
        for start in range(0, len(xy), 256):
            d = xy[start:start + 256, None, :] - a[None, :, :2]  # [K, F, 2]
            u = (d[..., 0] * e2[None, :, 1] - d[..., 1] * e2[None, :, 0]) / det
            v = (e1[None, :, 0] * d[..., 1] - e1[None, :, 1] * d[..., 0]) / det
            inside = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
            z = a[None, :, 2] + u * e1[None, :, 2] + v * e2[None, :, 2]
            out[start:start + 256] = np.where(inside, z, -np.inf).max(axis=1)
        out[np.isinf(out)] = np.nan
        return out

    def describe(self) -> dict:
        """The mesh's extent in x and y, for descriptors. (Its height range would include the underside of the
        terrain's border, which hangs 1 m below the surface.)"""
        return {"x_m": [float(self.lower[0]), float(self.upper[0])], "y_m": [float(self.lower[1]), float(self.upper[1])]}


def rest_base_height(ground: Ground, points: np.ndarray, radius: np.ndarray, xy, yaw: float = 0.0,
                     clearance: float = 0.0) -> float:
    """The height of an upright base at ``xy`` turned ``yaw`` about the vertical at which a robot's collision
    shapes just clear ``ground``. ``points`` ``[N, 3]`` and ``radius`` ``[N]`` bound the shapes from below in the base
    frame (:func:`~humanoid_sim.model.kinematics.collision_points_in_base`); each point must be above the surface
    under it, less its radius, ``clearance`` aside. Raises ValueError where the robot reaches past the terrain."""
    c, s = np.cos(yaw), np.sin(yaw)
    plan = np.stack([c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1]], axis=1) + np.asarray(xy)
    surface = ground.height(plan)
    if np.isnan(surface).any():
        raise ValueError(f"the robot at {tuple(float(v) for v in xy)} reaches past the terrain")
    return float((surface + radius - points[:, 2]).max() + clearance)
