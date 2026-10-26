"""Static boxes around the robot: a scene's fixed collision geometry (walls, furniture), read from a file.

The physics collides the robot with convex shapes and heightfields, not triangle meshes (MJWarp's collision
pipeline, see ``world/terrain.py``), so a scanned room enters the scene as boxes fitted to the scan. They stand on
the scene's own flat ground, which stays the walking surface.

The file is an ``.npz``, in metres and radians, z up, the floor at z = 0:

* ``center`` ``[N, 3]``: box centres in the scene file's frame;
* ``half`` ``[N, 3]``: half extents along the box axes;
* ``yaw``: rotation of each box about its centre's vertical axis, one value for every box or ``[N]``;
* ``rgb`` ``[N, 3]`` (optional): colours in [0, 1], for drawing;
* ``wall`` ``[N]`` bool (optional): which boxes are walls, drawn light grey and kept for the lidar over a scan mesh
  (``sensors/room_rays.py``);
* ``floor_corners`` ``[4, 2]`` (optional): the outline of a scan's floor, where its floor texture is drawn
  (``world/scan.py``).

Other arrays in the file are ignored. A scan's textured mesh can come with the boxes (``scan``), for drawing only.

Controls' hangar proxy (``ros2_ws/src/sim/ethrc_sim/environments/hangar/hangar_proxy.npz``) is such a file, and
:meth:`StaticBoxes.load` places it as Controls' MuJoCo scene does (``ethrc_sim/environments.py`` @ 44353e2b):
``spawn`` is the scene point put at the world origin, where the robot starts (Controls' ``HANGAR_SPAWN``), and
every box keeps its yaw about its own centre.

In an Isaac Lab scene the boxes are USD ``Cube`` prims with the collision API under :data:`STATIC_BOXES_PRIM`,
outside the env namespace: one set of static shapes that every env's solver world collides with, at the same
world coordinates. A scene with boxes therefore starts every env's robot at the world origin (each env is its own
solver world, so the robots never touch each other). Isaac Lab and pxr are imported lazily.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.world.scan import ScanFiles

STATIC_BOXES_PRIM = "/World/StaticBoxes"
WALL_RGB = (0.8, 0.78, 0.74)  # drawing only
DEFAULT_RGB = (0.62, 0.6, 0.57)


@dataclass(frozen=True)
class StaticBoxes:
    """Boxes placed in the world frame (``world = scene file frame - spawn``)."""

    center: np.ndarray  # [N, 3] world frame (m)
    half: np.ndarray  # [N, 3] half extents along the box axes (m)
    yaw: np.ndarray  # [N] rotation about z (rad)
    rgb: np.ndarray  # [N, 3] drawing colour in [0, 1]
    file: str
    sha256: str
    spawn_m: tuple[float, float]
    wall: np.ndarray | None = None  # [N] bool, which boxes are walls; None when the file does not say
    floor_corners: np.ndarray | None = None  # [4, 2] the file's floor outline, in the file's frame
    scan: ScanFiles | None = None  # the room's textured scan, for drawing

    @classmethod
    def load(cls, path: str | Path, spawn_m: Sequence[float] = (0.0, 0.0), *, scan: ScanFiles | None = None
             ) -> StaticBoxes:
        """The boxes of the file at ``path``, with the scene point ``spawn_m`` (x, y) at the world origin, and the
        room's ``scan`` for drawing."""
        path = Path(path)
        raw = path.read_bytes()
        if len(raw) < 1024 and raw.startswith(b"version https://git-lfs"):
            raise ValueError(f"{path} is a git-lfs pointer; run 'git lfs pull' where it comes from")
        if len(spawn_m) != 2:
            raise ValueError(f"spawn_m {tuple(spawn_m)}: the scene point (x, y) placed at the world origin")
        with np.load(path) as data:
            missing = {"center", "half", "yaw"} - set(data.files)
            if missing:
                raise ValueError(f"{path}: static boxes need {sorted(missing)}")
            center = np.asarray(data["center"], dtype=np.float64)
            half = np.asarray(data["half"], dtype=np.float64)
            n = len(center)
            yaw = np.broadcast_to(np.asarray(data["yaw"], dtype=np.float64), (n,)).copy()
            rgb = (np.asarray(data["rgb"], dtype=np.float64) if "rgb" in data.files
                   else np.tile(DEFAULT_RGB, (n, 1)))
            wall = np.asarray(data["wall"]) if "wall" in data.files else None
            if wall is not None:
                if wall.dtype != bool or wall.shape != (n,):
                    raise ValueError(f"{path}: wall must be [N] bool (N = {n}), got {wall.dtype} {wall.shape}")
                rgb = np.where(wall[:, None], WALL_RGB, rgb)
            floor = np.asarray(data["floor_corners"], dtype=np.float64) if "floor_corners" in data.files else None
        if center.shape != (n, 3) or half.shape != (n, 3) or rgb.shape != (n, 3):
            raise ValueError(f"{path}: center, half and rgb must be [N, 3] (N = {n})")
        if n == 0 or not (half > 0).all():
            raise ValueError(f"{path}: needs at least one box, every half extent positive")
        if not (np.isfinite(center).all() and np.isfinite(half).all() and np.isfinite(yaw).all()):
            raise ValueError(f"{path}: non-finite box values")
        offset = np.array([float(spawn_m[0]), float(spawn_m[1]), 0.0])
        return cls(center=center - offset, half=half, yaw=yaw, rgb=np.clip(rgb, 0.0, 1.0), file=str(path),
                   sha256=hashlib.sha256(raw).hexdigest(), spawn_m=(float(spawn_m[0]), float(spawn_m[1])),
                   wall=wall, floor_corners=floor, scan=scan)

    def __len__(self) -> int:
        return len(self.center)

    def quat_xyzw(self) -> np.ndarray:
        """``[N, 4]`` box orientations: the yaw about z."""
        q = np.zeros((len(self), 4))
        q[:, 2], q[:, 3] = np.sin(self.yaw / 2), np.cos(self.yaw / 2)
        return q

    def corners(self) -> np.ndarray:
        """``[N, 8, 3]`` world-frame box corners."""
        return box_corners(self.center, self.half, self.yaw)

    def clearance_m(self, xy: Sequence[float] = (0.0, 0.0)) -> float:
        """Horizontal distance from the world point ``xy`` to the nearest box's footprint, 0 inside one."""
        d = np.array([float(xy[0]), float(xy[1])]) - self.center[:, :2]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)  # into each box's axes
        outside = np.maximum(np.abs(local) - self.half[:, :2], 0.0)
        return float(np.linalg.norm(outside, axis=1).min())

    def describe(self) -> dict:
        """What a client needs to tell which scene it is in and where."""
        return {"file": self.file, "sha256": self.sha256, "boxes": len(self), "spawn_m": list(self.spawn_m),
                "spawn_clearance_m": self.clearance_m(), "scan": None if self.scan is None else self.scan.describe()}


def box_corners(center: np.ndarray, half: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """``[N, 8, 3]`` corners of boxes with ``center`` ``[N, 3]``, ``half`` ``[N, 3]`` turned by ``yaw`` ``[N]``
    about z."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    local = signs[None] * half[:, None, :]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x = c * local[..., 0] - s * local[..., 1]
    y = s * local[..., 0] + c * local[..., 1]
    return np.stack([x, y, local[..., 2]], axis=-1) + center[:, None, :]


def box_mesh(corners: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    """One triangle mesh of boxes given as :func:`box_corners`, four vertices per face so each face is lit flat:
    vertices ``[24N, 3]``, triangles ``[12N, 3]``, vertex normals ``[24N, 3]`` and vertex colours ``[24N, 3]``
    (uint8)."""
    # Corner k has signs (x, y, z) = bits (k >> 2, k >> 1, k) of k: faces as quads, wound outward.
    quads = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
    n = len(corners)
    verts = corners[:, quads.reshape(-1)]  # [N, 24, 3]
    normals = np.cross(verts[:, 1::4] - verts[:, 0::4], verts[:, 3::4] - verts[:, 0::4])  # [N, 6, 3]
    normals = np.repeat(normals / np.linalg.norm(normals, axis=-1, keepdims=True), 4, axis=1)
    quad_tris = np.array([[0, 1, 2], [0, 2, 3]])
    tris = (quad_tris[None, None] + 4 * np.arange(6)[None, :, None, None]
            + 24 * np.arange(n)[:, None, None, None]).reshape(-1, 3)
    colors = (np.repeat(np.clip(rgb, 0.0, 1.0), 24, axis=0) * 255).astype(np.uint8)
    return verts.reshape(-1, 3), tris, normals.reshape(-1, 3), colors


def box_prim_path(index: int, root: str = STATIC_BOXES_PRIM) -> str:
    """The ``Cube`` prim of the file's box ``index``."""
    return f"{root}/box_{index:04d}"


def spawn_static_boxes(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab spawner of :class:`StaticBoxesCfg`: an Xform at ``prim_path`` holding one physics material (the
    ground's friction, no restitution) and one ``Cube`` prim per box with the collision API, its display colour
    the box's."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    from pxr import Gf, UsdGeom, UsdPhysics

    if not (translation is None or np.allclose(translation, 0.0)) or not (
            orientation is None or np.allclose(orientation, (0.0, 0.0, 0.0, 1.0))):
        raise ValueError("static boxes are placed by their file and spawn point, not by an entity transform")
    boxes = StaticBoxes.load(cfg.file, cfg.spawn_m)
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    material = f"{prim_path}/PhysicsMaterial"
    sim_utils.spawn_rigid_body_material(material, RigidBodyMaterialBaseCfg(
        static_friction=cfg.friction, dynamic_friction=cfg.friction, restitution=0.0))
    quat = boxes.quat_xyzw()
    for i in range(len(boxes)):
        cube = UsdGeom.Cube.Define(stage, box_prim_path(i, prim_path))
        cube.CreateSizeAttr(2.0)  # unit half extents, scaled to the box's
        cube.AddTranslateOp().Set(Gf.Vec3d(*boxes.center[i]))
        cube.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(quat[i, 3]), *map(float, quat[i, :3])))
        cube.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*boxes.half[i]))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, boxes.rgb[i]))])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        sim_utils.bind_physics_material(cube.GetPath(), material, stage=stage)
    return root


def static_boxes_cfg(file: str | Path, spawn_m: Sequence[float] = (0.0, 0.0), *, friction: float):
    """The boxes in ``file`` (``spawn_m`` at the world origin) as an Isaac Lab scene entity at
    :data:`STATIC_BOXES_PRIM`, every box with the ground's ``friction``."""
    from dataclasses import MISSING

    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import SpawnerCfg
    from isaaclab.utils import configclass

    @configclass
    class StaticBoxesCfg(SpawnerCfg):
        func: Callable = spawn_static_boxes
        file: str = MISSING
        spawn_m: tuple[float, float] = (0.0, 0.0)
        friction: float = MISSING

    spawn = StaticBoxesCfg(file=str(file), spawn_m=(float(spawn_m[0]), float(spawn_m[1])), friction=float(friction))
    return AssetBaseCfg(prim_path=STATIC_BOXES_PRIM, spawn=spawn)


def stage_boxes(stage=None) -> tuple[np.ndarray, np.ndarray]:
    """The static boxes on the USD stage as drawn: world-frame corners ``[N, 8, 3]`` and colours ``[N, 3]``
    (empty without boxes)."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom

    stage = stage or sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(STATIC_BOXES_PRIM)
    corners, colors = [], []
    if root.IsValid():
        unit = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Cube):
                continue
            cube = UsdGeom.Cube(prim)
            xf = np.asarray(cube.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            local = unit * cube.GetSizeAttr().Get() / 2
            corners.append(local @ xf[:3, :3] + xf[3, :3])  # USD matrices are row-major, points as rows
            color = cube.GetDisplayColorAttr().Get()
            colors.append(np.asarray(color[0]) if color else np.asarray(DEFAULT_RGB))
    if not corners:
        return np.zeros((0, 8, 3)), np.zeros((0, 3))
    return np.stack(corners), np.stack(colors)
