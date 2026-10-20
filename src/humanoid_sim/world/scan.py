"""A scanned room's look, for drawing: a photo-textured mesh and a flat textured floor, placed with the room's static
boxes (``world/boxes.py``). The physics never sees it; the robot collides with the boxes.

Controls' hangar is such a scan: ``hangar_visual.obj`` (a texture coordinate on every face corner, the scan's own
floor cut out), its texture ``hangar_visual.png`` (the lighting baked in), and ``hangar_floor.png``, the floor seen
from above. The look follows Controls' MuJoCo ``hangar_scan`` environment (``ethrc_sim/environments.py`` @ 44353e2b):

* the mesh moves by ``-spawn`` like the boxes, without rotation or scale;
* MuJoCo flips an OBJ's v coordinate (the OBJ's origin is the image's bottom left) and keeps the image's first row
  at v = 0, so the texel a corner shows is at ``(u, 1 - v)`` from the image's top left, which is how Rerun reads a
  texture coordinate;
* the floor is a quad over the boxes file's ``floor_corners``, :data:`FLOOR_LIFT_M` above the ground, its texture's
  corners (0, 0), (1, 0), (1, 1), (0, 1) on the four corners in order;
* the ground plane itself is not drawn, the sky is a neutral :data:`SKY_RGB`, and the scan is not lit: its texture
  holds the light (Controls gives every OBJ normal +z so that MuJoCo lights every surface alike).

Two levels of detail (:data:`DETAILS`):

* ``full``: every triangle with its texture coordinates and the whole texture, for recordings to a file;
* ``live``: the vertices merged on a :data:`LIVE_CELL_M` grid and each coloured with the texture averaged over the
  face corners merged into it (the texture packs every triangle on its own patch, so a decimated mesh cannot keep
  texture coordinates), and the floor texture at most :data:`LIVE_FLOOR_PX` across, for viewers that stream.

Pillow reads the images, imported lazily.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DETAILS = ("full", "live")
FLOOR_LIFT_M = 0.001  # Controls' floor quad height (ethrc_sim/environments.py)
SKY_RGB = (0.78, 0.77, 0.74)  # Controls' SCAN_SKY
LIVE_CELL_M = 0.15  # about 100k vertices and 160k triangles of the hangar's 475k and 832k
LIVE_FLOOR_PX = 512


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        head = f.read(1024)
        if head.startswith(b"version https://git-lfs"):
            raise ValueError(f"{path} is a git-lfs pointer; run 'git lfs pull' where it comes from")
        h.update(head)
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ScanFiles:
    """A scan's files and their SHA-256 digests: the mesh (OBJ), its texture and, optionally, the floor's."""

    mesh: str
    texture: str
    floor_texture: str | None
    sha256: dict  # file -> digest

    @classmethod
    def of(cls, mesh: str | Path, texture: str | Path, floor_texture: str | Path | None = None) -> ScanFiles:
        files = [Path(mesh), Path(texture)] + ([Path(floor_texture)] if floor_texture is not None else [])
        return cls(mesh=str(files[0]), texture=str(files[1]),
                   floor_texture=None if floor_texture is None else str(files[2]),
                   sha256={str(f): _digest(f) for f in files})

    def describe(self) -> dict:
        def entry(f):
            return None if f is None else {"file": f, "sha256": self.sha256[f]}

        return {"mesh": entry(self.mesh), "texture": entry(self.texture), "floor_texture": entry(self.floor_texture)}


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An OBJ of triangles with a texture coordinate on every corner: positions ``[V, 3]``, texture coordinates
    ``[T, 2]`` as written (origin at the image's bottom left), and per face the position and texture coordinate
    indices ``[F, 3]`` (from 0). Normals, groups and materials are ignored."""
    v, vt, f = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line[2:])
            elif line.startswith("vt "):
                vt.append(line[3:])
            elif line.startswith("f "):
                f.append(line[2:])
    if not v or not vt or not f:
        raise ValueError(f"{path}: needs positions, texture coordinates and faces")
    positions = np.array(" ".join(v).split(), dtype=np.float64).reshape(len(v), -1)[:, :3]
    texcoords = np.array(" ".join(vt).split(), dtype=np.float64).reshape(len(vt), -1)[:, :2]
    first = f[0].split()[0]
    fields = first.count("/") + 1
    if fields < 2 or "//" in first:
        raise ValueError(f"{path}: every face corner needs a texture coordinate (v/vt or v/vt/vn)")
    numbers = " ".join(f).replace("/", " ").split()
    if len(numbers) != 3 * fields * len(f):
        raise ValueError(f"{path}: faces must all be triangles, every corner written like the first ({first})")
    idx = np.array(numbers, dtype=np.int64).reshape(len(f), 3, fields)[..., :2].copy()
    if (idx < 1).any():
        raise ValueError(f"{path}: only positive (1-based) indices are supported")
    idx -= 1
    if idx[..., 0].max() >= len(positions) or idx[..., 1].max() >= len(texcoords):
        raise ValueError(f"{path}: a face indexes past the positions or texture coordinates")
    return positions, texcoords, idx[..., 0], idx[..., 1]


def read_image(path: str | Path) -> np.ndarray:
    """An image as ``[H, W, 3]`` uint8, its first row the top."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def rerun_texcoords(texcoords: np.ndarray) -> np.ndarray:
    """OBJ texture coordinates (origin at the image's bottom left) as Rerun reads them (origin at the top left)."""
    out = np.array(texcoords, dtype=np.float64)
    out[:, 1] = 1.0 - out[:, 1]
    return out


@dataclass
class ScanMesh:
    """One drawable mesh: positions (world frame), optional triangles (None: consecutive triples), and either
    texture coordinates with a texture or vertex colours."""

    positions: np.ndarray  # [V, 3]
    triangles: np.ndarray | None  # [F, 3]
    texcoords: np.ndarray | None = None  # [V, 2], Rerun's convention
    texture: np.ndarray | None = None  # [H, W, 3] uint8
    colors: np.ndarray | None = None  # [V, 3] uint8

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.positions, self.triangles, self.texcoords, self.texture, self.colors)
                   if a is not None)


def scan_meshes(files: ScanFiles, spawn_m: tuple[float, float], floor_corners: np.ndarray | None, *,
                detail: str = "full") -> dict[str, ScanMesh]:
    """The scan (``"scan"``) and, with a floor texture and ``floor_corners`` (``[4, 2]``, the scene file's frame),
    its floor (``"floor"``), placed with ``spawn_m`` at the world origin, at ``detail`` (see :data:`DETAILS`)."""
    if detail not in DETAILS:
        raise ValueError(f"detail {detail!r}: one of {DETAILS}")
    offset = np.array([-float(spawn_m[0]), -float(spawn_m[1]), 0.0])
    positions, texcoords, fv, ft = read_obj(files.mesh)
    texture = read_image(files.texture)
    out = {}
    if detail == "full":
        out["scan"] = ScanMesh(positions=(positions[fv] + offset).reshape(-1, 3).astype(np.float32), triangles=None,
                               texcoords=rerun_texcoords(texcoords[ft].reshape(-1, 2)).astype(np.float32),
                               texture=texture)
    else:
        out["scan"] = _clustered(positions, texcoords, fv, ft, texture, LIVE_CELL_M, offset)
    if files.floor_texture is not None and floor_corners is not None:
        corners = np.asarray(floor_corners, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(f"floor_corners {corners.shape}: a textured floor needs 4 corners")
        floor_texture = read_image(files.floor_texture)
        if detail == "live":
            floor_texture = _downscaled(floor_texture, LIVE_FLOOR_PX)
        quad = np.c_[corners, np.full(4, FLOOR_LIFT_M)] + offset
        out["floor"] = ScanMesh(positions=quad.astype(np.float32), triangles=np.array([[0, 1, 2], [0, 2, 3]]),
                                texcoords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
                                texture=floor_texture)
    return out


def _texels(texture: np.ndarray, uv_rerun: np.ndarray) -> np.ndarray:
    """The texel at each texture coordinate (Rerun's convention), nearest."""
    h, w = texture.shape[:2]
    col = np.clip((uv_rerun[:, 0] * w).astype(np.int64), 0, w - 1)
    row = np.clip((uv_rerun[:, 1] * h).astype(np.int64), 0, h - 1)
    return texture[row, col]


def _clustered(positions, texcoords, fv, ft, texture, cell_m, offset) -> ScanMesh:
    """The mesh with its vertices merged per ``cell_m`` grid cell (each at the mean of its members), degenerate and
    repeated triangles dropped, each vertex coloured with the mean texel over the face corners merged into it."""
    keys = np.floor(positions / cell_m).astype(np.int64)
    _, cluster, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    cluster = cluster.ravel()
    n = len(counts)
    merged = np.zeros((n, 3))
    np.add.at(merged, cluster, positions)
    merged /= counts[:, None]
    corner_cluster = cluster[fv].ravel()
    texel = _texels(texture, rerun_texcoords(texcoords[ft].reshape(-1, 2))).astype(np.float64)
    color_sum = np.zeros((n, 3))
    np.add.at(color_sum, corner_cluster, texel)
    corner_count = np.bincount(corner_cluster, minlength=n)
    colors = np.round(color_sum / np.maximum(corner_count, 1)[:, None]).astype(np.uint8)
    tris = cluster[fv]
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
    tris = tris[np.sort(first)]
    used = np.unique(tris)
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return ScanMesh(positions=(merged[used] + offset).astype(np.float32), triangles=remap[tris].astype(np.uint32),
                    colors=colors[used])


def _downscaled(image: np.ndarray, max_px: int) -> np.ndarray:
    from PIL import Image

    h, w = image.shape[:2]
    scale = max_px / max(h, w)
    if scale >= 1.0:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
