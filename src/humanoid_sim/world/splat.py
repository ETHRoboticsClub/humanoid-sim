"""A scanned room's Gaussian splat as a USD ``ParticleField3DGaussianSplat``, for cameras to render.

Controls' hangar splat (``hangar_splat.ply``) is a standard 3D Gaussian Splatting point cloud: a binary
little-endian PLY whose vertices carry ``x y z``, ``f_dc_0..2`` (the degree-0 spherical harmonic per colour channel),
``f_rest_*`` (the higher degrees, all of the red channel's first, then green's, then blue's), ``opacity`` (a logit),
``scale_0..2`` (natural logs of the per-axis standard deviations) and ``rot_0..3`` (an unnormalized quaternion, real
part first). This module writes it as the OpenUSD 26.03 schema reads a splat
(``UsdVol.ParticleField3DGaussianSplat``):

* ``positions`` in the world frame: the PLY's frame moved by ``-spawn``, without rotation or scale, as the room's
  boxes and scan are (``world/boxes.py``, ``world/scan.py``) and as Controls' ``scripts/sim/splat_camera.py``
  moves the splat (``means = xyz - HANGAR_SPAWN``);
* ``orientations`` normalized, ``scales`` as ``exp(scale)`` and ``opacities`` as ``sigmoid(opacity)``, the
  conversions of Newton's ``Gaussian.create_from_ply`` and of Controls' loader;
* ``radiance:sphericalHarmonicsCoefficients``: per particle a contiguous block of ``(degree + 1)^2`` RGB
  coefficients, the degree-0 one first and then each degree's in turn (the PLY's ``f_rest`` order, which is the order
  3DGS and gsplat evaluate), with ``radiance:sphericalHarmonicsDegree``;
* ``extent`` over the centres widened by three standard deviations of the largest Gaussian;
* bound to Omniverse's ``ParticleFieldEmissive.mdl``, the material RTX renders splats with.

``sh_degree`` below the file's drops the higher degrees (a degree-0 splat keeps each Gaussian's base colour and loses
only view-dependent shading); ``half`` stores the coefficients as ``half3`` (``...Coefficientsh``), which the schema
and Newton read when the float version is absent. Physics never sees a splat: the robot collides with the boxes,
and a lidar sees the scan mesh and the boxes.

Parts of the room that move, such as a gantry crane's bridge, trolley and hook, can be cut out into prims of their
own (:func:`split`, ``parts``): each part's Gaussians go under an Xform ``/Splat/<part>`` whose translation (zero as
written) a renderer sets to move it. Gaussians in a ``drop`` box are left out, for things the scan caught that the
simulation draws itself (the robot hanging in the gantry, its rope). A part's boxes are given in the PLY's frame.

    humanoid-sim splat-usd hangar_splat.ply hangar_splat.usdc --spawn -4.42 3.48 [--parts crane_parts.json]
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRIM_PATH = "/Splat/Gaussians"
MATERIAL_PATH = "/Splat/Looks/ParticleFieldEmissive"
EXTENT_SIGMAS = 3.0  # a Gaussian's footprint for its bounds: three standard deviations
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4", "short": "<i2", "ushort": "<u2"}


@dataclass(frozen=True)
class Splat:
    """A splat's Gaussians in a frame of its own: ``positions`` ``[N, 3]`` (m), ``orientations_xyzw`` ``[N, 4]``
    (unit), ``scales`` ``[N, 3]`` (standard deviations, m), ``opacities`` ``[N]`` in [0, 1], ``sh`` ``[N, K, 3]``
    with ``K = (sh_degree + 1)^2`` (RGB per coefficient, degree 0 first)."""

    positions: np.ndarray
    orientations_xyzw: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray
    sh_degree: int

    def __len__(self) -> int:
        return len(self.positions)


def read_ply_header(path: str | Path) -> tuple[int, np.dtype, int]:
    """``(vertex count, vertex dtype, header bytes)`` of a binary little-endian PLY holding only vertices."""
    with Path(path).open("rb") as f:
        head = f.read(64)
        if head.startswith(b"version https://git-lfs"):
            raise ValueError(f"{path} is a git-lfs pointer; run 'git lfs pull' where it comes from")
        f.seek(0)
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        count, fields, element = None, [], None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            words = line.decode("ascii").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "format":
                if words[1] != "binary_little_endian":
                    raise ValueError(f"{path}: format {words[1]}, only binary_little_endian is read")
            elif words[0] == "element":
                element = words[1]
                if element != "vertex":
                    raise ValueError(f"{path}: element {element!r}: a splat PLY holds only vertices")
                count = int(words[2])
            elif words[0] == "property":
                if words[1] == "list":
                    raise ValueError(f"{path}: list property {words[-1]!r} in a splat PLY")
                if words[1] not in _PLY_TYPES:
                    raise ValueError(f"{path}: property type {words[1]!r}")
                fields.append((words[2], _PLY_TYPES[words[1]]))
            elif words[0] == "end_header":
                return count or 0, np.dtype(fields), f.tell()


def read_ply(path: str | Path) -> Splat:
    """The splat in a standard 3DGS PLY (see the module docstring), in the PLY's frame."""
    count, dtype, offset = read_ply_header(path)
    names = set(dtype.names)
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"}
    if missing := required - names:
        raise ValueError(f"{path}: not a 3DGS splat, no {sorted(missing)}")
    v = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))

    def stack(*keys) -> np.ndarray:
        return np.stack([np.asarray(v[k], dtype=np.float32) for k in keys], axis=1)

    n_rest = sum(1 for n in dtype.names if n.startswith("f_rest_"))
    degree = math.isqrt(n_rest // 3 + 1) - 1
    if 3 * ((degree + 1) ** 2 - 1) != n_rest:
        raise ValueError(f"{path}: {n_rest} f_rest values are no spherical harmonic degree's")
    dc = stack("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    if n_rest:
        # f_rest holds the red channel's coefficients, then green's, then blue's: [N, 3, K-1] -> [N, K-1, 3].
        rest = stack(*[f"f_rest_{i}" for i in range(n_rest)]).reshape(count, 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], axis=1)
    else:
        sh = dc
    wxyz = stack("rot_0", "rot_1", "rot_2", "rot_3")
    norm = np.linalg.norm(wxyz, axis=1, keepdims=True)
    if not np.all(norm > 0):
        raise ValueError(f"{path}: a Gaussian with a zero quaternion")
    xyzw = (wxyz / norm)[:, [1, 2, 3, 0]]
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    return Splat(positions=stack("x", "y", "z"), orientations_xyzw=np.ascontiguousarray(xyzw, dtype=np.float32),
                 scales=np.exp(stack("scale_0", "scale_1", "scale_2")), opacities=opacity.astype(np.float32),
                 sh=np.ascontiguousarray(sh, dtype=np.float32), sh_degree=degree)


def placed(splat: Splat, spawn_m: tuple[float, float] = (0.0, 0.0), sh_degree: int | None = None) -> Splat:
    """``splat`` in the world frame, its point ``spawn_m`` (x, y) at the origin, keeping at most ``sh_degree``."""
    degree = splat.sh_degree if sh_degree is None else sh_degree
    if not 0 <= degree <= splat.sh_degree:
        raise ValueError(f"sh_degree {sh_degree}: the splat has degrees 0..{splat.sh_degree}")
    offset = np.array([spawn_m[0], spawn_m[1], 0.0], dtype=np.float32)
    return Splat(positions=splat.positions - offset, orientations_xyzw=splat.orientations_xyzw,
                 scales=splat.scales, opacities=splat.opacities,
                 sh=np.ascontiguousarray(splat.sh[:, : (degree + 1) ** 2]), sh_degree=degree)


def extent(splat: Splat) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of the centres widened by :data:`EXTENT_SIGMAS` of the largest standard deviation."""
    pad = EXTENT_SIGMAS * float(splat.scales.max()) if len(splat) else 0.0
    return splat.positions.min(axis=0) - pad, splat.positions.max(axis=0) + pad


def select(splat: Splat, mask: np.ndarray) -> Splat:
    """The Gaussians where ``mask`` ``[N]`` is true."""
    return Splat(positions=splat.positions[mask], orientations_xyzw=splat.orientations_xyzw[mask],
                 scales=splat.scales[mask], opacities=splat.opacities[mask], sh=splat.sh[mask], sh_degree=splat.sh_degree)


def in_boxes(positions: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Which centres ``[N, 3]`` lie in any of ``boxes``: each ``center`` ``[3]``, ``half`` ``[3]`` (half extents) and
    ``yaw`` (rad, about the vertical, default 0)."""
    inside = np.zeros(len(positions), dtype=bool)
    for box in boxes:
        c, h, yaw = np.asarray(box["center"], float), np.asarray(box["half"], float), float(box.get("yaw", 0.0))
        d = positions - c
        cos, sin = math.cos(yaw), math.sin(yaw)
        local = np.stack([cos * d[:, 0] + sin * d[:, 1], -sin * d[:, 0] + cos * d[:, 1], d[:, 2]], axis=1)
        inside |= np.all(np.abs(local) <= h, axis=1)
    return inside


def split(splat: Splat, parts: dict[str, list[dict]]) -> tuple[Splat, dict[str, Splat]]:
    """The room without its parts, and each part (a name and its boxes, see :func:`in_boxes`); a Gaussian in
    several goes to the first. Gaussians in the ``drop`` part's boxes are in neither."""
    taken = np.zeros(len(splat), dtype=bool)
    out = {}
    for name, boxes in parts.items():
        if name != "drop" and (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in ("Gaussians", "Looks")):
            raise ValueError(f"part {name!r}: a USD prim name other than Gaussians and Looks")
        mask = in_boxes(splat.positions, boxes) & ~taken
        taken |= mask
        if name != "drop":
            out[name] = select(splat, mask)
    return select(splat, ~taken), out


def write_usd(splat: Splat, out: str | Path, *, half: bool = False, source: str | None = None,
              parts: dict[str, Splat] | None = None) -> Path:
    """Write ``splat`` (already placed) to a new USD layer at ``out`` (``.usd``, ``.usdc`` or ``.usda``), its
    default prim ``/Splat`` holding the Gaussians at :data:`PRIM_PATH` and each of ``parts`` at
    ``/Splat/<part>/Gaussians`` under an Xform with a zero translation. ``source`` is recorded in the layer's
    documentation."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Splat")
    stage.SetDefaultPrim(root.GetPrim())
    if source:
        stage.GetRootLayer().documentation = source

    material = UsdShade.Material.Define(stage, MATERIAL_PATH)
    shader = UsdShade.Shader.Define(stage, f"{MATERIAL_PATH}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("ParticleFieldEmissive.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("ParticleFieldEmissive", "mdl")
    out_port = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for name in ("surface", "volume", "displacement"):
        material.CreateOutput(f"mdl:{name}", Sdf.ValueTypeNames.Token).ConnectToSource(out_port)
    _write_gaussians(stage, PRIM_PATH, splat, half, material)
    for name, part in (parts or {}).items():
        UsdGeom.Xform.Define(stage, f"/Splat/{name}").AddTranslateOp()
        _write_gaussians(stage, f"/Splat/{name}/Gaussians", part, half, material)
    stage.GetRootLayer().Save()
    return out


def _write_gaussians(stage, path: str, splat: Splat, half: bool, material) -> None:
    from pxr import Gf, Sdf, UsdShade, Vt

    prim = stage.DefinePrim(path, "ParticleField3DGaussianSplat")
    T = Sdf.ValueTypeNames
    prim.CreateAttribute("positions", T.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(splat.positions))
    prim.CreateAttribute("orientations", T.QuatfArray).Set(Vt.QuatfArray.FromNumpy(splat.orientations_xyzw))
    prim.CreateAttribute("scales", T.Float3Array).Set(Vt.Vec3fArray.FromNumpy(splat.scales))
    prim.CreateAttribute("opacities", T.FloatArray).Set(Vt.FloatArray.FromNumpy(splat.opacities))
    flat = splat.sh.reshape(-1, 3)
    if half:
        prim.CreateAttribute("radiance:sphericalHarmonicsCoefficientsh", T.Half3Array).Set(
            Vt.Vec3hArray.FromNumpy(flat.astype(np.float16)))
    else:
        prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients", T.Float3Array).Set(
            Vt.Vec3fArray.FromNumpy(flat))
    prim.CreateAttribute("radiance:sphericalHarmonicsDegree", T.Int).Set(int(splat.sh_degree))
    lo, hi = extent(splat)
    prim.CreateAttribute("extent", T.Float3Array).Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)),
                                                                      Gf.Vec3f(*map(float, hi))]))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def convert(ply: str | Path, out: str | Path, *, spawn_m: tuple[float, float] = (0.0, 0.0),
            sh_degree: int | None = None, half: bool = False, parts: str | Path | None = None) -> dict:
    """Convert a 3DGS PLY to a USD splat placed with ``spawn_m`` at the origin, with the moving ``parts`` (a JSON
    object of part names to boxes in the PLY's frame, see :func:`split`) cut out; returns what was written."""
    ply = Path(ply)
    h = hashlib.sha256()
    with ply.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    room, cut = read_ply(ply), {}
    if parts is not None:
        room, cut = split(room, json.loads(Path(parts).read_text()))
    splat = placed(room, spawn_m, sh_degree)
    cut = {name: placed(part, spawn_m, sh_degree) for name, part in cut.items()}
    path = write_usd(splat, out, half=half, parts=cut,
                     source=f"{ply.name} sha256 {h.hexdigest()}, spawn {list(spawn_m)}, degree {splat.sh_degree}")
    lo, hi = extent(splat)
    return {"usd": str(path), "ply": str(ply), "ply_sha256": h.hexdigest(), "gaussians": len(splat),
            "parts": {name: len(part) for name, part in cut.items()},
            "sh_degree": splat.sh_degree, "half": half, "spawn_m": list(spawn_m), "extent_min": lo.tolist(),
            "extent_max": hi.tolist(), "bytes": path.stat().st_size}
