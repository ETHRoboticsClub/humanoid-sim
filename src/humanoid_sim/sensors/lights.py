"""A room's ceiling lights for the RTX renderer, from a file the room's owner writes: area lights facing straight
down, given in the scan's frame and moved into the world frame by ``-spawn`` like the room's boxes, scan and splat.

The file (JSON)::

    {"fixtures": [{"position_m": [x, y, z], "size_m": [w, l], "intensity": 3000, "color": [1, 1, 1]},
                  {"position_m": [x, y, z], "radius_m": 0.3, "intensity": 3000}],
     "dome_intensity": 100}

A fixture with ``size_m`` is a rectangle (USD ``RectLight``, ``w`` along x, ``l`` along y), one with ``radius_m`` a
disk (``DiskLight``); both emit along their -z, down. ``intensity`` is the emitting surface's luminance in nits,
``color`` linear RGB (white when left out). ``dome_intensity``, when given, replaces the session's dome light (the
ambient fill, nits; 0 for none). Isaac Lab has no rectangle light spawner, so the lights are authored with UsdLux.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LIGHTS_PRIM = "/World/CeilingLights"


@dataclass(frozen=True)
class Fixture:
    position_m: tuple[float, float, float]  # the scan's frame
    intensity: float  # nits
    size_m: tuple[float, float] | None = None  # rectangle (x, y)
    radius_m: float | None = None  # disk
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class CeilingLights:
    fixtures: tuple[Fixture, ...]
    dome_intensity: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> CeilingLights:
        return cls.parse(json.loads(Path(path).read_text()), source=str(path))

    @classmethod
    def parse(cls, data: dict, source: str = "lights") -> CeilingLights:
        unknown = set(data) - {"fixtures", "dome_intensity"}
        if unknown:
            raise ValueError(f"{source}: unknown keys {sorted(unknown)}")
        fixtures = tuple(_fixture(f, f"{source}: fixture {i}") for i, f in enumerate(data.get("fixtures") or []))
        if not fixtures:
            raise ValueError(f"{source}: no fixtures")
        dome = data.get("dome_intensity")
        if dome is not None and not (math.isfinite(dome) and dome >= 0):
            raise ValueError(f"{source}: dome_intensity {dome}: a finite value >= 0")
        return cls(fixtures, None if dome is None else float(dome))

    def placed(self, spawn_m: tuple[float, float]) -> list[tuple[float, float, float]]:
        """Every fixture's position in the world frame, the scan's point ``spawn_m`` at the origin."""
        return [(f.position_m[0] - spawn_m[0], f.position_m[1] - spawn_m[1], f.position_m[2]) for f in self.fixtures]


def _fixture(f: dict, where: str) -> Fixture:
    unknown = set(f) - {"position_m", "size_m", "radius_m", "intensity", "color"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if ("size_m" in f) == ("radius_m" in f):
        raise ValueError(f"{where}: give one of size_m (rectangle) and radius_m (disk)")
    position = _floats(f.get("position_m"), 3, f"{where}: position_m")
    size = _floats(f["size_m"], 2, f"{where}: size_m") if "size_m" in f else None
    radius = float(f["radius_m"]) if "radius_m" in f else None
    color = _floats(f.get("color", (1.0, 1.0, 1.0)), 3, f"{where}: color")
    intensity = float(f.get("intensity", math.nan))
    if any(not (math.isfinite(v) and v > 0) for v in (*(size or ()), *(() if radius is None else (radius,)))):
        raise ValueError(f"{where}: size_m and radius_m must be > 0")
    if not (math.isfinite(intensity) and intensity >= 0):
        raise ValueError(f"{where}: intensity (nits) must be given, finite and >= 0")
    if any(v < 0 for v in color):
        raise ValueError(f"{where}: color must be >= 0")
    return Fixture(position, intensity, size, radius, color)


def _floats(value, n: int, where: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{where}: {n} numbers")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{where}: finite numbers")
    return out


def ceiling_light_prims(stage, lights: CeilingLights, spawn_m: tuple[float, float],
                        prim_path: str = LIGHTS_PRIM) -> list:
    """Author ``lights`` under ``prim_path`` on ``stage``, placed with ``spawn_m``; returns the light prims."""
    from pxr import Gf, UsdGeom, UsdLux

    UsdGeom.Xform.Define(stage, prim_path)
    prims = []
    for i, (fixture, position) in enumerate(zip(lights.fixtures, lights.placed(spawn_m), strict=True)):
        path = f"{prim_path}/Fixture_{i:03d}"
        if fixture.size_m is not None:
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateWidthAttr(fixture.size_m[0])
            light.CreateHeightAttr(fixture.size_m[1])
        else:
            light = UsdLux.DiskLight.Define(stage, path)
            light.CreateRadiusAttr(fixture.radius_m)
        light.CreateIntensityAttr(fixture.intensity)
        light.CreateColorAttr(Gf.Vec3f(*fixture.color))
        UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(*position))  # unrotated: emits along -z
        prims.append(light.GetPrim())
    return prims


__all__ = ["LIGHTS_PRIM", "CeilingLights", "Fixture", "ceiling_light_prims"]
