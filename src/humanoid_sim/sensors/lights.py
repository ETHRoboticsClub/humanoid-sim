"""A room's ceiling lights for the RTX renderer, from a file the room's owner writes: area lights facing straight
down, given in the scan's frame and moved into the world frame by ``-spawn`` like the room's boxes, scan and splat.

The file (JSON)::

    {"fixtures": [{"position_m": [x, y, z], "size_m": [w, l], "intensity": 3000, "color": [1, 1, 1]},
                  {"position_m": [x, y, z], "radius_m": 0.3, "intensity": 3000}],
     "dome_intensity": 100, "dome_temperature_k": 4500,
     "sun": {"azimuth_deg": 135, "elevation_deg": 40, "intensity": 3000, "angle_deg": 0.53}}

A fixture with ``size_m`` is a rectangle (USD ``RectLight``, ``w`` along x, ``l`` along y), one with ``radius_m`` a
disk (``DiskLight``); both emit along their -z, down. ``intensity`` is the emitting surface's luminance in nits,
``color`` linear RGB (white when left out). ``dome_intensity``, when given, replaces the session's dome light (the
ambient fill, nits; 0 for none). A fixture with ``sphere_radius_m`` is a point-like sphere light (``SphereLight``,
radiating in all directions). ``temperature_k`` on a fixture, and ``dome_temperature_k`` for the dome, set a colour
temperature in Kelvin (USD ``enableColorTemperature``, multiplied with ``color``); left out, the light is not tinted.
``sun`` adds a ``DistantLight`` for outdoor scenes: ``azimuth_deg`` (the sun's bearing in the world frame, from +x
towards +y) and ``elevation_deg`` (above the horizon, 0 to 90) say where the sun is in the sky, ``intensity`` is its
illuminance (USD distant light units), ``angle_deg`` its angular diameter (0.53 for the real sun; larger is softer
shadows) and ``color`` / ``temperature_k`` as for a fixture. The sun is in the world frame, not moved by the spawn. A
file may have no fixtures when it gives a dome setting or a sun. Isaac Lab has no rectangle light spawner, so the lights are authored with UsdLux.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LIGHTS_PRIM = "/World/CeilingLights"
SUN_PRIM = "/World/Sun"
TEMPERATURE_RANGE_K = (1000.0, 40000.0)  # USD's colour temperature range


@dataclass(frozen=True)
class Fixture:
    position_m: tuple[float, float, float]  # the scan's frame
    intensity: float  # nits
    size_m: tuple[float, float] | None = None  # rectangle (x, y)
    radius_m: float | None = None  # disk
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sphere_radius_m: float | None = None  # sphere (point-like) light
    temperature_k: float | None = None  # colour temperature; None: untinted


@dataclass(frozen=True)
class Sun:
    azimuth_deg: float  # bearing of the sun in the world frame, from +x towards +y
    elevation_deg: float  # above the horizon
    intensity: float  # illuminance (USD distant light units)
    angle_deg: float = 0.53  # angular diameter
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temperature_k: float | None = None

    def toward_sun(self) -> tuple[float, float, float]:
        """Unit vector from the scene to the sun, world frame."""
        az, el = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


@dataclass(frozen=True)
class CeilingLights:
    fixtures: tuple[Fixture, ...]
    dome_intensity: float | None = None
    dome_temperature_k: float | None = None
    sun: Sun | None = None

    @classmethod
    def load(cls, path: str | Path) -> CeilingLights:
        return cls.parse(json.loads(Path(path).read_text()), source=str(path))

    @classmethod
    def parse(cls, data: dict, source: str = "lights") -> CeilingLights:
        unknown = set(data) - {"fixtures", "dome_intensity", "dome_temperature_k", "sun"}
        if unknown:
            raise ValueError(f"{source}: unknown keys {sorted(unknown)}")
        fixtures = tuple(_fixture(f, f"{source}: fixture {i}") for i, f in enumerate(data.get("fixtures") or []))
        dome = data.get("dome_intensity")
        if dome is not None and not (math.isfinite(dome) and dome >= 0):
            raise ValueError(f"{source}: dome_intensity {dome}: a finite value >= 0")
        dome_k = _temperature(data.get("dome_temperature_k"), f"{source}: dome_temperature_k")
        sun = _sun(data["sun"], f"{source}: sun") if data.get("sun") is not None else None
        if not (fixtures or dome is not None or dome_k is not None or sun):
            raise ValueError(f"{source}: no fixtures, dome setting or sun")
        return cls(fixtures, None if dome is None else float(dome), dome_k, sun)

    def placed(self, spawn_m: tuple[float, float]) -> list[tuple[float, float, float]]:
        """Every fixture's position in the world frame, the scan's point ``spawn_m`` at the origin."""
        return [(f.position_m[0] - spawn_m[0], f.position_m[1] - spawn_m[1], f.position_m[2]) for f in self.fixtures]


def dome_temperature(session_k: float | None, lights: CeilingLights | None) -> float | None:
    """The dome's colour temperature (Kelvin): the lights file's when it gives one, else the session's; None: white."""
    return session_k if lights is None or lights.dome_temperature_k is None else lights.dome_temperature_k


def dome_nits(session_nits: float, lights: CeilingLights | None) -> float | None:
    """The dome light a session renders with (nits): the lights file's ``dome_intensity`` when it gives one, else the
    session's; None for no dome."""
    nits = session_nits if lights is None or lights.dome_intensity is None else lights.dome_intensity
    return nits or None


def _fixture(f: dict, where: str) -> Fixture:
    unknown = set(f) - {"position_m", "size_m", "radius_m", "sphere_radius_m", "intensity", "color", "temperature_k"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if sum(k in f for k in ("size_m", "radius_m", "sphere_radius_m")) != 1:
        raise ValueError(f"{where}: give one of size_m (rectangle), radius_m (disk) and sphere_radius_m (sphere)")
    position = _floats(f.get("position_m"), 3, f"{where}: position_m")
    size = _floats(f["size_m"], 2, f"{where}: size_m") if "size_m" in f else None
    radius = float(f["radius_m"]) if "radius_m" in f else None
    sphere = float(f["sphere_radius_m"]) if "sphere_radius_m" in f else None
    color = _floats(f.get("color", (1.0, 1.0, 1.0)), 3, f"{where}: color")
    intensity = float(f.get("intensity", math.nan))
    if any(not (math.isfinite(v) and v > 0) for v in (*(size or ()), *(v for v in (radius, sphere) if v is not None))):
        raise ValueError(f"{where}: size_m, radius_m and sphere_radius_m must be > 0")
    if not (math.isfinite(intensity) and intensity >= 0):
        raise ValueError(f"{where}: intensity (nits) must be given, finite and >= 0")
    if any(v < 0 for v in color):
        raise ValueError(f"{where}: color must be >= 0")
    return Fixture(position, intensity, size, radius, color, sphere,
                   _temperature(f.get("temperature_k"), f"{where}: temperature_k"))


def _temperature(value, where: str) -> float | None:
    if value is None:
        return None
    lo, hi = TEMPERATURE_RANGE_K
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (lo <= value <= hi):
        raise ValueError(f"{where}: Kelvin in [{lo:g}, {hi:g}]")
    return float(value)


def _sun(s: dict, where: str) -> Sun:
    if not isinstance(s, dict):
        raise ValueError(f"{where}: an object")
    unknown = set(s) - {"azimuth_deg", "elevation_deg", "intensity", "angle_deg", "color", "temperature_k"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("azimuth_deg", "elevation_deg", "intensity"):
        if key not in s:
            raise ValueError(f"{where}: {key} must be given")
    az, el, intensity, angle = (float(s.get(k, d)) for k, d in (("azimuth_deg", 0), ("elevation_deg", 0),
                                                                ("intensity", 0), ("angle_deg", 0.53)))
    if not all(map(math.isfinite, (az, el, intensity, angle))):
        raise ValueError(f"{where}: finite numbers")
    if not 0 <= el <= 90:
        raise ValueError(f"{where}: elevation_deg in [0, 90]")
    if intensity < 0:
        raise ValueError(f"{where}: intensity must be >= 0")
    if not 0 <= angle <= 90:
        raise ValueError(f"{where}: angle_deg in [0, 90]")
    color = _floats(s.get("color", (1.0, 1.0, 1.0)), 3, f"{where}: color")
    if any(v < 0 for v in color):
        raise ValueError(f"{where}: color must be >= 0")
    return Sun(az, el, intensity, angle, color, _temperature(s.get("temperature_k"), f"{where}: temperature_k"))


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
        elif fixture.sphere_radius_m is not None:
            light = UsdLux.SphereLight.Define(stage, path)
            light.CreateRadiusAttr(fixture.sphere_radius_m)
        else:
            light = UsdLux.DiskLight.Define(stage, path)
            light.CreateRadiusAttr(fixture.radius_m)
        light.CreateIntensityAttr(fixture.intensity)
        light.CreateColorAttr(Gf.Vec3f(*fixture.color))
        set_color_temperature(light.GetPrim(), fixture.temperature_k)
        UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(*position))  # unrotated: emits along -z
        prims.append(light.GetPrim())
    return prims


def set_color_temperature(prim, kelvin: float | None) -> None:
    """Tint a ``UsdLux`` light prim to ``kelvin``; None leaves it untouched (white, as without this)."""
    from pxr import UsdLux

    if kelvin is None:
        return
    light = UsdLux.LightAPI(prim)
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateColorTemperatureAttr(float(kelvin))


def sun_prim(stage, sun: Sun, prim_path: str = SUN_PRIM):
    """Author ``sun`` as a ``DistantLight`` on ``stage`` (world frame), turned to shine away from the sun."""
    from pxr import Gf, UsdGeom, UsdLux

    light = UsdLux.DistantLight.Define(stage, prim_path)
    light.CreateIntensityAttr(sun.intensity)
    light.CreateAngleAttr(sun.angle_deg)
    light.CreateColorAttr(Gf.Vec3f(*sun.color))
    set_color_temperature(light.GetPrim(), sun.temperature_k)
    shine = -Gf.Vec3d(*sun.toward_sun())  # a distant light emits along its -z
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), shine)
    UsdGeom.Xformable(light.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(rotation, Gf.Vec3d(0)))
    return light.GetPrim()


__all__ = ["LIGHTS_PRIM", "SUN_PRIM", "CeilingLights", "Fixture", "Sun", "ceiling_light_prims",
           "dome_nits", "dome_temperature", "set_color_temperature", "sun_prim"]
