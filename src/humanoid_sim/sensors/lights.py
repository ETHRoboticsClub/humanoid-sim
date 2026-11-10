"""A scene's lighting for the RTX renderer beyond the session's white dome: the dome's intensity and colour
temperature, RTX's ambient light, extra lights (fixtures) and a sun. A :class:`Lighting` is built in code by whoever
chooses the look (a task, a low-light preset) or read from a file a room's owner writes; both are checked the same
way.

The file (JSON), every key optional though it must give one::

    {"fixtures": [{"position_m": [x, y, z], "size_m": [w, l], "intensity": 3000, "color": [1, 1, 1]},
                  {"position_m": [x, y, z], "radius_m": 0.3, "intensity": 3000},
                  {"position_m": [x, y, z], "sphere_radius_m": 0.05, "intensity": 800, "temperature_k": 2700}],
     "dome_intensity": 100, "dome_temperature_k": 4500, "ambient_intensity": 1.0,
     "sun": {"azimuth_deg": 135, "elevation_deg": 40, "intensity": 3000, "angle_deg": 0.53}}

Fixtures are given in the scan's frame and moved into the world frame by ``-spawn`` like the room's boxes, scan and
splat. A fixture with ``size_m`` is a rectangle (USD ``RectLight``, ``w`` along x, ``l`` along y), one with
``radius_m`` a disk (``DiskLight``); both emit along their -z, down, like a ceiling panel. One with
``sphere_radius_m`` is a point-like sphere light (``SphereLight``) radiating in all directions. ``intensity`` is the
emitting surface's luminance in nits, ``color`` linear RGB (white when left out). ``temperature_k`` on a fixture or the
sun, and ``dome_temperature_k`` for the dome, set a colour temperature in Kelvin (USD ``enableColorTemperature``,
multiplied with ``color``); left out, the light is not tinted. ``dome_intensity``, when given, replaces the session's
dome light (nits; 0 for none). ``ambient_intensity``, when given, replaces the level of RTX's ambient light
(``rtx.AMBIENT_INTENSITY``, dimensionless), a uniform fill without direction or shadows that lights every surface
whether or not there is a dome: a dark scene needs it at 0 as well as the dome.

``sun`` is a ``DistantLight`` in the world frame, not moved by the spawn: ``azimuth_deg`` (the sun's bearing, from +x
towards +y) and ``elevation_deg`` (above the horizon, 0 to 90) say where the sun is in the sky, ``intensity`` is its
illuminance (USD distant light units), ``angle_deg`` its angular diameter (0.53 for the real sun; larger is softer
shadows). Isaac Lab has no rectangle light spawner, so fixtures and sun are authored with UsdLux.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

LIGHTS_PRIM = "/World/Lights"
SUN_PRIM = "/World/Sun"
TEMPERATURE_RANGE_K = (1000.0, 40000.0)  # USD's colour temperature range
WHITE = (1.0, 1.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class Fixture:
    position_m: tuple[float, float, float]  # the scan's frame
    intensity: float  # nits
    size_m: tuple[float, float] | None = None  # rectangle (x, y)
    radius_m: float | None = None  # disk
    sphere_radius_m: float | None = None  # sphere (point-like)
    color: tuple[float, float, float] = WHITE
    temperature_k: float | None = None  # None: untinted

    def __post_init__(self):
        if sum(v is not None for v in (self.size_m, self.radius_m, self.sphere_radius_m)) != 1:
            raise ValueError("give one of size_m (rectangle), radius_m (disk) and sphere_radius_m (sphere)")
        _set(self, "position_m", _vector(self.position_m, 3, "position_m"))
        if self.size_m is not None:
            _set(self, "size_m", _vector(self.size_m, 2, "size_m", positive=True))
        for key in ("radius_m", "sphere_radius_m"):
            if getattr(self, key) is not None:
                _set(self, key, _number(getattr(self, key), key, positive=True))
        _set(self, "intensity", _number(self.intensity, "intensity (nits)", non_negative=True))
        _set(self, "color", _vector(self.color, 3, "color", non_negative=True))
        _set(self, "temperature_k", _kelvin(self.temperature_k, "temperature_k"))


@dataclass(frozen=True)
class Sun:
    azimuth_deg: float  # bearing of the sun in the world frame, from +x towards +y
    elevation_deg: float  # above the horizon
    intensity: float  # illuminance (USD distant light units)
    angle_deg: float = 0.53  # angular diameter
    color: tuple[float, float, float] = WHITE
    temperature_k: float | None = None  # None: untinted

    def __post_init__(self):
        _set(self, "azimuth_deg", _number(self.azimuth_deg, "azimuth_deg"))
        _set(self, "elevation_deg", _number(self.elevation_deg, "elevation_deg", within=(0.0, 90.0)))
        _set(self, "intensity", _number(self.intensity, "intensity", non_negative=True))
        _set(self, "angle_deg", _number(self.angle_deg, "angle_deg", within=(0.0, 90.0)))
        _set(self, "color", _vector(self.color, 3, "color", non_negative=True))
        _set(self, "temperature_k", _kelvin(self.temperature_k, "temperature_k"))

    def toward_sun(self) -> tuple[float, float, float]:
        """Unit vector from the scene to the sun, world frame."""
        az, el = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


@dataclass(frozen=True, kw_only=True)
class Lighting:
    fixtures: tuple[Fixture, ...] = ()
    dome_intensity: float | None = None  # nits, replaces the session's; 0: no dome; None: the session's
    dome_temperature_k: float | None = None  # None: a white dome
    ambient_intensity: float | None = None  # RTX's ambient light level; None: the session's; 0: none
    sun: Sun | None = None

    def __post_init__(self):
        _set(self, "fixtures", tuple(self.fixtures))
        if not all(isinstance(f, Fixture) for f in self.fixtures):
            raise TypeError("fixtures must be Fixture")
        if self.sun is not None and not isinstance(self.sun, Sun):
            raise TypeError("sun must be a Sun")
        if self.dome_intensity is not None:
            _set(self, "dome_intensity", _number(self.dome_intensity, "dome_intensity", non_negative=True))
        _set(self, "dome_temperature_k", _kelvin(self.dome_temperature_k, "dome_temperature_k"))
        if self.ambient_intensity is not None:
            _set(self, "ambient_intensity", _number(self.ambient_intensity, "ambient_intensity", non_negative=True))

    @classmethod
    def load(cls, path: str | Path) -> Lighting:
        return cls.parse(json.loads(Path(path).read_text()), source=str(path))

    @classmethod
    def parse(cls, data: dict, source: str = "lights") -> Lighting:
        """A lighting file's contents; it must give at least one setting."""
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{source}: an object with at least one of fixtures, dome_intensity, "
                             f"dome_temperature_k, ambient_intensity and sun")
        fixtures = data.get("fixtures") or []
        if not isinstance(fixtures, list):
            raise ValueError(f"{source}: fixtures must be a list")
        fields = {k: v for k, v in data.items() if k != "fixtures"}
        fields["fixtures"] = tuple(_build(Fixture, f, f"{source}: fixture {i}") for i, f in enumerate(fixtures))
        if fields.get("sun") is not None:
            fields["sun"] = _build(Sun, fields["sun"], f"{source}: sun")
        lighting = _build(cls, fields, source)
        if lighting == cls():
            raise ValueError(f"{source}: no fixtures, dome or ambient setting, or sun")
        return lighting

    def placed(self, spawn_m: tuple[float, float]) -> list[tuple[float, float, float]]:
        """Every fixture's position in the world frame, the scan's point ``spawn_m`` at the origin."""
        return [(f.position_m[0] - spawn_m[0], f.position_m[1] - spawn_m[1], f.position_m[2]) for f in self.fixtures]


def as_lighting(value: str | Path | Lighting | None) -> Lighting | None:
    """A session's ``lights``: a lighting file's path, a :class:`Lighting`, or None for the session's dome alone."""
    if value is None or isinstance(value, Lighting):
        return value
    if isinstance(value, (str, Path)):
        return Lighting.load(value)
    raise TypeError(f"lights: a path or a Lighting, not {type(value).__name__}")


def dome_nits(session_nits: float, lighting: Lighting | None) -> float | None:
    """The dome light a session renders with (nits): the lighting's ``dome_intensity`` when it gives one, else the
    session's; None for no dome."""
    nits = session_nits if lighting is None or lighting.dome_intensity is None else lighting.dome_intensity
    return nits or None


def ambient_level(session_level: float, lighting: Lighting | None) -> float:
    """The level of RTX's ambient light a session renders with: the lighting's ``ambient_intensity`` when it gives
    one, else the session's (``rtx.AMBIENT_INTENSITY``); 0 for none."""
    return session_level if lighting is None or lighting.ambient_intensity is None else lighting.ambient_intensity


def _build(cls, fields, where: str):
    """``cls(**fields)``, every complaint (an unknown or missing key, a bad value) a ValueError naming ``where``."""
    if not isinstance(fields, dict):
        raise ValueError(f"{where}: an object")
    try:
        return cls(**fields)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: {e}") from None


def _set(obj, name: str, value) -> None:
    object.__setattr__(obj, name, value)


def _number(value, what: str, *, positive=False, non_negative=False, within=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what}: a finite number, not {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{what} must be > 0")
    if non_negative and value < 0:
        raise ValueError(f"{what} must be >= 0")
    if within is not None and not within[0] <= value <= within[1]:
        raise ValueError(f"{what} in [{within[0]:g}, {within[1]:g}]")
    return float(value)


def _vector(value, n: int, what: str, **bounds) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{what}: {n} numbers")
    return tuple(_number(v, what, **bounds) for v in value)


def _kelvin(value, what: str) -> float | None:
    return None if value is None else _number(value, f"{what} (Kelvin)", within=TEMPERATURE_RANGE_K)


def fixture_prims(stage, lighting: Lighting, spawn_m: tuple[float, float], prim_path: str = LIGHTS_PRIM) -> list:
    """Author ``lighting``'s fixtures under ``prim_path`` on ``stage``, placed with ``spawn_m``; returns their prims."""
    from pxr import Gf, UsdGeom, UsdLux

    UsdGeom.Xform.Define(stage, prim_path)
    prims = []
    for i, (fixture, position) in enumerate(zip(lighting.fixtures, lighting.placed(spawn_m), strict=True)):
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


def lighting_prims(stage, lighting: Lighting, spawn_m: tuple[float, float]) -> list:
    """Author ``lighting``'s fixtures and sun on ``stage`` (the dome is the session's entity); returns their prims."""
    prims = fixture_prims(stage, lighting, spawn_m)
    return prims + ([sun_prim(stage, lighting.sun)] if lighting.sun is not None else [])


def set_color_temperature(prim, kelvin: float | None) -> None:
    """Tint a ``UsdLux`` light prim to ``kelvin``; None leaves it untouched (white, as without this)."""
    from pxr import UsdLux

    if kelvin is None:
        return
    light = UsdLux.LightAPI(prim)
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateColorTemperatureAttr(float(kelvin))


__all__ = ["LIGHTS_PRIM", "SUN_PRIM", "TEMPERATURE_RANGE_K", "Fixture", "Lighting", "Sun", "ambient_level",
           "as_lighting", "dome_nits", "fixture_prims", "lighting_prims", "set_color_temperature", "sun_prim"]
