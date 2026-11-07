"""A scene's lighting (``sensors/lights.py``): what a file or code may say, where each fixture lands in the world
frame (the scan's frame moved by ``-spawn``), the dome a session keeps, and the UsdLux prims authored on a stage."""

from __future__ import annotations

import json
import math

import pytest

from humanoid_sim.sensors.lights import (
    Fixture,
    Lighting,
    Sun,
    as_lighting,
    dome_nits,
    fixture_prims,
    lighting_prims,
    sun_prim,
)

SPAWN = (-4.42, 3.48)
FILE = {"fixtures": [{"position_m": [1.0, 2.0, 7.5], "size_m": [0.6, 1.2], "intensity": 2500.0,
                      "color": [1.0, 0.9, 0.8]},
                     {"position_m": [-3.0, 0.5, 7.0], "radius_m": 0.3, "intensity": 1000.0}],
        "dome_intensity": 50.0}
SUN = {"azimuth_deg": 90.0, "elevation_deg": 30.0, "intensity": 3000.0, "angle_deg": 1.0, "temperature_k": 5800}
LOOK = {"fixtures": [{"position_m": [0, 0, 3], "sphere_radius_m": 0.1, "intensity": 800, "temperature_k": 2700}],
        "dome_intensity": 20.0, "dome_temperature_k": 9000, "sun": SUN}


def test_a_file_is_read_with_its_fixtures_and_dome(tmp_path):
    path = tmp_path / "lights.json"
    path.write_text(json.dumps(FILE))
    lighting = Lighting.load(path)
    assert lighting.dome_intensity == 50.0
    rect, disk = lighting.fixtures
    assert (rect.size_m, rect.radius_m, rect.intensity, rect.color) == ((0.6, 1.2), None, 2500.0, (1.0, 0.9, 0.8))
    assert (disk.size_m, disk.radius_m, disk.color) == (None, 0.3, (1.0, 1.0, 1.0))  # white by default
    assert Lighting.parse({"fixtures": FILE["fixtures"]}).dome_intensity is None  # the session's dome stays
    assert as_lighting(path) == as_lighting(str(path)) == lighting


def test_a_file_may_give_only_the_dome_or_the_sun():
    assert Lighting.parse({"dome_temperature_k": 4000}).dome_temperature_k == 4000.0
    assert Lighting.parse({"sun": SUN}).sun == Sun(90.0, 30.0, 3000.0, 1.0, (1.0, 1.0, 1.0), 5800.0)
    lighting = Lighting.parse(LOOK)
    (bulb,) = lighting.fixtures
    assert (lighting.dome_temperature_k, bulb.sphere_radius_m, bulb.temperature_k) == (9000.0, 0.1, 2700.0)
    assert Lighting.parse(FILE).sun is None
    assert all(f.temperature_k is None for f in Lighting.parse(FILE).fixtures)  # untinted unless asked


def test_code_builds_the_same_lighting_a_file_gives():
    built = Lighting(fixtures=[Fixture(position_m=[0, 0, 3], sphere_radius_m=0.1, intensity=800, temperature_k=2700)],
                     dome_intensity=20, dome_temperature_k=9000, sun=Sun(90, 30, 3000, angle_deg=1, temperature_k=5800))
    assert built == Lighting.parse(LOOK)
    assert as_lighting(built) is built and as_lighting(None) is None
    with pytest.raises(TypeError):
        as_lighting(LOOK)  # a dict is neither a path nor a Lighting
    with pytest.raises(TypeError):
        Lighting(fixtures=[LOOK["fixtures"][0]])
    with pytest.raises(ValueError):
        Lighting(dome_temperature_k=50)  # checked in code as in a file


@pytest.mark.parametrize("file_dome,session,used", [
    (50.0, 600.0, 50.0),  # the lighting's dome replaces the session's
    (None, 600.0, 600.0),  # lighting without one keeps the session's
    (0.0, 600.0, None),  # the lighting turns the dome off
    (None, 0.0, None),  # no dome in the session, none in the lighting
])
def test_the_dome_a_lighting_leaves_in_the_session(file_dome, session, used):
    data = {"fixtures": FILE["fixtures"]} | ({} if file_dome is None else {"dome_intensity": file_dome})
    assert dome_nits(session, Lighting.parse(data)) == used


def test_without_lighting_the_session_dome_stays():
    assert (dome_nits(600.0, None), dome_nits(0.0, None)) == (600.0, None)


def test_the_dome_is_white_unless_a_temperature_tints_it():
    from humanoid_sim.sensors.rtx import dome_light_args

    assert dome_light_args(20.0) == {"intensity": 20.0, "color": (1.0, 1.0, 1.0)}  # no colour temperature enabled
    assert dome_light_args(20.0, 9000) == {"intensity": 20.0, "color": (1.0, 1.0, 1.0),
                                           "enable_color_temperature": True, "color_temperature": 9000.0}


def test_fixtures_are_placed_like_the_rooms_boxes():
    lighting = Lighting.parse(FILE)
    assert lighting.placed(SPAWN) == [pytest.approx((1.0 + 4.42, 2.0 - 3.48, 7.5)),
                                      pytest.approx((-3.0 + 4.42, 0.5 - 3.48, 7.0))]
    assert lighting.placed((0.0, 0.0)) == [(1.0, 2.0, 7.5), (-3.0, 0.5, 7.0)]


def test_the_sun_faces_the_horizon_bearing():
    assert Sun(0, 0, 1).toward_sun() == pytest.approx((1, 0, 0))
    assert Sun(90, 0, 1).toward_sun() == pytest.approx((0, 1, 0))
    assert Sun(0, 90, 1).toward_sun() == pytest.approx((0, 0, 1))


@pytest.mark.parametrize("bad", [
    {},
    [],
    {"fixtures": []},
    {"fixtures": {"position_m": [0, 0, 3]}},
    {"fixtures": FILE["fixtures"], "ambient": 3},
    {"fixtures": [{"position_m": [0, 0, 3], "intensity": 1}]},  # no shape
    {"fixtures": [{"position_m": [0, 0, 3], "size_m": [1, 1], "radius_m": 1, "intensity": 1}]},  # two shapes
    {"fixtures": [{"position_m": [0, 0, 3], "sphere_radius_m": 0.1, "radius_m": 0.1, "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0], "radius_m": 1, "intensity": 1}]},
    {"fixtures": [{"radius_m": 1, "intensity": 1}]},  # no position
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 0, "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "sphere_radius_m": 0, "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "size_m": [1, math.inf], "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1}]},  # no intensity
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1, "intensity": -1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1, "intensity": 1, "color": [1, -1, 1]}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1, "intensity": 1, "watts": 40}]},
    {"fixtures": [{"position_m": [0, 0, 3], "sphere_radius_m": 1, "intensity": 1, "temperature_k": 0}]},
    {"fixtures": FILE["fixtures"], "dome_intensity": -1},
    {"dome_intensity": True},
    {"dome_temperature_k": 500},
    {"dome_temperature_k": 1e6},
    {"dome_temperature_k": "warm"},
    {"sun": [90, 30, 3000]},
    {"sun": {**SUN, "elevation_deg": -5}},
    {"sun": {**SUN, "elevation_deg": 95}},
    {"sun": {**SUN, "intensity": -1}},
    {"sun": {**SUN, "angle_deg": math.nan}},
    {"sun": {**SUN, "shadow": 1}},
    {"sun": {"azimuth_deg": 0, "elevation_deg": 10}},  # no intensity
    {"sun": {**SUN, "temperature_k": 50}},
])
def test_malformed_lighting_is_refused(bad):
    with pytest.raises(ValueError):
        Lighting.parse(bad)


def test_the_stage_gets_downward_area_lights_at_the_placed_positions():
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    lighting = Lighting.parse(FILE)
    rect, disk = fixture_prims(stage, lighting, SPAWN)
    assert rect.IsA(UsdLux.RectLight) and disk.IsA(UsdLux.DiskLight)
    r, d = UsdLux.RectLight(rect), UsdLux.DiskLight(disk)
    assert (r.GetWidthAttr().Get(), r.GetHeightAttr().Get()) == pytest.approx((0.6, 1.2))
    assert d.GetRadiusAttr().Get() == pytest.approx(0.3)
    assert (r.GetIntensityAttr().Get(), d.GetIntensityAttr().Get()) == (2500.0, 1000.0)
    assert tuple(r.GetColorAttr().Get()) == pytest.approx((1.0, 0.9, 0.8))
    assert not rect.GetAttribute("inputs:enableColorTemperature").HasAuthoredValue()  # untinted
    for prim, expected in zip((rect, disk), lighting.placed(SPAWN), strict=True):
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert tuple(world.ExtractTranslation()) == pytest.approx(expected)
        # UsdLux area lights emit along their -z: unrotated, that is straight down.
        down = world.TransformDir(pxr.Gf.Vec3d(0, 0, -1))
        assert tuple(down) == pytest.approx((0.0, 0.0, -1.0))


def test_the_stage_gets_a_tinted_sphere_light_and_a_sun():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    lighting = Lighting.parse(LOOK)
    sphere, sun = lighting_prims(stage, lighting, SPAWN)
    assert sphere.IsA(UsdLux.SphereLight) and UsdLux.SphereLight(sphere).GetRadiusAttr().Get() == pytest.approx(0.1)
    api = UsdLux.LightAPI(sphere)
    assert api.GetEnableColorTemperatureAttr().Get() is True and api.GetColorTemperatureAttr().Get() == 2700.0

    light = UsdLux.DistantLight(sun)
    assert (light.GetIntensityAttr().Get(), light.GetAngleAttr().Get()) == (3000.0, 1.0)
    world = UsdGeom.Xformable(sun).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    shine = world.TransformDir(Gf.Vec3d(0, 0, -1))  # a distant light emits along its -z
    assert tuple(shine) == pytest.approx(tuple(-c for c in lighting.sun.toward_sun()))
    assert sun_prim(Usd.Stage.CreateInMemory(), lighting.sun).IsA(UsdLux.DistantLight)
