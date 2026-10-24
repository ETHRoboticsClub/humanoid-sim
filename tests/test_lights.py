"""Ceiling lights from a file (``sensors/lights.py``): what the file may say, where each fixture lands in the world
frame (the scan's frame moved by ``-spawn``), and the UsdLux prims authored on a stage."""

from __future__ import annotations

import json
import math

import pytest

from humanoid_sim.sensors.lights import CeilingLights, ceiling_light_prims

SPAWN = (-4.42, 3.48)
FILE = {"fixtures": [{"position_m": [1.0, 2.0, 7.5], "size_m": [0.6, 1.2], "intensity": 2500.0,
                      "color": [1.0, 0.9, 0.8]},
                     {"position_m": [-3.0, 0.5, 7.0], "radius_m": 0.3, "intensity": 1000.0}],
        "dome_intensity": 50.0}


def test_a_file_is_read_with_its_fixtures_and_dome(tmp_path):
    path = tmp_path / "lights.json"
    path.write_text(json.dumps(FILE))
    lights = CeilingLights.load(path)
    assert lights.dome_intensity == 50.0
    rect, disk = lights.fixtures
    assert (rect.size_m, rect.radius_m, rect.intensity, rect.color) == ((0.6, 1.2), None, 2500.0, (1.0, 0.9, 0.8))
    assert (disk.size_m, disk.radius_m, disk.color) == (None, 0.3, (1.0, 1.0, 1.0))  # white by default
    assert CeilingLights.parse({"fixtures": FILE["fixtures"]}).dome_intensity is None  # the session's dome stays


def test_fixtures_are_placed_like_the_rooms_boxes():
    lights = CeilingLights.parse(FILE)
    assert lights.placed(SPAWN) == [pytest.approx((1.0 + 4.42, 2.0 - 3.48, 7.5)),
                                    pytest.approx((-3.0 + 4.42, 0.5 - 3.48, 7.0))]
    assert lights.placed((0.0, 0.0)) == [(1.0, 2.0, 7.5), (-3.0, 0.5, 7.0)]


@pytest.mark.parametrize("bad", [
    {"fixtures": []},
    {"fixtures": FILE["fixtures"], "ambient": 3},
    {"fixtures": [{"position_m": [0, 0, 3], "intensity": 1}]},  # neither shape
    {"fixtures": [{"position_m": [0, 0, 3], "size_m": [1, 1], "radius_m": 1, "intensity": 1}]},  # both
    {"fixtures": [{"position_m": [0, 0], "radius_m": 1, "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 0, "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "size_m": [1, math.inf], "intensity": 1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1}]},  # no intensity
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1, "intensity": -1}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1, "intensity": 1, "color": [1, -1, 1]}]},
    {"fixtures": [{"position_m": [0, 0, 3], "radius_m": 1, "intensity": 1, "watts": 40}]},
    {"fixtures": FILE["fixtures"], "dome_intensity": -1},
])
def test_malformed_files_are_refused(bad):
    with pytest.raises(ValueError):
        CeilingLights.parse(bad)


def test_the_stage_gets_downward_area_lights_at_the_placed_positions():
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    lights = CeilingLights.parse(FILE)
    rect, disk = ceiling_light_prims(stage, lights, SPAWN)
    assert rect.IsA(UsdLux.RectLight) and disk.IsA(UsdLux.DiskLight)
    r, d = UsdLux.RectLight(rect), UsdLux.DiskLight(disk)
    assert (r.GetWidthAttr().Get(), r.GetHeightAttr().Get()) == pytest.approx((0.6, 1.2))
    assert d.GetRadiusAttr().Get() == pytest.approx(0.3)
    assert (r.GetIntensityAttr().Get(), d.GetIntensityAttr().Get()) == (2500.0, 1000.0)
    assert tuple(r.GetColorAttr().Get()) == pytest.approx((1.0, 0.9, 0.8))
    for prim, expected in zip((rect, disk), lights.placed(SPAWN), strict=True):
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert tuple(world.ExtractTranslation()) == pytest.approx(expected)
        # UsdLux area lights emit along their -z: unrotated, that is straight down.
        down = world.TransformDir(pxr.Gf.Vec3d(0, 0, -1))
        assert tuple(down) == pytest.approx((0.0, 0.0, -1.0))
