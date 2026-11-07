"""A scene's lighting (``sensors/lights.py``) as the RTX cameras see it: the G1 welded at the origin with the
``g1-cameras-d435`` rig, its D435i's colour stream looking down at a grey plate on the floor ahead. A warm dome tints
the image. With the dome and the ambient light off, the scene is black but for its lights: a rectangle, a disk and a
sphere fixture, given in the scan's frame and moved by ``-spawn``, each light the floor where it is placed, and a sun
throws a pillar's shadow away from its bearing. In the splat room of ``tests/rtx_oracles.py``, ceiling fixtures and a
sun leave the splat as bright as the dome alone; the robot's shadow on it is as deep as the share of the stage's lights
the robot hides. Where a point lands in the image is worked out from the G1's URDF and the
D435i's datasheet (``tests/rtx_oracles.py``), without the renderer.

Each lighting is rendered by ``tests/rtx_lights_render.py`` in a process of its own (Kit starts once a process); this
process starts neither Kit nor USD. Needs the Isaac Sim image (``docker/Dockerfile.kit``) on an RTX GPU:

    pytest -m rtx tests/test_rtx_lights.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaacsim", reason="RTX sensors need the Isaac Sim image (docker/Dockerfile.kit)")
if shutil.which("nvidia-smi") is None:
    pytest.skip("RTX sensors need an NVIDIA GPU with RT cores", allow_module_level=True)

from rtx_oracles import (  # noqa: E402
    D435_COLOR_AT,
    D435_COLOR_F,
    D435_CX_PX,
    D435_CY_PX,
    D435_JOINT,
    OPTICAL,
    WALLS,
    box_mask,
    device_pose,
    pinhole_pixel,
    splat_positions,
    torso_mount,
    write_splat_ply,
)

pytestmark = pytest.mark.rtx

REPO = Path(__file__).resolve().parents[1]
RGB = "sim-d435/rgb"
LENS = (D435_COLOR_F, D435_COLOR_F, D435_CX_PX, D435_CY_PX, None)
LUMA_BT601 = np.array([0.299, 0.587, 0.114])
SPAWN = (2.0, -1.0)  # the scan's point at the world's origin: fixtures and boxes are given in the scan's frame

# The scene, world frame (centre, half extents): a grey plate 2 mm thick on the floor ahead of the robot's feet, an
# even surface for the lights to fall on (the ground is chequered), and a pillar standing on it for the sun. The plate
# is RTX's default material, glossy: a light's pool on it peaks between the point under the light, where a matte
# floor is brightest, and the light's mirror image in the floor, where a mirror is.
GREY = (0.5, 0.5, 0.5)
PLATE = ((2.0, 0.0, 0.001), (1.5, 1.5, 0.001))
FLOOR_Z = 0.002
PILLAR = ((1.5, 0.0, FLOOR_Z + 0.4), (0.05, 0.05, 0.4))
DARK = {"dome_intensity": 0, "ambient_intensity": 0}
# Fixtures low over the plate, world frame, each lighting the floor under it about as brightly as the default dome:
# a panel of luminance L and area A at height h gives L A / h^2 (lux) under it, a sphere of radius r L pi r^2 / h^2.
FIXTURES = {"rect": {"position_m": (1.3, 0.45, 0.35), "size_m": [0.3, 0.3], "intensity": 3000},
            "disk": {"position_m": (1.3, -0.45, 0.35), "radius_m": 0.17, "intensity": 3000},
            "sphere": {"position_m": (1.6, 0.0, 0.35), "sphere_radius_m": 0.05, "intensity": 30000}}
SUN_ELEVATION_DEG = 30.0  # the pillar's shadow is 0.8 / tan(30 deg) = 1.4 m long
SUN = {"elevation_deg": SUN_ELEVATION_DEG, "intensity": 3000}
SHADOW_SIDE_Y = 0.5  # a patch of floor this far either side of the pillar, inside its shadow on that side
PATCH_HALF = (0.03, 0.2, 0.001)
# The splat room's lights: panels under the walls' tops and a sun over the room.
CEILING = [{"position_m": [x, y, 2.8], "size_m": [0.6, 1.2], "intensity": 3000} for x in (-1.5, 0.0, 1.5)
           for y in (-1.2, 1.2)]
ROOM_SUN = {"azimuth_deg": 135, "elevation_deg": 40, "intensity": 3000}

# Tolerances (8-bit levels and shares of a brightness). The brightest pixels' centroid lands within a few pixels of
# the segment joining the images of the point under the fixture and of its mirror image.
BRIGHTEST = 0.995  # the brightest half percent of the pixels
POOL_TOL_PX = 4.0
BLACK_LEVELS = 1.0
LIT_LEVELS = 50.0
WARMER_RB = 0.2  # a 3000 K dome raises the red-to-blue ratio by more than this over the white one
SHADOWED = 0.3  # a patch in the pillar's shadow renders under this share of the same patch lit
# A camera's mean brightness in the splat room with more lights against the dome alone: the robot, which the lights
# do light, and its soft shadows move it by under a percent; hiding lights from the splat darkens it by tens.
SAME_SHARE = 0.02
SUN_ALONE_DARK = 0.3  # pixels the sun alone renders under this share of the dome's: the robot's shadow
HALF_SHADOW = (0.4, 0.6)  # under the dome and a sun, the robot hides one light in two from its shadow


def scan(p):
    """A world point in the scan's frame."""
    return [p[0] + SPAWN[0], p[1] + SPAWN[1], p[2]]


def boxes(path, items, wall):
    np.savez(path, center=np.array([scan(c) for c, _ in items]), half=np.array([h for _, h in items]),
             yaw=np.zeros(len(items)), rgb=np.array([GREY] * len(items)), wall=np.array(wall, dtype=bool))


def luma(image):
    image = image.astype(float)
    return image @ LUMA_BT601 if image.ndim == 3 else image


def d435(run):
    R, t = device_pose((run["torso_R"], run["torso_t"]), torso_mount(D435_JOINT))
    return t + R @ np.asarray(D435_COLOR_AT), R @ OPTICAL


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("lights")
    boxes(d / "plate.npz", [PLATE], [False])
    boxes(d / "pillar.npz", [PLATE, PILLAR], [False, False])
    np.savez(d / "room.npz", center=np.array([c for c, _ in WALLS]), half=np.array([h for _, h in WALLS]),
             yaw=np.zeros(len(WALLS)), wall=np.ones(len(WALLS), dtype=bool))  # the splat room: spawn 0
    write_splat_ply(d / "room.ply", splat_positions())
    subprocess.run([sys.executable, "-m", "humanoid_sim.cli", "splat-usd", str(d / "room.ply"), str(d / "room.usdc")],
                   cwd=REPO, check=True, capture_output=True)
    plate = ["--scene", str(d / "plate.npz"), "--spawn", *map(str, SPAWN)]
    pillar = ["--scene", str(d / "pillar.npz"), "--spawn", *map(str, SPAWN)]
    room = ["--scene", str(d / "room.npz"), "--splat", str(d / "room.usdc")]
    configs = {"default": (plate, None), "warm": (plate, {"dome_temperature_k": 3000}),
               "dome_off": (plate, {"dome_intensity": 0}), "dark": (plate, DARK),
               **{kind: (plate, {**DARK, "fixtures": [{**f, "position_m": scan(f["position_m"])}]})
                  for kind, f in FIXTURES.items()},
               "sun+y": (pillar, {**DARK, "sun": {**SUN, "azimuth_deg": 90}}),
               "sun-y": (pillar, {**DARK, "sun": {**SUN, "azimuth_deg": -90}}),
               "splat": (room, None), "splat_ceiling": (room, {"fixtures": CEILING}),
               "splat_sun": (room, {"sun": ROOM_SUN}),
               "splat_sun_alone": (room, {"dome_intensity": 0, "sun": ROOM_SUN})}
    out = {}
    for name, (scene, lights) in configs.items():
        args = list(scene)
        if lights is not None:
            (d / f"{name}.lights.json").write_text(json.dumps(lights))
            args += ["--lights", str(d / f"{name}.lights.json")]
        npz = d / f"{name}.npz"
        done = subprocess.run([sys.executable, "tests/rtx_lights_render.py", "--out", str(npz), *args], cwd=REPO,
                              capture_output=True, text=True)
        assert done.returncode == 0, f"{name}\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}"
        with np.load(npz) as z:
            out[name] = {k: z[k] for k in z.files}
        out[name]["summary"] = json.loads(npz.with_suffix(".json").read_text())
    return out


def test_a_warm_dome_tints_the_image(runs):
    def red_to_blue(run):
        mean = run[RGB].reshape(-1, 3).astype(float).mean(axis=0)
        return mean[0] / mean[2]

    assert runs["warm"]["summary"]["scene"]["dome_temperature_k"] == 3000.0
    assert red_to_blue(runs["warm"]) > red_to_blue(runs["default"]) + WARMER_RB


def test_the_ambient_light_lights_the_scene_until_it_is_turned_off(runs):
    ambient = "/rtx/sceneDb/ambientLightIntensity"
    assert runs["dome_off"]["summary"]["rtx_post"][ambient] == pytest.approx(1.0)
    assert np.percentile(luma(runs["dome_off"][RGB]), 25) > LIT_LEVELS  # no dome: the ambient light still lights it
    dark = runs["dark"]
    assert dark["summary"]["rtx_post"][ambient] == 0.0 and dark["summary"]["scene"]["ambient_intensity"] == 0.0
    assert dark["summary"]["scene"]["dome_light_nits"] is None
    assert np.percentile(luma(dark[RGB]), 99) < BLACK_LEVELS


@pytest.mark.parametrize("kind", list(FIXTURES))
def test_each_fixture_lights_the_floor_where_it_is_placed(runs, kind):
    run = runs[kind]
    y = luma(run[RGB])
    assert y.max() > LIT_LEVELS, f"{kind}: no light in view"
    vs, us = np.nonzero(y >= np.quantile(y, BRIGHTEST))
    eye, optical = d435(run)
    light = np.asarray(FIXTURES[kind]["position_m"], dtype=float)
    under = np.array([light[0], light[1], FLOOR_Z])
    mirrored = np.array([light[0], light[1], 2 * FLOOR_Z - light[2]])
    mirror = eye + (eye[2] - FLOOR_Z) / (eye[2] - mirrored[2]) * (mirrored - eye)  # where the eye sees it, on the floor
    a, b = (pinhole_pixel(*LENS[:4], (p - eye) @ optical)[0] for p in (under, mirror))
    peak = np.array([us.mean(), vs.mean()])
    along = np.clip(np.dot(peak - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    off = np.linalg.norm(peak - (a + along * (b - a)))
    assert off < POOL_TOL_PX, f"{kind}: peak {peak.round(1)}, {off:.1f} px off {a.round(1)} to {b.round(1)}"


def test_the_sun_throws_the_pillars_shadow_away_from_its_bearing(runs):
    def patch(run, side_y):
        eye, optical = d435(run)
        mask = box_mask(run[RGB].shape[:2], LENS, eye, optical, np.array([PILLAR[0][0], side_y, FLOOR_Z]),
                        np.array(PATCH_HALF))
        assert mask.sum() > 100
        return luma(run[RGB])[mask].mean()

    for sun_side, run in ((1, runs["sun+y"]), (-1, runs["sun-y"])):  # the sun's bearing: +y (90 deg), -y (-90 deg)
        towards, away = patch(run, sun_side * SHADOW_SIDE_Y), patch(run, -sun_side * SHADOW_SIDE_Y)
        assert towards > LIT_LEVELS and away < SHADOWED * towards, f"sun at {sun_side:+d}y: {towards:.0f}, {away:.0f}"


def test_lights_leave_the_splat_room_as_bright(runs):
    base = runs["splat"]
    streams = [k for k in base if "/" in k and "depth" not in k]
    assert streams
    for stream in streams:
        for name in ("splat_ceiling", "splat_sun"):
            ratio = luma(runs[name][stream]).mean() / luma(base[stream]).mean()
            assert ratio == pytest.approx(1.0, abs=SAME_SHARE), f"{name} {stream}: {ratio:.3f} of the dome alone"


def test_the_robots_shadow_on_the_splat_is_the_share_of_the_lights_it_hides(runs):
    base, alone, sun = (luma(runs[n][RGB]) for n in ("splat", "splat_sun_alone", "splat_sun"))
    shadow = alone < SUN_ALONE_DARK * base  # the D435i looks down at the splat's floor: only the robot's shadow
    assert shadow.sum() > 1000
    assert sun[shadow].mean() / base[shadow].mean() == pytest.approx(np.mean(HALF_SHADOW),
                                                                       abs=np.ptp(HALF_SHADOW) / 2)
