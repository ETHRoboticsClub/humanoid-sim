"""The G1's RTX sensors against geometry worked out independently: a room of boxes with targets, the robot welded at
the origin, its rig's Mid-360 and the cameras the ``g1-cameras`` extra rig adds. Each camera's image of a target is
the set of pixels whose rays, cast through the mount (the numbers humanoid-urdf's ``PROVENANCE.md`` records) and the
lens (the ZED Mini's factory calibration, the D435i's datasheet fields of view), meet it; the D435i's depth is the
analytic distance to the room; every lidar return lies at the range where its ray meets the room, fired at its
pattern's time.

Needs the Isaac Sim image (``docker/Dockerfile.kit``) on an RTX GPU, and its own process (Kit starts once):

    pytest -m rtx tests/test_rtx_sensors.py
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaacsim", reason="RTX sensors need the Isaac Sim image (docker/Dockerfile.kit)")
# Kit must start before anything initialises CUDA in the process (it crashes otherwise): no torch.cuda here.
if shutil.which("nvidia-smi") is None:
    pytest.skip("RTX sensors need an NVIDIA GPU with RT cores", allow_module_level=True)

from rtx_oracles import box_coverage, box_mask, mask_centroid, ray_box_distance, ray_floor_distance  # noqa: E402

pytestmark = pytest.mark.rtx

G1 = "g1_29dof_with_hand_rev_1_0"
CAMERAS = "g1-cameras"  # sensors/rigs.py EXTRAS
SETTLE_S = 1.0  # the welded robot's arms and waist settle on their hold within this; images and sweeps after it

# Mounts in torso_link, humanoid-urdf PROVENANCE.md: position (m) and URDF rpy (rad).
ZED_MOUNT = ((0.1068, 0.0406, 0.457), (0.0, 0.0, 0.0))
D435_MOUNT = ((0.0576235, 0.01753, 0.42987), (0.0, 0.8307767239493009, 0.0))
MID360_MOUNT = ((0.0002835, 0.00003, 0.428434), (math.pi, 0.0836, 0.0))
# ZED Mini SN19945568 at HD720, the factory file as Controls' test_zed_backend.py carries it.
ZED_LEFT = (771.4, 771.14, 629.115, 353.0685, (-0.0358597, 0.0185157, -0.00496448, 0.00194007))
ZED_RIGHT = (772.01, 771.55, 641.75, 346.6085, (-0.0368591, 0.0234426, -0.0128113, 0.00625172))
ZED_RIGHT_EYE = (0.000233734, -0.0628272, 0.000181381)  # device frame: TZ, -Baseline, -TY
# D435i, D400 datasheet: colour 69 deg across 640 px, depth 87 x 58 deg at 640 x 480; colour 15 mm right of depth.
D435_COLOR_F = 320.0 / math.tan(math.radians(69.0 / 2))
D435_DEPTH_F = (320.0 / math.tan(math.radians(87.0 / 2)), 240.0 / math.tan(math.radians(58.0 / 2)))
D435_COLOR_AT = (0.0, -0.015, 0.0)

# The room: walls 0.2 m thick and 3 m tall around the robot, and targets (centre, half extents, colour): plates 2 mm
# thick, so a camera sees one face of each and nothing of their sides. The ZED Mini's eyes are grey: its target is a
# white plate on a black one behind it; the D435i's colour stream sees a green plate on the floor.
WALLS = [((3.1, 0.0, 1.5), (0.1, 2.7, 1.5)), ((-3.1, 0.0, 1.5), (0.1, 2.7, 1.5)),
         ((0.0, 2.6, 1.5), (3.2, 0.1, 1.5)), ((0.0, -2.6, 1.5), (3.2, 0.1, 1.5))]
WHITE, BLACK, GREEN = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
TARGETS = [((1.6, 0.05, 1.3), (0.001, 0.06, 0.06), WHITE),     # upright ahead, facing the ZED Mini
           ((1.65, 0.05, 1.3), (0.001, 0.2, 0.2), BLACK),      # behind it
           ((1.0, 0.12, 0.001), (0.06, 0.06, 0.001), GREEN)]    # flat on the floor ahead, below the D435i

# Tolerances. A target's image is weighed against the share of each pixel whose rays meet it, so the two agree to the
# renderer's anti-aliasing and lens resampling: a fraction of a pixel in the centroid, a few percent in the area.
# Depth and range: the renderer's precision.
PIXEL_TOL = 0.3
COUNT_TOL = 0.05
SURROUND_MARGIN_PX = 3
CONTRAST_FLOOR = 0.1
DEPTH_TOL_M = 0.01
RANGE_TOL_M = 0.01
SELF_HIT_M = 1.0  # rays that meet the robot's own body first: shorter than this, the room is farther


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_matrix_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical axes in a device frame


def device_pose(torso, mount):
    """World pose (R, t) of a device frame on the torso."""
    (R_t, t_t), (t_m, rpy_m) = torso, mount
    return R_t @ rpy(*rpy_m), R_t @ np.asarray(t_m) + t_t


def to_optical(device, offset, points_w):
    """World points in the optical frame of the stream at ``offset`` in ``device``."""
    R, t = device
    origin = t + R @ np.asarray(offset)
    return (np.asarray(points_w) - origin) @ (R @ OPTICAL)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from humanoid_sim.sensors.rtx import start_kit

    room = tmp_path_factory.mktemp("room") / "room.npz"
    boxes = WALLS + [(c, h) for c, h, _ in TARGETS]
    np.savez(room, center=np.array([c for c, _ in boxes]), half=np.array([h for _, h in boxes]),
             yaw=np.zeros(len(boxes)), rgb=np.array([(0.6, 0.6, 0.6)] * len(WALLS) + [c for *_, c in TARGETS]),
             wall=np.array([True] * len(WALLS) + [False] * len(TARGETS)),
             floor_corners=np.array([[-3, -2.5], [3, -2.5], [3, 2.5], [-3, 2.5]]))
    kit = start_kit("cuda:0")
    from humanoid_sim.sensors.session import SensorSession, SessionCfg

    session = SensorSession(SessionCfg(robot=G1, sensors=str(CAMERAS), scene=str(room), lidar_noise=False,
                                       keep_sweeps=3))
    session.run(SETTLE_S + 0.4)
    robot = session.sim.backend.robot
    i = robot.body_names.index("torso_link")
    torso = (quat_matrix_xyzw(robot.data.body_quat_w[0, i].cpu().numpy().astype(float)),
             robot.data.body_pos_w[0, i].cpu().numpy().astype(float))
    out = {"images": dict(session.last_images), "sweeps": list(session.sweeps), "torso": torso,
           "room": (np.array([c for c, _ in boxes]), np.array([h for _, h in boxes]))}
    dump = os.environ.get("HUMANOID_SIM_RTX_DUMP")  # keep what the sensors saw, to look at a failure
    if dump:
        Path(dump).mkdir(parents=True, exist_ok=True)
        np.savez(Path(dump) / "oracle.npz", torso_R=torso[0], torso_t=torso[1],
                 **{f"{c}_{s}": v for (c, s), v in out["images"].items()},
                 **{f"sweep{i}_{k}": getattr(w, k) for i, w in enumerate(out["sweeps"])
                    for k in ("points_m", "hit", "time_s", "origin_w", "rotation_w")},
                 sweep_starts=np.array([w.start_ns for w in out["sweeps"]]))
    yield out
    session.close()
    # Closing Kit ends the process: leave it to the interpreter's exit, after pytest has reported.
    atexit.register(kit.close)


def target_weights(image, lens, device, offset, target, surround):
    """The target in the image and where it should be: per pixel, how much of the target the image shows (its
    contrast against the ``surround`` box's pixels, 0 to 1, the faintest tenth dropped as noise), and the share of
    the pixel whose rays meet the target. A grey image's contrast is its brightness (a white target on a black
    surround); a colour image's is how far green stands above red and blue (a green target)."""
    R, t = device
    eye, optical = t + R @ np.asarray(offset), R @ OPTICAL
    expected = box_coverage(image.shape[:2], lens, eye, optical, *map(np.asarray, target[:2]))
    window = box_mask(image.shape[:2], lens, eye, optical, *map(np.asarray, surround[:2]))
    for _ in range(SURROUND_MARGIN_PX):  # off the surround's edge, where anti-aliasing mixes in what is behind it
        window = (window & np.roll(window, 1, 0) & np.roll(window, -1, 0)
                  & np.roll(window, 1, 1) & np.roll(window, -1, 1))
    img = image.astype(float)
    contrast = img if img.ndim == 2 else img[..., 1] - img[..., [0, 2]].mean(axis=-1)
    lo, hi = np.percentile(contrast[window], [5, 95])
    floor = lo + CONTRAST_FLOOR * (hi - lo)
    return np.where(window, np.clip((contrast - floor) / (hi - floor), 0.0, 1.0), 0.0), expected


def camera_sees(seen, expected):
    (u, v, n), (mu, mv, mn) = mask_centroid(expected), mask_centroid(seen)
    assert n > 200 and mn > 0, f"target not in view: {mn:.0f} px seen, {n:.0f} expected"
    assert (mu, mv) == pytest.approx((u, v), abs=PIXEL_TOL)
    assert mn == pytest.approx(n, rel=COUNT_TOL)


@pytest.mark.parametrize("eye,lens,offset", [("left", ZED_LEFT, (0.0, 0.0, 0.0)), ("right", ZED_RIGHT, ZED_RIGHT_EYE)])
def test_the_zed_mini_sees_the_target_where_its_lens_projects_it(run, eye, lens, offset):
    image = run["images"][("sim-zed", eye)]
    assert image.shape == (720, 1280) and image.dtype == np.uint8  # grey, as Controls' zed backend publishes
    camera_sees(*target_weights(image, lens, device_pose(run["torso"], ZED_MOUNT), offset, TARGETS[0], TARGETS[1]))


def test_the_d435i_colour_sees_the_target_where_its_lens_projects_it(run):
    image = run["images"][("sim-d435", "rgb")]
    assert image.shape == (480, 640, 3)
    lens = (D435_COLOR_F, D435_COLOR_F, 320.0, 240.0, None)
    surround = (TARGETS[2][0], tuple(np.multiply(TARGETS[2][1], (2.0, 2.0, 1.0))))  # the floor around the plate
    camera_sees(*target_weights(image, lens, device_pose(run["torso"], D435_MOUNT), D435_COLOR_AT, TARGETS[2],
                                surround))


def test_the_d435i_depth_is_the_distance_to_the_room(run):
    depth = run["images"][("sim-d435", "depth")]
    assert depth.shape == (480, 640)
    R, t = device_pose(run["torso"], D435_MOUNT)
    fx, fy = D435_DEPTH_F
    v, u = np.mgrid[0:480, 0:640]
    rays_o = np.stack([(u.ravel() - 320.0) / fx, (v.ravel() - 240.0) / fy, np.ones(u.size)], axis=1)  # z = 1
    rays_w = rays_o @ (R @ OPTICAL).T
    norm = np.linalg.norm(rays_w, axis=1)
    unit = rays_w / norm[:, None]
    centre, half = run["room"]
    dist = np.minimum(ray_floor_distance(t, unit), ray_box_distance(t, unit, centre, half, np.zeros(len(centre))))
    expected = (dist / norm).reshape(480, 640)  # along the optical axis: the ray's z is 1 per unit of norm
    measured = np.nan_to_num(depth, nan=np.inf, posinf=np.inf)
    seen = np.isfinite(expected) & (expected < 10.0)
    occluded = measured < expected - 0.05  # the robot's own body in front of the room
    room = seen & ~occluded
    assert room.sum() > 0.5 * seen.sum()
    err = np.abs(measured[room] - expected[room])
    assert np.median(err) < DEPTH_TOL_M / 2 and np.percentile(err, 99) < DEPTH_TOL_M, (
        f"median {np.median(err):.4f} m, p99 {np.percentile(err, 99):.4f} m over {room.sum()} px")


def test_every_lidar_return_lies_where_its_ray_meets_the_room(run):
    sweeps = run["sweeps"]
    assert len(sweeps) >= 2
    centre, half = run["room"]
    for sweep in sweeps:
        R, t = device_pose(run["torso"], MID360_MOUNT)
        np.testing.assert_allclose(sweep.origin_w, t, atol=1e-3)
        np.testing.assert_allclose(sweep.rotation_w, R, atol=1e-3)
        dirs_s = sweep.points_m / np.maximum(np.linalg.norm(sweep.points_m, axis=1, keepdims=True), 1e-9)
        hit = sweep.hit
        unit = dirs_s[hit] @ R.T
        expected = np.minimum(ray_floor_distance(t, unit), ray_box_distance(t, unit, centre, half,
                                                                            np.zeros(len(centre))))
        measured = np.linalg.norm(sweep.points_m[hit], axis=1)
        self_hit = (measured < SELF_HIT_M) & (measured < expected - 0.05)
        room = ~self_hit
        err = np.abs(measured[room] - expected[room])
        assert hit.mean() > 0.5, f"only {hit.mean():.0%} of rays returned"
        assert self_hit.mean() < 0.3
        assert np.percentile(err, 99) < RANGE_TOL_M, f"p99 range error {np.percentile(err, 99):.4f} m"


def test_lidar_rays_follow_the_pattern_and_its_timing(run):
    from humanoid_sim.sensors.models import LivoxMid360

    model = LivoxMid360
    sweeps = run["sweeps"]
    assert len(sweeps) >= 2
    period = model.frame_period_ns / 1e9
    starts = np.array([s.start_ns for s in sweeps])
    np.testing.assert_allclose(np.diff(starts), period * 1e9, atol=1e3)
    for sweep in sweeps:
        assert sweep.time_s.min() >= 0.0 and sweep.time_s.max() < period
        assert len(sweep.points_m) == model.point_rate_hz * period
        dirs = sweep.points_m[sweep.hit] / np.linalg.norm(sweep.points_m[sweep.hit], axis=1, keepdims=True)
        lo, hi = model.elevation_min_rad, model.elevation_max_rad
        el = np.arcsin(np.clip(dirs[:, 2], -1, 1))
        assert el.min() > lo - 1e-3 and el.max() < hi + 1e-3
