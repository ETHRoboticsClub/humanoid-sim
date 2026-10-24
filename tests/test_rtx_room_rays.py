"""What the RTX lidar and depth camera see of a scanned room (``sensors/room_rays.py``): a room of wall boxes with a
furniture box ahead of the G1, and a scan mesh that has only the piece of furniture, smaller than its box (a scan
with holes everywhere else). Without a splat both sensors see the walls and the mesh, never the furniture box; every
return and depth pixel lies where its ray meets those, worked out without the renderer (``tests/rtx_oracles.py``).
The physics keeps every box.

Needs the Isaac Sim image (``docker/Dockerfile.kit``) on an RTX GPU, and its own process (Kit starts once):

    pytest -m rtx tests/test_rtx_room_rays.py
"""

from __future__ import annotations

import atexit
import shutil

import numpy as np
import pytest

pytest.importorskip("isaacsim", reason="RTX sensors need the Isaac Sim image (docker/Dockerfile.kit)")
if shutil.which("nvidia-smi") is None:
    pytest.skip("RTX sensors need an NVIDIA GPU with RT cores", allow_module_level=True)

from rtx_oracles import (  # noqa: E402
    CAMERAS,
    D435_DEPTH_MAX_M,
    D435_HEIGHT_PX,
    D435_JOINT,
    D435_WIDTH_PX,
    DEPTH_TOL_M,
    G1,
    MID360_JOINT,
    NEARER_M,
    OPTICAL,
    RANGE_TOL_M,
    SELF_HIT_M,
    SETTLE_S,
    WALLS,
    d435_depth_rays,
    device_pose,
    quat_matrix_xyzw,
    ray_box_distance,
    ray_floor_distance,
    torso_mount,
    write_box_obj,
)

pytestmark = pytest.mark.rtx

FURNITURE = ((1.5, 0.0, 0.5), (0.35, 0.35, 0.5))  # the furniture's collision box, ahead of the robot
SCANNED = ((1.5, 0.0, 0.3), (0.2, 0.2, 0.3))  # the furniture as the scan has it: inside its box
SHADOW_M = NEARER_M  # a ray the furniture box would stop this much short of the room is one it would have shadowed
# More than this many rays or pixels through the furniture box's shadow: the 99th percentile of their errors is then
# at most the second largest (np.percentile's rank 0.99 (n - 1)), so a single ray grazing the box's edge cannot fail
# the check, while a visible box, wrong on every one of them, does.
MIN_SHADOWED = 100


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from humanoid_sim.sensors.rtx import start_kit

    d = tmp_path_factory.mktemp("scanned_room")
    room, scan = d / "room.npz", d / "scan.obj"
    boxes = WALLS + [FURNITURE]
    np.savez(room, center=np.array([c for c, _ in boxes]), half=np.array([h for _, h in boxes]),
             yaw=np.zeros(len(boxes)), wall=np.array([True] * len(WALLS) + [False]))
    write_box_obj(scan, *SCANNED)
    kit = start_kit("cuda:0")
    from isaaclab_newton.physics import NewtonManager

    from humanoid_sim.sensors.session import SensorSession, SessionCfg
    from humanoid_sim.world.boxes import STATIC_BOXES_PRIM

    session = SensorSession(SessionCfg(robot=G1, sensors=str(CAMERAS), scene=str(room), scan_mesh=str(scan),
                                       lidar_noise=False, keep_sweeps=3))
    session.run(SETTLE_S + 0.4)
    robot = session.sim.backend.robot
    i = robot.body_names.index("torso_link")
    torso = (quat_matrix_xyzw(robot.data.body_quat_w[0, i].cpu().numpy().astype(float)),
             robot.data.body_pos_w[0, i].cpu().numpy().astype(float))
    collided = sum(label.startswith(STATIC_BOXES_PRIM) for label in NewtonManager.get_model().shape_label)
    out = {"depth": session.last_images[("sim-d435", "depth")], "sweeps": list(session.sweeps), "torso": torso,
           "rays": session.room_rays, "collided": collided, "boxes": len(boxes)}
    yield out
    session.close()
    atexit.register(kit.close)


def seen(origin, unit):
    """Where each ray meets what the sensors should see: the floor, the wall boxes and the scanned furniture."""
    centre, half = np.array([c for c, _ in WALLS] + [SCANNED[0]]), np.array([h for _, h in WALLS] + [SCANNED[1]])
    return np.minimum(ray_floor_distance(origin, unit), ray_box_distance(origin, unit, centre, half,
                                                                          np.zeros(len(centre))))


def shadowed(origin, unit, expected):
    """Rays the furniture box would stop short of what they should see."""
    box = ray_box_distance(origin, unit, np.array([FURNITURE[0]]), np.array([FURNITURE[1]]), np.zeros(1))
    return box < expected - SHADOW_M


def test_the_furniture_box_is_hidden_and_still_collides(run):
    rays = run["rays"]
    assert rays.hidden == (len(WALLS),) and rays.lidar_only == () and not rays.scan_hidden_from_cameras
    assert run["collided"] == run["boxes"]


def test_the_lidar_sees_the_walls_and_the_scan_not_the_furniture_box(run):
    R, t = device_pose(run["torso"], torso_mount(MID360_JOINT))
    through_box = []  # range errors of the returns the furniture box would have stopped short
    for sweep in run["sweeps"]:
        hit = sweep.hit
        dirs = sweep.points_m[hit] / np.linalg.norm(sweep.points_m[hit], axis=1, keepdims=True)
        unit = dirs @ R.T
        expected = seen(t, unit)
        measured = np.linalg.norm(sweep.points_m[hit], axis=1)
        room = ~((measured < SELF_HIT_M) & (measured < expected - NEARER_M))  # not the robot's own body
        err = np.abs(measured - expected)
        assert np.percentile(err[room], 99) < RANGE_TOL_M, f"p99 range error {np.percentile(err[room], 99):.4f} m"
        through_box.append(err[shadowed(t, unit, expected) & room])
    through_box = np.concatenate(through_box)
    assert through_box.size > MIN_SHADOWED, f"only {through_box.size} returns passed through the furniture box"
    p99 = np.percentile(through_box, 99)
    assert p99 < RANGE_TOL_M, f"p99 range error {p99:.4f} m over {through_box.size} returns through the furniture box"


def test_without_a_splat_the_depth_camera_sees_what_the_lidar_sees(run):
    depth = run["depth"]
    R, t = device_pose(run["torso"], torso_mount(D435_JOINT))
    rays_w = d435_depth_rays() @ (R @ OPTICAL).T
    norm = np.linalg.norm(rays_w, axis=1)
    unit = rays_w / norm[:, None]
    dist = seen(t, unit)
    expected = (dist / norm).reshape(D435_HEIGHT_PX, D435_WIDTH_PX)
    measured = np.nan_to_num(depth, nan=np.inf, posinf=np.inf)
    in_range = np.isfinite(expected) & (expected < D435_DEPTH_MAX_M)
    room = in_range & ~(measured < expected - NEARER_M)  # the robot's own body in front of the room
    assert room.sum() > 0.5 * in_range.sum()
    err = np.abs(measured[room] - expected[room])
    assert np.percentile(err, 99) < DEPTH_TOL_M, f"p99 {np.percentile(err, 99):.4f} m over {room.sum()} px"
    through_box = shadowed(t, unit, dist).reshape(D435_HEIGHT_PX, D435_WIDTH_PX) & room
    assert through_box.sum() > MIN_SHADOWED, f"only {through_box.sum()} px see through the furniture box"
