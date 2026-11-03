"""What the RTX sensors see of a moving object: a tall box waits off to the G1's side, then slides in front of it. The
D435i's depth image and the Mid-360's sweeps see the box where its path has put it, and not before. Every depth pixel
that should see the box does, at the box's range, worked out without the renderer (``tests/rtx_oracles.py``).

Needs the Isaac Sim image (``docker/Dockerfile.kit``) on an RTX GPU, and its own process (Kit starts once):

    pytest -m rtx tests/test_rtx_objects.py
"""

from __future__ import annotations

import atexit
import json
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
    d435_depth_rays,
    device_pose,
    quat_matrix_xyzw,
    ray_box_distance,
    ray_floor_distance,
    torso_mount,
)

pytestmark = pytest.mark.rtx

BOX = ((1.5, 0.0, 1.0), (0.2, 0.4, 1.0))  # where the box ends up: centre, half extents (m)
ARRIVES_S = 1.5  # the box is off to the side until 1 s, then slides in


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from humanoid_sim.sensors.rtx import start_kit

    d = tmp_path_factory.mktemp("moving_box")
    path = {"times_s": [0.0, 1.0, ARRIVES_S, 4.0],
            "positions_m": [[1.5, 5.0, 1.0], [1.5, 5.0, 1.0], [1.5, 0.0, 1.0], [1.5, 0.0, 1.0]]}
    objects = d / "objects.json"
    objects.write_text(json.dumps({"moving": [{"name": "box", "shape": "box", "size": [2 * h for h in BOX[1]],
                                               "friction": 0.5, "rgb": [1.0, 0.0, 1.0], "path": path}]}))
    kit = start_kit("cuda:0")
    from humanoid_sim.sensors.session import SensorSession, SessionCfg

    session = SensorSession(SessionCfg(robot=G1, sensors=str(CAMERAS), objects=str(objects), lidar_noise=False,
                                       keep_sweeps=12))
    robot = session.sim.backend.robot
    out = {}
    for label, until_s in (("before", 0.9), ("after", ARRIVES_S + 1.0)):
        session.run(until_s - session.sim.t_ns / 1e9)
        i = robot.body_names.index("torso_link")
        out[label] = {
            "depth": session.last_images[("sim-d435", "depth")].copy(), "sweeps": list(session.sweeps),
            "torso": (quat_matrix_xyzw(robot.data.body_quat_w[0, i].cpu().numpy().astype(float)),
                      robot.data.body_pos_w[0, i].cpu().numpy().astype(float)),
            "box_y": float(session.sim.backend.objects.state().moving_position_m[0, 0, 1])}
    yield out
    session.close()
    atexit.register(kit.close)


def test_the_physics_has_the_box_where_its_path_puts_it(run):
    assert run["before"]["box_y"] == pytest.approx(5.0, abs=0.02)
    assert run["after"]["box_y"] == pytest.approx(0.0, abs=0.02)


def _expected_depth(snapshot):
    R, t = device_pose(snapshot["torso"], torso_mount(D435_JOINT))
    rays_w = d435_depth_rays() @ (R @ OPTICAL).T
    norm = np.linalg.norm(rays_w, axis=1)
    unit = rays_w / norm[:, None]
    box = ray_box_distance(t, unit, np.array([BOX[0]]), np.array([BOX[1]]), np.zeros(1))
    floor = ray_floor_distance(t, unit)
    return (box / norm).reshape(D435_HEIGHT_PX, D435_WIDTH_PX), (floor / norm).reshape(D435_HEIGHT_PX, D435_WIDTH_PX)


def test_the_depth_camera_sees_the_box_once_it_has_slid_in(run):
    box, floor = _expected_depth(run["after"])
    sees_box = np.isfinite(box) & (box < floor) & (box < D435_DEPTH_MAX_M)
    assert sees_box.sum() > 2000, f"only {sees_box.sum()} pixels should see the box: the test's geometry is off"
    measured = np.nan_to_num(run["after"]["depth"], nan=np.inf, posinf=np.inf)
    error = np.abs(measured[sees_box] - box[sees_box])
    assert np.percentile(error, 95) < 2 * DEPTH_TOL_M, f"p95 {np.percentile(error, 95):.4f} m over {sees_box.sum()} px"
    near = np.nan_to_num(run["before"]["depth"], nan=np.inf, posinf=np.inf) < box[sees_box].min() + 0.1
    assert near.sum() < 0.01 * D435_WIDTH_PX * D435_HEIGHT_PX  # before it arrived nothing stood where it ends up


def _on_box_surface(snapshot, tol=0.03):
    """Lidar returns of ``snapshot`` that lie within ``tol`` of the box's surface where it ends up."""
    R, t = device_pose(snapshot["torso"], torso_mount(MID360_JOINT))
    count = 0
    for sweep in snapshot["sweeps"]:
        points = sweep.points_m[sweep.hit] @ R.T + t
        d = np.abs(points - np.array(BOX[0])) - np.array(BOX[1])
        sdf = np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(d.max(axis=1), 0.0)
        count += int((np.abs(sdf) < tol).sum())
    return count


def test_the_lidar_returns_from_the_box_only_once_it_is_there(run):
    after = _on_box_surface(run["after"])
    assert after > 300, f"{after} returns on the box's surface"
    assert _on_box_surface(run["before"]) == 0


def test_a_moving_box_hides_nothing_it_should_not(run):
    """The sensors still see the floor either side of the box."""
    box, floor = _expected_depth(run["after"])
    sees_floor = np.isfinite(floor) & (floor < box) & (floor < 3.0)
    measured = np.nan_to_num(run["after"]["depth"], nan=np.inf, posinf=np.inf)
    ok = measured[sees_floor] > floor[sees_floor] - NEARER_M
    assert ok.mean() > 0.9  # the robot's own body may stand in front of some of them
