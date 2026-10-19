"""Static boxes around the robot: loaded and placed as Controls' MuJoCo hangar places them, collided in every env.

The placement oracle is MuJoCo's own kinematics on boxes added as Controls adds the hangar's
(``ethrc_sim/environments.py`` @ 44353e2b: a box geom at ``center - spawn``, turned by the file's yaw about its
centre), compared corner by corner. The physics tests run in the Isaac Lab image.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.world.boxes import STATIC_BOXES_PRIM, StaticBoxes, box_corners, box_mesh

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"
SPAWN = (-4.42, 3.48)
PLATFORM_TOP_M = 0.25
SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)


def _controls_file(path: Path, yaw=np.float32(0.3)) -> Path:
    """A file laid out as Controls' hangar_proxy.npz: float32 boxes, one shared yaw, colours, wall flags and the
    scan floor's outline. A platform under the spawn point, a wall 2.5 m ahead of it, a cabinet behind."""
    sx, sy = SPAWN
    center = np.array([[sx + 0.1, sy - 0.05, PLATFORM_TOP_M / 2], [sx + 2.5, sy, 1.0], [sx - 1.58, sy + 1.42, 0.6]],
                      dtype=np.float32)
    half = np.array([[0.6, 0.5, PLATFORM_TOP_M / 2], [0.1, 2.0, 1.0], [0.3, 0.4, 0.6]], dtype=np.float32)
    rgb = np.array([[0.5, 0.4, 0.3], [0.9, 0.9, 0.9], [0.2, 0.3, 0.4]], dtype=np.float32)
    floor = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float32)
    np.savez(path, center=center, half=half, rgb=rgb, wall=np.array([False, True, False]), yaw=yaw,
             floor_corners=floor)
    return path


def _mujoco_corners(path: Path, spawn) -> np.ndarray:
    """``[N, 8, 3]`` corners of the file's boxes added to a MuJoCo world as Controls adds the hangar's."""
    mujoco = pytest.importorskip("mujoco")
    with np.load(path) as data:
        center, half = data["center"].astype(np.float64), data["half"].astype(np.float64)
        yaw = np.broadcast_to(data["yaw"].astype(np.float64), (len(center),))
    spec = mujoco.MjSpec()
    offset = -np.array([spawn[0], spawn[1], 0.0])
    for c, h, y in zip(center, half, yaw, strict=True):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=h.tolist(), pos=(c + offset).tolist(),
                                quat=[np.cos(y / 2), 0.0, 0.0, np.sin(y / 2)])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return np.stack([data.geom_xpos[i] + (SIGNS * model.geom_size[i]) @ data.geom_xmat[i].reshape(3, 3).T
                     for i in range(model.ngeom)])


def test_boxes_are_placed_where_controls_places_them(tmp_path):
    path = _controls_file(tmp_path / "proxy.npz")
    boxes = StaticBoxes.load(path, SPAWN)
    np.testing.assert_allclose(boxes.corners(), _mujoco_corners(path, SPAWN), atol=1e-6)
    assert len(boxes) == 3
    assert boxes.describe()["spawn_m"] == list(SPAWN) and len(boxes.describe()["sha256"]) == 64
    np.testing.assert_allclose(boxes.rgb[1], (0.8, 0.78, 0.74))  # a wall


def test_one_yaw_per_box_and_no_optional_arrays(tmp_path):
    path = tmp_path / "boxes.npz"
    np.savez(path, center=np.array([[1.0, 2.0, 0.5], [-3.0, 0.5, 1.5]]), half=np.array([[0.5, 0.2, 0.5],
             [1.0, 0.1, 1.5]]), yaw=np.array([0.7, -1.2]))
    boxes = StaticBoxes.load(path)
    np.testing.assert_allclose(boxes.corners(), _mujoco_corners(path, (0.0, 0.0)), atol=1e-9)
    assert boxes.rgb.shape == (2, 3)


def test_clearance_of_the_spawn_point(tmp_path):
    path = tmp_path / "boxes.npz"
    np.savez(path, center=np.array([[3.0, 0.0, 0.5], [0.0, 5.0, 0.5]]), half=np.array([[1.0, 1.0, 0.5]] * 2),
             yaw=np.array([np.pi / 4, 0.0]))
    # The first box points a corner at the origin: its footprint reaches 3 - sqrt(2) m from it.
    assert StaticBoxes.load(path).clearance_m() == pytest.approx(3.0 - np.sqrt(2.0))
    assert StaticBoxes.load(path, (0.0, 3.5)).clearance_m() == pytest.approx(0.5)  # 1.5 m below the second box
    assert StaticBoxes.load(_controls_file(tmp_path / "proxy.npz"), SPAWN).clearance_m() == 0.0  # on the platform


def test_files_that_are_not_static_boxes_are_refused(tmp_path):
    good = dict(center=np.zeros((1, 3)), half=np.ones((1, 3)), yaw=0.0)
    for name, change in (("no_yaw", {"yaw": None}), ("flat", {"half": np.array([[1.0, 1.0, 0.0]])}),
                         ("shape", {"half": np.ones((2, 3))}), ("nan", {"center": np.full((1, 3), np.nan)})):
        arrays = {k: v for k, v in (good | change).items() if v is not None}
        np.savez(tmp_path / f"{name}.npz", **arrays)
        with pytest.raises(ValueError):
            StaticBoxes.load(tmp_path / f"{name}.npz")
    pointer = tmp_path / "pointer.npz"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 8202\n")
    with pytest.raises(ValueError, match="git-lfs"):
        StaticBoxes.load(pointer)
    np.savez(tmp_path / "ok.npz", **good)
    with pytest.raises(ValueError, match="spawn"):
        StaticBoxes.load(tmp_path / "ok.npz", (1.0, 2.0, 0.0))


def test_the_drawn_mesh_closes_each_box_with_outward_faces():
    corners = box_corners(np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 2.0]]), np.array([[0.5, 0.2, 0.5], [1.0, 2.0, 2.0]]),
                          np.array([0.4, -2.0]))
    verts, tris, normals, colors = box_mesh(corners, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert tris.shape == (24, 3) and verts.shape == normals.shape == colors.shape == (48, 3)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normal = np.cross(b - a, c - a)
    centre = corners.mean(axis=1)[tris[:, 0] // 24]
    assert (np.einsum("ij,ij->i", normal, (a + b + c) / 3 - centre) > 0).all()
    # The vertex normals are the faces' unit normals.
    for k in range(3):
        np.testing.assert_allclose(normals[tris[:, k]], normal / np.linalg.norm(normal, axis=1, keepdims=True),
                                   atol=1e-12)
    # Each box's faces enclose its volume: (1/6) sum of n . a over the faces.
    volume = np.add.reduceat(np.einsum("ij,ij->i", normal, a), np.arange(0, 24, 12)) / 6
    np.testing.assert_allclose(volume, [8 * 0.5 * 0.2 * 0.5, 8 * 1.0 * 2.0 * 2.0])


def test_a_spawn_point_inside_a_box_is_refused(tmp_path):
    from humanoid_sim.runtime.host import RuntimeHost

    boxes = StaticBoxes.load(_controls_file(tmp_path / "proxy.npz"), SPAWN)
    with pytest.raises(ValueError, match="inside a box"):
        RuntimeHost(tmp_path / "plant", tmp_path / "unused.usd", device="cpu", scene=boxes)


# -- physics (Isaac Lab image) ------------------------------------------------------------------------------------


@pytest.fixture
def usd():
    pytest.importorskip("isaaclab_newton")
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


@pytest.mark.physics
def test_the_solver_collides_with_the_boxes_where_controls_places_them(tmp_path, usd, request):
    """One set of static box shapes for every env, at Controls' corners; the drawing reads the same ones."""
    from isaaclab_newton.physics import NewtonManager
    from newton import GeoType

    from humanoid_sim.sim import SimCore
    from humanoid_sim.world.boxes import stage_boxes

    path = _controls_file(tmp_path / "proxy.npz")
    sim = SimCore(usd, device="cpu", num_envs=2, fixed_base=True, scene=StaticBoxes.load(path, SPAWN))
    request.addfinalizer(sim.close)
    model = NewtonManager.get_model()
    ours = sorted((label, i) for i, label in enumerate(model.shape_label) if label.startswith(STATIC_BOXES_PRIM))
    ids = [i for _, i in ours]
    assert len(ids) == 3  # shared by both envs, not one set each
    assert (model.shape_type.numpy()[ids] == int(GeoType.BOX)).all() and (model.shape_body.numpy()[ids] == -1).all()
    tf, half = model.shape_transform.numpy()[ids].astype(np.float64), model.shape_scale.numpy()[ids].astype(np.float64)
    from scipy.spatial.transform import Rotation

    corners = np.stack([tf[k, :3] + Rotation.from_quat(tf[k, 3:]).apply(SIGNS * half[k]) for k in range(3)])
    expected = _mujoco_corners(path, SPAWN)
    np.testing.assert_allclose(corners, expected, atol=1e-5)
    drawn, colors = stage_boxes()
    np.testing.assert_allclose(drawn, expected, atol=1e-5)
    assert colors.shape == (3, 3)


@pytest.mark.physics
def test_robots_in_every_env_stand_on_a_box(tmp_path, usd):
    """Dropped onto the platform under the spawn point, both envs' robots stand on its top, a platform's height above
    where they stand on the floor."""
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.utils import instantiate

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.model.poses import initial_pose
    from humanoid_sim.train.plant import RobotPlant
    from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg

    _, height = initial_pose(usd, "ethrc_v1", "home")
    sim = sim_utils.SimulationContext(robot_sim_cfg(device="cpu"))
    scene = instantiate(robot_scene_cfg(2, usd, scene=StaticBoxes.load(_controls_file(tmp_path / "p.npz"), SPAWN)))
    sim.reset()
    try:
        robot = scene["robot"]
        assert torch.equal(scene.env_origins, torch.zeros_like(scene.env_origins))
        plant = RobotPlant(robot, imu=None, contacts=scene["contacts"])
        default = robot.data.default_joint_pos.torch
        q = default[:, [list(robot.joint_names).index(n) for n in plant.core.map.joint_names]]
        drive_q, _, _ = plant.core.map.drive_state(q, torch.zeros_like(q))
        hold = DriveDemand.pd(drive_q, plant.core.drive_stiffness.expand(2, -1),
                              plant.core.drive_damping.expand(2, -1))
        root = robot.data.default_root_pose.torch.clone()
        root[:, 2] += PLATFORM_TOP_M + 0.02
        plant.reset(torch.arange(2), root_pose=root, root_vel=torch.zeros(2, 6), demand=hold)
        for _ in range(int(1.0e9 / plant.period_ns)):
            plant.pre_physics()
            scene.write_data_to_sim()
            sim.step()
            scene.update(plant.period_ns / 1e9)
        pose = plant.truth().root_pose_w
        assert ((pose[:, 2] - (height + PLATFORM_TOP_M)).abs() < 0.03).all(), pose[:, 2]
        assert (pose[:, 3:6].abs().amax(dim=1) < 0.05).all()  # upright
    finally:
        sim_utils.SimulationContext.clear_instance()


@pytest.mark.physics
def test_the_runtime_host_stands_the_robot_among_the_boxes(tmp_path, usd):
    """A free session starts at the spawn point, 1.2 m in front of the platform here, and names its scene in the
    descriptor; the robot stands there holding its start pose."""
    import torch

    from humanoid_sim.runtime.abi import plant_abi as abi
    from humanoid_sim.runtime.host import RuntimeHost

    boxes = StaticBoxes.load(_controls_file(tmp_path / "proxy.npz"), (SPAWN[0], SPAWN[1] - 1.2))
    assert boxes.clearance_m() > 0.4
    desc = Description.load()
    host = RuntimeHost(tmp_path / "plant", usd, device="cpu", support="free", pose="home", compile=False,
                       truth_every=1, scene=boxes)
    try:
        client = abi.Segment.open(tmp_path / "plant", robot_id="ethrc_v1", description_digest=desc.digest, wait_s=1)
        info = client.descriptor()
        assert info["scene"] == json.loads(json.dumps(boxes.describe()))
        q, _ = host._named_pose("home")
        hold = host.sim.plant.map.drive_state(q.to(host.sim.plant.cfg.dtype), torch.zeros_like(q))[0][0].numpy()
        drives = info["drives"]
        demand = abi.pd_demand(hold, [desc.drives[d].baseline_stiffness for d in drives],
                               [desc.drives[d].baseline_damping for d in drives])
        z0 = float(host.robot.data.root_link_pose_w.torch[0, 2])
        for cycle in range(1, 126):  # 0.5 s
            client.regions["demands"].write(abi.demands_record(demand, cycle=cycle, session_id=client.session_id))
            host.step()
        pose = host.robot.data.root_link_pose_w.torch[0]
        assert abs(float(pose[2]) - z0) < 0.02 and pose[:2].norm().item() < 0.02  # standing where it started
        client.close()
    finally:
        host.close()
