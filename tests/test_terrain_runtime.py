"""Heightfield terrain in replays (SimCore) and the runtime host, against training's (Warp CPU device). Runs in the
Isaac Lab image."""

import json

import numpy as np
import pytest
from conftest import PHYSICS_ROBOTS, built_usd

pytestmark = pytest.mark.physics


@pytest.fixture(autouse=True)
def _isaac_lab():
    pytest.importorskip("isaaclab_newton")

SIZE_M = 4.0
SCALE_M = 0.1
SEED = 7


def _bumps_cfg():
    """Training's call: the tiles a task lays out, here a 2 by 2 grid of bumpy tiles."""
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg

    from humanoid_sim.world.terrain import heightfield_terrain_cfg

    bumpy = {"bumps": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, 0.025), noise_step=0.005)}
    return heightfield_terrain_cfg(bumpy, size_m=SIZE_M, rows=2, cols=2, border_m=2.0, curriculum=False, seed=SEED)


def _bumps_spec(path):
    """The same tiles as a file."""
    path.write_text(json.dumps({
        "size_m": SIZE_M, "rows": 2, "cols": 2, "border_m": 2.0, "curriculum": False, "seed": SEED,
        "tiles": {"bumps": {"type": "HfRandomUniformTerrainCfg", "proportion": 1.0, "noise_range": [0.0, 0.025],
                            "noise_step": 0.005}}}))
    return path


def _lattice():
    axis = np.linspace(-3.9, 3.9, 40)
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)


def _surface(ground):
    return ground.height(_lattice())


def _two_levels(path, high_m=0.03):
    """A tile 8 m across, flat at 0 for y < 0 and ``high_m`` high from y = 0 on, the robot's feet on either side."""
    from humanoid_sim.world.terrain_io import write_heightfield_file

    n = 81
    heights = np.zeros((n, n))
    heights[:, n // 2:] = high_m
    write_heightfield_file(path, heights, horizontal_scale_m=SCALE_M)
    return heights


def test_replay_and_runtime_terrain_has_the_heights_of_trainings_for_the_same_tiles(tmp_path):
    """The scene a replay or the runtime builds from the tiles training passes (or from the tiles as a file) has
    the surface of the training scene's terrain, and the same tile origins. The generator's seed is the state of
    NumPy's global generator when the tiles are drawn: training seeds it, a SimCore seeds it itself."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg
    from humanoid_sim.world.terrain_io import Ground, load_terrain

    usd = built_usd("ethrc_v1")
    sim = sim_utils.SimulationContext(robot_sim_cfg("ethrc_v1", device="cpu"))
    np.random.seed(SEED)  # Isaac Lab's height-field functions draw from NumPy's global state
    try:
        with robot_build("ethrc_v1"):
            scene = instantiate(robot_scene_cfg(1, usd, terrain=_bumps_cfg()))
            sim.reset()
        trained, origin = _surface(Ground.from_stage()), scene.env_origins[0].cpu().numpy()
    finally:
        sim_utils.SimulationContext.clear_instance()
    assert np.isfinite(trained).all() and trained.max() > 0.01  # bumpy, and the lattice is on the terrain

    for terrain in (_bumps_cfg(), load_terrain(_bumps_spec(tmp_path / "bumps.json"))):
        replay = SimCore(usd, device="cpu", terrain=terrain, fixed_base=True)
        try:
            assert _surface(replay.ground) == pytest.approx(trained, abs=1e-6)
            assert replay.env_origins[0].cpu().numpy() == pytest.approx(origin, abs=1e-6)
        finally:
            replay.close()


def test_a_heights_file_is_the_terrain_it_holds(tmp_path):
    from humanoid_sim.sim import SimCore
    from humanoid_sim.world.terrain_io import load_terrain, write_heightfield_file

    heights = np.random.default_rng(1).uniform(0.0, 0.04, (81, 81))
    heights[30:51, 30:51] = np.maximum(heights[30:51, 30:51], 0.02)
    write_heightfield_file(tmp_path / "t.npz", heights, horizontal_scale_m=SCALE_M)
    terrain = load_terrain(tmp_path / "t.npz")
    assert terrain.source == str(tmp_path / "t.npz") and len(terrain.sha256) == 64
    sim = SimCore(built_usd("ethrc_v1"), device="cpu", terrain=terrain, fixed_base=True)
    try:
        axis = np.arange(81) * SCALE_M - 4.0
        grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        assert sim.ground.height(grid).reshape(81, 81) == pytest.approx(heights, abs=6e-4)  # the 1 mm steps
        near = (np.abs(axis[:, None]) <= 1.0) & (np.abs(axis[None, :]) <= 1.0)
        assert sim.env_origins[0].cpu().numpy() == pytest.approx([0.0, 0.0, heights[near].max()], abs=2e-3)
        assert sim.ground.height([[6.0, 0.0]])[0] == pytest.approx(0.0)  # the flat apron around the tile
    finally:
        sim.close()


@pytest.mark.parametrize("fixed_base", [True, False])
def test_a_replay_robot_starts_above_its_tiles_origin(tmp_path, fixed_base):
    import torch

    from humanoid_sim.sim import SimCore
    from humanoid_sim.world.terrain_io import load_terrain

    _two_levels(tmp_path / "t.npz", high_m=0.05)
    terrain = load_terrain(tmp_path / "t.npz")
    sim = SimCore(built_usd("ethrc_v1"), device="cpu", terrain=terrain, fixed_base=fixed_base, base_height_m=1.0)
    try:
        sim.step()
        root = sim.backend.robot.data.root_link_pose_w.torch[0]
        origin = sim.env_origins[0]
        assert torch.allclose(root[:2], origin[:2], atol=0.01)
        assert float(root[2]) == pytest.approx(float(origin[2]) + 1.0, abs=0.01 if fixed_base else 0.05)
        assert float(origin[2]) == pytest.approx(0.05, abs=1e-3)
    finally:
        sim.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_the_runtime_starts_the_robot_on_the_ground_under_its_feet(tmp_path, robot):
    """On a tile that is 3 cm higher on one side of the robot, the start pose sits 3 cm above the flat standing
    height, every sole at the same clearance; the descriptor and the trajectory name the terrain; resets and the next
    session put the robot on the ground again."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.model.poses import STANDING_CLEARANCE_M
    from humanoid_sim.runtime.host import RuntimeHost
    from humanoid_sim.runtime.trajectory import read_header
    from humanoid_sim.world.terrain_io import load_terrain

    _two_levels(tmp_path / "t.npz", high_m=0.03)
    terrain = load_terrain(tmp_path / "t.npz")
    host = RuntimeHost(tmp_path / "plant", built_usd(robot), device="cpu", robot=robot, support="free", compile=False,
                       terrain=terrain, trajectory=tmp_path / "traj.bin", truth_every=1)
    try:
        flat_z = host._poses[host.start_pose][1] + STANDING_CLEARANCE_M
        descriptor = host.seg.descriptor()["terrain"]
        assert descriptor["sha256"] == terrain.sha256 and descriptor["source"] == str(tmp_path / "t.npz")
        assert descriptor["spawn_m"] == pytest.approx([0.0, 0.0, 0.03], abs=1e-3)  # the tile's highest point near 0
        assert descriptor["extent_m"]["x_m"] == pytest.approx([-8.0, 8.0], abs=0.2)  # the tile and its border
        assert descriptor["size_m"] == pytest.approx(8.0) and descriptor["horizontal_scale_m"] == SCALE_M

        def check_standing():
            host.step()
            host._publish_truth()
            _, truth = host.seg.regions["truth"].read()
            assert truth["base_position_m"][:2] == pytest.approx([0.0, 0.0], abs=0.02)
            assert truth["base_position_m"][2] == pytest.approx(flat_z + 0.03, abs=0.01)
            assert truth["sole"]["lowest_z_m"] == pytest.approx(
                [STANDING_CLEARANCE_M + 0.03] * 2, abs=0.004)  # level soles, at the platform's height

        check_standing()
        host._reset({"pose": "home"})
        check_standing()
        host.new_session()
        check_standing()
    finally:
        host.close()
    header = read_header(tmp_path / "traj.bin")
    assert header["terrain"]["sha256"] == terrain.sha256 and header["terrain"]["spawn_m"] == descriptor["spawn_m"]


def test_a_flat_runtime_names_no_terrain(tmp_path):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost
    from humanoid_sim.runtime.trajectory import read_header

    host = RuntimeHost(tmp_path / "plant", built_usd("ethrc_v1"), device="cpu", support="free", compile=False,
                       trajectory=tmp_path / "traj.bin", truth_every=1)
    try:
        assert host.seg.descriptor()["terrain"] is None
    finally:
        host.close()
    assert read_header(tmp_path / "traj.bin")["terrain"] is None


def test_a_terrain_refuses_a_scene_of_boxes(tmp_path):
    """The boxes stand on the flat ground."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost
    from humanoid_sim.world.boxes import StaticBoxes
    from humanoid_sim.world.terrain_io import load_terrain

    np.savez(tmp_path / "boxes.npz", center=np.array([[3.0, 0.0, 0.5]]), half=np.full((1, 3), 0.5), yaw=0.0)
    _two_levels(tmp_path / "t.npz")
    with pytest.raises(ValueError, match="no scene or gantry"):
        RuntimeHost(tmp_path / "plant", built_usd("ethrc_v1"), device="cpu", support="free", compile=False,
                    terrain=load_terrain(tmp_path / "t.npz"), scene=StaticBoxes.load(tmp_path / "boxes.npz"))
