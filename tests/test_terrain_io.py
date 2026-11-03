"""Terrain files and the ground height under the robot (native: NumPy, and Newton for the kinematics)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import PHYSICS_ROBOTS, built_usd

from humanoid_sim.world.terrain_io import (
    Ground,
    Terrain,
    as_terrain,
    read_heightfield_file,
    rest_base_height,
    write_heightfield_file,
)


def _grid_mesh(heights: np.ndarray, scale: float, origin=(0.0, 0.0)):
    """The triangle mesh of a height grid, as Isaac Lab's height-field conversion lays it out."""
    n, m = heights.shape
    xx, yy = np.meshgrid(np.arange(n) * scale + origin[0], np.arange(m) * scale + origin[1], indexing="ij")
    vertices = np.stack([xx.ravel(), yy.ravel(), heights.ravel()], axis=1)
    index = np.arange(n * m).reshape(n, m)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return vertices, faces


def test_ground_height_is_the_surface_of_the_triangles():
    heights = np.add.outer(np.arange(5) * 0.02, np.arange(5) * 0.01)  # a plane: z = 0.2 x + 0.1 y
    ground = Ground(*_grid_mesh(heights, 0.1, origin=(-0.2, -0.2)))
    xy = np.array([[0.0, 0.0], [-0.2, -0.2], [0.13, -0.07], [0.19, 0.19]])
    assert ground.height(xy) == pytest.approx(0.2 * xy[:, 0] + 0.1 * xy[:, 1] + 0.06, abs=1e-12)
    assert np.isnan(ground.height([[0.5, 0.0], [0.0, -0.3]])).all()  # off the mesh
    assert ground.height(np.zeros((0, 2))).shape == (0,)


def test_ground_height_takes_the_highest_of_overlapping_triangles():
    low = _grid_mesh(np.zeros((2, 2)), 1.0)
    high = _grid_mesh(np.full((2, 2), 0.3), 1.0)
    ground = Ground(np.concatenate([low[0], high[0]]), np.concatenate([low[1], high[1] + 4]))
    assert ground.height([[0.4, 0.4]]) == pytest.approx([0.3])


def test_ground_describes_its_extent():
    ground = Ground(*_grid_mesh(np.array([[0.0, 0.1], [0.2, -0.05]]), 2.0, origin=(-1.0, -1.0)))
    assert ground.describe() == {"x_m": [-1.0, 1.0], "y_m": [-1.0, 1.0]}


def test_rest_base_height_clears_every_point_with_its_radius():
    ground = Ground(*_grid_mesh(np.zeros((9, 9)), 0.5, origin=(-2.0, -2.0)))
    points = np.array([[0.1, 0.0, -0.9], [-0.1, 0.0, -0.95], [0.0, 0.1, -0.5]])
    radius = np.array([0.0, 0.0, 0.1])
    assert rest_base_height(ground, points, radius, (0.0, 0.0)) == pytest.approx(0.95)
    assert rest_base_height(ground, points, radius, (0.0, 0.0), clearance=0.001) == pytest.approx(0.951)
    assert rest_base_height(ground, points[:1], radius[:1], (0.0, 0.0)) == pytest.approx(0.9)


def test_rest_base_height_follows_the_ground_under_each_point_and_the_heading():
    heights = np.zeros((9, 9))
    heights[6:, :] = 0.04  # a rise from x = 0.5 (0) to x = 1 (0.04)
    ground = Ground(*_grid_mesh(heights, 0.5, origin=(-2.0, -2.0)))
    points = np.array([[0.2, 0.0, -1.0], [-0.2, 0.0, -1.0]])
    radius = np.zeros(2)
    assert rest_base_height(ground, points, radius, (0.0, 0.0)) == pytest.approx(1.0)
    assert rest_base_height(ground, points, radius, (1.3, 0.0)) == pytest.approx(1.04)  # both points on the high part
    assert rest_base_height(ground, points, radius, (0.9, 0.0)) == pytest.approx(1.04)  # forward point at x = 1.1
    # Turned a quarter turn both points are at x = 0.9, where the rise has reached 0.032.
    assert rest_base_height(ground, points, radius, (0.9, 0.0), yaw=np.pi / 2) == pytest.approx(1.032)


def test_rest_base_height_refuses_a_robot_reaching_past_the_terrain():
    ground = Ground(*_grid_mesh(np.zeros((3, 3)), 0.5))
    with pytest.raises(ValueError, match="past the terrain"):
        rest_base_height(ground, np.array([[0.4, 0.0, -1.0]]), np.zeros(1), (0.8, 0.5))


def test_heights_file_round_trip_and_refusals(tmp_path):
    heights = np.random.default_rng(0).uniform(-0.02, 0.02, (6, 6))
    write_heightfield_file(tmp_path / "t.npz", heights, horizontal_scale_m=0.1)
    got, scale = read_heightfield_file(tmp_path / "t.npz")
    assert scale == 0.1 and got == pytest.approx(heights, abs=1e-6)
    for bad, why in [(np.zeros((4, 5)), "square"), (np.zeros((1, 1)), "square"), (np.full((3, 3), np.nan), "finite"),
                     (np.zeros(4), "square")]:
        with pytest.raises(ValueError, match=why):
            write_heightfield_file(tmp_path / "bad.npz", bad, horizontal_scale_m=0.1)
    with pytest.raises(ValueError, match="positive"):
        write_heightfield_file(tmp_path / "bad.npz", heights, horizontal_scale_m=0.0)
    np.savez(tmp_path / "no_scale.npz", heights=heights)
    with pytest.raises(ValueError, match="horizontal_scale_m"):
        read_heightfield_file(tmp_path / "no_scale.npz")


def test_terrain_describes_the_layout_its_config_states():
    gen = SimpleNamespace(size=(8.0, 8.0), num_rows=2, num_cols=4, border_width=5.0, curriculum=True,
                          horizontal_scale=0.1, vertical_scale=0.005, seed=3, sub_terrains={"waves": 1, "bumps": 1})
    cfg = SimpleNamespace(terrain_generator=gen, physics_material=SimpleNamespace(static_friction=0.7))
    assert as_terrain(cfg) == Terrain(cfg)
    assert as_terrain(Terrain(cfg)) == Terrain(cfg) and as_terrain(None) is None
    assert Terrain(cfg, source="a.json", sha256="ab").describe() == {
        "source": "a.json", "sha256": "ab", "type": "heightfield", "size_m": 8.0, "rows": 2, "cols": 4,
        "border_m": 5.0, "curriculum": True, "horizontal_scale_m": 0.1, "vertical_scale_m": 0.005, "seed": 3,
        "tiles": ["bumps", "waves"], "friction": 0.7}


def test_terrain_with_mesh_tiles_describes_itself_as_a_mesh():
    gen = SimpleNamespace(size=(4.0, 4.0), num_rows=1, num_cols=1, border_width=2.0, curriculum=False,
                          horizontal_scale=0.1, vertical_scale=0.005, seed=None,
                          sub_terrains={"stairs": SimpleNamespace(convert_to_heightfield=False),
                                        "bumps": SimpleNamespace(convert_to_heightfield=True)})
    cfg = SimpleNamespace(terrain_generator=gen, physics_material=SimpleNamespace(static_friction=1.0))
    assert Terrain(cfg).describe()["type"] == "mesh"
    gen.sub_terrains["stairs"].convert_to_heightfield = True
    assert Terrain(cfg).describe()["type"] == "heightfield"


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_resting_on_flat_ground_is_the_standing_height(robot):
    """The collision shapes the host rests on a terrain give the standing height on a plane, and the extra points on
    the soles' faces change nothing there."""
    from humanoid_sim.model.description import Description
    from humanoid_sim.model.kinematics import collision_points_in_base
    from humanoid_sim.model.poses import initial_pose

    usd = built_usd(robot)
    desc = Description.load(robot)
    q, height = initial_pose(usd, robot, "home")
    points, radius = collision_points_in_base(usd, desc, q)
    plane = Ground(*_grid_mesh(np.zeros((41, 41)), 0.1, origin=(-2.0, -2.0)))
    assert rest_base_height(plane, points, radius, (0.0, 0.0)) == pytest.approx(height, abs=1e-9)
    assert rest_base_height(plane, points, radius, (0.3, -0.2), yaw=1.0) == pytest.approx(height, abs=1e-9)
    raised = Ground(*_grid_mesh(np.full((41, 41), 0.07), 0.1, origin=(-2.0, -2.0)))
    assert rest_base_height(raised, points, radius, (0.0, 0.0)) == pytest.approx(height + 0.07, abs=1e-9)
    assert isinstance(q, torch.Tensor)


def test_a_bump_under_the_middle_of_a_sole_lifts_the_base():
    """The points on a sole's faces catch a bump that the corners straddle."""
    from humanoid_sim.model.kinematics import _box_face_grid

    half = np.array([0.125, 0.045, 0.01])
    grid = _box_face_grid(half, 0.01)
    assert np.abs(grid).max(axis=0) == pytest.approx(half)
    assert len(grid) > 8 * 20
    heights = np.zeros((41, 41))
    heights[20, 20] = 0.02  # at (0, 0), a bump 2 cm high and 0.2 m across at the base
    ground = Ground(*_grid_mesh(heights, 0.1, origin=(-2.0, -2.0)))
    corners = np.array([[x, y, -0.01] for x in (-0.125, 0.125) for y in (-0.045, 0.045)])
    sole = np.concatenate([corners, grid[grid[:, 2] < 0]])
    assert rest_base_height(ground, corners, np.zeros(4), (0.0, 0.0)) == pytest.approx(0.01)
    assert rest_base_height(ground, sole, np.zeros(len(sole)), (0.0, 0.0)) == pytest.approx(0.03, abs=0.003)  # the peak is 2 cm, between the grid's points
