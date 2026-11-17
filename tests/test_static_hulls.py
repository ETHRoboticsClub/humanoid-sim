"""Static convex pieces around the robot: loaded and placed like the static boxes, refused when degenerate.

Fixtures are synthetic pieces (a box's corners, a tetrahedron, a sheet), never a room's file.
"""

import numpy as np
import pytest

from humanoid_sim.world.boxes import box_corners
from humanoid_sim.world.hulls import (
    MIN_THICKNESS_M,
    StaticHulls,
    convex_polygon,
    hull_triangles,
    polygon_distance,
    thickness,
)

SPAWN = (-4.42, 3.48)


def _box(center, half, yaw=0.0) -> np.ndarray:
    return box_corners(np.array([center], dtype=np.float64), np.array([half], dtype=np.float64),
                       np.array([yaw]))[0]


def _write(path, pieces, **extra):
    np.savez(path, points=np.concatenate(pieces), counts=np.array([len(p) for p in pieces]), **extra)
    return path


def test_pieces_are_placed_by_the_spawn_point(tmp_path):
    wall = _box((SPAWN[0] + 2.5, SPAWN[1], 1.0), (0.1, 2.0, 1.0), 0.3)
    tetra = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) + [SPAWN[0] - 2, SPAWN[1], 0]
    hulls = StaticHulls.load(_write(tmp_path / "hulls.npz", [wall, tetra]), SPAWN)
    assert len(hulls) == 2
    offset = np.array([SPAWN[0], SPAWN[1], 0.0])
    np.testing.assert_allclose(hulls.pieces()[0], wall - offset)
    np.testing.assert_allclose(hulls.pieces()[1], tetra - offset)
    d = hulls.describe()
    assert d["pieces"] == 2 and d["spawn_m"] == list(SPAWN) and len(d["sha256"]) == 64
    assert hulls.rgb.shape == (2, 3)


def test_clearance_of_the_spawn_point(tmp_path):
    # A box turned by 45 degrees points a corner at the origin from 3 m away: its footprint reaches 3 - sqrt(2) m.
    hulls = StaticHulls.load(_write(tmp_path / "h.npz", [_box((3.0, 0.0, 0.5), (1.0, 1.0, 0.5), np.pi / 4)]))
    assert hulls.clearance_m() == pytest.approx(3 - np.sqrt(2))
    assert hulls.clearance_m((3.0, 0.0)) == 0.0


def test_a_spawn_point_inside_a_piece_is_refused(tmp_path):
    from humanoid_sim.runtime.host import RuntimeHost

    hulls = StaticHulls.load(_write(tmp_path / "h.npz", [_box((SPAWN[0], SPAWN[1], 0.5), (0.3, 0.3, 0.5))]), SPAWN)
    with pytest.raises(ValueError, match="inside a convex piece"):
        RuntimeHost(tmp_path / "plant", tmp_path / "unused.usd", device="cpu", collision=hulls)


def test_thin_and_malformed_pieces_are_refused(tmp_path):
    sheet = _box((0.0, 0.0, 1.0), (1.0, MIN_THICKNESS_M / 4, 1.0))
    with pytest.raises(ValueError, match="thinner"):
        StaticHulls.load(_write(tmp_path / "sheet.npz", [sheet]))
    with pytest.raises(ValueError, match="fewer than 4"):
        StaticHulls.load(_write(tmp_path / "few.npz", [np.eye(3)]))
    np.savez(tmp_path / "count.npz", points=np.zeros((8, 3)), counts=np.array([5]))
    with pytest.raises(ValueError, match="sum"):
        StaticHulls.load(tmp_path / "count.npz")
    bad = _box((0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        StaticHulls.load(_write(tmp_path / "nan.npz", [bad]))
    (tmp_path / "lfs.npz").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    with pytest.raises(ValueError, match="git-lfs"):
        StaticHulls.load(tmp_path / "lfs.npz")


def test_room_config_refuses_boxes_and_pieces_placed_differently(tmp_path):
    from humanoid_sim.world.boxes import StaticBoxes
    from humanoid_sim.world.hulls import add_room

    np.savez(tmp_path / "b.npz", center=np.array([[3.0, 0.0, 0.5]]), half=np.array([[0.5, 0.5, 0.5]]), yaw=0.0)
    boxes = StaticBoxes.load(tmp_path / "b.npz", (1.0, 0.0))
    hulls = StaticHulls.load(_write(tmp_path / "h.npz", [_box((3.0, 0.0, 0.5), (0.5, 0.5, 0.5))]), (0.0, 0.0))
    with pytest.raises(ValueError, match="spawn"):
        add_room(object(), boxes, hulls, friction=1.0)


def test_polygon_helpers():
    square = convex_polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float))
    assert len(square) == 4
    assert polygon_distance(square, np.array([0.5, 0.5])) == 0.0
    assert polygon_distance(square, np.array([2.0, 0.5])) == pytest.approx(1.0)
    assert polygon_distance(square, np.array([2.0, 2.0])) == pytest.approx(np.sqrt(2))
    assert thickness(_box((0, 0, 0), (1.0, 0.2, 3.0), 0.7)) == pytest.approx(0.4)


def test_hull_triangles_wind_outward():
    pytest.importorskip("scipy")
    pts = _box((1.0, -2.0, 0.5), (0.5, 0.3, 0.5), 0.4)
    tris = hull_triangles(pts)
    assert len(tris) == 12
    normal = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
    assert (np.einsum("ij,ij->i", normal, pts[tris[:, 0]] - pts.mean(axis=0)) > 0).all()
