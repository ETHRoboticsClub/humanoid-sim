"""Which of a room's surfaces each ray class sees (``sensors/room_rays.py``), worked out by hand for five boxes: walls
0 and 2, furniture 1, 3 and 4. No Kit needed."""

import numpy as np
import pytest

from humanoid_sim.sensors.room_rays import room_rays
from humanoid_sim.world.boxes import StaticBoxes

WALL = np.array([True, False, True, False, False])
ALL = (0, 1, 2, 3, 4)
SENSORS = {"cameras": True, "lidar": True}


@pytest.mark.parametrize("splat,scan,hidden,lidar_only", [
    # The splat for the cameras, the mesh and the walls for the lidar: furniture boxes for no ray.
    (True, True, (1, 3, 4), (0, 2)),
    # The splat for the cameras, every box for the lidar.
    (True, False, (), ALL),
    # No splat: the cameras see what the lidar sees, the mesh and the walls.
    (False, True, (1, 3, 4), ()),
    # Boxes alone: every ray sees every box.
    (False, False, (), ()),
])
def test_each_ray_class_sees_its_rooms_best_surface(splat, scan, hidden, lidar_only):
    rays = room_rays(WALL, 5, splat=splat, scan=scan, **SENSORS)
    assert (rays.hidden, rays.lidar_only) == (hidden, lidar_only)
    assert rays.scan_hidden_from_cameras is splat
    assert rays.walls_known


def test_all_walls_and_no_walls():
    walls = room_rays(np.ones(5, dtype=bool), 5, splat=True, scan=True, **SENSORS)
    assert (walls.hidden, walls.lidar_only) == ((), ALL)
    furniture = room_rays(np.zeros(5, dtype=bool), 5, splat=True, scan=True, **SENSORS)
    assert (furniture.hidden, furniture.lidar_only) == (ALL, ())


def test_without_wall_flags_every_box_bounds_the_room():
    # Hiding unflagged boxes would let the lidar see out through the mesh's holes; it keeps them, and says so.
    rays = room_rays(None, 5, splat=True, scan=True, **SENSORS)
    assert (rays.hidden, rays.lidar_only) == ((), ALL)
    assert not rays.walls_known and rays.describe()["walls_known"] is False


def test_a_scan_mesh_without_boxes_has_no_walls_to_know():
    # --scan-mesh without --scene: nothing to hide, and no boxes file whose wall flags could be missing.
    rays = room_rays(None, 0, splat=False, scan=True, **SENSORS)
    assert (rays.hidden, rays.lidar_only) == ((), ())
    assert rays.walls_known and not rays.scan_hidden_from_cameras


def test_a_box_no_sensor_needs_is_hidden_from_every_ray():
    no_lidar = room_rays(WALL, 5, cameras=True, splat=True, lidar=False, scan=False)
    assert (no_lidar.hidden, no_lidar.lidar_only) == (ALL, ())
    no_cameras = room_rays(WALL, 5, cameras=False, splat=False, lidar=True, scan=True)
    assert (no_cameras.hidden, no_cameras.lidar_only) == ((1, 3, 4), (0, 2))
    cameras_only = room_rays(WALL, 5, cameras=True, splat=False, lidar=False, scan=False)
    assert (cameras_only.hidden, cameras_only.lidar_only) == ((), ())


def test_refused_inputs():
    with pytest.raises(ValueError, match="only when a lidar"):
        room_rays(WALL, 5, cameras=True, splat=False, lidar=False, scan=True)
    with pytest.raises(ValueError, match="bool"):
        room_rays(WALL.astype(int), 5, splat=True, scan=True, **SENSORS)
    with pytest.raises(ValueError, match=r"\[4\]"):
        room_rays(WALL, 4, splat=True, scan=True, **SENSORS)


def test_the_wall_flags_come_from_the_boxes_file(tmp_path):
    path = tmp_path / "room.npz"
    np.savez(path, center=np.array([[0.0, 3.0, 1.0], [1.0, 0.0, 0.4], [0.0, -3.0, 1.0]]),
             half=np.array([[3.0, 0.1, 1.0], [0.3, 0.3, 0.4], [3.0, 0.1, 1.0]]), yaw=np.float32(0.0),
             wall=np.array([True, False, True]))
    boxes = StaticBoxes.load(path)
    np.testing.assert_array_equal(boxes.wall, [True, False, True])
    rays = room_rays(boxes.wall, len(boxes), splat=True, scan=True, **SENSORS)
    assert (rays.hidden, rays.lidar_only) == ((1,), (0, 2))


@pytest.mark.parametrize("hidden", [True, False])
def test_a_scan_mesh_hidden_from_the_cameras_casts_no_shadow_into_them(tmp_path, hidden):
    Usd = pytest.importorskip("pxr.Usd")
    from humanoid_sim.sensors.lidar import scan_mesh_prim

    obj = tmp_path / "scan.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n")
    stage = Usd.Stage.CreateInMemory()
    prim = scan_mesh_prim(stage, obj, (0.0, 0.0), hidden_from_cameras=hidden)
    # RTX's primvars: hidden from primary camera rays, and from shadow rays.
    flags = [bool(prim.GetAttribute(a).Get()) for a in ("primvars:hideForCamera", "primvars:doNotCastShadows")]
    assert flags == [hidden, hidden]
