"""Which of a room's surfaces each RTX ray class sees. No Kit: ``SensorSession`` applies the answer to the stage.

A scanned room can have three surfaces: its collision boxes (walls and furniture, ``world/boxes.py``), its scan
mesh (added for the lidar, ``--scan-mesh``) and its Gaussian splat (for the cameras, ``--splat``). Each ray class
sees the most faithful of them it has:

* The lidar sees the scan mesh and the wall boxes. The furniture boxes enclose the scanned furniture by up to tens
  of centimetres and would shadow it; the walls stay because the mesh has holes at windows and doors and the walls
  keep every return inside the room. Without the mesh it sees every box.
* The cameras see the splat. Without one they see what the lidar sees (the depth stream and the lidar agree).

A box no ray sees is hidden from every ray. The physics collides with every box whatever is hidden: its model is
built from the stage before this is applied (``SensorSession``).

A boxes file without wall flags does not say which boxes bound the room; every box is then taken as a wall, so a
scan keeps its returns inside the room at the cost of furniture boxes in front of the mesh (``walls_known`` False,
recorded in the session summary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoomRays:
    hidden: tuple[int, ...]  # box indices hidden from every ray
    lidar_only: tuple[int, ...]  # box indices hidden from camera rays, seen by the lidar
    scan_hidden_from_cameras: bool  # the scan mesh, when there is one
    walls_known: bool  # False: a boxes file had no wall flags, every box was taken as a wall

    def describe(self) -> dict:
        return {"boxes_hidden": len(self.hidden), "boxes_lidar_only": len(self.lidar_only),
                "scan_hidden_from_cameras": self.scan_hidden_from_cameras, "walls_known": self.walls_known}


def room_rays(wall: np.ndarray | None, n_boxes: int, *, cameras: bool, splat: bool, lidar: bool,
              scan: bool) -> RoomRays:
    """What each ray class sees of ``n_boxes`` boxes with wall flags ``wall`` (``[n_boxes]`` bool, None when the
    file has none or there are no boxes), given whether cameras and a lidar are simulated, a splat covers the cameras
    and a scan mesh was added for the lidar."""
    if scan and not lidar:
        raise ValueError("a scan mesh is added as lidar surfaces only when a lidar is simulated")
    if wall is not None:
        wall = np.asarray(wall)
        if wall.dtype != bool or wall.shape != (n_boxes,):
            raise ValueError(f"wall flags must be [{n_boxes}] bool, got {wall.dtype} {wall.shape}")
    walls = np.ones(n_boxes, dtype=bool) if wall is None else wall
    lidar_sees = lidar & (walls | (not scan))
    cameras_see = (cameras and not splat) & (walls | (not scan))
    return RoomRays(hidden=tuple(int(i) for i in np.flatnonzero(~lidar_sees & ~cameras_see)),
                    lidar_only=tuple(int(i) for i in np.flatnonzero(lidar_sees & ~cameras_see)),
                    scan_hidden_from_cameras=splat, walls_known=wall is not None or n_boxes == 0)
