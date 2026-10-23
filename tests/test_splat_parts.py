"""Cutting a splat's moving parts out (``world/splat.py`` ``split``), without pxr."""

import numpy as np
import pytest

from humanoid_sim.world import splat as sp


def _splat(points) -> sp.Splat:
    n = len(points)
    return sp.Splat(positions=np.asarray(points, dtype=np.float32), orientations_xyzw=np.tile([0, 0, 0, 1.0], (n, 1)),
                    scales=np.full((n, 3), 0.01), opacities=np.full(n, 0.9), sh=np.zeros((n, 1, 3)), sh_degree=0)


def test_parts_take_the_gaussians_in_their_boxes_and_drop_boxes_remove_them():
    room = _splat([[0.0, 0.0, 3.1], [0.0, 0.9, 3.1], [2.0, 0.0, 3.1], [0.0, 0.0, 1.0], [5.0, 5.0, 0.0]])
    # A bridge 2 m long along y turned 90 degrees (so along x), the trolley on it, the hanging robot dropped.
    parts = {"trolley": [{"center": [0.0, 0.0, 3.1], "half": [0.2, 0.2, 0.2]}],
             "bridge": [{"center": [1.0, 0.0, 3.1], "half": [0.1, 1.2, 0.1], "yaw": np.pi / 2}],
             "drop": [{"center": [0.0, 0.0, 1.0], "half": [0.3, 0.3, 1.0]}]}
    rest, cut = sp.split(room, parts)
    assert cut["trolley"].positions.tolist() == [[0.0, 0.0, pytest.approx(3.1)]]  # first part wins
    assert cut["bridge"].positions[:, 0].tolist() == [2.0]
    assert rest.positions.tolist() == [[0.0, pytest.approx(0.9), pytest.approx(3.1)], [5.0, 5.0, 0.0]]


def test_part_names_must_be_prim_names():
    with pytest.raises(ValueError, match="prim name"):
        sp.split(_splat([[0.0, 0.0, 0.0]]), {"the bridge": []})
