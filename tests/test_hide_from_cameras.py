"""Geometry hidden from the cameras (``rtx.hide_from_cameras``: a splat room's boxes, the ground, the lidar's scan
mesh) is hidden from shadow rays too, so it cannot darken the splat that stands in for it."""

from __future__ import annotations

import pytest


def test_hidden_geometry_neither_shows_nor_casts_shadows():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from humanoid_sim.sensors.rtx import CASTS_NO_SHADOWS, HIDDEN_FROM_CAMERAS, hide_from_cameras

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Boxes")
    walls = [UsdGeom.Cube.Define(stage, f"/World/Boxes/Wall_{i}").GetPrim() for i in range(2)]
    other = UsdGeom.Cube.Define(stage, "/World/Robot").GetPrim()
    assert hide_from_cameras(stage, "/World/Boxes") == 2
    for wall in walls:
        assert wall.GetAttribute(HIDDEN_FROM_CAMERAS).Get() and wall.GetAttribute(CASTS_NO_SHADOWS).Get()
    assert not other.HasAttribute(HIDDEN_FROM_CAMERAS) and not other.HasAttribute(CASTS_NO_SHADOWS)
