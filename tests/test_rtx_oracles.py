"""The RTX tests' analytic helpers (``tests/rtx_oracles.py``) on cases worked out by hand."""

import math

import numpy as np
import pytest
from rtx_oracles import (
    box_coverage,
    box_mask,
    fisheye_pixel,
    fisheye_ray,
    mask_centroid,
    pinhole_pixel,
    ray_box_distance,
    ray_floor_distance,
)


def test_rays_meet_a_room_where_its_walls_are():
    # A wall 0.2 m thick centred 4.1 m ahead (its face at 4.0 m), and one turned 90 deg 3.1 m to the left.
    center = np.array([[4.1, 0.0, 1.5], [0.0, 3.1, 1.5]])
    half = np.array([[0.1, 3.0, 1.5], [0.1, 4.0, 1.5]])
    yaw = np.array([0.0, math.pi / 2])
    origin = np.array([0.0, 0.0, 1.0])
    s = math.sqrt(0.5)
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [s, 0, -s], [-1.0, 0, 0], [s, s, 0]])
    d = ray_box_distance(origin, dirs, center, half, yaw)
    assert d[0] == pytest.approx(4.0) and d[1] == pytest.approx(3.0)
    assert math.isinf(d[2])  # 45 deg down from 1 m: below the wall's foot (z = 0) by the time x = 4
    assert math.isinf(d[3])
    assert d[4] == pytest.approx(3.0 / s)
    f = ray_floor_distance(origin, dirs)
    assert f[2] == pytest.approx(1.0 / s) and math.isinf(f[0])


def test_projections():
    assert pinhole_pixel(100, 100, 50, 40, np.array([0.5, -0.2, 1.0]))[0] == pytest.approx([100, 20])
    th = math.radians(30)
    (u, v), = fisheye_pixel(200, 200, 0, 0, (0.1, 0, 0, 0), np.array([math.tan(th), 0, 1.0]))
    assert u == pytest.approx(200 * th * (1 + 0.1 * th**2)) and v == 0


def test_a_pixel_ray_projects_back_to_its_pixel():
    k = (-0.0358597, 0.0185157, -0.00496448, 0.00194007)
    u, v = np.array([10.0, 640.0, 1200.5]), np.array([20.0, 360.0, 700.25])
    for kk, project in ((k, lambda p: fisheye_pixel(771.4, 771.14, 629.1, 353.1, k, p)),
                        (None, lambda p: pinhole_pixel(771.4, 771.14, 629.1, 353.1, p))):
        uv = project(fisheye_ray(771.4, 771.14, 629.1, 353.1, kk, u, v))
        np.testing.assert_allclose(uv, np.stack([u, v], axis=1), atol=1e-6)


def test_a_box_mask_is_the_pixels_whose_rays_meet_the_box():
    # A camera at the origin looking along world +x (optical z), f = 100 px, centre (50, 50): a 0.2 m square plate
    # 1 m ahead covers pixels 40..60 in both directions (centres within +-10 px of the axis).
    optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    mask = box_mask((101, 101), (100.0, 100.0, 50.0, 50.0, None), np.zeros(3), optical,
                    np.array([1.0, 0.0, 0.0]), np.array([0.001, 0.1, 0.1]))
    u, v, n = mask_centroid(mask)
    assert (u, v, n) == (50.0, 50.0, 21 * 21)


def test_a_coverage_is_the_share_of_each_pixel_the_box_covers():
    # The plate of the box-mask test moved by a quarter pixel (2.5 mm at 1 m, f = 100 px) to the right: its edges at
    # 40.25 and 60.25 px. Edge pixels 40 and 60 are a quarter and three quarters covered.
    optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    cov = box_coverage((101, 101), (100.0, 100.0, 50.0, 50.0, None), np.zeros(3), optical,
                       np.array([1.0, -0.0025, 0.0]), np.array([0.001, 0.1, 0.1]))
    assert cov[50, 40] == 0.25 and cov[50, 41] == 1.0 and cov[50, 60] == 0.75 and cov[50, 61] == 0.0
    u, v, n = mask_centroid(cov)
    assert u == pytest.approx(50.25) and v == pytest.approx(50.0) and n == pytest.approx(20 * 20)
