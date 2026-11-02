"""Stereo depth errors (sensors/stereo_depth.py) on synthetic depth images: noise that grows with the square of the
range and on plain surfaces, spread over neighbouring pixels, disparity in steps, the occlusion band beside a
foreground edge and at the image border, no depth on plain surfaces, the median filter, and the stereo geometry of a
camera model. No Kit, no description."""

from __future__ import annotations

import numpy as np
import pytest

from humanoid_sim.sensors import geometry
from humanoid_sim.sensors.models import Camera, OakDW, RealSenseD435i
from humanoid_sim.sensors.stereo_depth import (
    blurred_noise,
    disparity_px,
    median_filter,
    occluded,
    stereo_depth,
    stereo_pair,
    texture_std,
)

FOCAL_PX = 400.0
BASELINE_M = 0.05  # f · b = 20 px m
SIGMA_PX = 0.08


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize("z", [1.0, 2.0, 4.0])
def test_noise_grows_with_the_square_of_the_range(z):
    depth = np.full((200, 400), z)
    out = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=SIGMA_PX, rng=_rng())
    seen = np.isfinite(out)
    expected = z**2 * SIGMA_PX / (FOCAL_PX * BASELINE_M)  # the D400 datasheet's RMS error
    assert np.std(out[seen]) == pytest.approx(expected, rel=0.05)
    assert abs(np.mean(out[seen]) - z) < 0.1 * expected  # unbiased to well within its spread


def test_whole_pixel_steps_leave_only_the_depths_of_whole_disparities():
    depth = np.linspace(0.8, 4.0, 400)[None, :].repeat(50, axis=0)
    out = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=SIGMA_PX, rng=_rng(),
                       disparity_step_px=1.0)
    k = FOCAL_PX * BASELINE_M / out[np.isfinite(out)]
    np.testing.assert_allclose(k, np.round(k), atol=1e-9)
    exact = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng(),
                         disparity_step_px=1.0)
    seen = np.isfinite(exact)
    d = FOCAL_PX * BASELINE_M / depth[seen]
    np.testing.assert_allclose(exact[seen], FOCAL_PX * BASELINE_M / np.round(d))  # the nearest step, without noise
    eighth = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng(),
                          disparity_step_px=0.125)
    assert np.abs(eighth[seen] - depth[seen]).max() < np.abs(exact[seen] - depth[seen]).max() / 4


def test_another_depth_focal_length_scales_every_depth_by_the_ratio():
    depth = np.full((10, 100), 2.0)
    out = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng(),
                       depth_focal_px=FOCAL_PX * 1.005)
    seen = np.isfinite(out)
    assert seen.any()
    np.testing.assert_allclose(out[seen], 2.0 * 1.005)


def test_without_errors_the_depth_is_exact_where_the_second_imager_sees():
    depth = np.full((10, 100), 2.0)
    out = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng())
    border = int(np.ceil(FOCAL_PX * BASELINE_M / 2.0))  # columns that fall outside the second imager's view
    assert np.isnan(out[:, :border]).all()
    np.testing.assert_array_equal(out[:, border:], 2.0)


def test_a_foreground_edge_casts_its_occlusion_band_away_from_the_second_imager():
    near_m, far_m, a, b = 1.0, 4.0, 60, 80
    depth = np.full((4, 120), far_m)
    depth[:, a:b] = near_m
    d_near, d_far = FOCAL_PX * BASELINE_M / near_m, FOCAL_PX * BASELINE_M / far_m  # 20 and 5 px
    band = d_near - d_far  # 15 px of background hidden from the second imager
    out = occluded(disparity_px(depth, FOCAL_PX, BASELINE_M), second_on_right=True)
    row = out[0]
    hidden = np.flatnonzero(row[int(np.ceil(d_far)):])  + int(np.ceil(d_far))  # past the image-border band
    assert hidden.min() == pytest.approx(a - band, abs=1) and hidden.max() == a - 1
    assert not row[a:].any()  # the foreground and the background past it are seen
    mirrored = occluded(disparity_px(depth, FOCAL_PX, BASELINE_M), second_on_right=False)[0]
    hidden = np.flatnonzero(mirrored[:-int(np.ceil(d_far))])
    assert hidden.min() == b and hidden.max() == pytest.approx(b - 1 + band, abs=1)


def test_no_surface_hides_no_surface():
    depth = np.full((2, 60), np.inf)  # nothing in view right of column 40: it cannot hide the surface left of it
    depth[:, :40] = 1.0
    out = occluded(disparity_px(depth, FOCAL_PX, BASELINE_M))
    border = int(FOCAL_PX * BASELINE_M / 1.0)  # 20 px outside the second imager's view
    assert out[:, :border].all() and not out[:, border:40].any()


def test_plain_surfaces_get_no_depth_and_textured_ones_keep_it():
    depth = np.full((40, 120), 1.5)
    grey = np.full((40, 120), 128.0)
    grey[:, 60:] += _rng(1).normal(0.0, 20.0, (40, 60))  # the right half textured
    out = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng(),
                       reference=grey, min_texture=2.0)
    assert np.isnan(out[:, 20:55]).all()
    assert np.isfinite(out[:, 65:]).all()


def test_plain_surfaces_get_more_noise_held_within_the_texture_range():
    depth = np.full((200, 400), 2.0)
    grey = np.full((200, 400), 128.0)
    grey[:, 200:] += _rng(1).normal(0.0, 40.0, (200, 200))  # the right half textured, far above the range
    texture_range, exponent = (0.5, 8.0), 0.5
    out = stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=SIGMA_PX, rng=_rng(),
                       reference=grey, texture_exponent=exponent, texture_range=texture_range)
    d = FOCAL_PX * BASELINE_M / out
    plain, textured = np.std(d[:, 60:190]), np.std(d[:, 210:])  # past the border band and the patch overlap
    assert plain == pytest.approx(SIGMA_PX * texture_range[0] ** -exponent, rel=0.05)
    assert textured == pytest.approx(SIGMA_PX * texture_range[1] ** -exponent, rel=0.05)


def test_blurred_noise_keeps_unit_variance_and_correlates_neighbours():
    for blur_px in (0.0, 1.0, 2.0):
        e = blurred_noise(_rng(), (300, 300), blur_px)
        assert np.std(e) == pytest.approx(1.0, rel=0.05)
        lag1 = np.mean(e[:, 1:] * e[:, :-1]) / np.var(e)
        # A Gaussian blur of sigma b correlates pixels one apart by exp(-1 / (4 b²)); white noise not at all.
        expected = np.exp(-1.0 / (4 * blur_px**2)) if blur_px else 0.0
        assert lag1 == pytest.approx(expected, abs=0.03)


def test_the_median_fills_small_holes_and_removes_small_islands():
    depth = np.full((40, 120), 2.0)
    depth[20, 60] = np.inf  # a one-pixel hole in the surface
    depth[:, 90:] = np.inf  # no surface on the right ...
    depth[10, 105] = 2.0  # ... but for one pixel
    kw = dict(focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng())
    raw = stereo_depth(depth, **kw)
    filtered = stereo_depth(depth, median_px=5, **kw)
    assert np.isnan(raw[20, 60]) and filtered[20, 60] == pytest.approx(2.0)
    assert np.isfinite(raw[10, 105]) and np.isnan(filtered[10, 105])
    np.testing.assert_array_equal(filtered[5:35, 30:85], 2.0)  # the surface whole and unchanged


def test_median_filter_is_the_window_median():
    image = _rng().normal(size=(9, 11))
    out = median_filter(image, 3)
    assert out[4, 5] == np.median(image[3:6, 4:7])
    assert out.shape == image.shape


def test_texture_std_is_the_patch_spread():
    grey = np.zeros((20, 20))
    grey[:, 1::2] = 10.0  # alternating columns: every 7 x 7 patch holds 3 or 4 of 7 bright columns
    s = texture_std(grey)
    assert np.all((s > 4.5) & (s < 5.1))
    assert np.allclose(texture_std(np.full((9, 9), 3.0)), 0.0)


def test_the_noise_is_reproducible_from_its_seed():
    depth = np.full((20, 50), 3.0)
    kw = dict(focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=SIGMA_PX)
    a = stereo_depth(depth, rng=_rng(7), **kw)
    np.testing.assert_array_equal(a, stereo_depth(depth, rng=_rng(7), **kw))
    assert not np.array_equal(np.nan_to_num(a), np.nan_to_num(stereo_depth(depth, rng=_rng(8), **kw)))


def test_bad_inputs_are_refused():
    depth = np.ones((4, 4))
    with pytest.raises(ValueError, match="baseline"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=0.0, disparity_sigma_px=0.1, rng=_rng())
    with pytest.raises(ValueError, match="focal lengths"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.1, rng=_rng(),
                     depth_focal_px=0.0)
    with pytest.raises(ValueError, match="disparity_step_px"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.1, rng=_rng(),
                     disparity_step_px=-1.0)
    with pytest.raises(ValueError, match="reference image"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng(),
                     min_texture=1.0)
    with pytest.raises(ValueError, match="differ in size"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.0, rng=_rng(),
                     reference=np.ones((3, 3)), min_texture=1.0)
    with pytest.raises(ValueError, match="texture_range"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.1, rng=_rng(),
                     reference=np.ones((4, 4)), texture_exponent=0.5)
    with pytest.raises(ValueError, match="reference image"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.1, rng=_rng(),
                     texture_exponent=0.5, texture_range=(1.0, 10.0))
    with pytest.raises(ValueError, match="noise_blur_px"):
        stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.1, rng=_rng(),
                     noise_blur_px=-1.0)
    for median_px in (1, 4, -3):
        with pytest.raises(ValueError, match="median_px"):
            stereo_depth(depth, focal_px=FOCAL_PX, baseline_m=BASELINE_M, disparity_sigma_px=0.1, rng=_rng(),
                         median_px=median_px)


@pytest.mark.parametrize("figures, match", [
    ({"depth_noise_texture_exponent": 0.3}, "depth_noise_texture_range"),
    ({"depth_noise_texture_exponent": 0.3, "depth_noise_texture_range": (2.0, 1.0)}, "depth_noise_texture_range"),
    ({"depth_noise_blur_px": -1.0}, ">= 0"),
    ({"depth_median_px": 4}, "depth_median_px"),
])
def test_a_camera_model_with_bad_depth_figures_is_refused(figures, match):
    with pytest.raises(ValueError, match=match):
        type("BadCamera", (Camera,), {"streams": ("mono_left", "depth"), **figures})


@pytest.mark.parametrize("model", [RealSenseD435i, OakDW])
def test_a_stereo_camera_matches_its_left_imager_against_the_right_one(model):
    pinhole = (300.0, 300.0, 320.0, 200.0)
    camera = model("camera_link", width=640, height=400, frame_period_ns=33_333_333,
                   intrinsics={s: pinhole for s in model.streams})
    camera.name = "cam"
    spec = geometry.camera_spec(camera)
    reference, baseline_m, on_right = stereo_pair(spec.streams, spec.stream("depth"), model.stereo_baseline_m)
    assert reference == "mono_left" and on_right
    assert baseline_m == pytest.approx(model.stereo_baseline_m)


def test_every_stereo_camera_has_depth_error_figures():
    for model in (RealSenseD435i, OakDW):
        assert model.depth_disparity_sigma_px > 0
        assert model.depth_disparity_step_px >= 0 and model.depth_min_texture >= 0
