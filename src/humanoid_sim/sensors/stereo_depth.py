"""What a stereo camera's depth stream reports, from the exact depth RTX renders (``distance_to_image_plane``).

A stereo camera does not measure depth: it matches each pixel of its reference imager against its second imager
and measures the disparity ``d = f · b / z`` (pixels; ``f`` the focal length in pixels, ``b`` the baseline). Its
depth errors follow from that, and this module applies them to an exact depth image:

* **Noise.** The matcher finds ``d`` to a fraction of a pixel, ``σ_d`` (the model's ``depth_disparity_sigma_px``).
  Through ``z = f · b / d`` the depth error grows with the square of the range: ``σ_z ≈ z² · σ_d / (f · b)``, the
  RMS error formula of Intel's D400 datasheet. The noise is drawn per pixel, independently.
* **Steps.** A matcher reports ``d`` in steps (the model's ``depth_disparity_step_px``): whole pixels without
  sub-pixel refinement, so the depth takes only the values ``f · b / k`` (at 2 m with ``f · b`` = 21 px m, 1.93 m or
  2.12 m). The noisy disparity is rounded to the nearest step; 0 leaves it continuous.
* **Occlusion.** A pixel the second imager cannot see has no match: the band beside a foreground edge, on the side
  away from the second imager (``d_near − d_far`` pixels wide), and the image border on that side, where the second
  imager's view ends. A pixel is occluded when a nearer pixel on the second imager's side lands on or past it in
  the second imager.
* **No texture.** A matcher compares image patches; where the reference image is too uniform (a plain wall, a dark
  surface) it finds no match. A pixel whose reference-image patch has a grey-level standard deviation below the
  model's ``depth_min_texture`` gets no depth.

Pixels without depth are NaN; :func:`~humanoid_sim.sensors.feeds.depth_mm` reports them as 0, as the cameras do.
The depth range (``depth_range_m``) is applied there too.

Not modeled: the spatial correlation of the matcher's errors (its block size), noise that grows on plain surfaces
(a camera that still reports depth there flickers more), foreground edges widened by the matcher's window,
grazing-angle failures, flying pixels at edges, and the cameras' own post-processing filters.
"""

from __future__ import annotations

import numpy as np

TEXTURE_WINDOW_PX = 7  # side of the square patch whose grey-level spread decides "no texture"; modeled


def disparity_px(depth_m: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """Disparity (pixels) of depth (m along the optical axis); 0 where there is no surface (NaN, inf or ≤ 0)."""
    z = np.asarray(depth_m, dtype=np.float64)
    ok = np.isfinite(z) & (z > 0)
    return np.where(ok, focal_px * baseline_m / np.where(ok, z, 1.0), 0.0)


def occluded(disparity: np.ndarray, *, second_on_right: bool = True) -> np.ndarray:
    """Pixels ``[H, W]`` the second imager does not see: occluded by a nearer surface on its side, or outside its
    view. ``disparity`` ``[H, W]`` is the reference imager's (0: no surface). With the second imager on the right a
    pixel at column ``u`` appears at ``u − d`` in it."""
    d = np.asarray(disparity, dtype=np.float64)
    if not second_on_right:
        return occluded(d[:, ::-1], second_on_right=True)[:, ::-1]
    u = np.arange(d.shape[1], dtype=np.float64)
    where = u[None, :] - d  # column in the second imager
    # The leftmost second-imager column reached by any pixel to the right of each one (exclusive).
    after = np.minimum.accumulate(where[:, ::-1], axis=1)[:, ::-1]
    after = np.concatenate([after[:, 1:], np.full((d.shape[0], 1), np.inf)], axis=1)
    return (after < where) | (where < 0)


def texture_std(image: np.ndarray, window: int = TEXTURE_WINDOW_PX) -> np.ndarray:
    """Grey-level standard deviation over a ``window`` square around each pixel (edges padded by reflection)."""
    g = np.asarray(image, dtype=np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    r = window // 2
    p = np.pad(g, r, mode="reflect")

    def box_mean(x):
        c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), axis=0), axis=1)
        return (c[window:, window:] - c[:-window, window:] - c[window:, :-window] + c[:-window, :-window]) / window**2

    mean = box_mean(p)
    return np.sqrt(np.maximum(box_mean(p * p) - mean * mean, 0.0))


def stereo_pair(streams, depth, baseline_m: float) -> tuple[str | None, float, bool]:
    """A camera's stereo geometry for its ``depth`` stream (both :class:`~humanoid_sim.sensors.geometry.Stream`):
    the reference imager (the grey stream at the depth's optical centre, None without one), the baseline (m) to the
    other grey stream (``baseline_m``, the model's, without one) and whether that imager is on the reference's
    right in the image. Camera frames have y left, so an imager at smaller y is on the right."""
    monos = [s for s in streams if s.kind == "mono"]
    reference = next((s for s in monos if np.allclose(s.translation_m, depth.translation_m)), None)
    second = next((s for s in monos if s is not reference), None)
    if reference is None or second is None:
        return (reference.name if reference else None), baseline_m, True
    dy = second.translation_m[1] - reference.translation_m[1]
    return reference.name, float(np.linalg.norm(np.subtract(second.translation_m, reference.translation_m))), dy < 0


def stereo_depth(depth_m: np.ndarray, *, focal_px: float, baseline_m: float, disparity_sigma_px: float,
                 rng: np.random.Generator, disparity_step_px: float = 0.0, reference: np.ndarray | None = None,
                 min_texture: float = 0.0, second_on_right: bool = True) -> np.ndarray:
    """The depth (m) a stereo camera reports for the exact depth ``depth_m`` ``[H, W]``: noisy through its disparity,
    rounded to its ``disparity_step_px`` (0: continuous), NaN where its second imager is occluded or its reference
    image ``reference`` (grey, ``[H, W]``) has less texture than ``min_texture``. ``baseline_m`` and ``focal_px`` are
    the depth stream's."""
    if baseline_m <= 0 or focal_px <= 0:
        raise ValueError(f"stereo depth needs a positive baseline and focal length, got {baseline_m}, {focal_px}")
    if disparity_step_px < 0:
        raise ValueError(f"disparity_step_px must be >= 0, got {disparity_step_px}")
    d = disparity_px(depth_m, focal_px, baseline_m)
    surface = d > 0
    noisy = d + rng.normal(0.0, disparity_sigma_px, d.shape) if disparity_sigma_px > 0 else d
    if disparity_step_px > 0:
        noisy = np.round(noisy / disparity_step_px) * disparity_step_px
    valid = surface & (noisy > 0) & ~occluded(d, second_on_right=second_on_right)
    if min_texture > 0:
        if reference is None:
            raise ValueError("min_texture needs the reference image")
        if reference.shape[:2] != d.shape:
            raise ValueError(f"reference image {reference.shape[:2]} and depth {d.shape} differ in size")
        valid &= texture_std(reference) >= min_texture
    return np.where(valid, focal_px * baseline_m / np.where(valid, noisy, 1.0), np.nan)
