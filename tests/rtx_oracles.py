"""Analytic answers the RTX sensor tests compare against: where a ray first meets a room of boxes and its floor, and
where a point lands in a camera image. Plain numpy, written from the geometry, sharing nothing with the renderer or
with ``sensors/rig.py`` beyond the lens parameters it is handed."""

from __future__ import annotations

import numpy as np


def ray_box_distance(origin: np.ndarray, dirs: np.ndarray, center: np.ndarray, half: np.ndarray,
                     yaw: np.ndarray) -> np.ndarray:
    """``[R]`` distance along each unit ray ``dirs`` ``[R, 3]`` from ``origin`` ``[3]`` to the nearest of boxes
    (``center``, ``half`` ``[N, 3]``, ``yaw`` ``[N]`` about z), ``inf`` where none is hit. Slab method in each box's
    frame."""
    best = np.full(len(dirs), np.inf)
    for c, h, a in zip(center, half, yaw, strict=True):
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])  # box to world
        o = R.T @ (origin - c)
        d = dirs @ R  # each ray in the box frame
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - o) / d
            t2 = (h - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 0.0))
        t = np.where(tmin > 0, tmin, tmax)
        best = np.where(hit & (t > 0) & (t < best), t, best)
    return best


def ray_floor_distance(origin: np.ndarray, dirs: np.ndarray, floor_z: float = 0.0) -> np.ndarray:
    """``[R]`` distance along each ray to the plane ``z = floor_z`` below, ``inf`` for rays that never reach it."""
    with np.errstate(divide="ignore"):
        t = (floor_z - origin[2]) / dirs[:, 2]
    return np.where((dirs[:, 2] < 0) & (t > 0), t, np.inf)


def pinhole_pixel(fx: float, fy: float, cx: float, cy: float, p_optical: np.ndarray) -> np.ndarray:
    """Pixel of a point in a distortion-free camera's optical frame (x right, y down, z forward)."""
    p = np.atleast_2d(p_optical)
    return np.stack([fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy], axis=1)


def fisheye_pixel(fx: float, fy: float, cx: float, cy: float, k: tuple[float, float, float, float],
                  p_optical: np.ndarray) -> np.ndarray:
    """Pixel of a point through OpenCV's fisheye (equidistant) model with coefficients ``k``."""
    p = np.atleast_2d(p_optical)
    r = np.hypot(p[:, 0], p[:, 1])
    theta = np.arctan2(r, p[:, 2])
    td = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)
    s = np.where(r > 0, td / np.where(r > 0, r, 1.0), 0.0)
    return np.stack([fx * s * p[:, 0] + cx, fy * s * p[:, 1] + cy], axis=1)


def fisheye_ray(fx: float, fy: float, cx: float, cy: float, k: tuple[float, float, float, float] | None,
                u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unit rays ``[..., 3]`` in the optical frame through pixel coordinates ``u``, ``v``: OpenCV's fisheye model
    inverted (Newton's method on its angle polynomial), or a pinhole's with ``k`` None."""
    x, y = (np.asarray(u, dtype=float) - cx) / fx, (np.asarray(v, dtype=float) - cy) / fy
    if k is None:
        r = np.stack([x, y, np.ones_like(x)], axis=-1)
    else:
        td = np.hypot(x, y)
        th = td.copy()
        for _ in range(20):
            f = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8) - td
            th -= f / (1 + 3 * k[0] * th**2 + 5 * k[1] * th**4 + 7 * k[2] * th**6 + 9 * k[3] * th**8)
        s = np.where(td > 0, np.sin(th) / np.where(td > 0, td, 1.0), 1.0)
        r = np.stack([x * s, y * s, np.cos(th)], axis=-1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def box_mask(shape: tuple[int, int], lens: tuple, eye_w: np.ndarray, optical_w: np.ndarray, center: np.ndarray,
             half: np.ndarray, at: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """``[H, W]``: the pixels whose centre's ray (OpenCV's convention: pixel (u, v) centred at u, v), or the ray
    through the point ``at`` from it, meets the box ``center``, ``half`` (world, axis aligned), for a camera at
    ``eye_w`` whose optical axes are the columns of ``optical_w``; ``lens`` is ``(fx, fy, cx, cy, k or None)``."""
    v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    rays = fisheye_ray(*lens, u.ravel() + at[0], v.ravel() + at[1]) @ np.asarray(optical_w).T
    d = ray_box_distance(np.asarray(eye_w, dtype=float), rays, np.array([center], dtype=float),
                         np.array([half], dtype=float), np.zeros(1))
    return np.isfinite(d).reshape(shape)


def box_coverage(shape: tuple[int, int], lens: tuple, eye_w: np.ndarray, optical_w: np.ndarray, center: np.ndarray,
                 half: np.ndarray, samples: int = 4) -> np.ndarray:
    """``[H, W]``: the share of each pixel's area (``samples`` by ``samples`` rays over it) whose rays meet the box."""
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    return np.mean([box_mask(shape, lens, eye_w, optical_w, center, half, (du, dv))
                    for du in offsets for dv in offsets], axis=0)


def mask_centroid(mask: np.ndarray) -> tuple[float, float, float]:
    """Centroid (u, v) of a mask or of per-pixel weights, and their sum."""
    w = np.asarray(mask, dtype=float)
    vs, us = np.mgrid[0:w.shape[0], 0:w.shape[1]]
    n = w.sum()
    return float((w * us).sum() / n), float((w * vs).sum() / n), float(n)
