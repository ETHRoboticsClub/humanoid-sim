"""Analytic answers the RTX sensor tests compare against: where a ray first meets a room of boxes and its floor, and
where a point lands in a camera image; and the G1 rig and test room those tests share. Plain numpy, written from the
geometry and the datasheets; the sensors' mounts are read from the G1's URDF in humanoid-urdf (found as the
simulation finds it, ``model/description.py``) and parsed here, sharing nothing with the renderer or with
``sensors/geometry.py``'s mounts."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

G1 = "g1_29dof_with_hand_rev_1_0"
CAMERAS = "g1-cameras-d435"  # sensors/rigs.py EXTRAS
SETTLE_S = 1.0  # the welded robot's arms and waist settle on their hold within this; images and sweeps after it

# The URDF joints that fix the sensors on torso_link.
D435_JOINT, MID360_JOINT, ZED_JOINT = "d435_joint", "mid360_joint", "camera0_joint"
# D435i, every stream as Controls' realsense adapter opens it: 640 x 480, the principal point at the image's centre.
D435_WIDTH_PX, D435_HEIGHT_PX = 640, 480
D435_CX_PX, D435_CY_PX = D435_WIDTH_PX / 2, D435_HEIGHT_PX / 2
# D435i depth, D400 datasheet: 87 x 58 deg, depth reported out to 10 m.
D435_DEPTH_F = (D435_CX_PX / math.tan(math.radians(87.0 / 2)), D435_CY_PX / math.tan(math.radians(58.0 / 2)))
D435_DEPTH_MAX_M = 10.0

# The test room: walls 0.2 m thick and 3 m tall around the robot (centre, half extents in m).
WALLS = [((3.1, 0.0, 1.5), (0.1, 2.7, 1.5)), ((-3.1, 0.0, 1.5), (0.1, 2.7, 1.5)),
         ((0.0, 2.6, 1.5), (3.2, 0.1, 1.5)), ((0.0, -2.6, 1.5), (3.2, 0.1, 1.5))]

# Depth and range: the renderer's precision.
DEPTH_TOL_M = 0.01
RANGE_TOL_M = 0.01
# A depth or range this much short of the analytic one met something nearer than the room (the robot's own body):
# five times the tolerance, so a renderer error is never mistaken for it.
NEARER_M = 0.05
SELF_HIT_M = 1.0  # rays that meet the robot's own body first: shorter than this, the room is farther

OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical axes in a device frame


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_matrix_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def torso_mount(joint: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The fixed ``joint``'s origin in the G1's URDF: position (m) and rpy (rad) in torso_link, its parent."""
    from humanoid_sim.model.description import Description

    urdf = Description.load(G1).urdf_path
    found = [j for j in ET.parse(urdf).getroot().findall("joint") if j.get("name") == joint]
    assert len(found) == 1, f"{urdf.name}: {len(found)} joints named {joint}"
    j = found[0]
    assert (j.get("type"), j.find("parent").get("link")) == ("fixed", "torso_link"), f"{joint} is not fixed on the torso"
    origin = j.find("origin")
    xyz, rpy_ = (tuple(float(v) for v in origin.get(k, "0 0 0").split()) for k in ("xyz", "rpy"))
    return xyz, rpy_


def device_pose(torso, mount):
    """World pose (R, t) of a device frame on the torso."""
    (R_t, t_t), (t_m, rpy_m) = torso, mount
    return R_t @ rpy(*rpy_m), R_t @ np.asarray(t_m) + t_t


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


def d435_depth_rays() -> np.ndarray:
    """``[H * W, 3]``: the ray through each D435i depth pixel's centre in its optical frame, row by row, scaled to
    ``z = 1`` (a pixel's depth is the distance along its ray divided by the ray's norm)."""
    fx, fy = D435_DEPTH_F
    v, u = np.mgrid[0:D435_HEIGHT_PX, 0:D435_WIDTH_PX]
    return np.stack([(u.ravel() - D435_CX_PX) / fx, (v.ravel() - D435_CY_PX) / fy, np.ones(u.size)], axis=1)


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


def write_box_obj(path, center, half):
    """A closed box as an OBJ of outward-wound triangles, every corner with a texture coordinate (``read_obj``)."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    v = np.asarray(center) + signs * np.asarray(half)
    # Corner k has signs (x, y, z) = bits (k >> 2, k >> 1, k): each face's quad, wound outward.
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    lines = [f"v {x} {y} {z}" for x, y, z in v] + ["vt 0 0"]
    for a, b, c, d in quads:
        lines += [f"f {a + 1}/1 {b + 1}/1 {c + 1}/1", f"f {a + 1}/1 {c + 1}/1 {d + 1}/1"]
    path.write_text("\n".join(lines) + "\n")
