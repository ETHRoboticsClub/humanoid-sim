"""Where a rig's cameras and lidars sit and what they see, built from the rig's sensor models
(``sensors/models.py``, ``sensors/rigs.py``): the RTX sensors (``sensors/rtx.py``, ``sensors/lidar.py``) render them and the feeds
(``sensors/feeds.py``) publish them in the layouts Controls reads. Nothing about a product or a unit is written here.

* :class:`Mount`: a sensor's frame (its rig ``frame``, a URDF link) composed through fixed joints onto the body the
  physics keeps (the asset build merges fixed links into their parent body: ``d435_link`` rides on ``torso_link``).
* :func:`camera_spec`: a rig camera's streams, each with its kind and encoding as Controls' backend publishes it, its
  lens (the rig's calibration of the unit: intrinsics and distortion) and its optical centre in the camera frame
  (the unit's calibration, else the model's nominal place).
* :func:`scan_pattern`: a lidar model's ray directions and firing times, sweep by sweep.

Frames. A camera's frame has x forward, y left and z up, its origin at its reference imager's optical centre
(Controls' ``ethrc_cameras.backend.Calibration``); each stream's optical frame has x right, y down and z forward. A
lidar's frame is its rig frame's link (the G1's Mid-360 at ``mid360_link``, mounted upside down: roll pi).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Optical frame (x right, y down, z forward) to device or link frame (x forward, y left, z up).
LINK_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy as a rotation matrix: ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion (x, y, z, w) of a rotation matrix, w >= 0."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t)
        q = ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = (*v, (R[k, j] - R[j, k]) / s)
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[3] >= 0 else -q))


@dataclass(frozen=True)
class Mount:
    """A sensor frame fixed on a body the physics keeps: ``link``, and the frame's pose in it."""

    sensor_link: str  # the description link that is the sensor's frame
    link: str  # the body it rides on
    translation_m: tuple[float, float, float]
    rotation: tuple[tuple[float, ...], ...]  # 3x3, sensor frame to ``link``

    @property
    def R(self) -> np.ndarray:
        return np.array(self.rotation)

    @property
    def rotation_xyzw(self) -> tuple[float, float, float, float]:
        return quat_xyzw(self.R)

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.translation_m
        return T

    @classmethod
    def from_urdf(cls, urdf: str | Path, sensor_link: str, kept_links: set[str] | None = None) -> Mount:
        """The pose of ``sensor_link`` in the nearest ancestor reached through a non-fixed joint (or the root), or in
        the nearest ancestor in ``kept_links``."""
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        if sensor_link not in by_child:
            raise ValueError(f"{Path(urdf).name}: no joint places {sensor_link!r}")
        T = np.eye(4)
        link = sensor_link
        while link in by_child and by_child[link].get("type") == "fixed" and not (kept_links and link in kept_links
                                                                                   and link != sensor_link):
            j = by_child[link]
            origin = j.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            J = np.eye(4)
            J[:3, :3], J[:3, 3] = rpy_matrix(*rpy), xyz
            T = J @ T
            link = j.find("parent").get("link")
        return cls(sensor_link, link, tuple(float(v) for v in T[:3, 3]),
                   tuple(tuple(float(v) for v in row) for row in T[:3, :3]))


@dataclass(frozen=True)
class Lens:
    """A stream's lens at its output resolution: a pinhole with ``distortion_model`` ``plumb_bob`` (k1 k2 p1 p2
    k3), ``equidistant`` (k1..k4, OpenCV's fisheye) or ``none``."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion_model: str = "none"
    distortion: tuple[float, ...] = ()

    @classmethod
    def from_fov(cls, width_px: int, height_px: int, hfov_deg: float, vfov_deg: float | None = None) -> Lens:
        """A centred pinhole with these fields of view (the vertical one from square pixels when not given)."""
        fx = 0.5 * width_px / math.tan(math.radians(hfov_deg) / 2)
        fy = fx if vfov_deg is None else 0.5 * height_px / math.tan(math.radians(vfov_deg) / 2)
        return cls(width_px, height_px, fx, fy, width_px / 2, height_px / 2)

    def project(self, p_optical: np.ndarray) -> np.ndarray:
        """Pixel coordinates ``[N, 2]`` (u right, v down; OpenCV's convention, pixel (u, v) centred at u, v) of points
        ``[N, 3]`` in the optical frame, in front of the lens."""
        p = np.atleast_2d(np.asarray(p_optical, dtype=float))
        a, b = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
        if self.distortion_model == "equidistant":
            k1, k2, k3, k4 = self.distortion
            r = np.hypot(a, b)
            theta = np.arctan(r)
            td = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
            scale = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 1.0)
            a, b = a * scale, b * scale
        elif self.distortion_model == "plumb_bob":
            k1, k2, p1, p2, k3 = self.distortion
            r2 = a * a + b * b
            radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            a, b = a * radial + 2 * p1 * a * b + p2 * (r2 + 2 * a * a), b * radial + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
        elif self.distortion_model != "none":
            raise ValueError(f"distortion model {self.distortion_model!r}")
        return np.stack([self.fx_px * a + self.cx_px, self.fy_px * b + self.cy_px], axis=1)

    def contract_model(self) -> tuple[str, tuple[float, ...]]:
        """The model as Controls' ``Intrinsics`` names it (``none`` is ``plumb_bob`` with zeros)."""
        if self.distortion_model == "none":
            return "plumb_bob", (0.0, 0.0, 0.0, 0.0, 0.0)
        return self.distortion_model, tuple(self.distortion)


@dataclass(frozen=True)
class Stream:
    """One image stream of a camera: ``kind`` ``color`` ``mono`` or ``depth``, its encoding as Controls'
    ``FRAME_ENCODINGS`` names it, its lens and its optical frame's pose in the camera frame."""

    name: str
    kind: str
    encoding: str
    rate_hz: float
    lens: Lens
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = quat_xyzw(LINK_FROM_OPTICAL)
    sync_group: str = ""
    depth_source: str = ""
    range_m: tuple[float, float] = (0.0, math.inf)  # depth: what the sensor reports, else 0
    shutter: str = "global"  # or "rolling"

    def optical_in_device(self) -> np.ndarray:
        """4x4 pose of the optical frame in the device frame."""
        from scipy.spatial.transform import Rotation

        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.rotation_xyzw).as_matrix()
        T[:3, 3] = self.translation_m
        return T


@dataclass(frozen=True)
class CameraSpec:
    """A rig camera as its feed publishes it."""

    camera_id: str  # the feed's device directory, and the device serial Controls lists
    product: str
    sensor_link: str  # the rig frame: the description link that is the camera frame
    streams: tuple[Stream, ...]
    calibration_id: str

    def stream(self, name: str) -> Stream:
        return next(s for s in self.streams if s.name == name)


ENCODINGS = {"color": "bgr8", "mono": "mono8", "depth": "16UC1"}  # Controls' FRAME_ENCODINGS, by stream kind


def camera_spec(camera) -> CameraSpec:
    """The camera spec of a rig camera (:class:`~humanoid_sim.sensors.models.Camera`) the RTX sensors can simulate
    (``Camera.rtx_ready``)."""
    c = camera
    if not c.name:
        raise ValueError(f"{c!r} is in no rig: its feed takes the rig's name")
    if not c.rtx_ready:
        raise ValueError(f"camera {c.name!r} ({c.model}): not every stream has a kind, a place and intrinsics")
    stereo = c.stereo_baseline_m is not None
    streams = []
    for name in c.streams:
        kind = c.stream_kinds[name]
        fx, fy, cx, cy = c.intrinsics[name]
        model, coefficients = c.distortion.get(name, ("none", ()))
        lens = Lens(c.width, c.height, fx, fy, cx, cy, model, tuple(coefficients))
        streams.append(Stream(name, kind, ENCODINGS[kind], 1e9 / c.frame_period_ns, lens,
                              tuple(c.stream_offsets_m[name]),
                              sync_group="stereo" if stereo and kind == "mono" else "",
                              depth_source="stereo" if stereo and kind == "depth" else "",
                              range_m=tuple(c.depth_range_m) if kind == "depth" else (0.0, math.inf),
                              shutter="rolling" if name in c.rolling_shutter else "global"))
    return CameraSpec(f"sim-{c.name}", c.product, c.frame, tuple(streams),
                      f"{c.name}-{c.model}-{c.width}x{c.height}")


def scan_pattern(model, sweep: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit ray directions ``[N, 3]`` in the sensor frame and each ray's firing time ``[N]`` (s after the sweep's
    start, evenly over the sweep) for sweep number ``sweep`` of a lidar (:class:`~humanoid_sim.sensors.models.Lidar`): its points per sweep
    over its band of elevation and its horizontal field. The Mid-360 scans a non-repetitive rosette; this
    approximates it by a 2-D Halton sequence (bases 2 and 3) mapped uniformly over the band's solid angle and continued
    from sweep to sweep, so no direction repeats and coverage fills in over time as the Mid-360's does."""
    period_s = model.frame_period_ns / 1e9
    n = int(round(model.point_rate_hz * period_s))
    idx = np.arange(sweep * n, (sweep + 1) * n, dtype=np.int64) + 1
    u, v = _halton(idx, 2), _halton(idx, 3)
    lo, hi = model.elevation_min_rad, model.elevation_max_rad
    el = np.arcsin(np.sin(lo) + v * (np.sin(hi) - np.sin(lo)))  # uniform over the band's area
    az = (u - 0.5) * model.horizontal_fov_rad
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    t = (np.arange(n) + 0.5) / n * period_s
    return d, t


def _halton(index: np.ndarray, base: int) -> np.ndarray:
    out = np.zeros(len(index))
    f = 1.0
    i = index.copy()
    while np.any(i > 0):
        f /= base
        out += f * (i % base)
        i //= base
    return out
