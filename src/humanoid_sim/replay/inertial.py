"""Offline IMU-to-base acceleration for equivalent inertial loading, not odometry.

Capture vectors are already expressed in base axes, but measured at the IMU.
No ROS, physics imports, position integration, estimated bias or fitted gain.
"""

from __future__ import annotations

import hashlib
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.replay.base import euler_zyx, sample_times_ns
from humanoid_sim.replay.exc import ImuTrace


@dataclass(frozen=True)
class AccelerationCfg:
    window_ns: int
    max_gap_ns: int = 50_000_000

    def __post_init__(self):
        if type(self.window_ns) is not int or self.window_ns < 0:
            raise ValueError("acceleration smoothing window must be nonnegative integer ns")
        if type(self.max_gap_ns) is not int or self.max_gap_ns <= 0:
            raise ValueError("acceleration maximum gap must be positive integer ns")

    def validate_mode(self, mode):
        if mode != "orientation":
            raise ValueError("IMU acceleration requires --base orientation")


def rotate(q, v):
    """Rotate vectors by unit xyzw quaternions; NumPy broadcasting."""
    xyz, w = q[..., :3], q[..., 3:]
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + w * v)


def _polynomial(t_ns, values, window_ns):
    """Quadratic local fits at supported interior samples, including derivatives.

    A zero window disables smoothing, using each sample's two neighbours only
    for a three-point derivative. Fit times are centred/scaled before least squares.
    """
    indices, smooth, deriv = [], [], []
    radius = window_ns // 2
    for i in range(1, len(t_ns) - 1):
        if window_ns:
            if t_ns[i] - radius < t_ns[0] or t_ns[i] + radius > t_ns[-1]:
                continue
            lo, hi = np.searchsorted(t_ns, [t_ns[i] - radius, t_ns[i] + radius], side="left")
            hi = int(np.searchsorted(t_ns, t_ns[i] + radius, side="right"))
        else:
            lo, hi = i - 1, i + 2
        if hi - lo < 3:
            raise ValueError("smoothing window has fewer than three IMU samples")
        dt = (t_ns[lo:hi] - t_ns[i]).astype(np.float64) * 1e-9
        scale = np.max(np.abs(dt))
        x = dt / scale
        coeff = np.linalg.lstsq(np.stack([np.ones_like(x), x, x * x], axis=1), values[lo:hi], rcond=None)[0]
        indices.append(i)
        smooth.append(coeff[0] if window_ns else values[i])
        deriv.append(coeff[1] / scale)
    if len(indices) < 2:
        raise ValueError("insufficient IMU coverage after derivative/filter edge trimming")
    return np.asarray(indices), np.asarray(smooth), np.asarray(deriv)


@dataclass(frozen=True)
class AccelerationSignal:
    t_ns: np.ndarray
    acceleration_w: np.ndarray
    raw_acceleration_w: np.ndarray
    rotational_acceleration_b: np.ndarray
    specific_force_b: np.ndarray
    metadata: dict

    def intervals(self, start_ns, end_ns):
        """Exact interval means of the piecewise-linear acceleration signal."""
        starts, ends = np.asarray(start_ns), np.asarray(end_ns)
        if (starts.dtype != np.int64 or ends.dtype != np.int64 or starts.shape != ends.shape
                or np.any(ends <= starts) or np.any(starts < self.t_ns[0]) or np.any(ends > self.t_ns[-1])):
            raise ValueError("acceleration intervals must be covered, increasing int64 ns pairs")
        t = (self.t_ns - self.t_ns[0]).astype(np.float64) * 1e-9
        y = self.acceleration_w
        h = np.diff(t)
        area = np.vstack([np.zeros(3), np.cumsum(0.5 * (y[1:] + y[:-1]) * h[:, None], axis=0)])

        def integral(stamps):
            x = (stamps - self.t_ns[0]).astype(np.float64) * 1e-9
            k = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 2)
            dt = x - t[k]
            slope = (y[k + 1] - y[k]) / h[k, None]
            return area[k] + y[k] * dt[:, None] + 0.5 * slope * dt[:, None] ** 2

        return (integral(ends) - integral(starts)) / ((ends - starts).astype(float) * 1e-9)[:, None]


def prepare_acceleration(trace: ImuTrace, desc, gravity_w, cfg: AccelerationCfg) -> AccelerationSignal:
    """Recover acceleration at base origin from a rigid-mounted, base-axis IMU.

    f_B = R_WB.T (a_sensor_W - g_W). Remove alpha×r + omega×(omega×r)
    before rotating. Constant initial yaw alignment matches BaseMotion.
    """
    n = len(trace.t_ns)
    arrays = (trace.t_ns, trace.seq, trace.orientation_xyzw, trace.angular_velocity, trace.specific_force)
    shapes = ((n,), (n,), (n, 4), (n, 3), (n, 3))
    if n < 5 or any(np.shape(a) != s or not np.isfinite(a).all() for a, s in zip(arrays, shapes, strict=True)):
        raise ValueError("finite IMU stamps, sequence, xyzw orientation, gyro and specific force required")
    if trace.t_ns.dtype != np.int64 or trace.seq.dtype != np.int64 or np.any(np.diff(trace.t_ns) < 0):
        raise ValueError("IMU stamps/sequence must be ordered integer ns/counts")
    norm = np.linalg.norm(trace.orientation_xyzw, axis=1)
    if np.any(np.abs(norm - 1) > 1e-3):
        raise ValueError("IMU orientation must contain unit quaternions")
    gravity = np.asarray(gravity_w, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("finite simulation world gravity [3] required")
    if np.any(gravity[:2] != 0) or gravity[2] >= 0:
        raise ValueError("initial-yaw-aligned IMU replay requires z-up world gravity")
    # imu_mount supplies the point relative to its parent. Refuse a moving parent.
    parent, offset, _ = desc.imu_mount()
    tree = ET.parse(desc.urdf_path).getroot()
    children = {j.find("child").get("link") for j in tree.findall("joint")}
    roots = set(desc.link_names()) - children
    if len(roots) != 1:
        raise ValueError("description must have one physical root")
    root = roots.pop()
    if parent != root:
        raise ValueError(f"IMU mount parent {parent!r} is not physical base {root!r}; origin shift unsupported")
    r = np.asarray(offset, dtype=float)
    stamps = sample_times_ns(trace)
    if np.any(np.diff(stamps) <= 0) or np.any(np.diff(stamps) > cfg.max_gap_ns):
        raise ValueError("IMU reconstructed times are nonmonotonic or exceed maximum gap")
    ids, omega, alpha = _polynomial(stamps, trace.angular_velocity, cfg.window_ns)
    rotational = np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
    q = trace.orientation_xyzw / norm[:, None]
    raw = rotate(q[ids], trace.specific_force[ids] - rotational) + gravity
    yaw0 = euler_zyx(q[:1])[0, 0]
    c, s = np.cos(yaw0), np.sin(yaw0)
    align = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    raw = raw @ align.T
    if cfg.window_ns:
        keep, acceleration, _ = _polynomial(stamps[ids], raw, cfg.window_ns)
        ids, raw, rotational = ids[keep], raw[keep], rotational[keep]
    else:
        acceleration = raw.copy()
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    meta = {
        "mode": "imu", "modeled": True, "settings": asdict(cfg),
        "input_frame": root, "input_origin": desc.imu_link(), "offset_b_m": list(offset),
        "gravity_w_m_s2": gravity.tolist(), "initial_yaw_rad": float(yaw0),
        "raw_span_ns": [int(stamps[0]), int(stamps[-1])],
        "valid_span_ns": [int(stamps[ids[0]]), int(stamps[ids[-1]])],
        "samples": n, "retained_samples": len(ids), "imu_sha256": digest.hexdigest(),
        "acceleration_rms_m_s2": np.sqrt(np.mean(acceleration ** 2, axis=0)).tolist(),
        "derivative": "centered quadratic local polynomial", "filter": "offline centered; no bias removal",
        **desc.identity(),
    }
    return AccelerationSignal(stamps[ids], acceleration, raw, rotational, trace.specific_force[ids], meta)


def source_identity(path):
    """Record the owning Git checkout; no credentials, remote calls or mutations."""
    path = Path(path).resolve()
    root = next((p for p in path.parents if (p / ".git").exists()), None)
    if root is None:
        return {"revision": None, "dirty": None, "available": False, "reason": "not a Git checkout"}
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        # A bind-mounted worktree may lack its external .git directory. Do not
        # invent an identity; byte hashes remain available in capture metadata.
        return {"revision": None, "dirty": None, "available": False, "reason": "Git metadata inaccessible"}
    return {"revision": revision, "dirty": bool(dirty), "available": True}


def prepare_capture_acceleration(capture, desc, gravity_w, cfg):
    """Prepare once with immutable input fingerprints and simulator provenance."""
    signal = prepare_acceleration(capture.imu(), desc, gravity_w, cfg)
    files = {}
    root = getattr(capture, "root", None)
    if root is not None:
        root = Path(root)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".yaml", ".yml", ".mcap", ".db3"):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[str(path.relative_to(root))] = digest.hexdigest()
    code = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in ("replay/inertial.py", "replay/replay.py", "sim.py", "physics/isaaclab.py"):
        digest.update(name.encode())
        digest.update((code / name).read_bytes())
    signal.metadata.update(capture_id=capture.dataset_id, capture_files_sha256=files,
                           sim_source=source_identity(__file__), mechanism_sha256=digest.hexdigest(),
                           producer_contract="Controls ethrc_ssot/imu.py rotated(), b6ff5e5a; base axes, IMU origin")
    return signal


def add_acceleration_args(parser):
    parser.add_argument("--base-acceleration", choices=("none", "imu"), default="none")
    parser.add_argument("--imu-window-ms", type=float, default=None,
                        help="explicit centered smoothing duration; 0 for ideal synthetic inputs")
    parser.add_argument("--imu-max-gap-ms", type=float, default=50.0)


def acceleration_config(args):
    if args.base_acceleration == "none":
        if args.imu_window_ms is not None:
            raise ValueError("--imu-window-ms requires --base-acceleration imu")
        return None
    if args.imu_window_ms is None or not np.isfinite(args.imu_window_ms):
        raise ValueError("IMU acceleration requires an explicit finite --imu-window-ms")
    if not np.isfinite(args.imu_max_gap_ms):
        raise ValueError("finite --imu-max-gap-ms required")
    cfg = AccelerationCfg(int(round(args.imu_window_ms * 1e6)), int(round(args.imu_max_gap_ms * 1e6)))
    cfg.validate_mode(args.base)
    return cfg
