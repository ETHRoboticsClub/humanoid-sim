"""Sim-versus-robot metrics over 2 s windows, from signals on the 200 Hz grid (:mod:`humanoid_sim.replay.grid`).

Every metric is computed per window (2 s, every 1 s, the windows of the robot logs) and reported as the mean, std,
min and max over the windows, robot and sim on the same windows. Metric ids are dotted:

========================================  ==========================================================================
``tracking_rms.<drive>``                  RMS of the drive's position minus the target in force, mean removed, rad
                                          (robot and sim)
``tracking_offset.<drive>``               the mean of that error, rad (robot and sim; information, no band)
``sim_real_rms.<drive>``                  RMS of sim minus robot position, mean removed, rad (sim only)
``hip_q_over_target.<band>.gain``         symmetric hip flexion position over its target: gain and phase, per band
``hip_tau_over_target.<band>.gain``       symmetric hip flexion torque over its target (N m per rad)
``pitch_rate_over_hip_target.<band>``     torso pitch rate (IMU gyro y) over the symmetric hip flexion target
``pitch_rate_over_hip_tau.<band>``        torso pitch rate over the symmetric hip flexion torque
``imu.pitch_sway_deg``, ``imu.roll_sway_deg``  std of the IMU pitch / roll angle in the window
``imu.pitch_rate_rms.<band>``             RMS of the pitch rate in the band, about its mean, rad/s
========================================  ==========================================================================

The tracking error is dynamic: its mean over the window is removed and reported apart as the offset. With the base
held the sim's legs do not carry the robot's weight, so the static deflection a loaded drive shows (tens of mrad at the
hips and ankles) is not the plant's to reproduce, while the lag and gain of the drive following its target are.

Each response ``….<band>.gain`` has a ``….<band>.phase_deg`` beside it. The symmetric hip flexion is
``(left - right) / 2`` of the hip flexion joints (the left and right joints are mirrored, so the difference is the
common flexion); phase is the angle of output over input, negative when the output lags. A response is estimated
per window from Hann-windowed spectra pooled over the band's bins: ``H = sum(conj(U) Y) / sum(|U|^2)``; a window whose
input has less than ``MIN_INPUT`` RMS in the band, or whose coherence ``|sum(conj(U) Y)|^2 / (sum|U|^2 sum|Y|^2)`` is
below ``MIN_COHERENCE``, is left out: there is nothing to estimate from. Phase statistics are
circular. The IMU and pitch-rate families need the sim's own torso motion: with the base prescribed from the IMU they
have no sim value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from humanoid_sim.replay.grid import GRID_HZ

WINDOW_S = 2.0
STEP_S = 1.0
RESPONSE_BANDS_HZ = ((1.0, 3.0), (3.0, 6.0), (6.0, 12.0))
RATE_BANDS_HZ = ((0.5, 3.0), (3.0, 6.0), (6.0, 12.0))
# Input RMS in the band below which a window has no response to estimate from (measurement floor).
MIN_INPUT = {"target": 2e-4, "tau": 0.05}  # rad, N m
# Coherence below which a window's response is not estimated (the output is mostly something else). One window pools
# only a few bins (4 to 12), so its coherence is biased up by about one over that count: 0.5 is a loose gate.
MIN_COHERENCE = 0.5
NEEDS_FREE_BASE = ("pitch_rate_over_hip_target", "pitch_rate_over_hip_tau", "imu")


def band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}Hz"


def is_phase(metric: str) -> bool:
    return metric.endswith(".phase_deg")


def family(metric: str) -> str:
    return metric.split(".", 1)[0]


def needs_free_base(metric: str) -> bool:
    return family(metric) in NEEDS_FREE_BASE


def wrap_deg(x):
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


@dataclass
class HipPair:
    """Where the hip flexion drives are in the drive axis, and each one's joint sign (``q_joint = sign * q_drive``)."""

    left: int
    right: int
    sign_left: float
    sign_right: float

    def symmetric(self, x: np.ndarray) -> np.ndarray:
        """Symmetric flexion ``(left - right) / 2`` in the joint frame of a drive-frame array ``[T, D]``."""
        return 0.5 * (self.sign_left * x[:, self.left] - self.sign_right * x[:, self.right])


@dataclass
class Signals:
    """One side's signals on the grid. ``target`` is the same for robot and sim: the frame in force."""

    t_ns: np.ndarray  # [T] grid
    drives: list[str]
    target: np.ndarray  # [T, D] drive frame, rad
    position: np.ndarray  # [T, D] drive frame, rad
    torque: np.ndarray | None = None  # [T, D] N m
    gyro: np.ndarray | None = None  # [T, 3] rad/s, base frame
    angles: np.ndarray | None = None  # [T, 3] yaw, pitch, roll, rad


@dataclass
class Stat:
    mean: float
    std: float
    min: float
    max: float
    n: int

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "min": self.min, "max": self.max, "n": self.n}


def stat(values: np.ndarray, *, phase: bool = False) -> Stat | None:
    """Mean, std, min and max of the finite values; None without any. ``phase``: degrees, circular: the mean
    direction, and the std, min and max of the wrapped offsets from it (min and max as angles)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    if not phase:
        return Stat(float(v.mean()), float(v.std()), float(v.min()), float(v.max()), len(v))
    mean = float(np.degrees(np.angle(np.exp(1j * np.radians(v)).mean())))
    off = wrap_deg(v - mean)
    return Stat(mean, float(np.sqrt((off**2).mean())), float(wrap_deg(mean + off.min())),
                float(wrap_deg(mean + off.max())), len(v))


def window_slices(n: int, *, fs: float = GRID_HZ, window_s: float = WINDOW_S, step_s: float = STEP_S) -> list[slice]:
    size, step = round(window_s * fs), round(step_s * fs)
    return [slice(i, i + size) for i in range(0, n - size + 1, step)]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def band_rms(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """RMS of the part of ``x`` in ``[lo, hi)`` Hz, its mean removed, no taper."""
    spec = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1 / fs)
    spec[(f < lo) | (f >= hi)] = 0
    return rms(np.fft.irfft(spec, n=len(x)))


def response(u: np.ndarray, y: np.ndarray, fs: float, lo: float, hi: float, min_input: float) -> tuple[float, float]:
    """Gain and phase (deg) of ``y`` over ``u`` in ``[lo, hi)`` Hz from one window; NaN, NaN when the input has less
    than ``min_input`` RMS there or the coherence is below ``MIN_COHERENCE``."""
    if band_rms(u, fs, lo, hi) < min_input:
        return np.nan, np.nan
    w = np.hanning(len(u))
    spec_u = np.fft.rfft((u - u.mean()) * w)
    spec_y = np.fft.rfft((y - y.mean()) * w)
    f = np.fft.rfftfreq(len(u), 1 / fs)
    s = (f >= lo) & (f < hi)
    suu = (np.abs(spec_u[s]) ** 2).sum()
    syy = (np.abs(spec_y[s]) ** 2).sum()
    suy = (np.conj(spec_u[s]) * spec_y[s]).sum()
    if syy == 0 or abs(suy) ** 2 / (suu * syy) < MIN_COHERENCE:
        return np.nan, np.nan
    return float(abs(suy) / suu), float(np.degrees(np.angle(suy)))


@dataclass
class WindowValues:
    """Per-window values of every metric for one side; NaN where a window can't give one."""

    values: dict[str, np.ndarray] = field(default_factory=dict)
    windows: int = 0


def side_values(s: Signals, hip: HipPair, *, fs: float = GRID_HZ, only: set[str] | None = None) -> WindowValues:
    """Every metric one side's signals give, per window. ``only``: families to compute (default all it can)."""
    sl = window_slices(len(s.t_ns), fs=fs)
    out = WindowValues(windows=len(sl))

    def want(fam: str) -> bool:
        return only is None or fam in only

    def put(metric: str, k: int, value: float) -> None:
        out.values.setdefault(metric, np.full(len(sl), np.nan))[k] = value

    sym_target = hip.symmetric(s.target)
    sym_q = hip.symmetric(s.position)
    sym_tau = hip.symmetric(s.torque) if s.torque is not None else None
    gyro_y = s.gyro[:, 1] if s.gyro is not None else None
    pitch_deg = np.degrees(s.angles[:, 1]) if s.angles is not None else None
    roll_deg = np.degrees(s.angles[:, 2]) if s.angles is not None else None
    for k, w in enumerate(sl):
        if want("tracking_rms"):
            err = s.position[w] - s.target[w]
            ok = np.isfinite(err).all(axis=0)
            for d, name in enumerate(s.drives):
                if ok[d]:
                    put(f"tracking_rms.{name}", k, rms(err[:, d] - err[:, d].mean()))
                    put(f"tracking_offset.{name}", k, float(err[:, d].mean()))
        pairs = [("hip_q_over_target", sym_q, "target"), ("hip_tau_over_target", sym_tau, "target"),
                 ("pitch_rate_over_hip_target", gyro_y, "target"), ("pitch_rate_over_hip_tau", gyro_y, "tau")]
        for fam, y_all, kind in pairs:
            if not want(fam) or y_all is None:
                continue
            u_all = sym_tau if kind == "tau" else sym_target
            if u_all is None:
                continue
            u, y = u_all[w], y_all[w]
            if not (np.isfinite(u).all() and np.isfinite(y).all()):
                continue
            for lo, hi in RESPONSE_BANDS_HZ:
                gain, phase = response(u, y, fs, lo, hi, MIN_INPUT[kind])
                put(f"{fam}.{band_label(lo, hi)}.gain", k, gain)
                put(f"{fam}.{band_label(lo, hi)}.phase_deg", k, phase)
        if want("imu") and gyro_y is not None and pitch_deg is not None:
            gy, pd, rd = gyro_y[w], pitch_deg[w], roll_deg[w]
            if np.isfinite(gy).all() and np.isfinite(pd).all() and np.isfinite(rd).all():
                put("imu.pitch_sway_deg", k, float(pd.std()))
                put("imu.roll_sway_deg", k, float(rd.std()))
                for lo, hi in RATE_BANDS_HZ:
                    put(f"imu.pitch_rate_rms.{band_label(lo, hi)}", k, band_rms(gy, fs, lo, hi))
    return out


@dataclass
class MetricResult:
    metric: str
    robot: Stat | None
    sim: Stat | None  # None: no sim value (needs the sim's own torso motion, or no common windows)
    windows: int  # windows both sides had a value on (the robot's, for robot-only metrics)

    def as_dict(self) -> dict:
        return {"robot": None if self.robot is None else self.robot.as_dict(),
                "sim": None if self.sim is None else self.sim.as_dict(), "windows": self.windows}


def compare(robot: Signals, sim: Signals | None, hip: HipPair) -> dict[str, MetricResult]:
    """Every metric, robot and sim on the windows both have a value for. Without ``sim`` the robot's alone."""
    rv = side_values(robot, hip)
    sv = side_values(sim, hip) if sim is not None else WindowValues()
    out: dict[str, MetricResult] = {}
    for metric, r in rv.values.items():
        s = sv.values.get(metric)
        phase = is_phase(metric)
        if s is None:
            out[metric] = MetricResult(metric, stat(r, phase=phase), None, int(np.isfinite(r).sum()))
            continue
        common = np.isfinite(r) & np.isfinite(s)
        out[metric] = MetricResult(metric, stat(r[common], phase=phase), stat(s[common], phase=phase),
                                   int(common.sum()))
    if sim is not None:  # sim minus robot position, on the windows both are defined on
        sl = window_slices(len(robot.t_ns))
        for d, name in enumerate(robot.drives):
            vals = np.full(len(sl), np.nan)
            for k, w in enumerate(sl):
                err = sim.position[w, d] - robot.position[w, d]
                if np.isfinite(err).all():
                    vals[k] = rms(err - err.mean())
            out[f"sim_real_rms.{name}"] = MetricResult(f"sim_real_rms.{name}", None, stat(vals),
                                                       int(np.isfinite(vals).sum()))
    return out
