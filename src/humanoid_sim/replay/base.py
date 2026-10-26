"""Base motion of a capture, from its IMU, for replays on base joints (robot hanging on a gantry).

``imu.csv`` is already in the ``base_link`` frame. Its stamps are host-receipt times: the IMU delivers in
bursts (5-6 samples 13 µs apart, about every 15 ms), so a stamp lags its measurement by up to one burst
period. Sample times are rebuilt per burst: a burst holds the samples measured since the previous burst, so
they are spread evenly over that interval, the last one at its own stamp (the least delayed one).

Orientation: base yaw, pitch and roll (``R = Rz Ry Rx``) in the filter's gravity-aligned frame. The filter
has no heading reference, so the yaw at the first sample is removed. Rates are the Euler-angle rates of the
measured angular velocity. The filter's tilt leans with linear acceleration, so ``yaw`` mode keeps only the
yaw (base upright). On the excitation captures the gantry swings mostly in yaw (tens of degrees), and yaw
alone gave the best replay in the original comparison. Default replay ignores linear acceleration;
opt-in equivalent loading is prepared in replay/inertial.py with explicit filtering because specific force
also carries vibration the base does not follow rigidly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humanoid_sim.replay.exc import ImuTrace, sample_linear

BURST_GAP_NS = 1_000_000  # samples closer than this arrived in one burst


def sample_times_ns(trace: ImuTrace) -> np.ndarray:
    """Measurement time of every sample: each burst spread evenly back to the previous burst's last stamp."""
    if np.any(np.diff(trace.seq) != 1):
        raise ValueError("IMU sequence has gaps; sample times cannot be rebuilt from it")
    t = trace.t_ns
    last = np.flatnonzero(np.append(np.diff(t) > BURST_GAP_NS, True))
    out = t.copy()  # samples of the first burst keep their stamps
    for prev, end in zip(last[:-1], last[1:]):
        n = end - prev
        out[prev + 1:end + 1] = t[prev] + np.round((t[end] - t[prev]) * np.arange(1, n + 1) / n).astype(np.int64)
    return out


def euler_zyx(q_xyzw: np.ndarray) -> np.ndarray:
    """Yaw, pitch, roll ``[N, 3]`` of unit quaternions ``[N, 4]`` (``R = Rz(yaw) Ry(pitch) Rx(roll)``)."""
    x, y, z, w = q_xyzw.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.stack([yaw, pitch, roll], axis=1)


def euler_zyx_rates(angles: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Yaw, pitch, roll rates from the body-frame angular velocity."""
    _, pitch, roll = angles.T
    wx, wy, wz = omega_body.T
    sr, cr, cp, tp = np.sin(roll), np.cos(roll), np.cos(pitch), np.tan(pitch)
    return np.stack([(sr * wy + cr * wz) / cp, cr * wy - sr * wz, wx + sr * tp * wy + cr * tp * wz], axis=1)


BASE_MODES = ("yaw", "orientation")


@dataclass
class BaseMotion:
    angles: np.ndarray  # [T, 3] yaw, pitch, roll (rad)
    rates: np.ndarray  # [T, 3] rad/s

    @classmethod
    def from_imu(cls, trace: ImuTrace, t_ns: np.ndarray, mode: str = "yaw") -> BaseMotion:
        """Base orientation and rates at times ``t_ns``."""
        if mode not in BASE_MODES:
            raise ValueError(f"base mode {mode!r}; expected one of {BASE_MODES}")
        ts = sample_times_ns(trace)
        angles = euler_zyx(trace.orientation_xyzw)
        angles[:, 0] = np.unwrap(angles[:, 0])
        angles[:, 0] -= angles[0, 0]
        rates = euler_zyx_rates(angles, trace.angular_velocity)

        def at(x):
            return np.stack([sample_linear(ts, x[:, k], t_ns) for k in range(x.shape[1])], axis=1)

        out = cls(at(angles), at(rates))
        if mode == "yaw":
            out.angles[:, 1:] = 0.0
            out.rates[:, 1:] = 0.0
        return out
