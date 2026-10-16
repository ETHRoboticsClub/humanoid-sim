"""Base motion from an IMU trace: Euler angles and rates, burst sample times."""

import numpy as np
import pytest

from humanoid_sim.replay.base import BaseMotion, euler_zyx, euler_zyx_rates, sample_times_ns
from humanoid_sim.replay.exc import ImuTrace


def _quat_zyx(yaw, pitch, roll):
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1)).as_quat()


def test_euler_zyx_round_trip_and_rates():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    a = rng.uniform([-3, -1.2, -3], [3, 1.2, 3], size=(200, 3))
    assert np.allclose(euler_zyx(_quat_zyx(*a.T)), a, atol=1e-9)
    # Rates: finite-difference the angles of a known body-rate motion.
    from scipy.spatial.transform import Rotation

    omega = np.array([0.3, -0.2, 0.9])
    dt = 1e-5
    r0 = Rotation.from_euler("ZYX", [0.4, 0.2, -0.3])
    r1 = r0 * Rotation.from_rotvec(omega * dt)  # body-frame rotation over dt
    fd = (r1.as_euler("ZYX") - r0.as_euler("ZYX")) / dt
    assert np.allclose(euler_zyx_rates(r0.as_euler("ZYX")[None], omega[None])[0], fd, atol=1e-4)


def _trace(stamps_ms, yaw):
    n = len(stamps_ms)
    q = _quat_zyx(np.asarray(yaw), np.zeros(n), np.zeros(n))
    return ImuTrace(np.asarray(stamps_ms, dtype=np.int64) * 1_000_000 + 0, np.arange(1, n + 1), q,
                    np.zeros((n, 3)), np.tile([0.0, 0.0, 9.81], (n, 1)))


def test_burst_samples_spread_back_to_previous_burst():
    pytest.importorskip("scipy")
    # Bursts at 0 (1 sample), 15 ms (3 samples, 13 µs apart): spread over (0, 15] ms.
    tr = _trace([0, 15, 15, 15], [0, 0, 0, 0])
    tr.t_ns[2:] += np.array([13_000, 26_000])
    ts = sample_times_ns(tr)
    assert ts.tolist() == [0, 5_008_667, 10_017_333, 15_026_000]
    tr.seq[2] = 10
    with pytest.raises(ValueError, match="gaps"):
        sample_times_ns(tr)


def test_yaw_mode_keeps_yaw_relative_to_start():
    pytest.importorskip("scipy")
    stamps = np.arange(0, 100, 2.5)
    yaw = 1.0 + 0.5 * np.sin(stamps / 20.0)
    tr = _trace(np.round(stamps).astype(int), yaw)
    bm = BaseMotion.from_imu(tr, np.array([0, 50_000_000], dtype=np.int64))
    assert bm.angles[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert bm.angles[1, 0] == pytest.approx(0.5 * np.sin(50 / 20.0), abs=0.02)
    assert not bm.angles[:, 1:].any() and not bm.rates[:, 1:].any()
    with pytest.raises(ValueError, match="base mode"):
        BaseMotion.from_imu(tr, np.array([0]), mode="roll")
