"""The sim-versus-robot metrics on synthetic signals with known answers."""

import numpy as np
import pytest

from humanoid_sim.replay.grid import GRID_HZ
from humanoid_sim.replay.metrics import (
    HipPair,
    Signals,
    band_rms,
    compare,
    needs_free_base,
    response,
    side_values,
    stat,
    window_slices,
    wrap_deg,
)

FS = GRID_HZ
DRIVES = ["hip_l", "hip_r", "knee"]
HIP = HipPair(left=0, right=1, sign_left=-1.0, sign_right=-1.0)


def _t(seconds):
    return np.arange(0, int(seconds * FS)) / FS


def _signals(seconds=6.0, *, delay_s=0.0, gain=1.0, offset=0.0, imu=True):
    t = _t(seconds)
    sym = 0.02 * np.sin(2 * np.pi * 4.0 * t)  # symmetric hip flexion target, 4 Hz, joint frame
    target = np.zeros((len(t), 3))
    target[:, 0], target[:, 1] = -sym, sym  # drive frame: sign -1; right is mirrored, so left minus right is 2 sym
    target[:, 2] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    position = np.stack([np.interp(t - delay_s, t, target[:, k]) * gain for k in range(3)], axis=1) + offset
    torque = 100.0 * (target - position)
    gyro = angles = None
    if imu:
        gyro = np.zeros((len(t), 3))
        gyro[:, 1] = 0.2 * np.sin(2 * np.pi * 1.5 * t)
        angles = np.zeros((len(t), 3))
        angles[:, 1] = np.radians(1.0) * np.sin(2 * np.pi * 1.5 * t)
        angles[:, 2] = np.radians(0.2) * np.sin(2 * np.pi * 1.5 * t)
    return Signals((t * 1e9).astype(np.int64), DRIVES, target, position, torque, gyro, angles)


def test_windows_are_two_seconds_every_one_second():
    w = window_slices(6 * FS)
    assert [(s.start, s.stop) for s in w] == [(i * FS, i * FS + 2 * FS) for i in range(5)]
    assert window_slices(FS) == []


def test_circular_stat_handles_the_wrap_and_none_without_values():
    s = stat(np.array([175.0, -175.0, 179.0]), phase=True)
    assert abs(wrap_deg(s.mean - 180.0)) < 3 and s.std < 6 and s.n == 3
    assert stat(np.array([np.nan])) is None
    plain = stat(np.array([1.0, 2.0, 3.0, np.nan]))
    assert (plain.mean, plain.min, plain.max, plain.n) == (2.0, 1.0, 3.0, 3)


def test_band_rms_is_the_rms_of_the_band_only():
    t = _t(2.0)
    x = 0.3 * np.sin(2 * np.pi * 4.0 * t) + 5.0 + 1.0 * np.sin(2 * np.pi * 9.0 * t)
    assert band_rms(x, FS, 3.0, 6.0) == pytest.approx(0.3 / np.sqrt(2), rel=1e-6)
    assert band_rms(x, FS, 6.0, 12.0) == pytest.approx(1.0 / np.sqrt(2), rel=1e-6)


def test_response_reads_gain_and_phase_and_refuses_noise_and_tiny_inputs():
    t = _t(2.0)
    u = np.sin(2 * np.pi * 4.0 * t)
    y = 2.0 * np.sin(2 * np.pi * 4.0 * t - np.radians(60))
    gain, phase = response(u, y, FS, 3.0, 6.0, 1e-3)
    assert gain == pytest.approx(2.0, rel=0.02) and phase == pytest.approx(-60.0, abs=1.0)
    rng = np.random.default_rng(0)
    assert np.isnan(response(u, rng.standard_normal(len(t)), FS, 3.0, 6.0, 1e-3)[0])  # incoherent output
    assert np.isnan(response(1e-6 * u, y, FS, 3.0, 6.0, 1e-3)[0])  # no input to estimate from


def test_tracking_error_is_dynamic_and_the_offset_reported_apart():
    s = _signals(offset=0.05)
    v = side_values(s, HIP).values
    assert v["tracking_offset.knee"] == pytest.approx(0.05, abs=1e-9)
    assert v["tracking_rms.knee"] == pytest.approx(0.0, abs=1e-9)  # follows its target exactly, bar the offset
    assert len(v["tracking_rms.hip_l"]) == 5


def test_hip_response_of_a_lagging_sim_reads_the_lag_and_the_imu_metrics():
    robot = _signals(delay_s=0.010, gain=0.9)
    sim = _signals(delay_s=0.020, gain=0.9, imu=False)
    out = compare(robot, sim, HIP)
    q = "hip_q_over_target.3-6Hz"
    # symmetric flexion is (left - right) / 2 in joint frame: the target's own phase is 0, so the phase is the lag
    assert out[f"{q}.gain"].robot.mean == pytest.approx(0.9, rel=0.03)
    assert out[f"{q}.phase_deg"].robot.mean == pytest.approx(-360 * 4.0 * 0.010, abs=2.0)
    assert out[f"{q}.phase_deg"].sim.mean == pytest.approx(-360 * 4.0 * 0.020, abs=2.0)
    assert out[f"{q}.gain"].windows == 5
    # robot-only families: the sim has no torso motion of its own
    assert out["imu.pitch_sway_deg"].robot.mean == pytest.approx(1.0 / np.sqrt(2), rel=0.02)
    assert out["imu.pitch_sway_deg"].sim is None and needs_free_base("imu.pitch_sway_deg")
    assert out["imu.pitch_rate_rms.0.5-3Hz"].robot.mean == pytest.approx(0.2 / np.sqrt(2), rel=0.02)
    assert needs_free_base("pitch_rate_over_hip_tau.3-6Hz.gain") and not needs_free_base("tracking_rms.knee")
    # sim minus robot position, mean removed: a 10 ms later copy of a 1 Hz 0.1 rad sine on the knee
    expected = 0.9 * 0.1 / np.sqrt(2) * 2 * np.sin(np.pi * 1.0 * 0.010)
    assert out["sim_real_rms.knee"].sim.mean == pytest.approx(expected, rel=0.05)


def test_windows_with_missing_samples_are_left_out():
    s = _signals()
    s.position[3 * FS:3 * FS + 10, 2] = np.nan
    n = np.isfinite(side_values(s, HIP).values["tracking_rms.knee"]).sum()
    assert n == 5 - 2  # the two windows that hold the gap
