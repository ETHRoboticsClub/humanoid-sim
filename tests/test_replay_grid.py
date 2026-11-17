"""Recording signals on the 200 Hz grid by source stamps: measurements interpolated, targets held from their frames."""

import numpy as np
import pytest

from humanoid_sim.replay.grid import GRID_HZ, GRID_NS, HOLD_NS, hold, interpolate, make_grid
from humanoid_sim.replay.metrics import response

T0 = 1_794_000_000_000_000_000


def test_grid_is_uniform_int64_and_half_open():
    g = make_grid(T0, T0 + 1_000_000_000)
    assert g.dtype == np.int64 and len(g) == GRID_HZ and np.all(np.diff(g) == GRID_NS) and g[0] == T0
    with pytest.raises(ValueError):
        make_grid(T0, T0)


def test_interpolate_is_linear_on_unsorted_duplicated_stamps_and_nan_outside():
    t = np.array([T0 + 20_000_000, T0, T0 + 10_000_000, T0 + 10_000_000], dtype=np.int64)
    x = np.array([2.0, 0.0, 1.0, 99.0])  # the repeated stamp keeps its first sample
    grid = np.array([T0 - 1, T0, T0 + 5_000_000, T0 + 15_000_000, T0 + 20_000_000, T0 + 20_000_001], dtype=np.int64)
    out = interpolate(t, x, grid)
    assert np.isnan(out[[0, 5]]).all()
    assert out[1:5] == pytest.approx([0.0, 0.5, 1.5, 2.0])


def test_interpolate_columns_and_large_stamps_stay_exact():
    t = T0 + np.arange(0, 100, 5, dtype=np.int64) * 1_000_000
    x = np.stack([np.arange(len(t), dtype=float), -np.arange(len(t), dtype=float)], axis=1)
    out = interpolate(t, x, t[:-1] + 2_500_000)  # midway between samples
    assert out.shape == (len(t) - 1, 2)
    assert out[:, 0] == pytest.approx(np.arange(len(t) - 1) + 0.5, abs=1e-9)


def test_hold_keeps_the_frame_in_force_and_expires_after_the_last():
    t = np.array([T0, T0 + 20_000_000, T0 + 40_000_000], dtype=np.int64)
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    grid = T0 + np.array([-1, 0, 19_999_999, 20_000_000, 40_000_000, 40_000_000 + HOLD_NS,
                          40_000_000 + HOLD_NS + 1], dtype=np.int64)
    out = hold(t, x, grid)
    assert np.isnan(out[0]).all() and np.isnan(out[-1]).all()
    assert out[1:6, 0].tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_grid_phase_is_the_command_path_less_the_holds_half_frame():
    """A 4 Hz target sent in 50 Hz frames, answered 5.3 ms later (the command path). The held target is the frame in
    force, which lags the signal it samples by half a frame (10 ms) on average, so on the grid the response over it
    reads delay minus 10 ms: a few degrees of lead at 4 Hz. Robot and sim read their targets the same way, so the
    comparison between them is exact; pairing each new target with a sample taken 7 ms before it would instead shift
    the phase by about 10 degrees at 4 Hz, and 20 to 27 degrees at the frequencies and delays the pitch fit found."""
    f, delay_s = 4.0, 0.0053
    frames = T0 + np.arange(0, 10_000_000_000, 20_000_000, dtype=np.int64)
    u_frames = np.sin(2 * np.pi * f * (frames - T0) / 1e9)
    meas = T0 + np.arange(0, 10_000_000_000, 5_000_000, dtype=np.int64)  # the drive's own stamps
    y_meas = np.sin(2 * np.pi * f * ((meas - T0) / 1e9 - delay_s))
    grid = make_grid(T0 + 1_000_000_000, T0 + 9_000_000_000)
    u, y = hold(frames, u_frames, grid), interpolate(meas, y_meas, grid)
    gain, phase = response(u[:400], y[:400], GRID_HZ, 3.0, 6.0, 1e-3)
    assert gain == pytest.approx(1.0, abs=0.1)
    assert phase == pytest.approx(-360 * f * (delay_s - 0.010), abs=5.0)
