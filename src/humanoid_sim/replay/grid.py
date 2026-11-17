"""Signals of a recording on one uniform time grid, by their own source stamps.

The robot's streams arrive on different clocks and rates: drive feedback every 5 ms with each drive's source stamp,
the IMU in bursts stamped on receipt, the policy's targets as frames at 50 Hz. Comparing them (or the sim with them)
only means something when each signal is read at the instant it describes:

* a measurement (drive position or torque, gyro) is the linear interpolation between its own samples at the grid time;
  outside its recorded span it is NaN, never extended;
* a target is held from its frame's stamp, the instant the loop host sent it, until the next frame: the target in
  force at the grid time. It is not moved by any command latency: the latency belongs to the plant that answers it
  (5.3 ms on the robot, applied in the sim), and moving the target would hide it from the comparison. After the last
  frame it holds for ``HOLD_NS`` and is then NaN.

Pairing a policy tick's new target with the gyro read a few ms before it, or the sim's step-end state with the
step-start target, skews phase by tens of degrees at 3 to 12 Hz (about 20 to 27 degrees at 4.5 Hz for 7 ms); the grid
removes that. 200 Hz is the drives' feedback rate.
"""

from __future__ import annotations

import numpy as np

GRID_HZ = 200
GRID_NS = 1_000_000_000 // GRID_HZ
HOLD_NS = 40_000_000  # a target stays in force this long after the last frame (two policy periods)


def make_grid(start_ns: int, end_ns: int, step_ns: int = GRID_NS) -> np.ndarray:
    """Grid times in ``[start, end)``, ``step_ns`` apart, int64 ns on the source clock."""
    if end_ns <= start_ns:
        raise ValueError("grid end must follow its start")
    return np.arange(int(start_ns), int(end_ns), int(step_ns), dtype=np.int64)


def _sorted_unique(t_ns: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_ns = np.asarray(t_ns, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] != t_ns.shape[0]:
        raise ValueError("a signal needs one row per stamp")
    order = np.argsort(t_ns, kind="stable")
    t_ns, x = t_ns[order], x[order]
    keep = np.concatenate([[True], np.diff(t_ns) > 0])  # a repeated stamp keeps its first sample
    return t_ns[keep], x[keep]


def interpolate(t_ns: np.ndarray, x: np.ndarray, grid_ns: np.ndarray) -> np.ndarray:
    """``x`` ``[N]`` or ``[N, K]`` at ``t_ns`` linearly interpolated to the grid; NaN outside ``[t[0], t[-1]]``."""
    t_ns, x = _sorted_unique(t_ns, x)
    flat = x.reshape(len(t_ns), -1)
    rel = (grid_ns - t_ns[0]).astype(np.float64)  # offsets from the first stamp keep float64 exact at ns
    tr = (t_ns - t_ns[0]).astype(np.float64)
    out = np.stack([np.interp(rel, tr, flat[:, k], left=np.nan, right=np.nan) for k in range(flat.shape[1])], axis=1)
    return out.reshape((len(grid_ns), *x.shape[1:]))


def hold(t_ns: np.ndarray, x: np.ndarray, grid_ns: np.ndarray, *, max_age_ns: int = HOLD_NS) -> np.ndarray:
    """The frame in force at each grid time: the last of ``x`` ``[F]`` or ``[F, K]`` stamped at or before it. NaN
    before the first frame and more than ``max_age_ns`` after the last."""
    t_ns, x = _sorted_unique(t_ns, x)
    i = np.searchsorted(t_ns, grid_ns, side="right") - 1
    out = x[np.clip(i, 0, None)].copy()
    out[(i < 0) | (grid_ns > t_ns[-1] + max_age_ns)] = np.nan
    return out
