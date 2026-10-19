"""Reader for Controls excitation captures (``ethrc-excitation/v8``, ``datasets/humanoid-v1/exc-*``).

Streams used for drive-level replay (``pdo_feedback.csv``, ~200 Hz per drive):

* ``position_demand_rad``: 0x6062, the drive's position demand after master slew/blend and drive
  interpolation, in the drive's native frame (cranks: crank angle). This is what the sim's drive bank
  is fed during replay.
* ``native_position_rad``, ``velocity_rad_s``, ``torque_nm``, ``current_a``, ``current_demand_a``: what the
  drive measured. The replay is scored against position, velocity and torque; the currents are read but unused.

``imu.csv`` is Controls' ``/rt/imu``: the IMU relay's output, already rotated into the ``base_link`` frame
(orientation from the Madgwick filter, angular velocity, specific force), stamped on host receipt.

Times are ``t_source_ros_ns`` (ROS epoch, integer ns). ``t_recv_runtime_ns`` is a different clock and is
never mixed in. ``gains.json`` holds the drive gains in force (raw JVPT counts and SI).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "ethrc-excitation/v8"
PDO_FIELDS = ("native_position_rad", "velocity_rad_s", "torque_nm", "current_a", "position_demand_rad",
              "current_demand_a", "motor_position_rad")


@dataclass
class DriveTrace:
    device: str
    t_ns: np.ndarray  # int64, t_source_ros_ns, strictly increasing
    fields: dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]


@dataclass
class DriveGains:
    stiffness: float  # Nm/rad
    damping: float  # Nm s/rad
    p_gain_raw: int
    d_gain_raw: int


@dataclass
class ImuTrace:
    t_ns: np.ndarray  # int64, t_source_ros_ns (host receipt), sorted
    seq: np.ndarray  # int64, the relay's sample sequence
    orientation_xyzw: np.ndarray  # [N, 4], base_link in the filter's gravity-aligned frame
    angular_velocity: np.ndarray  # [N, 3] rad/s, base_link frame
    specific_force: np.ndarray  # [N, 3] m/s², base_link frame


class ExcitationCapture:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{self.root}: format {self.manifest.get('format')!r} != {FORMAT}")
        if not self.manifest.get("complete", False):
            raise ValueError(f"{self.root}: capture is not complete")
        self._pdo: dict[str, DriveTrace] | None = None

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]

    @property
    def name(self) -> str:
        return self.manifest.get("display_name", self.dataset_id)

    def gains(self) -> dict[str, DriveGains]:
        g = json.loads((self.root / "gains.json").read_text())
        start, end = g["start"]["actuators"], g["end"]["actuators"]
        out = {}
        for dev, a in start.items():
            if (a["raw_p_gain"], a["raw_d_gain"]) != (end[dev]["raw_p_gain"], end[dev]["raw_d_gain"]):
                raise ValueError(f"{self.dataset_id}: gains of {dev} changed during the capture")
            out[dev] = DriveGains(float(a["stiffness_nm_per_rad"]), float(a["damping_nm_s_per_rad"]),
                                  int(a["raw_p_gain"]), int(a["raw_d_gain"]))
        return out

    def pdo(self) -> dict[str, DriveTrace]:
        """Per-drive PDO feedback traces, sorted by source time, duplicates dropped."""
        if self._pdo is None:
            rows: dict[str, list[list[str]]] = {}
            with (self.root / "pdo_feedback.csv").open(newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                col = {name: i for i, name in enumerate(header)}
                for r in reader:
                    rows.setdefault(r[col["device"]], []).append(r)
            traces = {}
            for dev, rs in rows.items():
                t = np.array([int(r[col["t_source_ros_ns"]]) for r in rs], dtype=np.int64)
                order = np.argsort(t, kind="stable")
                t = t[order]
                keep = np.concatenate([[True], np.diff(t) > 0])
                fields = {}
                for name in PDO_FIELDS:
                    v = np.array([float(r[col[name]]) if r[col[name]] != "" else np.nan for r in rs])
                    fields[name] = v[order][keep]
                traces[dev] = DriveTrace(dev, t[keep], fields)
            self._pdo = traces
        return self._pdo

    def imu(self) -> ImuTrace:
        with (self.root / "imu.csv").open(newline="") as f:
            reader = csv.reader(f)
            col = {name: i for i, name in enumerate(next(reader))}
            rows = list(reader)

        def num(*names, dtype=np.float64):
            return np.array([[r[col[n]] for n in names] for r in rows], dtype=dtype)

        t = num("t_source_ros_ns", dtype=np.int64)[:, 0]
        order = np.argsort(t, kind="stable")
        return ImuTrace(
            t[order], num("source_seq", dtype=np.int64)[order, 0],
            num("orientation_x", "orientation_y", "orientation_z", "orientation_w")[order],
            num("angular_velocity_x_rad_s", "angular_velocity_y_rad_s", "angular_velocity_z_rad_s")[order],
            num("specific_force_x_m_s2", "specific_force_y_m_s2", "specific_force_z_m_s2")[order])

    def time_span_ns(self) -> tuple[int, int]:
        """Interval covered by every drive's PDO trace."""
        traces = self.pdo().values()
        return max(int(tr.t_ns[0]) for tr in traces), min(int(tr.t_ns[-1]) for tr in traces)


def sample_hold(trace_t: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Zero-order hold: the latest sample at or before each ``t`` (first sample before the trace starts)."""
    idx = np.searchsorted(trace_t, t, side="right") - 1
    return values[np.clip(idx, 0, len(values) - 1)]


def sample_linear(trace_t: np.ndarray, values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation in integer-ns time (computed relative to the trace start for precision)."""
    t0 = trace_t[0]
    return np.interp((t - t0).astype(np.float64), (trace_t - t0).astype(np.float64), values)
