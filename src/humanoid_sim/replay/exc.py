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
never mixed in. ``gains.json`` holds the drive gains in force, in SI (the replay uses these) and as the drives' raw JVPT counts.

A capture the run left incomplete is refused unless opened with ``allow_incomplete``; :meth:`ExcitationCapture.sequences`
then cuts it into the sweeps it finished (and optionally each at a top frequency), which a fit replays as separate
captures. The sweep timing is the manifest's ``signals`` (Controls' ``ChirpSpec``): sweep ``k`` of a joint runs
over signal time ``start_offset + k (duration + gap)`` for ``duration``, at ``f(t) = f0 (f1/f0)^(t/duration)``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
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
class CommandTrace:
    """Joint-space command frames as they left the loop host: semantic radians (relative to the standing pose),
    one row per frame, each joint held at its last commanded value (NaN until first commanded)."""

    t_ns: np.ndarray  # int64, the frame stamps, strictly increasing
    joints: list[str]
    positions_rad: np.ndarray  # [F, J]

    @classmethod
    def from_frames(cls, frames: list[tuple[int, dict[str, float]]]) -> CommandTrace:
        """From ``(stamp, {joint: rad})`` frames in any order; a repeated stamp keeps the later frame."""
        frames = sorted(frames, key=lambda f: f[0])
        joints = sorted({j for _, f in frames for j in f})
        held = dict.fromkeys(joints, np.nan)
        t, rows = [], []
        for stamp, f in frames:
            held |= f
            if t and stamp == t[-1]:
                rows[-1] = [held[j] for j in joints]
                continue
            t.append(stamp)
            rows.append([held[j] for j in joints])
        return cls(np.asarray(t, dtype=np.int64), joints, np.asarray(rows, dtype=np.float64).reshape(len(t), -1))


@dataclass
class ImuTrace:
    t_ns: np.ndarray  # int64, t_source_ros_ns (host receipt), sorted
    seq: np.ndarray  # int64, the relay's sample sequence
    orientation_xyzw: np.ndarray  # [N, 4], base_link in the filter's gravity-aligned frame
    angular_velocity: np.ndarray  # [N, 3] rad/s, base_link frame
    specific_force: np.ndarray  # [N, 3] m/s², base_link frame


class ExcitationCapture:
    def __init__(self, root: Path, *, allow_incomplete: bool = False):
        """``allow_incomplete``: open a capture the run did not finish; only :meth:`sequences` makes it fit data."""
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{self.root}: format {self.manifest.get('format')!r} != {FORMAT}")
        self.complete = bool(self.manifest.get("complete", False))
        if not self.complete and not allow_incomplete:
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
            if a["stiffness_nm_per_rad"] is None:  # no position gains, e.g. the hands' actuators
                continue
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

    def commands(self) -> CommandTrace:
        """``commands.csv``: the frames the robot API published, ``q_nominal_rad`` at ``t_frame_ros_ns``."""
        return self._frames(None)

    def applied_commands(self) -> CommandTrace | None:
        """The command frames stamped when the controller applied them (``acks.csv``: receiver ``controller``,
        state ``applied``, ``t_ack_ros_ns``), so a replay of them leaves out the software path before the
        controller. Frames the controller never applied are left out. None without such acknowledgements."""
        applied = self._applied()
        return self._frames(applied) if applied else None

    def _applied(self) -> dict[int, int]:
        path = self.root / "acks.csv"
        applied: dict[int, int] = {}
        if not path.is_file():
            return applied
        with path.open(newline="") as f:
            reader = csv.reader(f)
            col = {name: i for i, name in enumerate(next(reader))}
            for r in reader:
                if r[col["receiver_id"]] == "controller" and r[col["state"]] == "applied":
                    applied[int(r[col["seq"]])] = int(r[col["t_ack_ros_ns"]])
        return applied

    def _frames(self, applied: dict[int, int] | None, lo_ns: int | None = None,
                hi_ns: int | None = None) -> CommandTrace:
        """Frames stamped at ``t_frame_ros_ns``, or at their APPLIED time with ``applied`` (unapplied ones left
        out), kept by frame time in ``[lo_ns, hi_ns]``: the bounds are frame times, so a window is the same frames
        whichever stamp a replay uses."""
        frames: dict[int, dict[str, float]] = {}
        with (self.root / "commands.csv").open(newline="") as f:
            reader = csv.reader(f)
            col = {name: i for i, name in enumerate(next(reader))}
            for r in reader:
                t_frame = int(r[col["t_frame_ros_ns"]])
                if (lo_ns is not None and t_frame < lo_ns) or (hi_ns is not None and t_frame > hi_ns):
                    continue
                if applied is None:
                    stamp = t_frame
                else:
                    stamp = applied.get(int(r[col["seq"]]))
                    if stamp is None:
                        continue
                frames.setdefault(stamp, {})[r[col["joint_id"]]] = float(r[col["q_nominal_rad"]])
        return CommandTrace.from_frames(list(frames.items()))

    def _signal_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Every frame's ``(t_frame_ros_ns, signal time in ns)``, in frame order."""
        seen: dict[int, int] = {}
        with (self.root / "commands.csv").open(newline="") as f:
            reader = csv.reader(f)
            col = {name: i for i, name in enumerate(next(reader))}
            for r in reader:
                seen[int(r[col["t_frame_ros_ns"]])] = round(float(r[col["t_signal_s"]]) * 1e9)
        t = np.array(sorted(seen), dtype=np.int64)
        return t, np.array([seen[int(x)] for x in t], dtype=np.int64)

    def sequences(self, max_frequency_hz: float | None = None) -> list[ExcitationSequence]:
        """The sweeps the run finished, each a capture of its own (sweep ``k`` from the rest before it, the first
        from the capture's start); ``max_frequency_hz`` ends each where its chirp reaches that frequency. A sweep
        is finished when a command frame reached its end, cut or not. Raises when none finished."""
        signals = self.manifest.get("signals") or []
        if not signals:
            raise ValueError(f"{self.dataset_id}: the manifest names no signals, so its sweeps are unknown")
        timing = ("f0_hz", "f1_hz", "duration_s", "sequence_gap_s", "start_offset_ns")
        ref = signals[0]
        for s in signals[1:]:
            if len(s["sequence_scales"]) != len(ref["sequence_scales"]) or not all(
                    math.isclose(float(s[k]), float(ref[k]), rel_tol=1e-9, abs_tol=1e-9) for k in timing):
                raise ValueError(f"{self.dataset_id}: joints swept on different schedules ({ref['joint_id']}, "
                                 f"{s['joint_id']}); a sweep window needs one")
        f0, f1 = float(ref["f0_hz"]), float(ref["f1_hz"])
        duration_ns = round(float(ref["duration_s"]) * 1e9)
        period_ns = duration_ns + round(float(ref["sequence_gap_s"]) * 1e9)
        offset_ns = int(ref["start_offset_ns"])
        cut_ns = duration_ns
        if max_frequency_hz is not None:
            if not f0 < max_frequency_hz:
                raise ValueError(f"{self.dataset_id}: a cut at {max_frequency_hz} Hz is not above the sweep's "
                                 f"f0 {f0} Hz")
            if max_frequency_hz < f1:
                cut_ns = round(duration_ns * math.log(max_frequency_hz / f0) / math.log(f1 / f0))
        t_frame, t_signal = self._signal_frames()
        out = []
        for k, scale in enumerate(ref["sequence_scales"]):
            start_ns = offset_ns + k * period_ns
            if t_signal.size == 0 or t_signal.max() < start_ns + duration_ns:
                break
            lo = None if k == 0 else int(t_frame[np.searchsorted(t_signal, start_ns, side="left")])
            hi = int(t_frame[np.searchsorted(t_signal, start_ns + cut_ns, side="right") - 1])
            out.append(ExcitationSequence(self, k, float(scale), lo, hi, {
                "capture": self.dataset_id, "capture_complete": self.complete, "sequence": k,
                "sequence_scale": float(scale), "signal_ns": [start_ns, start_ns + cut_ns],
                "frames_ros_ns": [lo, hi], "max_frequency_hz": max_frequency_hz,
                "top_frequency_hz": f0 * (f1 / f0) ** (cut_ns / duration_ns)}))
        if not out:
            raise ValueError(f"{self.dataset_id}: the run finished none of its sweeps")
        return out

    def time_span_ns(self) -> tuple[int, int]:
        """Interval covered by every drive's PDO trace."""
        traces = self.pdo().values()
        return max(int(tr.t_ns[0]) for tr in traces), min(int(tr.t_ns[-1]) for tr in traces)


@dataclass
class ExcitationSequence:
    """One finished sweep of an :class:`ExcitationCapture` (:meth:`ExcitationCapture.sequences`), read through the
    capture's interface and clipped to its window: frames by frame time in ``[lo_ns, hi_ns]`` plus the last one before
    ``lo_ns`` (in force as the window opens, the replay's first demand), feedback and IMU over the same span. ``lo_ns`` None: from the
    capture's start. ``provenance`` is what a fit records about the window."""

    capture: ExcitationCapture
    index: int
    scale: float
    lo_ns: int | None
    hi_ns: int
    provenance: dict = field(default_factory=dict)

    @property
    def dataset_id(self) -> str:
        return f"{self.capture.dataset_id}#sequence{self.index}"

    @property
    def name(self) -> str:
        return f"{self.capture.name} sequence {self.index}"

    def gains(self) -> dict[str, DriveGains]:
        return self.capture.gains()

    def _in_force_ns(self) -> int | None:
        if self.lo_ns is None:
            return None
        t_frame, _ = self.capture._signal_frames()
        # Strictly before: applied-stamped, a frame at lo_ns takes effect after the window opens.
        return int(t_frame[max(np.searchsorted(t_frame, self.lo_ns, side="left") - 1, 0)])

    def commands(self) -> CommandTrace:
        return self.capture._frames(None, self._in_force_ns(), self.hi_ns)

    def applied_commands(self) -> CommandTrace | None:
        applied = self.capture._applied()
        return self.capture._frames(applied, self._in_force_ns(), self.hi_ns) if applied else None

    def _keep(self, t: np.ndarray) -> np.ndarray:
        return (t <= self.hi_ns) if self.lo_ns is None else (t >= self.lo_ns) & (t <= self.hi_ns)

    def pdo(self) -> dict[str, DriveTrace]:
        out = {}
        for dev, tr in self.capture.pdo().items():
            keep = self._keep(tr.t_ns)
            out[dev] = DriveTrace(dev, tr.t_ns[keep], {k: v[keep] for k, v in tr.fields.items()})
        return out

    def imu(self) -> ImuTrace:
        tr = self.capture.imu()
        keep = self._keep(tr.t_ns)
        return ImuTrace(tr.t_ns[keep], tr.seq[keep], tr.orientation_xyzw[keep], tr.angular_velocity[keep],
                        tr.specific_force[keep])

    def time_span_ns(self) -> tuple[int, int]:
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
