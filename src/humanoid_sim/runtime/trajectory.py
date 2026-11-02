"""The runtime's truth, recorded at every physics step: the ground truth of a run and the input the sensors are
rendered from afterwards (:mod:`humanoid_sim.sensors.dataset`).

A trajectory file is one JSON header line, then fixed-size records. Each record is the host's truth at one plant time:
base pose and twist in the world frame, the joint positions in URDF joint coordinates (the header names them) and the
gantry hook's position. Plant time is integer ns, the same clock as the runtime's feedback and IMU samples. The file
is append-only and every record is whole once written, so a run that is killed still leaves a readable file.

Base: the floating root link of the description (``Trajectory.base_frame``), pose as (position, quaternion x y z w).
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import numpy as np

SCHEMA = "ethrc-humanoid-sim-trajectory/v1"
IMU_SCHEMA = "ethrc-humanoid-sim-imu/v1"
LOG = logging.getLogger(__name__)
FLUSH_RECORDS = 250  # about a second of records at 4 ms steps


def record_dtype(num_joints: int) -> np.dtype:
    return np.dtype([
        ("sim_time_ns", "<i8"), ("session_id", "<u8"),
        ("base_position_m", "<f8", (3,)), ("base_quat_xyzw", "<f8", (4,)),
        ("base_lin_vel_m_s", "<f8", (3,)), ("base_ang_vel_rad_s", "<f8", (3,)),
        ("joint_pos_rad", "<f8", (num_joints,)), ("gantry_position_m", "<f8", (3,)),
    ])


class TrajectoryWriter:
    def __init__(self, path: str | Path, *, robot: str, description_digest: str, joints: list[str], period_ns: int,
                 base_frame: str, scene: dict | None = None, gantry: dict | None = None):
        self.path = Path(path)
        self.dtype = record_dtype(len(joints))
        header = self.header = {
                  "schema": SCHEMA, "recording_id": uuid.uuid4().hex,
                  "robot": robot, "description_digest": description_digest, "joints": list(joints),
                  "period_ns": int(period_ns), "base_frame": base_frame, "scene": scene, "gantry": gantry,
                  "frame": "world: the host's scene frame, z up", "quaternion": "x y z w",
                  "units": {"time": "ns, plant time", "position": "m", "velocity": "m/s, rad/s",
                            "joint_pos": "rad, URDF joint coordinates"}}
        self._file = open(self.path, "wb")
        self._file.write(json.dumps(header).encode() + b"\n")
        self._file.flush()
        self._last_ns = -1
        self._since_flush = 0
        self.count = 0

    def append(self, truth, session_id: int = 0) -> bool:
        """Write the host's truth record (``plant_abi.TRUTH``) unless its plant time was already written (a truth
        published after a command shares its step's time, and a paused host stops the clock). Returns whether it
        was written."""
        t = int(truth["sim_time_ns"])
        if t <= self._last_ns:
            return False
        nj = self.dtype["joint_pos_rad"].shape[0]
        rec = np.zeros((), self.dtype)
        rec["sim_time_ns"], rec["session_id"] = t, session_id
        for name in ("base_position_m", "base_quat_xyzw", "base_lin_vel_m_s", "base_ang_vel_rad_s",
                     "gantry_position_m"):
            rec[name] = truth[name]
        rec["joint_pos_rad"] = truth["joint_pos_rad"][:nj]
        self._file.write(rec.tobytes())
        self._last_ns = t
        self.count += 1
        self._since_flush += 1
        if self._since_flush >= FLUSH_RECORDS:
            self._file.flush()
            self._since_flush = 0
        return True

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


IMU_RECORD = np.dtype([("seq", "<u8"), ("sample_ns", "<i8"), ("stamp_ns", "<i8"),
                       ("gyro_rad_s", "<f8", (3,)), ("specific_force_m_s2", "<f8", (3,))])


class ImuWriter:
    """The body IMU's delivered samples of a run, next to its trajectory (``<trajectory>.imu``): the same header line
    (plus ``imu``: the descriptor's IMU block) and one record per sample the host delivered, oldest first, as a client
    reads them from the feedback region: raw gyro (rad/s) and specific force (m/s^2) in the sensor frame, ``sample_ns``
    the measurement time and ``stamp_ns`` the host-receipt time, both plant time."""

    def __init__(self, path: str | Path, header: dict, imu: dict):
        self.path = Path(path)
        self._file = open(self.path, "wb")
        self._file.write(json.dumps({**header, "schema": IMU_SCHEMA, "imu": imu}).encode() + b"\n")
        self._file.flush()
        self._last_seq = 0
        self._since_flush = 0
        self.count = 0

    def append(self, samples) -> int:
        """Write the samples of ``samples`` (the feedback's IMU array, newest first, valid entries only) not written
        yet. Returns how many."""
        new = samples[samples["seq"] > self._last_seq][::-1]
        if not len(new):
            return 0
        if self._last_seq and int(new["seq"][0]) != self._last_seq + 1:
            LOG.warning("IMU file: samples %d..%d were not in the feedback window and are missing from %s",
                        self._last_seq + 1, int(new["seq"][0]) - 1, self.path)
        rec = np.zeros(len(new), IMU_RECORD)
        for name in IMU_RECORD.names:
            rec[name] = new[name]
        self._file.write(rec.tobytes())
        self._last_seq = int(new["seq"][-1])
        self.count += len(new)
        self._since_flush += len(new)
        if self._since_flush >= FLUSH_RECORDS:
            self._file.flush()
            self._since_flush = 0
        return len(new)

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


def read_imu(path: str | Path) -> tuple[dict, np.ndarray]:
    """An IMU side file's header and its records."""
    with open(path, "rb") as f:
        header = json.loads(f.readline())
        if header.get("schema") != IMU_SCHEMA:
            raise ValueError(f"{path}: not an IMU file (schema {header.get('schema')!r})")
        data = f.read()
    whole = len(data) // IMU_RECORD.itemsize
    return header, np.frombuffer(data[:whole * IMU_RECORD.itemsize], dtype=IMU_RECORD).copy()


def read_header(path: str | Path) -> dict:
    """A trajectory file's header alone (the robot, joints, scene and crane it was recorded with)."""
    with open(path, "rb") as f:
        header = json.loads(f.readline())
    if header.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a trajectory file (schema {header.get('schema')!r})")
    return header


def _slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Spherical interpolation of unit quaternions ``[N, 4]`` at fractions ``u`` ``[N]``."""
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where((dot < 0)[:, None], -q1, q1)
    dot = np.abs(dot).clip(0.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    near = sin < 1e-6
    a = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(near, 1.0, sin))
    b = np.where(near, u, np.sin(u * theta) / np.where(near, 1.0, sin))
    q = a[:, None] * q0 + b[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class Trajectory:
    """A recorded run: ``records`` (a structured array) and the header's ``joints``, ``robot``, ``period_ns``."""

    def __init__(self, header: dict, records: np.ndarray):
        if header.get("schema") != SCHEMA:
            raise ValueError(f"trajectory schema {header.get('schema')!r}, expected {SCHEMA!r}")
        self.header, self.records = header, records
        self.joints = list(header["joints"])
        self.robot = header["robot"]
        self.base_frame = header["base_frame"]
        self.period_ns = int(header["period_ns"])
        if len(records) < 2:
            raise ValueError("a trajectory needs at least two records")
        if np.any(np.diff(records["sim_time_ns"]) <= 0):
            raise ValueError("trajectory plant times are not strictly increasing")

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        with open(path, "rb") as f:
            header = json.loads(f.readline())
            if header.get("schema") != SCHEMA:
                raise ValueError(f"{path}: not a trajectory file (schema {header.get('schema')!r})")
            dtype = record_dtype(len(header["joints"]))
            data = f.read()
        whole = len(data) // dtype.itemsize  # a record cut off by a kill is dropped
        return cls(header, np.frombuffer(data[:whole * dtype.itemsize], dtype=dtype).copy())

    @property
    def start_ns(self) -> int:
        return int(self.records["sim_time_ns"][0])

    @property
    def end_ns(self) -> int:
        return int(self.records["sim_time_ns"][-1])

    def at(self, t_ns) -> dict[str, np.ndarray]:
        """The truth at plant times ``t_ns`` (an int or an array), linearly interpolated (spherically for the base
        orientation); times outside the recording are refused. A gantry position that is NaN (no crane) stays NaN."""
        t = np.atleast_1d(np.asarray(t_ns, dtype=np.int64))
        times = self.records["sim_time_ns"]
        if t.min() < times[0] or t.max() > times[-1]:
            raise ValueError(f"time {int(t.min())}..{int(t.max())} ns is outside the trajectory "
                             f"{int(times[0])}..{int(times[-1])} ns")
        hi = np.searchsorted(times, t, side="left").clip(1, len(times) - 1)
        lo = hi - 1
        u = ((t - times[lo]) / (times[hi] - times[lo])).clip(0.0, 1.0)
        r0, r1 = self.records[lo], self.records[hi]

        def lerp(name):
            return r0[name] + (r1[name] - r0[name]) * u.reshape((-1,) + (1,) * (r0[name].ndim - 1))

        out = {name: lerp(name) for name in ("base_position_m", "base_lin_vel_m_s", "base_ang_vel_rad_s",
                                             "joint_pos_rad", "gantry_position_m")}
        out["base_quat_xyzw"] = _slerp(r0["base_quat_xyzw"], r1["base_quat_xyzw"], u)
        out["sim_time_ns"] = t
        return out
