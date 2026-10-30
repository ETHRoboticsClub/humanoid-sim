"""Reader for Controls rosbag2 recordings (``ros2 bag record --all``, sqlite3 or MCAP), without ROS.

A bag recorded while the robot runs an excitation on the gantry holds what an ``ethrc-excitation`` capture holds,
and :class:`BagCapture` serves it through the same interface as :class:`~humanoid_sim.replay.exc.ExcitationCapture`:

* ``/ethrc/drive_feedback`` (``msg_ethrc/msg/DriveFeedback``, 200 Hz): every drive's ``source_stamp_ns``, measured
  position, velocity and torque, and its position demand 0x6062 (``position_demand_rad``), as ``pdo_feedback.csv``;
* ``/drive_gains/state`` (``msg_ethrc/msg/DriveGains``, latched, every 250 ms and on a change): the gains in force;
* ``/rt/imu`` (``sensor_msgs/msg/Imu``): orientation and angular velocity in ``base_link``, stamped on receipt;
* ``/rt/body_control/device`` (``msg_ethrc/msg/BodyControlFrame``, 50 Hz while commanding): the router's
  COMMAND frames to the EtherCAT master, every drive's joint reference in semantic radians, stamped by the loop
  host when it sent them. No topic carries the per-drive targets the master writes to the bus; the replay rebuilds
  them from these frames in its Controls model and supplies a resolved drive stream to replay.

Times are the messages' own stamps (``source_stamp_ns``, ``header.stamp``: ROS time, integer ns), never the bag's
receipt times. Message definitions come from the bag when it carries them (MCAP, and sqlite3 from ROS Iron on);
otherwise from the copies in ``msgs/`` (Controls ``ros2_ws/src/contracts/msg_ethrc/msg`` @ d35a2fa99).

A bag has no ``complete`` flag or gantry declaration, so the reader checks what it can: every drive streams and
the gains hold still and are applied. ``sensor_msgs/Imu`` carries no sample counter, so a dropped IMU sample goes
unseen. That the robot hung on the gantry is the caller's to know. ``rosbags`` is the ``bag`` extra.
"""

from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

import numpy as np

from humanoid_sim.replay.exc import CommandTrace, DriveGains, DriveTrace, ImuTrace

FEEDBACK_TOPIC = "/ethrc/drive_feedback"
GAINS_TOPIC = "/drive_gains/state"
IMU_TOPIC = "/rt/imu"
COMMAND_TOPIC = "/rt/body_control/device"
KIND_COMMAND = 2  # BodyControlFrame.KIND_COMMAND
MSGS = Path(__file__).with_name("msgs")
FEEDBACK_FIELDS = {"native_position_rad": "position_rad", "velocity_rad_s": "velocity_rad_s",
                   "torque_nm": "torque_nm", "current_a": "current_a", "position_demand_rad": "position_demand_rad",
                   "current_demand_a": "current_demand_a", "motor_position_rad": "motor_position_rad"}


def _ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _typestore(connections):
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    store = get_typestore(Stores.LATEST)
    types = {}
    for c in connections:
        if c.msgtype in store.types:
            continue
        if c.msgdef is not None and c.msgdef.data and c.msgdef.format.name == "MSG":
            types |= get_types_from_msg(c.msgdef.data, c.msgtype)
        else:
            path = MSGS / f"{c.msgtype.rsplit('/', 1)[-1]}.msg"
            if path.is_file():
                types |= get_types_from_msg(path.read_text(), c.msgtype)
    store.register(types)
    return store


class BagCapture:
    """One rosbag2 directory, read once on first use."""

    def __init__(self, root: Path, *, drives: list[str] | None = None):
        self.root = Path(root)
        if not (self.root / "metadata.yaml").is_file():
            raise ValueError(f"{self.root}: not a rosbag2 directory (no metadata.yaml)")
        self._drives = drives

    @cached_property
    def dataset_id(self) -> str:
        """``bag-`` and the first 16 hex digits of the sha256 over the bag's files (names and contents)."""
        h = hashlib.sha256()
        for p in sorted(self.root.iterdir()):
            if p.is_file():
                h.update(p.name.encode())
                with p.open("rb") as f:
                    h.update(hashlib.file_digest(f, "sha256").digest())
        return f"bag-{h.hexdigest()[:16]}"

    @property
    def name(self) -> str:
        return self.root.name

    @cached_property
    def _messages(self) -> dict[str, list]:
        from rosbags.rosbag2 import Reader

        wanted = (FEEDBACK_TOPIC, GAINS_TOPIC, IMU_TOPIC, COMMAND_TOPIC)
        out: dict[str, list] = {t: [] for t in wanted}
        with Reader(self.root) as reader:
            conns = [c for c in reader.connections if c.topic in wanted]
            missing = sorted(set((FEEDBACK_TOPIC, GAINS_TOPIC)) - {c.topic for c in conns})
            if missing:
                raise ValueError(f"{self.root}: no {', '.join(missing)} in the bag")
            store = _typestore(conns)
            for conn, _, raw in reader.messages(connections=conns):
                out[conn.topic].append(store.deserialize_cdr(raw, conn.msgtype))
        return out

    def pdo(self) -> dict[str, DriveTrace]:
        """Per-drive feedback traces, sorted by source time, duplicates dropped (as ``ExcitationCapture.pdo``)."""
        return self._pdo

    @cached_property
    def _pdo(self) -> dict[str, DriveTrace]:
        rows: dict[str, list[tuple]] = {}
        for m in self._messages[FEEDBACK_TOPIC]:
            n = len(m.device)
            cols = []
            for field in FEEDBACK_FIELDS.values():
                values = getattr(m, field, None)
                if field in ("current_a", "current_demand_a") and (values is None or len(values) == 0):
                    values = np.full(n, np.nan)
                cols.append(np.asarray(values, dtype=np.float64))
            if any(len(c) != n for c in cols) or len(m.source_stamp_ns) != n:
                raise ValueError(f"{self.root}: a DriveFeedback message with sequences of different lengths")
            for i, dev in enumerate(m.device):
                rows.setdefault(dev, []).append((int(m.source_stamp_ns[i]), *(float(c[i]) for c in cols)))
        if self._drives is not None:
            missing = sorted(set(self._drives) - set(rows))
            if missing:
                raise ValueError(f"{self.root}: no feedback from drives {missing}")
        traces = {}
        for dev, rs in rows.items():
            a = np.array(rs, dtype=np.float64)
            t = np.array([r[0] for r in rs], dtype=np.int64)
            order = np.argsort(t, kind="stable")
            t, a = t[order], a[order]
            keep = np.concatenate([[True], np.diff(t) > 0])
            traces[dev] = DriveTrace(dev, t[keep], {k: a[keep, i + 1] for i, k in enumerate(FEEDBACK_FIELDS)})
        return traces

    def gains(self) -> dict[str, DriveGains]:
        """The gains in force on each drive. Refuses a bag whose gains change, ramp, fail or are not applied on
        every drive while the drives stream, or that lacks a streaming drive (raw counts are not in the message:
        0)."""
        t0, t1 = self.time_span_ns()
        msgs = self._messages[GAINS_TOPIC]
        stamp = [_ns(m.header.stamp) for m in msgs]
        # The state in force at the start (the latest at or before it) and every state during the span.
        before = [m for m, s in zip(msgs, stamp, strict=True) if s <= t0]
        during = [m for m, s in zip(msgs, stamp, strict=True) if t0 < s <= t1]
        states = before[-1:] + during
        if not states:
            raise ValueError(f"{self.root}: no {GAINS_TOPIC} message at or after the feedback starts")
        ref = None
        for m in states:
            if m.failed or m.failed_motors or not m.applied or not all(m.motor_applied) or m.ramp_remaining_s > 0:
                raise ValueError(f"{self.root}: gains not applied, ramping or failed during the recording")
            pairs = {d: (float(kp), float(kd)) for d, kp, kd in
                     zip(m.motors, m.kp_nm_per_rad, m.kd_nm_s_per_rad, strict=True)}
            if ref is None:
                ref = pairs
            elif pairs != ref:
                raise ValueError(f"{self.root}: gains changed during the recording")
        missing = sorted(set(self.pdo()) - set(ref))
        if missing:
            raise ValueError(f"{self.root}: no gains for streaming drives {missing}")
        return {d: DriveGains(kp, kd, 0, 0) for d, (kp, kd) in ref.items()}

    def imu(self) -> ImuTrace:
        msgs = self._messages[IMU_TOPIC]
        if not msgs:
            raise ValueError(f"{self.root}: no {IMU_TOPIC} in the bag (a moving-base replay needs it)")
        t = np.array([_ns(m.header.stamp) for m in msgs], dtype=np.int64)
        order = np.argsort(t, kind="stable")
        q = np.array([[m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w] for m in msgs])
        w = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in msgs])
        a = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in msgs])
        # sensor_msgs/Imu has no sample counter: number the samples in stamp order (a dropped one goes unseen).
        return ImuTrace(t[order], np.arange(len(t), dtype=np.int64), q[order], w[order], a[order])

    def commands(self) -> CommandTrace:
        frames = [(_ns(m.header.stamp), dict(zip(m.joint_ids, (float(p) for p in m.positions_rad), strict=True)))
                  for m in self._messages[COMMAND_TOPIC] if m.kind == KIND_COMMAND]
        if not frames:
            raise ValueError(f"{self.root}: no COMMAND frames on {COMMAND_TOPIC}")
        return CommandTrace.from_frames(frames)

    def applied_commands(self) -> None:
        """A bag carries no controller acknowledgements: fits replay the frame stamps."""
        return None

    def time_span_ns(self) -> tuple[int, int]:
        traces = self.pdo().values()
        return max(int(tr.t_ns[0]) for tr in traces), min(int(tr.t_ns[-1]) for tr in traces)
