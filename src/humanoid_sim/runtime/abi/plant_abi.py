"""Runtime segment ABI: the shared-memory file between humanoid-sim's real-time host and a client that plays the
robot's devices (Controls' fake EtherCAT bank for the ETHRC V1, its fake Unitree board for the G1).

Self-contained (Python >= 3.10, NumPy only) so Controls can vendor it unchanged next to ``plant_abi.h``, which
declares the same layout for C and C++.

Layout: a header, then six regions, each with one writer:

==========  ======  ==========================================================================================
region      writer  contents
==========  ======  ==========================================================================================
header      sim     magic, ABI version, layout hash, sizes, session id, robot id, digests, periods, state,
                    heartbeat
descriptor  sim     JSON: drive and joint names, units, profile, revisions, fidelity notes
demands     client  bus cycle, client clock, session echo, one SI ``DriveDemand`` per drive (law, targets, gains)
feedback    sim     sim time, physical drive feedback per drive, lateness, the latest raw IMU samples
mailbox     client  one command at a time: sequence number, command code, JSON arguments
ack         sim     the last command handled: its sequence number, a result code and a message
truth       sim     the oracle tests need: base pose and twist, centre of mass, joint state, gantry tension,
                    per-sole pose, lowest point and normal force
==========  ======  ==========================================================================================

Every region starts with ``seq`` (u64) and ``crc`` (u32). A writer makes ``seq`` odd, writes the payload, stores
the CRC-32 of the payload and makes ``seq`` even again; a reader copies the payload between two equal, even
reads of ``seq`` and accepts it only if its CRC matches. The CRC makes a torn copy detectable without relying
on memory ordering, which Python does not give. Values are little-endian, at natural alignment; every region
starts on a 64-byte boundary.

Sessions: the header's ``session_id`` names one client session. A client attaches by writing demands that echo
it. The session ends (state ended) when the client sends DETACH, when its demands are older than the sim's client
timeout, or when the sim stops. A sim that serves sessions one after another creates each on a new file it
renames over the same path: a client still mapping the old file sees it ended, a client that opens the path gets
the new session.

Time: ``*_mono_ns`` fields are ``CLOCK_MONOTONIC`` of the writing process; ``sim_time_ns`` and the IMU samples'
``sample_ns`` and ``stamp_ns`` are plant time. The feedback region maps plant time ``t`` to the sim host's clock:
``written_mono_ns - (sim_time_ns - t)``. Identity: the description digest is :func:`description_digest`.
Physical quantities are SI in the physical drive-output frame (crank angle for the ankle cranks). Demands are
vendor-neutral: a client converts its device's command (an EPOS4's CiA-402 mode and raw JVPT gains, a Unitree
motor's LowCmd entry) into a law with SI targets and gains at its own boundary (ABI 2.0).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ABI_MAJOR = 2
ABI_MINOR = 0
MAGIC = b"HSIMPLNT"
ALIGN = 64

# Sizes fixed by the ABI (a robot with more drives or joints needs a new major version).
MAX_DRIVES = 64
MAX_JOINTS = 64
IMU_SAMPLES = 32  # latest delivered samples kept in the feedback region, newest first
DESCRIPTOR_BYTES = 16384
MAILBOX_BYTES = 4096
ACK_MESSAGE_BYTES = 256
SOLES = 2  # left, right

# Header state.
STATE_CREATING, STATE_READY, STATE_ENDED = 0, 1, 2

# Mailbox commands. Arguments are a JSON object:
# RESET            {"pose": "home" | "standing" | "current"} or {"joint_pos": {name: rad}, "base_pose": [x,y,z,qx,qy,qz,qw]}
# PAUSE, RESUME    {}: while paused physics stops and feedback keeps being written
# SET_SUPPORT      {"support": "free"}; {"support": "gantry", "tension_breakpoints_s_n": [[s, N], ...],
#                  "attachment_local_m": [x,y,z] | null, "attachment_bodies": [name, ...] | null}; or a cable,
#                  {"support": "gantry", "attachment_local_m": [x,y,z], "cable": {"overhead_distance_m": m,
#                  "length_breakpoints_s_m": [[s, m], ...], "stiffness_n_m": k, "damping_n_s_m": c}}
# APPLY_DISTURBANCE {"body": name, "force_n": [x,y,z], "torque_nm": [x,y,z], "duration_s": s}, or a torque on one
#                  joint outside the drive's feedback: {"joint" | "drive" (1:1): name, "torque_nm": Nm,
#                  "frequency_hz": f (0: constant), "duration_s": s | null (until cleared)}; torque_nm 0 clears it
# INJECT_FAULT     {"fault": name, "drive": name | null, "value": number, "duration_s": s | null}
# PRESCRIBE_POSE   {"joint_pos": {...}, "base_pose": [...] | null, "floor_clearance_m": m | null}: hold a pose
#                  (the base dropped so its lowest sole corner sits that far above the floor); {"release": true}
# DETACH           {}: the client is leaving; the sim acks and ends the session
# SET_FIDELITY     {"imu": "ideal" | "real"}: the IMU without or with its model errors, for this session; every
#                  session starts real
CMD_NONE = 0
CMD_RESET = 1
CMD_PAUSE = 2
CMD_RESUME = 3
CMD_SET_SUPPORT = 4
CMD_APPLY_DISTURBANCE = 5
CMD_INJECT_FAULT = 6
CMD_PRESCRIBE_POSE = 7
CMD_DETACH = 8
CMD_SET_FIDELITY = 9
COMMANDS = {
    CMD_RESET: "reset",
    CMD_PAUSE: "pause",
    CMD_RESUME: "resume",
    CMD_SET_SUPPORT: "set_support",
    CMD_APPLY_DISTURBANCE: "apply_disturbance",
    CMD_INJECT_FAULT: "inject_fault",
    CMD_PRESCRIBE_POSE: "prescribe_pose",
    CMD_DETACH: "detach",
    CMD_SET_FIDELITY: "set_fidelity",
}

# Ack results.
ACK_OK = 0
ACK_REJECTED = 1  # malformed or not allowed now; message says why
ACK_UNKNOWN = 2  # unknown command code
ACK_FAILED = 3  # accepted but failed while applying

# Physical faults the sim can inject. Bus and protocol faults belong to Controls' fake bank, and so does a drive
# whose encoder stops reporting (on an EPOS4 that is a drive fault, not a plant state).
FAULTS = ("encoder_freeze", "imu_spike", "imu_bias", "power_sag")


def _region(*fields) -> np.dtype:
    return np.dtype([("seq", "<u8"), ("crc", "<u4"), ("_pad0", "<u4"), *fields], align=True)


HEADER = np.dtype(
    [
        ("magic", "S8"),
        ("abi_major", "<u2"),
        ("abi_minor", "<u2"),
        ("header_bytes", "<u4"),
        ("layout_hash", "<u8"),
        ("segment_bytes", "<u8"),
        ("session_id", "<u8"),
        ("state", "<u4"),
        ("num_drives", "<u4"),
        ("num_joints", "<u4"),
        ("imu_samples", "<u4"),
        ("physics_period_ns", "<u8"),
        ("bus_period_ns", "<u8"),
        ("heartbeat_mono_ns", "<u8"),
        ("robot_id", "S32"),
        ("description_digest", "S64"),
        ("sim_revision", "S48"),
        ("offsets", "<u8", (6,)),  # descriptor, demands, feedback, mailbox, ack, truth
    ],
    align=True,
)

# Drive laws (DriveDemand.law).
LAW_DISABLED = 0  # no torque
LAW_PD = 1  # kp (target - q) + kd (target_velocity - v) + feedforward
LAW_FREEZE = 2  # PD about the position the drive reads when it enters FREEZE, no velocity target or feedforward

DRIVE_DEMAND = np.dtype(
    [
        ("law", "u1"),
        ("_pad0", "u1", (7,)),
        ("target_position_rad", "<f8"),
        ("target_velocity_rad_s", "<f8"),
        ("feedforward_torque_nm", "<f8"),
        ("stiffness_nm_per_rad", "<f8"),
        ("damping_nm_s_per_rad", "<f8"),
        ("torque_limit_nm", "<f8"),  # a cap on the applied torque; inf: none
    ],
    align=True,
)

DRIVE_FEEDBACK = np.dtype(
    [
        ("position_rad", "<f8"),
        ("velocity_rad_s", "<f8"),
        ("applied_torque_nm", "<f8"),
        ("demanded_torque_nm", "<f8"),
    ],
    align=True,
)

IMU_SAMPLE = np.dtype(
    [
        ("seq", "<u8"),
        ("sample_ns", "<i8"),
        ("stamp_ns", "<i8"),
        ("gyro_rad_s", "<f8", (3,)),
        ("specific_force_m_s2", "<f8", (3,)),
    ],
    align=True,
)

SOLE = np.dtype(
    [
        ("position_m", "<f8", (3,)),  # the sole body's frame
        ("quat_xyzw", "<f8", (4,)),
        ("lowest_z_m", "<f8"),  # lowest corner of the sole plate
        ("normal_force_n", "<f8"),  # vertical contact force on the sole body
        ("plate_normal_w", "<f8", (3,)),  # the plate's face normal, world
        ("plate_face_height_span_m", "<f8"),  # height difference across the plate's face
        ("face_xy_m", "<f8", (4, 2)),  # the plate's bottom-face corners, world x and y
    ],
    align=True,
)

DESCRIPTOR = _region(("length", "<u4"), ("_pad1", "<u4"), ("json", "u1", (DESCRIPTOR_BYTES,)))
DEMANDS = _region(
    ("cycle", "<u8"),
    ("written_mono_ns", "<u8"),
    ("session_id", "<u8"),
    ("num_drives", "<u4"),
    ("_pad1", "<u4"),
    ("drive", DRIVE_DEMAND, (MAX_DRIVES,)),
)
FEEDBACK = _region(
    ("step", "<u8"),
    ("sim_time_ns", "<i8"),
    ("written_mono_ns", "<u8"),
    ("late_steps", "<u8"),
    ("keeping_up", "<u4"),
    ("paused", "<u4"),
    ("num_drives", "<u4"),
    ("imu_count", "<u4"),
    ("imu_total", "<u8"),
    ("drive", DRIVE_FEEDBACK, (MAX_DRIVES,)),
    ("imu", IMU_SAMPLE, (IMU_SAMPLES,)),
)
MAILBOX = _region(
    ("command", "<u4"),
    ("length", "<u4"),
    ("written_mono_ns", "<u8"),
    ("args", "u1", (MAILBOX_BYTES,)),
)
ACK = _region(
    ("acked", "<u8"), ("result", "<i4"), ("_pad1", "<u4"), ("message", f"S{ACK_MESSAGE_BYTES}")
)
TRUTH = _region(
    ("sim_time_ns", "<i8"),
    ("base_position_m", "<f8", (3,)),
    ("base_quat_xyzw", "<f8", (4,)),
    ("base_lin_vel_m_s", "<f8", (3,)),
    ("base_ang_vel_rad_s", "<f8", (3,)),
    ("com_m", "<f8", (3,)),
    ("gantry_tension_n", "<f8"),
    ("mass_kg", "<f8"),
    ("num_joints", "<u4"),
    ("_pad1", "<u4"),
    ("joint_pos_rad", "<f8", (MAX_JOINTS,)),
    ("joint_vel_rad_s", "<f8", (MAX_JOINTS,)),
    ("sole", SOLE, (SOLES,)),
)

REGIONS = (
    ("descriptor", DESCRIPTOR),
    ("demands", DEMANDS),
    ("feedback", FEEDBACK),
    ("mailbox", MAILBOX),
    ("ack", ACK),
    ("truth", TRUTH),
)


def _aligned(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def _layout() -> tuple[dict[str, int], int]:
    offsets, at = {}, _aligned(HEADER.itemsize)
    for name, dtype in REGIONS:
        offsets[name] = at
        at = _aligned(at + dtype.itemsize)
    return offsets, at


OFFSETS, SEGMENT_BYTES = _layout()


def _describe(dtype: np.dtype) -> list:
    out = []
    for name in dtype.names:
        field, offset = dtype.fields[name][:2]
        kind = _describe(field.base) if field.base.names else field.base.str
        out.append([name, offset, kind, list(field.shape)])
    return out


def layout_hash() -> int:
    """64-bit hash of every field's name, offset, type and shape, and the region offsets."""
    spec = {
        "header": _describe(HEADER),
        "regions": {n: [OFFSETS[n], _describe(d)] for n, d in REGIONS},
    }
    return int.from_bytes(
        hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).digest()[:8], "little"
    )


LAYOUT_HASH = layout_hash()


class AbiError(RuntimeError):
    pass


class StaleRead(RuntimeError):
    """A region kept changing (or failed its CRC) for every attempt: the writer is mid-write or stalled."""


def description_digest(urdf: str | Path, actuators: str | Path) -> str:
    """sha256 (hex) over the robot description: the URDF and ``actuators.yaml``, each as its file name and bytes.
    Both sides compute it from their own checkout of humanoid-urdf; equal digests mean the same description."""
    h = hashlib.sha256()
    for path in (Path(urdf), Path(actuators)):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def mono_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def _payload(dtype: np.dtype) -> slice:
    start = dtype.fields["_pad0"][1] + 4
    return slice(start, dtype.itemsize)


@dataclass
class Region:
    name: str
    dtype: np.dtype
    raw: np.ndarray  # u8 view of the region in the segment

    def __post_init__(self):
        self._seq = self.raw[0:8].view("<u8")
        self._crc = self.raw[8:12].view("<u4")
        self._body = self.raw[_payload(self.dtype)]

    @property
    def seq(self) -> int:
        return int(self._seq[0])

    def write(self, record: np.ndarray) -> int:
        """Publish ``record`` (a 0-d array of the region's dtype). Returns the new sequence number."""
        body = np.frombuffer(record.tobytes(), np.uint8)[_payload(self.dtype)]
        s = int(self._seq[0])
        if s % 2:
            s += 1  # a writer died mid-write; start a fresh even generation
        self._seq[0] = s + 1
        self._body[:] = body
        self._crc[0] = zlib.crc32(body)
        self._seq[0] = s + 2
        return s + 2

    def read(self, attempts: int = 1000) -> tuple[int, np.ndarray] | None:
        """``(seq, record)`` of the latest complete write, or None if never written. Raises :class:`StaleRead`
        when no attempt finds a complete write, as when the writer was descheduled mid-write: a periodic reader
        keeps its last record and reads again next cycle."""
        for _ in range(attempts):
            s0 = int(self._seq[0])
            if s0 == 0:
                return None
            if s0 % 2:
                continue
            body = self._body.copy()
            crc = int(self._crc[0])
            if int(self._seq[0]) == s0 and zlib.crc32(body) == crc:
                buf = bytearray(self.dtype.itemsize)
                np.frombuffer(buf, np.uint8)[_payload(self.dtype)] = body
                rec = np.frombuffer(buf, self.dtype).reshape(())
                rec["seq"], rec["crc"] = s0, crc
                return s0, rec
        raise StaleRead(f"{self.name}: no consistent copy after {attempts} attempts")


class Segment:
    """The mapped segment file. :meth:`create` (the sim) or :meth:`open` (the client)."""

    def __init__(self, path: Path, mm: mmap.mmap, fd: int):
        self.path, self._mm, self._fd = path, mm, fd
        self.buf = np.frombuffer(mm, dtype=np.uint8)
        self.header = self.buf[: HEADER.itemsize].view(
            HEADER
        )  # shape (1,): field writes go to the file
        self.regions = {
            name: Region(name, dtype, self.buf[OFFSETS[name] : OFFSETS[name] + dtype.itemsize])
            for name, dtype in REGIONS
        }

    # -- lifecycle --------------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        robot_id: str,
        description_digest: str,
        sim_revision: str,
        num_drives: int,
        num_joints: int,
        physics_period_ns: int,
        bus_period_ns: int,
        descriptor: dict,
        session_id: int | None = None,
        ready: bool = True,
    ) -> Segment:
        """Create (replacing any old file) and fill the header and descriptor. With ``ready=False`` the writer
        publishes its first records before calling :meth:`mark_ready`, so a client never sees empty regions."""
        if num_drives > MAX_DRIVES or num_joints > MAX_JOINTS:
            raise AbiError(f"at most {MAX_DRIVES} drives and {MAX_JOINTS} joints")
        path = Path(path)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o660)
        # Owner and group read-write whatever the umask, so a client in the group can attach.
        os.fchmod(fd, 0o660)
        os.ftruncate(fd, SEGMENT_BYTES)
        mm = mmap.mmap(fd, SEGMENT_BYTES)
        seg = cls(path, mm, fd)
        h = seg.header
        h["abi_major"], h["abi_minor"], h["header_bytes"] = ABI_MAJOR, ABI_MINOR, HEADER.itemsize
        h["layout_hash"], h["segment_bytes"] = LAYOUT_HASH, SEGMENT_BYTES
        h["session_id"] = (
            session_id if session_id is not None else int.from_bytes(os.urandom(8), "little") or 1
        )
        h["state"] = STATE_CREATING
        h["num_drives"], h["num_joints"], h["imu_samples"] = num_drives, num_joints, IMU_SAMPLES
        h["physics_period_ns"], h["bus_period_ns"] = physics_period_ns, bus_period_ns
        h["heartbeat_mono_ns"] = mono_ns()
        h["robot_id"] = robot_id.encode()
        h["description_digest"] = description_digest.encode()
        h["sim_revision"] = sim_revision.encode()[:48]
        h["offsets"] = [OFFSETS[name] for name, _ in REGIONS]
        seg.write_descriptor(descriptor)
        h["magic"] = MAGIC  # last: an opener never sees a half-filled header as valid
        os.replace(tmp, path)
        if ready:
            seg.mark_ready()
        return seg

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        robot_id: str | None = None,
        description_digest: str | None = None,
        wait_s: float = 0.0,
    ) -> Segment:
        """Attach to a ready segment, checking magic, ABI major, layout hash, size and the expected identity."""
        path = Path(path)
        deadline = time.monotonic() + wait_s
        while True:
            try:
                seg = cls._open_once(path)
                state = seg.state
                if state == STATE_READY:
                    break
                seg.close()
                err = AbiError(f"{path}: segment not ready (state {state})")
            except (FileNotFoundError, AbiError) as e:
                err = e
            if time.monotonic() >= deadline:
                raise err
            time.sleep(0.01)
        h = seg.header
        if robot_id is not None and h["robot_id"][0].decode() != robot_id:
            seg.close()
            raise AbiError(f"robot id {h['robot_id'][0].decode()!r} != expected {robot_id!r}")
        if (
            description_digest is not None
            and h["description_digest"][0].decode() != description_digest
        ):
            seg.close()
            raise AbiError(
                "description digest differs: the sim and the client read different descriptions"
            )
        return seg

    @classmethod
    def _open_once(cls, path: Path) -> Segment:
        fd = os.open(path, os.O_RDWR)
        size = os.fstat(fd).st_size
        if size != SEGMENT_BYTES:
            os.close(fd)
            raise AbiError(f"{path}: {size} bytes, this ABI expects {SEGMENT_BYTES}")
        seg = cls(path, mmap.mmap(fd, size), fd)
        h = seg.header
        problems = []
        if bytes(h["magic"][0]) != MAGIC:
            problems.append("bad magic")
        if int(h["abi_major"][0]) != ABI_MAJOR:
            problems.append(f"ABI {int(h['abi_major'][0])}.x, this side speaks {ABI_MAJOR}.x")
        if int(h["layout_hash"][0]) != LAYOUT_HASH:
            problems.append("layout hash differs")
        if problems:
            seg.close()
            raise AbiError(f"{path}: " + "; ".join(problems))
        return seg

    def close(self) -> None:
        self.header = None
        self.regions = {}
        self.buf = None
        try:
            self._mm.close()
        except BufferError:
            pass  # a view is still alive; the mapping goes when it does
        os.close(self._fd)

    # -- header -----------------------------------------------------------------------------------

    @property
    def state(self) -> int:
        return int(self.header["state"][0])

    @property
    def session_id(self) -> int:
        return int(self.header["session_id"][0])

    @property
    def num_drives(self) -> int:
        return int(self.header["num_drives"][0])

    @property
    def num_joints(self) -> int:
        return int(self.header["num_joints"][0])

    def heartbeat(self) -> None:
        self.header["heartbeat_mono_ns"] = mono_ns()

    def heartbeat_age_ns(self) -> int:
        return mono_ns() - int(self.header["heartbeat_mono_ns"][0])

    def mark_ready(self) -> None:
        self.header["state"] = STATE_READY

    def end(self) -> None:
        self.header["state"] = STATE_ENDED

    # -- regions ----------------------------------------------------------------------------------

    def write_descriptor(self, descriptor: dict) -> None:
        data = json.dumps(descriptor, sort_keys=True).encode()
        if len(data) > DESCRIPTOR_BYTES:
            raise AbiError(f"descriptor is {len(data)} bytes, at most {DESCRIPTOR_BYTES}")
        rec = np.zeros((), DESCRIPTOR)
        rec["length"] = len(data)
        rec["json"][: len(data)] = np.frombuffer(data, np.uint8)
        self.regions["descriptor"].write(rec)

    def descriptor(self) -> dict:
        got = self.regions["descriptor"].read()
        if got is None:
            raise AbiError("no descriptor")
        rec = got[1]
        return json.loads(bytes(rec["json"][: int(rec["length"])]).decode())

    def send(self, command: int, args: dict | None = None) -> int:
        """Client: post a command; returns its sequence number (wait for it with :meth:`wait_ack`)."""
        data = json.dumps(args or {}).encode()
        if len(data) > MAILBOX_BYTES:
            raise AbiError(f"command arguments are {len(data)} bytes, at most {MAILBOX_BYTES}")
        rec = np.zeros((), MAILBOX)
        rec["command"], rec["length"], rec["written_mono_ns"] = command, len(data), mono_ns()
        rec["args"][: len(data)] = np.frombuffer(data, np.uint8)
        return self.regions["mailbox"].write(rec)

    def poll_command(self, last_seq: int) -> tuple[int, int, dict] | None:
        """Sim: ``(seq, command, args)`` of a command newer than ``last_seq``, or None (also while the client is
        writing one). Malformed JSON gives ``args = None``."""
        try:
            got = self.regions["mailbox"].read()
        except StaleRead:
            return None
        if got is None or got[0] <= last_seq:
            return None
        seq, rec = got
        try:
            args = json.loads(bytes(rec["args"][: int(rec["length"])]).decode() or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            args = None
        return seq, int(rec["command"]), args

    def ack(self, seq: int, result: int, message: str = "") -> None:
        rec = np.zeros((), ACK)
        rec["acked"], rec["result"] = seq, result
        rec["message"] = message.encode()[:ACK_MESSAGE_BYTES]
        self.regions["ack"].write(rec)

    def wait_ack(self, seq: int, timeout_s: float = 5.0) -> tuple[int, str]:
        """Client: ``(result, message)`` once the sim has handled command ``seq``."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                got = self.regions["ack"].read()
            except StaleRead:  # the sim is writing it
                continue
            if got is not None and int(got[1]["acked"]) >= seq:
                rec = got[1]
                if int(rec["acked"]) == seq:
                    return int(rec["result"]), rec["message"].item().decode()
                raise AbiError(f"command {seq} was superseded before it was handled")
            time.sleep(0.001)
        raise TimeoutError(f"no ack for command {seq} within {timeout_s} s")


def pd_demand(
    target_position_rad, stiffness_nm_per_rad, damping_nm_s_per_rad, *, law: int = LAW_PD, **fields
) -> dict[str, np.ndarray]:
    """Per-drive arrays for :func:`demands_record`: every drive on ``law`` about its target at the given gains, no
    velocity target, feed-forward or torque limit unless ``fields`` names them."""
    target = np.asarray(target_position_rad, dtype=np.float64)
    n = len(target)
    demand = {name: np.zeros(n) for name in DRIVE_DEMAND.names if not name.startswith("_")}
    demand |= {
        "law": np.full(n, law),
        "target_position_rad": target.copy(),
        "torque_limit_nm": np.full(n, np.inf),
        "stiffness_nm_per_rad": np.broadcast_to(stiffness_nm_per_rad, n).astype(np.float64),
        "damping_nm_s_per_rad": np.broadcast_to(damping_nm_s_per_rad, n).astype(np.float64),
    }
    return demand | {k: np.broadcast_to(v, n).copy() for k, v in fields.items()}


def demands_record(demand: dict[str, np.ndarray], *, cycle: int, session_id: int) -> np.ndarray:
    """Build a demands record from per-drive arrays named as :data:`DRIVE_DEMAND`'s fields."""
    n = len(demand["law"])
    rec = np.zeros((), DEMANDS)
    rec["cycle"], rec["written_mono_ns"], rec["session_id"], rec["num_drives"] = (
        cycle,
        mono_ns(),
        session_id,
        n,
    )
    for name in DRIVE_DEMAND.names:
        if not name.startswith("_"):
            rec["drive"][name][:n] = demand[name]
    return rec
