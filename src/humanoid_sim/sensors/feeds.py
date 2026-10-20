"""Simulated camera devices as Controls' camera service reads them: the ``ethrc-camera-sim-feed/v1`` file rings
(``ethrc_cameras/sim_feed.py`` in Controls, the normative layout this module writes; ``RING_VERSION`` 2).

One directory per device under the feed root, named by its camera id, holding a ring file per stream and
``descriptor.json``, written last: rings first, each under a temporary name renamed into place, then the descriptor
the same way, so a present descriptor means a complete device. A ring is a 128-byte header and a fixed number of
slots; each slot is published under a sequence lock (odd ``seq_begin``, the stamp, entry number and payload, then
``seq_end`` and ``seq_begin`` even, then the header's ``published`` count). Every 8-byte field is written as one aligned
store; on x86-64 (total store order) that orders them as the contract requires, as Controls' Python reference
implementation relies on too. ``heartbeat_s`` is ``CLOCK_MONOTONIC``.

Stamps are exposure times on the clock the stack's ROS time runs on: version 2 of the rings stamps in ROS time, and
humanoid-sim has no ROS, so :func:`ros_time_s` moves a plant time onto the host's realtime clock (the stack's ROS
time when it runs on the wall clock, as the runtime does) through the plant's mapping to ``CLOCK_MONOTONIC``
(``runtime.md``, "The clock"). Controls' plan moves the rings to monotonic stamps (version 3); this module follows
when it does.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.sensors.geometry import CameraSpec, Stream

DESCRIPTOR_SCHEMA = "ethrc-camera-sim-feed/v1"
RING_MAGIC = 0x46534345  # "ECSF"
RING_VERSION = 2
HEADER = struct.Struct("<II64sIII16sIIIQd")  # 128 bytes
SEQ = struct.Struct("<Q")
STAMP_N = struct.Struct("<dQ")
F64 = struct.Struct("<d")
SLOT_PAYLOAD_OFFSET = 24
PUBLISHED_OFFSET = 112
HEARTBEAT_OFFSET = 120
SLOT_ALIGN = 64
BYTES_PER_PIXEL = {"bgr8": 3, "rgb8": 3, "mono8": 1, "16UC1": 2}
DEFAULT_CAPACITY = 8


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def ros_time_s(plant_ns: int, now_plant_ns: int) -> float:
    """The realtime clock's reading (s) at plant time ``plant_ns``, given that the plant is at ``now_plant_ns`` now:
    the plant runs at the monotonic clock's rate while it keeps up, so its age now is wall age."""
    return time.time() - (now_plant_ns - plant_ns) * 1e-9


class Ring:
    """The producer's side of one ring file."""

    def __init__(self, path: Path, session: str, width: int, height: int, step: int, encoding: str,
                 capacity: int = DEFAULT_CAPACITY):
        self.payload_bytes = step * height
        self.capacity = capacity
        self.seq_end = _round_up(SLOT_PAYLOAD_OFFSET + self.payload_bytes, 8)
        self.stride = _round_up(self.seq_end + SEQ.size, SLOT_ALIGN)
        size = HEADER.size + capacity * self.stride
        pending = path.with_name(path.name + ".tmp")
        with open(pending, "wb") as f:
            f.truncate(size)
            f.write(HEADER.pack(RING_MAGIC, RING_VERSION, session.encode("ascii"), width, height, step,
                                encoding.encode("ascii"), capacity, self.payload_bytes, 0, 0, time.monotonic()))
        os.replace(pending, path)
        self._file = open(path, "r+b")
        self._map = mmap.mmap(self._file.fileno(), size)
        self.published = 0

    def publish(self, stamp_s: float, payload: bytes | memoryview) -> None:
        if len(payload) != self.payload_bytes:
            raise ValueError(f"payload of {len(payload)} bytes; the ring carries {self.payload_bytes}")
        n = self.published
        at = HEADER.size + (n % self.capacity) * self.stride
        m = self._map
        SEQ.pack_into(m, at, 2 * n + 1)
        STAMP_N.pack_into(m, at + SEQ.size, stamp_s, n)
        m[at + SLOT_PAYLOAD_OFFSET: at + SLOT_PAYLOAD_OFFSET + self.payload_bytes] = payload
        SEQ.pack_into(m, at + self.seq_end, 2 * n + 2)
        SEQ.pack_into(m, at, 2 * n + 2)
        self.published = n + 1
        SEQ.pack_into(m, PUBLISHED_OFFSET, self.published)
        self.heartbeat()

    def heartbeat(self) -> None:
        F64.pack_into(self._map, HEARTBEAT_OFFSET, time.monotonic())

    def close(self) -> None:
        self._map.close()
        self._file.close()


def _intrinsics(stream: Stream) -> dict:
    model, coeffs = stream.lens.contract_model()
    lens = stream.lens
    return {"width_px": lens.width_px, "height_px": lens.height_px, "fx_px": float(lens.fx_px),
            "fy_px": float(lens.fy_px), "cx_px": float(lens.cx_px), "cy_px": float(lens.cy_px),
            "distortion_model": model, "distortion": [float(c) for c in coeffs]}


def descriptor(spec: CameraSpec, session: str) -> dict:
    """The device's ``descriptor.json`` document."""
    streams = []
    for s in spec.streams:
        entry = {"name": s.name, "kind": s.kind, "width_px": s.lens.width_px, "height_px": s.lens.height_px,
                 "rate_hz": float(s.rate_hz), "shutter": "global", "sync_group": s.sync_group,
                 "depth_source": s.depth_source, "encoding": s.encoding}
        streams.append(entry)
    return {"schema": DESCRIPTOR_SCHEMA, "camera_id": spec.camera_id, "product": spec.product, "session": session,
            "streams": streams,
            "calibration": {"calibration_id": spec.calibration_id,
                            "intrinsics": {s.name: _intrinsics(s) for s in spec.streams},
                            "sensor_poses": {s.name: {"translation_m": [float(v) for v in s.translation_m],
                                                      "rotation_xyzw": [float(v) for v in s.rotation_xyzw]} for s in spec.streams},
                            "withheld": {}, "not_provided": {}},
            "imu_rate_hz": None}


@dataclass
class _StreamRing:
    stream: Stream
    ring: Ring
    step: int


class CameraFeed:
    """One simulated camera device's directory under ``root``: its rings, then its descriptor."""

    def __init__(self, root: str | Path, spec: CameraSpec, capacity: int = DEFAULT_CAPACITY):
        self.spec = spec
        self.dir = Path(root) / spec.camera_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "descriptor.json").unlink(missing_ok=True)  # a restarted producer: incomplete until written
        self.session = f"humanoid-sim-{uuid.uuid4().hex[:16]}"
        self.rings: dict[str, _StreamRing] = {}
        for s in spec.streams:
            bpp = BYTES_PER_PIXEL[s.encoding]
            step = s.lens.width_px * bpp
            ring = Ring(self.dir / f"{s.name}.ring", self.session, s.lens.width_px, s.lens.height_px, step,
                        s.encoding, capacity)
            self.rings[s.name] = _StreamRing(s, ring, step)
        pending = self.dir / "descriptor.json.tmp"
        pending.write_text(json.dumps(descriptor(spec, self.session), indent=1))
        os.replace(pending, self.dir / "descriptor.json")

    def publish(self, stream: str, stamp_s: float, image: np.ndarray) -> None:
        """``image``: ``[H, W, 3]`` uint8 RGB for colour streams (written in the stream's encoding), ``[H, W]`` uint16
        millimetres for depth, ``[H, W]`` uint8 for mono."""
        entry = self.rings[stream]
        s = entry.stream
        if s.encoding == "bgr8":
            image = image[..., ::-1]
        expect = (s.lens.height_px, s.lens.width_px)
        if image.shape[:2] != expect:
            raise ValueError(f"{stream}: image {image.shape[:2]}, the stream is {expect}")
        entry.ring.publish(stamp_s, np.ascontiguousarray(image).tobytes())

    def heartbeat(self) -> None:
        for entry in self.rings.values():
            entry.ring.heartbeat()

    def close(self) -> None:
        """A clean stop: the descriptor goes, then the rings are released (their files stay until replaced)."""
        (self.dir / "descriptor.json").unlink(missing_ok=True)
        for entry in self.rings.values():
            entry.ring.close()


def depth_mm(depth_m: np.ndarray, range_m: tuple[float, float]) -> np.ndarray:
    """Depth along the optical axis (m) as the 16UC1 millimetres a stereo camera reports: 0 where it has no depth
    (outside its range, or no return)."""
    d = np.asarray(depth_m, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    valid = np.isfinite(d) & (d >= range_m[0]) & (d <= range_m[1])
    return np.where(valid, np.round(d * 1000.0), 0).clip(0, 65535).astype(np.uint16)

