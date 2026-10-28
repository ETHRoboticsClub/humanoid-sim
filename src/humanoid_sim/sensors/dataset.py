"""A rendered sensor dataset: what the rig's cameras and lidars saw along a recorded run, at their nominal rates, every
stamp the run's plant time (integer ns, the clock of the trajectory, the runtime's feedback and its IMU samples).

Layout under the dataset directory (``humanoid-sim-sensor-dataset/v1``):

* ``manifest.json``: the robot and description digest, the trajectory the sensors were rendered from, and per sensor
  its streams with their encoding, nominal rate, calibration (the camera feed descriptor's ``calibration``) and mount.
* ``cameras/<camera_id>/<stream>/index.csv`` and ``<seq>.png``: ``seq,stamp_ns,file``. Mono is 8-bit grey, colour
  8-bit RGB, depth 16-bit millimetres (0: no depth).
* ``lidars/<name>/index.csv`` and ``<seq>.npz``: one sweep each: ``points_m`` ``[N, 3]`` in the sensor frame in firing
  order (zero where a ray did not return), ``hit``, ``time_s`` (each ray after the sweep start), ``start_ns`` and
  ``origin_w`` / ``rotation_w``: the sensor's world pose at the render that completed the sweep, so at the sweep's END
  (about ``start_ns`` plus the sweep period), not at ``start_ns``. A ray fired at ``start_ns + time_s`` is best placed
  with the sensor's pose at that time, from the trajectory.

The camera frame's stamp is the plant time it was rendered at; a lidar sweep's is the plant time its first ray fired.
"""

from __future__ import annotations

import csv
import json
import struct
import zlib
from pathlib import Path

import numpy as np

SCHEMA = "humanoid-sim-sensor-dataset/v1"


def png_bytes(image: np.ndarray) -> bytes:
    """A PNG of ``image``: ``[H, W]`` uint8 (grey), ``[H, W]`` uint16 (16-bit grey) or ``[H, W, 3]`` uint8 (RGB)."""
    image = np.ascontiguousarray(image)
    if image.ndim == 2 and image.dtype == np.uint8:
        depth, color, rows = 8, 0, image
    elif image.ndim == 2 and image.dtype == np.uint16:
        depth, color, rows = 16, 0, image.astype(">u2")
    elif image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8:
        depth, color, rows = 8, 2, image
    else:
        raise ValueError(f"cannot write a {image.dtype} image of shape {image.shape} as PNG")
    height, width = image.shape[:2]
    raw = b"".join(b"\x00" + rows[y].tobytes() for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


class _Index:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.dir = directory
        self._file = open(directory / "index.csv", "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["seq", "stamp_ns", "file"])
        self.count = 0

    def add(self, stamp_ns: int, suffix: str, payload: bytes) -> None:
        name = f"{self.count:06d}.{suffix}"
        (self.dir / name).write_bytes(payload)
        self._csv.writerow([self.count, int(stamp_ns), name])
        self.count += 1

    def close(self) -> None:
        self._file.close()


class DatasetWriter:
    """Writes one dataset. A directory that already holds a manifest of the same run (robot, description, trajectory and
    time range) is added to: a render with the splat for the cameras and one without it for the lidar share a directory
    (a lidar must not trace the splat), each keeping the other's sensors in the manifest. A manifest of another run is
    an error."""

    def __init__(self, root: str | Path, manifest: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = {"schema": SCHEMA, **manifest, "cameras": {}, "lidars": {}}
        self._previous_counts: dict[str, int] = {}
        path = self.root / "manifest.json"
        if path.exists():
            old = json.loads(path.read_text())
            same = ("schema", "robot", "description_digest", "trajectory", "start_ns", "end_ns")
            differing = [k for k in same if old.get(k) != self.manifest.get(k)]
            if differing:
                raise ValueError(f"{path} is a dataset of another run (differs in {differing}): use a new directory")
            self.manifest["cameras"], self.manifest["lidars"] = old["cameras"], old["lidars"]
            self._previous_counts = old.get("counts", {})
        self._indexes: dict[tuple[str, str, str], _Index] = {}

    def _index(self, kind: str, *parts: str) -> _Index:
        key = (kind, *parts)
        if key not in self._indexes:
            self._indexes[key] = _Index(self.root.joinpath(kind, *parts))
        return self._indexes[key]

    def describe_camera(self, camera_id: str, record: dict) -> None:
        self.manifest["cameras"][camera_id] = record

    def describe_lidar(self, name: str, record: dict) -> None:
        self.manifest["lidars"][name] = record

    def camera_frame(self, camera_id: str, stream: str, stamp_ns: int, image: np.ndarray) -> None:
        self._index("cameras", camera_id, stream).add(stamp_ns, "png", png_bytes(image))

    def lidar_sweep(self, name: str, sweep) -> None:
        import io

        buf = io.BytesIO()
        np.savez_compressed(buf, points_m=sweep.points_m.astype(np.float32), hit=sweep.hit,
                            time_s=sweep.time_s.astype(np.float32), start_ns=np.int64(sweep.start_ns),
                            origin_w=np.asarray(sweep.origin_w, dtype=np.float64),
                            rotation_w=np.asarray(sweep.rotation_w, dtype=np.float64))
        self._index("lidars", name).add(sweep.start_ns, "npz", buf.getvalue())

    def counts(self) -> dict[str, int]:
        return {"/".join(k[1:]): idx.count for k, idx in self._indexes.items()}

    def close(self) -> None:
        for idx in self._indexes.values():
            idx.close()
        self.manifest["counts"] = {**self._previous_counts, **self.counts()}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=1))
