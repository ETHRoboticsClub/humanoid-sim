"""Explicit, sourced capture-to-description position translations, outside physics.

Only positions translate. Gains, velocities, effort, clocks and fitted encoder
biases retain their meanings. No robot-specific offsets are embedded here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "humanoid-sim-capture-coordinates/v1"
IDENTITY_KEYS = ("robot_id", "unit_uuid", "model_revision", "calibration_revision")


@dataclass(frozen=True)
class CaptureCoordinates:
    """Model-drive = capture-drive + offset; model-standing = capture-standing + offset."""

    source_identity: tuple[str, ...]
    target_digest: str
    drive_offsets: tuple[tuple[str, float], ...]
    command_offsets: tuple[tuple[str, float], ...]
    provenance_json: str
    sha256: str

    @classmethod
    def load(cls, path: str | Path, desc):
        return cls.from_document(json.loads(Path(path).read_text()), desc)

    @classmethod
    def from_document(cls, document, desc):
        if document.get("schema") != SCHEMA:
            raise ValueError("unsupported capture coordinate schema")
        source = document.get("source_identity", {})
        if any(not isinstance(source.get(k), str) or not source[k] for k in IDENTITY_KEYS):
            raise ValueError("capture coordinate source identity is incomplete")
        provenance = document.get("provenance")
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("capture coordinate provenance is required")
        values = []
        for key, names in (("drive_offsets_rad", desc.drives), ("command_offsets_rad", desc.joints)):
            table = document.get(key, {})
            if set(table) != set(names):
                raise ValueError(f"capture coordinate {key} inventory mismatch")
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                for v in table.values()
            ):
                raise ValueError("capture coordinate offsets must be finite numbers")
            values.append(tuple((n, float(table[n])) for n in names))
        # Crank neutral translation already belongs to AnkleTransmission, not a second motor offset.
        if any(dict(values[0])[n] != 0 for n, d in desc.drives.items() if d.crank is not None):
            raise ValueError("capture coordinate crank offsets must be zero; use the linkage neutral")
        canonical = json.dumps(document, sort_keys=True, allow_nan=False)
        result = cls(
            tuple(source[k] for k in IDENTITY_KEYS),
            document.get("target_description_digest", ""),
            *values,
            json.dumps(provenance, sort_keys=True),
            hashlib.sha256(canonical.encode()).hexdigest(),
        )
        result.validate_target(desc)
        return result

    def validate_target(self, desc):
        if self.target_digest != desc.digest:
            raise ValueError("capture coordinate target description digest mismatch")
        if set(dict(self.drive_offsets)) != set(desc.drives) or set(dict(self.command_offsets)) != set(
            desc.joints
        ):
            raise ValueError("capture coordinate target inventory mismatch")
        reference = desc.table.get("standing_reference", {})
        for key in ("robot_id", "unit_uuid"):
            if reference.get(key) != self.source_identity[IDENTITY_KEYS.index(key)]:
                raise ValueError(f"capture coordinate target {key} mismatch")

    def validate_source(self, manifest):
        identity = manifest.get("identity", {})
        if tuple(identity.get(k) for k in IDENTITY_KEYS) != self.source_identity:
            raise ValueError("capture coordinate source identity mismatch")

    def metadata(self):
        return {
            "schema": SCHEMA,
            "sha256": self.sha256,
            "source_identity": dict(zip(IDENTITY_KEYS, self.source_identity, strict=True)),
            "target_description_digest": self.target_digest,
            "drive_offsets_rad": dict(self.drive_offsets),
            "command_offsets_rad": dict(self.command_offsets),
            "provenance": json.loads(self.provenance_json),
            "output_frame": "current description drive/standing coordinates; raw files unchanged",
        }


def validate_capture_coordinates(capture, desc):
    """Fail closed on identified incompatible captures. Unidentified legacy/test readers retain their API."""
    coordinates = getattr(capture, "coordinates", None)
    identity = getattr(capture, "manifest", {}).get("identity")
    if coordinates is not None:
        coordinates.validate_source(capture.manifest)
        coordinates.validate_target(desc)
        return coordinates.metadata()
    if identity:
        target = desc.table.get("standing_reference", {})
        if any(identity.get(k) != target.get(k) or not identity.get(k) for k in IDENTITY_KEYS):
            raise ValueError(
                "capture identity differs from description; an explicit capture coordinate artifact is required"
            )
    return {"output_frame": "description native (no translation)", "identified": bool(identity)}
