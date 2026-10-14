"""Load a pinned V1 device inventory from the authoritative robot description.

This builds device configuration, not a fully qualified V1 physics scene. The
ankle drives remain explicit coupled devices; they are never assigned to a
single semantic ankle joint. No bus addresses are invented from list order.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .config import read_mapping
from .contracts import DriveConfig


@dataclass(frozen=True)
class DriveSpec:
    name: str
    family: str
    joints: tuple[str, ...]
    sign: int | None
    crank: str | None
    config: DriveConfig


@dataclass(frozen=True)
class RobotDescription:
    name: str
    source_revision: str
    urdf: Path
    drives: tuple[DriveSpec, ...]
    firmware_period_ns: int
    actuator_data: dict


def _verified_file(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("description path escapes root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"description content differs from pinned profile: {relative}")
    return path


def load_v1(description_root: str | Path, profile_path: str | Path) -> RobotDescription:
    """Read files from an explicit checkout, verifying their pinned content hashes.

    Canonical actuator_data retains torque-speed tables, calibration and source
    metadata. The first firmware slice uses peak caps only; it does not claim
    the complete table, thermal protection or calibrated ankle plant is active.
    """
    profile = read_mapping(profile_path)
    if profile.get("schema") != "humanoid-sim-robot/v1":
        raise ValueError("unsupported robot profile schema")
    root = Path(description_root)
    files = profile["files"]
    act_path = _verified_file(root, files["actuators"]["path"], files["actuators"]["sha256"])
    urdf = _verified_file(root, files["urdf"]["path"], files["urdf"]["sha256"])
    data = read_mapping(act_path)
    if data.get("schema") != "ethrc-actuators/v1":
        raise ValueError("unsupported actuator description schema")
    drives = []
    baselines = data["baseline_position_gains"]
    entries = []
    for joint, entry in data["joints"].items():
        if entry.get("device"):
            entries.append((entry["device"], entry, (joint,), entry["sign"], None))
    for name, entry in data["drives"].items():
        entries.append((name, entry, tuple(entry["joints"]), None, entry["crank"]))
    periods = set()
    for name, entry, joints, sign, crank in entries:
        rate = data["families"][entry["family"]]["controller_rate_hz"]
        if type(rate) is not int or rate <= 0 or 1_000_000_000 % rate:
            raise ValueError("controller rate must resolve to integer nanoseconds")
        periods.add(1_000_000_000 // rate)
        gains = baselines[name]
        config = DriveConfig(
            name=name,
            stiffness=gains["stiffness"],
            damping=gains["damping"],
            torque_limit=entry["effort"],
            watchdog_ns=profile["modeled_firmware"]["watchdog_ns"],
            command_delay_ns=profile["modeled_firmware"]["command_delay_ns"],
        )
        drives.append(DriveSpec(name, entry["family"], joints, sign, crank, config))
    if len(periods) != 1:
        raise NotImplementedError("initial bank requires one firmware rate")
    if len(drives) != profile["expected_drive_count"] or len({d.name for d in drives}) != len(drives):
        raise ValueError("device inventory mismatch")
    return RobotDescription(
        profile["robot"], profile["source"]["revision"], urdf, tuple(drives), periods.pop(), data
    )
