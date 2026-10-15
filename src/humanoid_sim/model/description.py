"""Robot description: humanoid-urdf, read hash-verified, never copied.

Every robot fact this package uses comes through here: URDF, ``actuators.yaml`` (drives, signs, gains,
envelopes, ankle geometry). The digest covers both files and is recorded with every output so a
result can be traced to the exact description.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from humanoid_sim.transmissions.ankle import AnkleTransmission, mirror_right

ACTUATORS_SCHEMA = "ethrc-actuators/v1"

# Robot id → directory in humanoid-urdf and the ament package inside it.
ROBOTS = {
    "ethrc_v1": ("ethrc_v1", "URDF_5_description"),
}


@dataclass(frozen=True)
class Drive:
    """One EtherCAT drive. ``joints`` are the URDF joints it moves (two for an ankle crank)."""

    name: str  # EtherCAT device name
    family: str
    joints: tuple[str, ...]
    sign: int  # device = sign * URDF joint (1:1 drives); +1 for cranks (crank frame is the device frame)
    crank: str | None  # "upper" | "lower" for ankle cranks
    effort_peak: float  # Nm at the drive output
    effort_nominal: float
    velocity: float  # rad/s no-load at 48 V
    position_limit_device: tuple[float, float] | None  # device frame, cranks only (joints: see Joint)
    baseline_stiffness: float  # Nm/rad at the drive output
    baseline_damping: float


@dataclass(frozen=True)
class Joint:
    name: str
    region: str
    device: str | None
    sign: int
    standing_offset_rad: float
    armature: float  # joint-space value (armature_joint for the ankle hinges)
    damping: float  # passive viscous, Nm s/rad
    friction: float  # passive Coulomb, Nm
    effort_peak: float
    velocity: float


class Description:
    def __init__(self, robot: str, root: Path):
        if robot not in ROBOTS:
            raise ValueError(f"unknown robot {robot!r}; known: {sorted(ROBOTS)}")
        self.robot = robot
        self.root = root
        directory, package = ROBOTS[robot]
        self.robot_dir = root / directory
        self.package_dir = self.robot_dir / package
        self.actuators_path = self.robot_dir / "actuators.yaml"
        self.table = yaml.safe_load(self.actuators_path.read_text())
        if self.table.get("schema") != ACTUATORS_SCHEMA:
            raise ValueError(f"{self.actuators_path}: schema {self.table.get('schema')!r} != {ACTUATORS_SCHEMA}")
        self.urdf_path = self.package_dir / self.table["urdf"]

    @staticmethod
    def default_root() -> Path:
        env = os.environ.get("HUMANOID_SIM_DESCRIPTION_ROOT")
        if env:
            return Path(env)
        return Path(__file__).resolve().parents[3] / "third_party" / "humanoid-urdf"

    @classmethod
    def load(cls, robot: str = "ethrc_v1", root: Path | None = None) -> Description:
        return cls(robot, Path(root) if root else cls.default_root())

    # -- identity ---------------------------------------------------------------------------------

    @cached_property
    def digest(self) -> str:
        """sha256 over the URDF and actuators.yaml bytes (meshes are covered by the git revision)."""
        h = hashlib.sha256()
        for path in (self.urdf_path, self.actuators_path):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    @cached_property
    def revision(self) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def identity(self) -> dict:
        return {"robot": self.robot, "description_digest": self.digest, "description_revision": self.revision}

    # -- inventory --------------------------------------------------------------------------------

    @cached_property
    def joints(self) -> dict[str, Joint]:
        out = {}
        for name, j in self.table["joints"].items():
            out[name] = Joint(
                name=name, region=j["region"], device=j.get("device"), sign=int(j["sign"]),
                standing_offset_rad=float(j["standing_offset_rad"]),
                armature=float(j.get("armature_joint", j["armature"])),
                damping=float(j["damping"]), friction=float(j["friction"]),
                effort_peak=float(j["effort"]), velocity=float(j["velocity"]),
            )
        return out

    @cached_property
    def drives(self) -> dict[str, Drive]:
        """All EtherCAT drives: the 1:1 joint drives, then the ankle cranks (description order)."""
        gains = self.table["baseline_position_gains"]
        out = {}
        for name, j in self.table["joints"].items():
            device = j.get("device")
            if device is None:
                continue
            out[device] = Drive(
                name=device, family=j["family"], joints=(name,), sign=int(j["sign"]), crank=None,
                effort_peak=float(j["effort"]), effort_nominal=float(j["effort_nominal"]),
                velocity=float(j["velocity"]), position_limit_device=None,
                baseline_stiffness=float(gains[device]["stiffness"]),
                baseline_damping=float(gains[device]["damping"]),
            )
        for device, d in self.table["drives"].items():
            out[device] = Drive(
                name=device, family=d["family"], joints=tuple(d["joints"]), sign=1, crank=d["crank"],
                effort_peak=float(d["effort"]), effort_nominal=float(d["effort_nominal"]),
                velocity=float(d["velocity"]), position_limit_device=tuple(d["position_limit"]),
                baseline_stiffness=float(gains[device]["stiffness"]),
                baseline_damping=float(gains[device]["damping"]),
            )
        missing = set(gains) - set(out)
        if missing:
            raise ValueError(f"baseline gains for unknown drives: {sorted(missing)}")
        return out

    def calibration_travel(self, name: str) -> tuple[float, float]:
        """Simulated travel in URDF (calibration) radians: the recorded mechanical stop where there is one,
        else the position limit. Same rule as Controls ``ethrc_ssot/joint_travel.py``."""
        entry = self.table["joints"][name]
        offset = float(entry["standing_offset_rad"])
        lo, hi = (float(b) + offset for b in entry["position_limit"])
        stop = entry.get("mechanical_stop_calibration_rad")
        if stop is None:
            return lo, hi
        stop_lo = lo if stop[0] is None else float(stop[0])
        stop_hi = hi if stop[1] is None else float(stop[1])
        if stop_lo > lo or stop_hi < hi:
            raise ValueError(f"{name}: mechanical stop lies inside the position limit")
        return stop_lo, stop_hi

    def imu_mount(self) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        """The IMU's parent link and its pose there (xyz in m, URDF rpy in rad): the URDF's ``imu_joint``."""
        import xml.etree.ElementTree as ET

        joint = next((j for j in ET.parse(self.urdf_path).getroot().findall("joint")
                      if j.get("name") == "imu_joint"), None)
        if joint is None:
            raise ValueError(f"{self.urdf_path.name} has no imu_joint")
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        return joint.find("parent").get("link"), xyz, rpy

    def torque_speed(self, family: str) -> list[tuple[float, float]]:
        return [tuple(p) for p in self.table["families"][family]["torque_speed"]]

    def family(self, family: str) -> dict:
        return self.table["families"][family]

    # -- ankle ------------------------------------------------------------------------------------

    def ankle_geometry(self, side: str) -> dict:
        right = self.table["ankle_model"]["right_geometry"]
        return right if side == "r" else mirror_right(right)

    def ankle_kinematic_sign(self, side: str) -> list[int]:
        return [int(s) for s in self.table["ankle_model"]["semantic_to_kinematic_sign"][side]]

    def ankle_transmission(self, side: str, *, device="cpu", dtype=None) -> AnkleTransmission:
        import torch

        return AnkleTransmission(self.ankle_geometry(side), self.ankle_kinematic_sign(side), device=device,
                                 dtype=dtype or torch.float32)

    def ankle_cranks(self, side: str) -> tuple[str, str]:
        """(upper, lower) crank drive names of one side."""
        joints = {f"foot_pitch_{side}", f"foot_roll_{side}"}
        by_crank = {d.crank: d.name for d in self.drives.values() if d.crank and set(d.joints) == joints}
        return by_crank["upper"], by_crank["lower"]

    def ankle_model_module(self):
        """humanoid-urdf's own ``ankle_model`` module (numpy reference), loaded from the submodule."""
        path = self.package_dir / self.package_dir.name / "ankle_model.py"
        spec = importlib.util.spec_from_file_location(f"_hu_ankle_model_{self.robot}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
