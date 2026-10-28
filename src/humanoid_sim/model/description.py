"""Robot description: humanoid-urdf, read hash-verified, never copied.

Every robot fact this package uses comes through here: URDF, ``actuators.yaml`` (drives, signs, gains,
envelopes, linkage geometry) and the named poses the description records. The digest covers the URDF and
``actuators.yaml``; it and the description's git revision are recorded with every output so a result can be traced
to the exact description.

A robot is a directory of the description root that holds an ``actuators.yaml`` commanding its joints; its id is
the directory name (``ethrc_v1``, ``g1_29dof_rev_1_0``). A description whose joints are only observed (a
``lowstate_index`` and no command address: the Go2, which walks under Unitree's controller) is not a robot here. The ament package is the subdirectory holding the URDF that ``actuators.yaml``
names. Both actuator layouts in humanoid-urdf load: drives named by EtherCAT device with a gain table and crank
linkages (ETHRC), and one motor per joint with gains on the joint and passive terms on the motor family (Unitree).
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from humanoid_sim.transmissions.ankle import AnkleTransmission, mirror_right

ACTUATORS_SCHEMA = "ethrc-actuators/v1"

PRESETS_SCHEMA = "ethrc-position-presets/v1"


def robots(root: Path | None = None) -> list[str]:
    """Robot ids under a description root: its directories whose ``actuators.yaml`` commands its joints."""
    root = Path(root) if root else Description.default_root()
    found = (d for d in root.iterdir() if (d / "actuators.yaml").is_file())
    return sorted(d.name for d in found if _commanded(d / "actuators.yaml"))


def _commanded(actuators: Path) -> bool:
    """Whether any joint has a command address: an EtherCAT device, a LowCmd slot or a coupled drive."""
    document = yaml.safe_load(actuators.read_text())
    if document.get("drives"):
        return True
    return any("device" in j or "motor_index" in j for j in document.get("joints", {}).values())


@dataclass(frozen=True)
class Drive:
    """One drive (an EtherCAT drive, a Unitree motor). ``joints`` are the URDF joints it moves (two for a crank)."""

    name: str  # EtherCAT device name, or the joint name for a robot whose motors have no device names
    family: str
    joints: tuple[str, ...]
    sign: int  # device = sign * URDF joint (1:1 drives); +1 for cranks (crank frame is the device frame)
    crank: str | None  # "upper" | "lower" for linkage cranks
    effort_peak: float  # Nm at the drive output
    effort_nominal: float | None
    velocity: float  # rad/s no-load
    position_limit_device: tuple[float, float] | None  # device frame, cranks only (joints: see Joint)
    baseline_stiffness: float  # Nm/rad at the drive output
    baseline_damping: float
    motor_index: int | None = None  # slot in the backend's motor array (Unitree LowCmd), when it has one


@dataclass(frozen=True)
class Joint:
    name: str
    region: str
    device: str | None
    sign: int
    standing_offset_rad: float
    armature: float  # joint-space value (armature_joint for the ankle hinges)
    damping: float  # passive viscous, Nm s/rad (the joint's, else its motor family's)
    friction: float  # passive Coulomb, Nm (the joint's, else its motor family's): static, what holds a joint at rest
    effort_peak: float
    velocity: float
    dynamic_friction: float  # Coulomb friction while sliding, Nm (``dynamic_friction``, else ``friction``)


@dataclass(frozen=True)
class Linkage:
    """Two crank drives that move two hinges together (an ETHRC ankle)."""

    side: str  # "l" | "r"
    upper: str  # crank drive names
    lower: str
    joints: tuple[str, str]  # (pitch, roll) hinges


class Description:
    def __init__(self, robot: str, root: Path):
        self.robot = robot
        self.root = root
        self.robot_dir = root / robot
        self.actuators_path = self.robot_dir / "actuators.yaml"
        if not self.actuators_path.is_file():
            raise ValueError(f"unknown robot {robot!r}; known: {robots(root)}")
        self.table = yaml.safe_load(self.actuators_path.read_text())
        if self.table.get("schema") != ACTUATORS_SCHEMA:
            raise ValueError(f"{self.actuators_path}: schema {self.table.get('schema')!r} != {ACTUATORS_SCHEMA}")
        rel = self.table["urdf"]
        packages = [d for d in sorted(self.robot_dir.iterdir()) if d.is_dir() and (d / rel).is_file()]
        if len(packages) != 1:
            raise ValueError(f"{robot}: {len(packages)} packages hold {rel}, expected one")
        self.package_dir = packages[0]
        self.urdf_path = self.package_dir / rel

    @cached_property
    def usd_path(self) -> Path | None:
        """Delivered USD selected explicitly by this robot's model profile, if any."""
        from humanoid_sim.model.asset import load_profile

        relative = load_profile(self.robot)["model"].get("usd")
        if relative is None:
            return None
        path = self.package_dir / relative
        if not path.is_file():
            raise ValueError(f"profile selects missing USD description {path}")
        return path

    @cached_property
    def usd_model(self):
        from humanoid_sim.model.usd import UsdDescription

        return None if self.usd_path is None else UsdDescription(self.usd_path)

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
        """sha256 over the URDF and actuators.yaml (meshes are covered by the git revision); the runtime
        segment's definition, so a client can check it from its own checkout."""
        from humanoid_sim.runtime.abi.plant_abi import description_digest

        return description_digest(self.urdf_path, self.actuators_path)

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
        families = self.table.get("families", {})
        for name, j in self.table["joints"].items():
            family = families.get(j.get("family"), {})

            def value(key, j=j, family=family, name=name):
                v = j.get(key, family.get(key))
                if v is None:
                    raise ValueError(f"{name}: no {key!r} on the joint or its family")
                return float(v)

            friction = value("friction")
            # A family's dynamic friction goes with the family's static one, never with a joint's own.
            own = "friction" in j
            dynamic = float(j.get("dynamic_friction", friction if own else family.get("dynamic_friction", friction)))
            if not 0.0 <= dynamic <= friction:
                raise ValueError(f"{name}: dynamic_friction {dynamic} outside [0, friction {friction}]")
            out[name] = Joint(
                name=name, region=j["region"], device=j.get("device", name), sign=int(j["sign"]),
                standing_offset_rad=float(j["standing_offset_rad"]),
                armature=float(j["armature_joint"]) if "armature_joint" in j else value("armature"),
                damping=value("damping"), friction=friction,
                effort_peak=float(j["effort"]), velocity=float(j["velocity"]), dynamic_friction=dynamic,
            )
        return out

    def preset_pose(self, preset_id: str) -> dict[str, float]:
        """URDF joint positions (rad) of a named preset in the robot's ``position-presets.yaml``; joints the preset
        does not list stay at zero (its ``base: zero``)."""
        path = self.robot_dir / "position-presets.yaml"
        data = yaml.safe_load(path.read_text())
        if data.get("schema") != PRESETS_SCHEMA:
            raise ValueError(f"{path}: schema {data.get('schema')!r} != {PRESETS_SCHEMA}")
        preset = next((p for p in data["presets"] if p["preset_id"] == preset_id), None)
        if preset is None:
            raise ValueError(f"{path} has no preset {preset_id!r}")
        if preset.get("base", "zero") != "zero":
            raise ValueError(f"{preset_id}: only presets based on zero are supported")
        unknown = sorted(set(preset["positions_rad"]) - set(self.joints))
        if unknown:
            raise ValueError(f"{preset_id} names joints the description lacks: {unknown}")
        return {n: float(preset["positions_rad"].get(n, 0.0)) for n in self.joints}

    @cached_property
    def drives(self) -> dict[str, Drive]:
        """All drives: the 1:1 joint drives in joint order, then the linkage cranks (description order).

        A joint with a ``device`` field is driven by that EtherCAT device (``device: null``: by linkage cranks
        under ``drives:``); a joint without one has its own motor, named after the joint. Baseline gains come from
        ``baseline_position_gains`` (by device) when the description has that table, else from the joint."""
        gains = self.table.get("baseline_position_gains")

        def gain(device, entry):
            if gains is not None:
                if device not in gains:
                    raise ValueError(f"no baseline_position_gains for {device}")
                return float(gains[device]["stiffness"]), float(gains[device]["damping"])
            return float(entry["baseline_stiffness"]), float(entry["baseline_damping"])

        out = {}
        for name, j in self.table["joints"].items():
            device = j.get("device", name)
            if device is None:
                continue
            k, d = gain(device, j)
            out[device] = Drive(
                name=device, family=j["family"], joints=(name,), sign=int(j["sign"]), crank=None,
                effort_peak=float(j["effort"]),
                effort_nominal=float(j["effort_nominal"]) if "effort_nominal" in j else None,
                velocity=float(j["velocity"]), position_limit_device=None, baseline_stiffness=k, baseline_damping=d,
                motor_index=int(j["motor_index"]) if "motor_index" in j else None,
            )
        for device, d in self.table.get("drives", {}).items():
            k, dmp = gain(device, d)
            out[device] = Drive(
                name=device, family=d["family"], joints=tuple(d["joints"]), sign=1, crank=d["crank"],
                effort_peak=float(d["effort"]),
                effort_nominal=float(d["effort_nominal"]) if "effort_nominal" in d else None,
                velocity=float(d["velocity"]), position_limit_device=tuple(d["position_limit"]),
                baseline_stiffness=k, baseline_damping=dmp,
            )
        missing = set(gains or ()) - set(out)
        if missing:
            raise ValueError(f"baseline gains for unknown drives: {sorted(missing)}")
        return out

    @cached_property
    def linkages(self) -> list[Linkage]:
        """The crank linkages, one per pair of hinges: the ETHRC ankles (left first); empty for a robot without."""
        out = []
        for side in ("l", "r"):
            joints = (tuple(self.ankle_model.payload["sides"][side]["joint_names"]) if self.usd_path else
                      (f"foot_pitch_{side}", f"foot_roll_{side}"))
            by_crank = {d.crank: d.name for d in self.drives.values() if d.crank and set(d.joints) == set(joints)}
            if by_crank:
                if set(by_crank) != {"upper", "lower"}:
                    raise ValueError(f"linkage {side}: cranks {sorted(by_crank)}, expected upper and lower")
                out.append(Linkage(side, by_crank["upper"], by_crank["lower"], joints))
        cranks = {d.name for d in self.drives.values() if d.crank}
        if cranks != {c for lk in out for c in (lk.upper, lk.lower)}:
            raise ValueError(f"crank drives outside an ankle linkage: {sorted(cranks)}")
        return out

    def braking_limit(self, drive: str) -> float:
        """The torque a drive holds against the motion: its family's ``braking_peak`` where the description states
        one, never above the drive's ``effort`` (Controls' ``ethrc_ssot.torque_envelope.braking_limit``)."""
        d = self.drives[drive]
        return min(float(self.family_value(d.family, "braking_peak", d.effort_peak)), d.effort_peak)

    def drive_delay_ns(self, family: str) -> int | None:
        """A motor family's ``drive_delay_s`` in integer ns: the controller's APPLIED acknowledgement to the drive's
        position demand starting to move. None when the description records none for the family."""
        if family not in self.table.get("families", {}):
            raise ValueError(f"{self.robot}: no drive family {family!r}")
        value = self.family_value(family, "drive_delay_s")
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{self.robot}: {family} drive_delay_s {value!r} must be finite and non-negative")
        return round(value * 1e9)

    def family_value(self, family: str, key: str, default=None):
        """A motor family's ``key``, or ``default`` when the family does not record it."""
        return self.table.get("families", {}).get(family, {}).get(key, default)

    def calibration_travel(self, name: str) -> tuple[float, float]:
        """Simulated travel in URDF (calibration) radians: the recorded mechanical stop where there is one,
        else the position limit. Same rule as Controls ``ethrc_ssot/joint_travel.py``. An ankle hinge has no limit
        of its own: its travel is the axis hull of what its cranks' stops allow (``ankle_model.bounds``)."""
        entry = self.table["joints"][name]
        linkage = next((lk for lk in self.linkages if name in lk.joints), None)
        if linkage is not None:
            return self._ankle_bounds(linkage.side)[linkage.joints.index(name)]
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

    def link_names(self) -> set[str]:
        import xml.etree.ElementTree as ET

        if self.usd_model is not None:
            return set(self.usd_model.bodies) | {p.GetName() for p in self.usd_model.stage.Traverse()
                                                if p.GetCustomDataByKey("fusion2usd:sensor")}
        return {link.get("name") for link in ET.parse(self.urdf_path).getroot().findall("link")}

    def _imu_joint(self):
        import xml.etree.ElementTree as ET

        joint = next((j for j in ET.parse(self.urdf_path).getroot().findall("joint")
                      if j.get("name") == "imu_joint"), None)
        if joint is None:
            raise ValueError(f"{self.urdf_path.name} has no imu_joint")
        return joint

    def imu_link(self) -> str:
        """The IMU's own link: the child of the URDF's ``imu_joint``."""
        if self.usd_model is not None:
            return self.usd_model.imu_frame()
        child = self._imu_joint().find("child")
        if child is None:
            raise ValueError(f"{self.urdf_path.name}: imu_joint has no child")
        return child.get("link")

    def imu_mount(self) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        """The IMU's parent link and its pose there (xyz in m, URDF rpy in rad): the URDF's ``imu_joint``."""
        if self.usd_model is not None:
            return self.usd_model.imu_mount()
        joint = self._imu_joint()
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
        if self.usd_path is not None:
            return {**self.ankle_model.payload["sides"][side]["geometry"],
                    "pitch_axis": [0, 1, 0], "roll_axis": [1, 0, 0]}
        right = self.table["ankle_model"]["right_geometry"]
        return right if side == "r" else mirror_right(right)

    def ankle_kinematic_sign(self, side: str) -> list[int]:
        if self.usd_path is not None:
            return [1, 1]
        return [int(s) for s in self.table["ankle_model"]["semantic_to_kinematic_sign"][side]]

    def ankle_neutral(self, side: str) -> list[float]:
        """Joint angles (pitch, roll) of the linkage neutral, where both cranks read zero."""
        if self.usd_path is not None:
            return self.ankle_model.neutral(side).tolist()
        return [float(v) for v in self.table["ankle_model"]["semantic_neutral_rad"][side]]

    def ankle_transmission(self, side: str, *, device="cpu", dtype=None) -> AnkleTransmission:
        import torch

        native = self.usd_path is not None
        return AnkleTransmission(self.ankle_geometry(side), self.ankle_kinematic_sign(side),
                                 (0.0, 0.0) if native else self.ankle_neutral(side),
                                 native_angles=native, device=device, dtype=dtype or torch.float32)

    def ankle_cranks(self, side: str) -> tuple[str, str]:
        """(upper, lower) crank drive names of one side."""
        linkage = next((lk for lk in self.linkages if lk.side == side), None)
        if linkage is None:
            raise ValueError(f"{self.robot} has no ankle linkage on side {side!r}")
        return linkage.upper, linkage.lower

    def _ankle_bounds(self, side: str):
        cache = self.__dict__.setdefault("_ankle_bounds_cache", {})
        if side not in cache:
            cache[side] = self.ankle_model.bounds(side)
        return cache[side]

    @cached_property
    def ankle_model(self):
        """humanoid-urdf's ``AnkleModel`` for this description: the cranks' stops and what they allow."""
        if self.usd_path is not None:
            return self.ankle_model_module().load(self.actuators_path, self.usd_path)
        return self.ankle_model_module().load(self.actuators_path)

    def ankle_model_module(self):
        """humanoid-urdf's own ``ankle_model`` module (numpy reference), loaded from the submodule."""
        path = self.package_dir / self.package_dir.name / "ankle_model.py"
        spec = importlib.util.spec_from_file_location(f"_hu_ankle_model_{self.robot}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
