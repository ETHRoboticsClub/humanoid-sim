"""Drive models: one class per drive product, as the sensors are (:mod:`humanoid_sim.sensors.models`).

A product's class names its ``actuators.yaml`` family and how it is driven. Its figures are robot facts, so the
description holds them (humanoid-urdf's ``actuators.yaml`` ``families``) and an instance reads them from it: one
instance is that product on one robot. :func:`drive_models` gives each of a description's drives its model; a
family without a class here is refused.

Timing: ``drive_delay_ns`` is the time from the controller staging a demand (its APPLIED acknowledgement) to the
drive acting on it: the bus master's staging, the bus and the drive's interpolation. The runtime host applies each
bus cycle's demand that long after the bus delivered it, and the training plant always adds it to the latency
before the drive. A family whose delay is unmeasured has ``drive_delay_ns`` 0 and ``delay_measured`` False.
"""

from __future__ import annotations

from typing import ClassVar


class DriveModel:
    family: ClassVar[str]  # actuators.yaml families key
    product: ClassVar[str]
    interface: ClassVar[str]  # how the robot's controller drives it

    def __init__(self, desc):
        delay = desc.drive_delay_ns(self.family)
        self.delay_measured = delay is not None
        self.drive_delay_ns = delay or 0
        self.delay_source = desc.family_value(self.family, "drive_delay_source") if delay is not None else None

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "family" in cls.__dict__ and not all(hasattr(cls, k) for k in ("product", "interface")):
            raise TypeError(f"{cls.__name__}: a product names its product and interface")

    def describe(self) -> dict:
        """The model as the runtime descriptor publishes it."""
        return {"model": type(self).__name__, "family": self.family, "product": self.product,
                "interface": self.interface, "drive_delay_ns": self.drive_delay_ns,
                "delay_measured": self.delay_measured}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(drive_delay_ns={self.drive_delay_ns})"


class Epos4Drive(DriveModel):
    """A maxon high-efficiency joint: motor, gearbox and EPOS4 controller, on EtherCAT in Controls' JVPT mode."""

    interface = "EtherCAT CiA-402, maxon EPOS4 JVPT (mode -64)"


class Hej90(Epos4Drive):
    family = "HEJ90-48-140"
    product = "maxon HEJ 90-48-140"


class Hej70(Epos4Drive):
    family = "HEJ70-48-50"
    product = "maxon HEJ 70-48-50"


class Hej50(Epos4Drive):
    family = "HEJ50-48-30"
    product = "maxon HEJ 50-48-30"


class UnitreeMotor(DriveModel):
    """A Unitree G1 joint motor, driven through the low-level command (LowCmd) at the board's rate."""

    interface = "Unitree LowCmd"


class UnitreeN7520_14(UnitreeMotor):
    family = "N7520-14.3"
    product = "Unitree N7520, 14.3:1"


class UnitreeN7520_22(UnitreeMotor):
    family = "N7520-22.5"
    product = "Unitree N7520, 22.5:1"


class UnitreeN5020(UnitreeMotor):
    family = "N5020-16"
    product = "Unitree N5020, 16:1"


class UnitreeW4010(UnitreeMotor):
    family = "W4010-25"
    product = "Unitree W4010, 25:1"


MODELS: dict[str, type[DriveModel]] = {cls.family: cls for cls in (
    Hej90, Hej70, Hej50, UnitreeN7520_14, UnitreeN7520_22, UnitreeN5020, UnitreeW4010)}


def drive_models(desc) -> dict[str, DriveModel]:
    """Each of the description's drives (by name, in its drive order) with its product's model; one instance per
    family."""
    by_family: dict[str, DriveModel] = {}
    out = {}
    for name, drive in desc.drives.items():
        if drive.family not in MODELS:
            raise ValueError(f"{desc.robot}: drive {name} is a {drive.family!r}, which has no drive model; "
                             f"known: {sorted(MODELS)}")
        if drive.family not in by_family:
            by_family[drive.family] = MODELS[drive.family](desc)
        out[name] = by_family[drive.family]
    return out
