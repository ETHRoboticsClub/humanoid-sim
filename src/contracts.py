"""SI process-data contracts for the behavioral drive model.

Angles, velocities and efforts refer to the actuator output shaft in rad,
rad/s and Nm. These are semantic PDO fields, not a raw EtherCAT wire codec.
Only the documented subset below is modeled; this is not HEJ firmware.
"""

from dataclasses import dataclass
from enum import IntEnum
import math

import torch

MODE_JVPT = -64  # Maxon SDK ModeOfOperationEnum.hpp, CyclicJVPTMode.
CONTROLWORD_DISABLE = 0
CONTROLWORD_QUICK_STOP = 2
CONTROLWORD_SHUTDOWN = 6
CONTROLWORD_SWITCH_ON = 7
CONTROLWORD_ENABLE = 15
CONTROLWORD_FAULT_RESET = 128
SUPPORTED_CONTROLWORDS = (0, 2, 6, 7, 15, 128)


class DriveState(IntEnum):
    SWITCH_ON_DISABLED = 0x40
    READY_TO_SWITCH_ON = 0x21
    SWITCHED_ON = 0x23
    OPERATION_ENABLED = 0x27
    QUICK_STOP_ACTIVE = 0x07
    FAULT = 0x08


@dataclass(frozen=True)
class DriveConfig:
    name: str
    mode: int = MODE_JVPT
    stiffness: float = 0.0
    damping: float = 0.0
    torque_limit: float = 1.0
    speed_limit: float | None = None
    watchdog_ns: int = 100_000_000
    command_delay_ns: int = 0
    torque_filter_tau_s: float = 0.0
    coulomb_friction: float = 0.0
    viscous_friction: float = 0.0
    armature: float = 0.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("drive name must be nonempty")
        if self.mode != MODE_JVPT:
            raise NotImplementedError("only cyclic JVPT mode -64 is modeled")
        for key in (
            "stiffness",
            "damping",
            "torque_filter_tau_s",
            "coulomb_friction",
            "viscous_friction",
            "armature",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(self.torque_limit) or self.torque_limit <= 0:
            raise ValueError("torque_limit must be finite and positive")
        if self.speed_limit is not None and (not math.isfinite(self.speed_limit) or self.speed_limit <= 0):
            raise ValueError("speed_limit must be finite and positive")
        for key in ("watchdog_ns", "command_delay_ns"):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= 2**63 - 1:
                raise ValueError(f"{key} must be a nonnegative int64 integer")


@dataclass(frozen=True)
class CommandBatch:
    """One complete frame per [environment, device], copied on submission."""

    controlword: torch.Tensor
    mode: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    effort: torch.Tensor


@dataclass(frozen=True)
class DriveFeedback:
    """Snapshot; measured_effort is physical feedback supplied by the backend."""

    timestamp_ns: int
    position: torch.Tensor
    velocity: torch.Tensor
    measured_effort: torch.Tensor
    commanded_position: torch.Tensor
    commanded_velocity: torch.Tensor
    commanded_effort: torch.Tensor
    demand_position: torch.Tensor
    demand_velocity: torch.Tensor
    demand_effort: torch.Tensor
    requested_effort: torch.Tensor
    actuator_effort: torch.Tensor
    friction_effort: torch.Tensor
    applied_effort: torch.Tensor
    statusword: torch.Tensor
    mode_display: torch.Tensor
    watchdog_expired: torch.Tensor
    firmware_timestamp_ns: int = -1
