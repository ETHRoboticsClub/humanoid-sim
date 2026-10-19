"""Drive demand: what a drive's control law is told, per env and drive, in SI and vendor-neutral.

Controls' device layer converts its vendor's command into this at its own boundary: an EPOS4 drive's CiA-402
mode and raw JVPT gains (mode -64 is PD, homing 6 and -62 are FREEZE, disabled is DISABLED), a Unitree motor's
``LowCmd`` entry (mode 1 is PD, mode 0 is DISABLED). The record is the plant segment's ``DriveDemand``
(Controls ``docs/sim-learning-api-plan.md`` §1.2):

* ``law``: DISABLED (no torque), PD (``τ = kp (target − q) + kd (v_target − v) + ff``) or FREEZE (PD about the
  position the drive reads when it enters FREEZE, ``v_target = 0``, no ff);
* positions in the physical drive-output frame (rad), velocity rad/s, feed-forward N m;
* ``stiffness`` N m/rad and ``damping`` N m s/rad, the drive's own gains;
* ``torque_limit_nm``: a cap on the applied torque, ``inf`` when there is none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

LAW_DISABLED = 0
LAW_PD = 1
LAW_FREEZE = 2
LAWS = {"disabled": LAW_DISABLED, "pd": LAW_PD, "freeze": LAW_FREEZE}


@dataclass
class DriveDemand:
    """One demand per env and drive, every field ``[E, D]``."""

    law: torch.Tensor  # int8: LAW_DISABLED, LAW_PD, LAW_FREEZE
    target_position_rad: torch.Tensor
    target_velocity_rad_s: torch.Tensor
    feedforward_torque_nm: torch.Tensor
    stiffness_nm_per_rad: torch.Tensor
    damping_nm_s_per_rad: torch.Tensor
    torque_limit_nm: torch.Tensor  # inf: none

    @classmethod
    def disabled(cls, num_envs: int, num_drives: int, *, device="cpu", dtype=torch.float32) -> DriveDemand:
        def z():
            return torch.zeros(num_envs, num_drives, device=device, dtype=dtype)

        return cls(torch.zeros(num_envs, num_drives, device=device, dtype=torch.int8), z(), z(), z(), z(), z(),
                   torch.full((num_envs, num_drives), math.inf, device=device, dtype=dtype))

    @classmethod
    def pd(cls, position: torch.Tensor, stiffness, damping, *, velocity: torch.Tensor | None = None,
           feedforward: torch.Tensor | None = None, torque_limit_nm: torch.Tensor | None = None,
           law: int = LAW_PD) -> DriveDemand:
        """An enabled demand (PD, or FREEZE with ``law=LAW_FREEZE``) from SI quantities; gains broadcast to
        ``position``'s shape."""
        zeros = torch.zeros_like(position)

        def full(x):
            return torch.as_tensor(x, device=position.device, dtype=position.dtype).expand(position.shape).clone()

        return cls(
            law=torch.full_like(position, law, dtype=torch.int8),
            target_position_rad=position.clone(),
            target_velocity_rad_s=zeros.clone() if velocity is None else velocity.clone(),
            feedforward_torque_nm=zeros.clone() if feedforward is None else feedforward.clone(),
            stiffness_nm_per_rad=full(stiffness), damping_nm_s_per_rad=full(damping),
            torque_limit_nm=full(math.inf) if torque_limit_nm is None else full(torque_limit_nm),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, torch.Tensor]) -> DriveDemand:
        return cls(**{f.name: d[f.name] for f in fields(cls)})
