"""Mechanical mappings, independent of device firmware and physics engine.

Jacobians always mean d(device-output angle)/d(joint angle). Virtual work is
therefore tau_joint = J.T @ tau_device. No constant ankle approximation is
silently substituted for a missing calibrated linkage model.
"""

from collections.abc import Callable
from typing import Protocol

import torch


class Transmission(Protocol):
    def device_state(self, q: torch.Tensor, dq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def joint_effort(self, q: torch.Tensor, effort: torch.Tensor) -> torch.Tensor: ...


class DirectTransmission:
    """One device per joint: device angle = sign * q + device_offset."""

    def __init__(self, signs: torch.Tensor, offsets: torch.Tensor | None = None):
        if signs.ndim != 1 or not torch.all((signs == 1) | (signs == -1)):
            raise ValueError("signs must be a vector of +1/-1")
        self.signs = signs.clone()
        self.offsets = torch.zeros_like(signs) if offsets is None else offsets.clone()
        if self.offsets.shape != signs.shape or not torch.isfinite(self.offsets).all():
            raise ValueError("offsets must be finite and match signs")

    def device_state(self, q, dq):
        return self.signs * q + self.offsets, self.signs * dq

    def joint_effort(self, q, effort):
        return self.signs * effort


class CoupledTransmission:
    """Caller supplies the calibrated nonlinear kinematics, batched over envs.

    kinematics(q) returns motor angles [E,M] and J [E,M,J]. Limits and feasible
    reset sampling belong to the concrete linkage model, not a box clamp here.
    """

    def __init__(self, kinematics: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]):
        self.kinematics = kinematics

    def device_state(self, q, dq):
        motor_q, jacobian = self.kinematics(q)
        return motor_q, torch.einsum("emj,ej->em", jacobian, dq)

    def joint_effort(self, q, effort):
        _, jacobian = self.kinematics(q)
        return torch.einsum("emj,em->ej", jacobian, effort)
