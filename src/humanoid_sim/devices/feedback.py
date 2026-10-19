"""Drive-frame feedback, the quantities a drive reports for its output."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DriveFeedback:
    """Physical drive-frame feedback ``[E, D]``: what Controls' fake bank accepts (``fake_bank_feedback``)."""

    position_rad: torch.Tensor
    velocity_rad_s: torch.Tensor
    applied_torque_nm: torch.Tensor
    demanded_torque_nm: torch.Tensor
