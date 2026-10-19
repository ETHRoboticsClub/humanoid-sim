"""Torque-speed envelope of each drive, batched.

Motoring (torque and velocity in the same direction): the datasheet curve at |velocity|, linearly
interpolated between the description's points and zero beyond the last one. Standstill: the curve's peak.
Braking (opposite signs): the family's ``braking_peak`` where the description states one, else the drive's
``effort``, never above ``effort`` (Controls' ``ethrc_ssot.torque_envelope.braking_limit``). The curves do not
model thermal derating. Source of every number: humanoid-urdf ``families.*.torque_speed`` and ``braking_peak``.
"""

from __future__ import annotations

import torch


class TorqueSpeedEnvelope:
    def __init__(self, curves: list[list[tuple[float, float]]], *, braking: list[float] | None = None, device="cpu",
                 dtype=torch.float32):
        """``curves[d]`` = [(speed rad/s, torque Nm), ...] for drive d, speeds ascending from 0; ``braking[d]`` the
        torque it holds against the motion (default: the curve's peak)."""
        width = max(len(c) for c in curves)
        speeds, torques = [], []
        for c in curves:
            if c[0][0] != 0 or any(b[0] <= a[0] for a, b in zip(c, c[1:])):
                raise ValueError(f"torque-speed curve must start at 0 rad/s and ascend: {c}")
            pad = [c[-1]] * (width - len(c))
            # padded points repeat the last speed with +inf spacing so interpolation never uses them
            speeds.append([p[0] for p in c] + [c[-1][0] + 1e9 * (i + 1) for i in range(len(pad))])
            torques.append([p[1] for p in c] + [c[-1][1]] * len(pad))
        self.speeds = torch.tensor(speeds, device=device, dtype=dtype)  # [D, P]
        self.torques = torch.tensor(torques, device=device, dtype=dtype)
        self.peak = self.torques[:, 0].clone()  # [D] standstill
        self.braking = self.peak.clone() if braking is None else torch.tensor(braking, device=device, dtype=dtype)
        self.max_speed = torch.tensor([c[-1][0] for c in curves], device=device, dtype=dtype)

    def motoring_limit(self, speed: torch.Tensor) -> torch.Tensor:
        """Max |torque| while motoring at |speed| ``[..., D]``."""
        s = speed.abs()
        lead = s.shape[:-1]
        sp = self.speeds.expand(*lead, *self.speeds.shape)
        tq = self.torques.expand(*lead, *self.torques.shape)
        idx = torch.searchsorted(sp.contiguous(), s.unsqueeze(-1).contiguous()).clamp(1, sp.shape[-1] - 1)
        s0, s1 = sp.gather(-1, idx - 1).squeeze(-1), sp.gather(-1, idx).squeeze(-1)
        t0, t1 = tq.gather(-1, idx - 1).squeeze(-1), tq.gather(-1, idx).squeeze(-1)
        lim = t0 + (t1 - t0) * ((s - s0) / (s1 - s0)).clamp(0, 1)
        return torch.where(s >= self.max_speed, torch.zeros_like(lim), lim)

    def limits(self, velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(lower, upper) torque bounds at ``velocity`` ``[..., D]``."""
        motoring = self.motoring_limit(velocity)
        peak, braking = self.peak.expand_as(velocity), self.braking.expand_as(velocity)
        upper = torch.where(velocity > 0, motoring, torch.where(velocity < 0, braking, peak))
        lower = torch.where(velocity < 0, -motoring, torch.where(velocity > 0, -braking, -peak))
        return lower, upper

    def clamp(self, torque: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        lower, upper = self.limits(velocity)
        return torch.maximum(torch.minimum(torque, upper), lower)
