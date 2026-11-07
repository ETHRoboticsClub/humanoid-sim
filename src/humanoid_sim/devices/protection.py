"""Drive protection: the output current limit and the torque overload trip.

Both follow what Controls does with a real drive, not a firmware of the drive's own (the sim has none to copy):

* Current limit. A drive's current loop saturates at its output current limit (EPOS4 ``0x3001:02``,
  ``max_current_a`` in ``actuators.yaml``). Its torque equivalent clips the applied torque, and caps the control law's
  output that the overload rule judges (Controls reads that as the drive's current demand, ``0x30D0``).
* Overload. Controls' torque guard (``TorqueOverload.hpp``, applied by ``JointSafetyMonitor::judgeOverload`` on the
  robot and by ``ControlCore::torque`` in its own simulator) trips a drive whose torque demand stays above the
  drive's torque limit (its ``effort``, the peak) continuously for ``backend.overload_dwell_s``. One sample under the
  limit restarts the count, and a drive that is off is not judged.

What the drive does once tripped is the ``response``: ``flag`` only reports it (the drive keeps its torque, as on
the robot until the host reacts), ``cut`` also applies no torque until the drive is turned off. Turning it off
(the law DISABLED in force for a step) clears the trip and the count, as Controls' re-enable does.

All state is ``[E, D]`` tensors written in place; nothing branches on data, so it runs inside a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

RESPONSES = ("flag", "cut")

# Bits of the per-drive status the runtime reports (plant ABI 3.6).
STATUS_OVERLOAD = 1


def profile_response(profile: dict) -> str | None:
    """The profile's ``devices.protection.response``: None (no protection) when the profile has none."""
    return (profile.get("devices", {}).get("protection") or {}).get("response")


@dataclass(frozen=True)
class DriveProtection:
    response: str  # one of RESPONSES
    dwell_ns: int  # the overload dwell
    limit_nm: torch.Tensor  # [D] torque above which a drive is overloaded: its peak effort
    current_limit_nm: torch.Tensor  # [D] torque at the output current limit; inf where the family records none

    def __post_init__(self):
        if self.response not in RESPONSES:
            raise ValueError(f"unknown drive protection response {self.response!r}; known: {list(RESPONSES)}")
        if self.dwell_ns <= 0:
            raise ValueError(f"overload dwell {self.dwell_ns} ns must be positive")

    @classmethod
    def for_drives(cls, desc, drive_names: list[str], response: str | None, *, device="cpu",
                   dtype=torch.float32) -> DriveProtection | None:
        """The protection for a description's drives, or None when ``response`` is None (no protection: the bank
        behaves as it did before). Raises when asked for but the description records no ``overload_dwell_s``."""
        if response is None:
            return None
        dwell = desc.overload_dwell_ns()
        if dwell is None:
            raise ValueError(f"{desc.robot}: drive protection needs backend.overload_dwell_s in actuators.yaml")
        drives = [desc.drives[n] for n in drive_names]
        limits = [desc.current_limit_torque_nm(n) for n in drive_names]
        return cls(response, dwell,
                   torch.tensor([d.effort_peak for d in drives], device=device, dtype=dtype),
                   torch.tensor([float("inf") if c is None else c for c in limits], device=device, dtype=dtype))


class OverloadState:
    """Per env and drive: when the current overload began (-1: none) and whether the drive has tripped."""

    def __init__(self, num_envs: int, num_drives: int, *, device="cpu"):
        self.over_since_ns = torch.full((num_envs, num_drives), -1, dtype=torch.int64, device=device)
        self.tripped = torch.zeros(num_envs, num_drives, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.over_since_ns.index_fill_(0, env_ids, -1)
        self.tripped.index_fill_(0, env_ids, False)

    def update(self, protection: DriveProtection, demanded: torch.Tensor, on: torch.Tensor,
               t_ns: int | torch.Tensor) -> torch.Tensor:
        """Judge the step starting at ``t_ns`` (an int, a 0-d tensor or ``[E]``): ``demanded`` ``[E, D]`` Nm is the
        control law's output, ``on`` ``[E, D]`` bool whether the drive is not off. Returns ``tripped`` ``[E, D]``,
        which includes a trip that happens at this sample."""
        t = torch.as_tensor(t_ns, device=demanded.device, dtype=torch.int64)
        t = t.reshape(-1, 1) if t.dim() == 1 else t
        current = protection.current_limit_nm.to(demanded.dtype)
        over = on & (torch.minimum(demanded.abs(), current) > protection.limit_nm.to(demanded.dtype))
        since = torch.where(over, torch.where(self.over_since_ns >= 0, self.over_since_ns, t.expand_as(over)),
                            torch.full_like(self.over_since_ns, -1))
        self.over_since_ns.copy_(since)
        tripped = (self.tripped | (over & (t - since >= protection.dwell_ns))) & on
        self.tripped.copy_(tripped)
        return tripped

    def status(self) -> torch.Tensor:
        """The per-drive status word ``[E, D]`` (uint32 values in an int64 tensor): :data:`STATUS_OVERLOAD`."""
        return self.tripped.to(torch.int64) * STATUS_OVERLOAD
