"""Motor winding I²t protection: a drive that runs above its nominal current falls back to it.

A maxon EPOS4 protects its motor's winding with an I²t limit computed from the nominal current (``0x3001:01``), the
output current limit (``0x3001:02``) and the winding's thermal time constant (``0x3001:04``); the HEJ datasheets
call it "Motor temperature i2t protection". Above the nominal current the winding heats, and once it has heated for
as long as the time constant allows, the drive limits its current to the nominal one until it has cooled.

The model, ``modeled`` until a capture of the drive's I²t level (``0x3200:01``) verifies it:

* The winding's heat is a first-order low-pass of the squared output torque with the winding's time constant ``τ``:
  ``heat ← sq + (heat − sq) · exp(−dt / τ)``, ``sq`` the time-weighted squared applied torque over the step. Torque
  stands for current (``τ = k · I``, gear losses ignored), so the nominal current is the nominal torque.
* While ``heat ≥ nominal²`` the applied torque is clipped to ``±nominal``; below it the drive has its full envelope.
  From cold, a constant torque ``T > nominal`` is limited after ``τ · ln(T² / (T² − nominal²))``.
* A drive that is off applies no torque and cools. A reset starts the env's windings cold.

The limit applies to the torque the drive applies, after the torque-speed envelope and the demand's cap; the
demanded torque (the control law's output) is not changed. All state is ``[E, D]`` tensors written in place, and
nothing branches on data, so it runs inside a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindingI2t:
    """Per drive: the torque at the nominal current (inf: no I²t limit) and the winding's thermal time constant."""

    nominal_nm: torch.Tensor  # [D] Nm
    time_constant_ns: torch.Tensor  # [D] int64 ns

    def __post_init__(self):
        if self.nominal_nm.dim() != 1 or self.nominal_nm.shape != self.time_constant_ns.shape:
            raise ValueError("nominal_nm and time_constant_ns must both be [drives]")
        if self.time_constant_ns.dtype != torch.int64:
            raise ValueError("time_constant_ns must be int64 nanoseconds")
        if not bool((self.nominal_nm > 0).all()):
            raise ValueError("nominal torques must be positive (inf: no limit)")
        if not bool((self.time_constant_ns > 0).all()):
            raise ValueError("winding time constants must be positive")


class WindingHeat:
    """Per env and drive: the winding's heat (Nm², the low-passed squared torque) and whether it limits the drive."""

    def __init__(self, num_envs: int, num_drives: int, *, device="cpu", dtype=torch.float32):
        self.heat = torch.zeros(num_envs, num_drives, device=device, dtype=dtype)
        self.limited = torch.zeros(num_envs, num_drives, device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.heat.index_fill_(0, env_ids, 0.0)
        self.limited.index_fill_(0, env_ids, False)

    def bound(self, winding: WindingI2t) -> torch.Tensor:
        """The torque bound ``[E, D]`` for the step about to run (inf where the winding does not limit), from the
        heat at its start; also writes ``limited``."""
        nominal = winding.nominal_nm.to(self.heat.dtype)
        limited = self.heat >= nominal * nominal
        self.limited.copy_(limited)
        return torch.where(limited, nominal.expand_as(self.heat), torch.full_like(self.heat, float("inf")))

    def advance(self, winding: WindingI2t, squared_torque: torch.Tensor, dt_ns: int | torch.Tensor) -> None:
        """Heat the windings over a step of ``dt_ns`` (an int or a 0-d int64 tensor) during which the drives applied
        a time-weighted squared torque ``squared_torque`` ``[E, D]`` (Nm²)."""
        dt = torch.as_tensor(dt_ns, device=self.heat.device, dtype=torch.float64)
        decay = torch.exp(-dt / winding.time_constant_ns.to(self.heat.device, torch.float64)).to(self.heat.dtype)
        sq = squared_torque.to(self.heat.dtype)
        self.heat.copy_(sq + (self.heat - sq) * decay)
