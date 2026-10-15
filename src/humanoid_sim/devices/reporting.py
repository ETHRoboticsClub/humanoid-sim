"""What a drive reports for its physical state: the training plant's mirror of Controls' fake drive.

In the runtime, Controls' ``Epos4Slave`` turns the plant's physical feedback into the drive's process data; in
training this module does the same for the measured drive feedback. Source: ETHRC-Humanoid-Controls
``ros2_ws/src/sim/ethrc_fake_ethercat/src/Epos4Slave.cpp`` @ 447b4acb (``setFeedback``, ``reportedPosition``,
``reportedVelocityMilliRpm``, ``torqueEstimate_mNm``):

* position: whole encoder counts of the drive's reading, ``round(position · cpr / 2π)``, plus a jitter of -1, 0
  or +1 count on every read; the master reads ``counts · 2π / cpr``;
* velocity: whole counts per controller sample, ``round(velocity · cpr / 2π · T)`` with ``T = 1 /
  controller_rate_hz``, back to rad/s (a coarse signal: one count per sample is 3.8 rad/s on a 4096-count
  encoder);
* torque: the applied torque in whole mNm (0x3672). The demanded torque passes through.

The drive's position reading includes its offset from the truth (the plant's ``encoder_offset``), as the drive
regulates; it is quantized here. Batched ``[E, D]``; the jitter is drawn per read from the module's generator.
"""

from __future__ import annotations

import math

import torch

from humanoid_sim.core import DriveFeedback


class DriveReporting:
    def __init__(self, counts_per_revolution: list[int], controller_rate_hz: list[float], *, device="cpu",
                 dtype=torch.float32, jitter: bool = True, seed: int = 0):
        self.device = torch.device(device)
        # float64 and Epos4Slave's order of operations, so halves round as it rounds them.
        self.cpr = torch.tensor(counts_per_revolution, device=self.device, dtype=torch.float64)  # [D]
        self.counts_per_rad = (self.cpr / (2 * math.pi)).to(dtype)
        self.sample_s = torch.tensor([1.0 / r for r in controller_rate_hz], device=self.device, dtype=torch.float64)
        self.jitter = jitter
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @classmethod
    def for_drives(cls, desc, drives: list[str], profile: dict, **kw) -> DriveReporting:
        """Counts per revolution from the profile's ``devices.encoders`` (per family), controller rate from the
        description's families."""
        encoders = profile["devices"]["encoders"]
        families = [desc.drives[d].family for d in drives]
        missing = sorted(set(families) - set(encoders))
        if missing:
            raise ValueError(f"profile devices.encoders lacks families {missing}")
        return cls([int(encoders[f]["counts_per_revolution"]) for f in families],
                   [float(desc.family(f)["controller_rate_hz"]) for f in families], **kw)

    def position(self, position_rad: torch.Tensor) -> torch.Tensor:
        counts = _round(position_rad.double() * self.cpr / (2 * math.pi))
        if self.jitter:
            counts = counts + torch.randint(-1, 2, counts.shape, generator=self.generator, device=self.device,
                                            dtype=counts.dtype)
        return (counts * (2 * math.pi) / self.cpr).to(position_rad.dtype)

    def velocity(self, velocity_rad_s: torch.Tensor) -> torch.Tensor:
        per_sample = _round(velocity_rad_s.double() * self.cpr / (2 * math.pi) * self.sample_s)
        return (per_sample / self.sample_s * (2 * math.pi) / self.cpr).to(velocity_rad_s.dtype)

    @staticmethod
    def torque(torque_nm: torch.Tensor) -> torch.Tensor:
        return _round(torque_nm * 1000.0) / 1000.0

    def report(self, fb: DriveFeedback) -> DriveFeedback:
        """The reported quantities of physical feedback ``[..., D]`` (position already including the drive's
        offset)."""
        return DriveFeedback(self.position(fb.position_rad), self.velocity(fb.velocity_rad_s),
                             self.torque(fb.applied_torque_nm), fb.demanded_torque_nm)


def _round(x: torch.Tensor) -> torch.Tensor:
    """Halves away from zero, as C's ``lround`` and ``std::round`` (``torch.round`` rounds them to even)."""
    whole = torch.trunc(x)
    return whole + torch.where((x - whole).abs() >= 0.5, torch.sign(x), torch.zeros_like(x))
