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

Two errors the fake drive does not have are added, each switchable (zero turns it off) and measured on the robot
(PDO feedback of the V1 excitation captures exc-53f09771 and exc-5fd2bcca, 2026-10-28, in windows where the drive
sat within 2 counts for 100 samples, about 0.5 s; the figures per family are the profile's ``devices.feedback_noise``):

* velocity dither: the drive's 0x606C reading is whole counts per 400 us sample (the captures confirm it to within 0.05 of
  a count at p95), but at rest 31 to 33 % of the samples are not zero but +-1 count, symmetrically, and 2 counts in 0.1 to 0.4 %. That is the rounding of the true counts per sample plus a Gaussian of ``velocity_dither_counts`` (sigma in
  counts per sample, 0.50 to 0.52 on every family): it gives 31 to 33 % nonzero and 0.3 % at 2 counts. The reading
  of a drive at rest then has an RMS of 2.2 rad/s on a HEJ90 (3.83 rad/s per count, 4096 counts) and 0.54 to 0.55 rad/s on a
  HEJ70/50 (0.96 rad/s per count). Without it a drive reads exactly 0 below half a count per sample (1.92 rad/s
  on a HEJ90, 0.48 rad/s on a HEJ70/50);
* torque noise: white Gaussian noise of ``torque_noise_nm`` on the applied torque before it is rounded to mNm
  (lag-1 autocorrelation 0.0). Standard deviation of the torque reading at rest, median over the windows: 0.46 Nm on
  the HEJ90 (per drive 0.24 to 0.65; the waist yaw, which carries load that drifts, 1.2), 0.11 Nm on the HEJ70, 0.05
  Nm on the HEJ50. It is measured at rest; in motion the reading varies far more (the drive's current follows its
  loop), which is torque, not noise, and is not modeled.

Position jitter is unchanged: the sim's -1, 0, +1 (uniform) differs from the window's median count in 67 % of the
samples at rest; the robot's differs in 25 to 30 %, about 2.5 times less. Counts and mNm rounding match the robot.

The drive's position reading includes its offset from the truth (the plant's ``encoder_offset``), as the drive
regulates; it is quantized here. Batched ``[E, D]``; the jitter is drawn per read from the module's generator.
"""

from __future__ import annotations

import math

import torch

from humanoid_sim.devices.feedback import DriveFeedback


class DriveReporting:
    def __init__(self, counts_per_revolution: list[int], controller_rate_hz: list[float], *, device="cpu",
                 dtype=torch.float32, jitter: bool = True, seed: int = 0,
                 velocity_dither_counts: list[float] | None = None, torque_noise_nm: list[float] | None = None):
        """``velocity_dither_counts`` / ``torque_noise_nm``: per drive ``[D]``, the sigma of the Gaussian added to
        the velocity in counts per controller sample and to the applied torque in Nm; None: none."""
        self.device = torch.device(device)
        # float64 and Epos4Slave's order of operations, so halves round as it rounds them.
        self.cpr = torch.tensor(counts_per_revolution, device=self.device, dtype=torch.float64)  # [D]
        self.counts_per_rad = (self.cpr / (2 * math.pi)).to(dtype)
        self.sample_s = torch.tensor([1.0 / r for r in controller_rate_hz], device=self.device, dtype=torch.float64)
        self.jitter = jitter
        self.velocity_dither, self._dither = self._sigma(velocity_dither_counts, len(counts_per_revolution), dtype)
        self.torque_noise, self._noisy = self._sigma(torque_noise_nm, len(counts_per_revolution), dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def _sigma(self, values, num_drives: int, dtype) -> tuple[torch.Tensor | None, bool]:
        if values is None:
            return None, False
        if len(values) != num_drives or any(not v >= 0 for v in values):
            raise ValueError(f"expected {num_drives} non-negative sigmas, got {values!r}")
        # Decided here, once, so that steady state never asks the device whether a sigma is nonzero.
        return torch.tensor(values, device=self.device, dtype=dtype), any(v > 0 for v in values)

    @classmethod
    def for_drives(cls, desc, drives: list[str], profile: dict, *, noise: bool = True, **kw) -> DriveReporting:
        """Counts per revolution from the profile's ``devices.encoders`` (per family), controller rate from the
        description's families. ``noise``: the velocity dither and torque noise of the profile's
        ``devices.feedback_noise`` (per family; none where the profile has none); False reports without them."""
        encoders = profile["devices"]["encoders"]
        families = [desc.drives[d].family for d in drives]
        missing = sorted(set(families) - set(encoders))
        if missing:
            raise ValueError(f"profile devices.encoders lacks families {missing}")
        measured = profile["devices"].get("feedback_noise") if noise else None
        if measured is not None:
            missing = sorted(set(families) - set(measured))
            if missing:
                raise ValueError(f"profile devices.feedback_noise lacks families {missing}")
            kw.setdefault("velocity_dither_counts", [float(measured[f]["velocity_dither_counts"]) for f in families])
            kw.setdefault("torque_noise_nm", [float(measured[f]["torque_nm"]) for f in families])
        return cls([int(encoders[f]["counts_per_revolution"]) for f in families],
                   [float(desc.family(f)["controller_rate_hz"]) for f in families], **kw)

    def position(self, position_rad: torch.Tensor) -> torch.Tensor:
        counts = _round(position_rad.double() * self.cpr / (2 * math.pi))
        if self.jitter:
            counts = counts + torch.randint(-1, 2, counts.shape, generator=self.generator, device=self.device,
                                            dtype=counts.dtype)
        return (counts * (2 * math.pi) / self.cpr).to(position_rad.dtype)

    def velocity(self, velocity_rad_s: torch.Tensor) -> torch.Tensor:
        counts = velocity_rad_s.double() * self.cpr / (2 * math.pi) * self.sample_s
        if self._dither:
            counts = counts + self.velocity_dither.double() * self._normal(counts)
        per_sample = _round(counts)
        return (per_sample / self.sample_s * (2 * math.pi) / self.cpr).to(velocity_rad_s.dtype)

    def torque(self, torque_nm: torch.Tensor) -> torch.Tensor:
        if self._noisy:
            torque_nm = torque_nm + self.torque_noise.to(torque_nm.dtype) * self._normal(torque_nm)
        return _round(torque_nm * 1000.0) / 1000.0

    def _normal(self, like: torch.Tensor) -> torch.Tensor:
        return torch.randn(like.shape, generator=self.generator, device=self.device, dtype=like.dtype)

    def report(self, fb: DriveFeedback) -> DriveFeedback:
        """The reported quantities of physical feedback ``[..., D]`` (position already including the drive's
        offset)."""
        return DriveFeedback(self.position(fb.position_rad), self.velocity(fb.velocity_rad_s),
                             self.torque(fb.applied_torque_nm), fb.demanded_torque_nm)


def _round(x: torch.Tensor) -> torch.Tensor:
    """Halves away from zero, as C's ``lround`` and ``std::round`` (``torch.round`` rounds them to even)."""
    whole = torch.trunc(x)
    return whole + torch.where((x - whole).abs() >= 0.5, torch.sign(x), torch.zeros_like(x))
