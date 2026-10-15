"""Raw IMU samples as the robot's host receives them: sensor errors, sample clock, burst delivery.

Input, once per physics step: the ideal angular velocity and specific force in the sensor frame at the step
boundary (Isaac Lab's IMU sensor at the description's mount). Output: per env, the most recent delivered
samples, newest first, each with its measurement time and its host-receipt stamp.

* **Sampling.** Samples are measured on the sensor's own clock every ``sample_period_ns`` (a random phase per
  env), linearly interpolated between step boundaries.
* **Errors.** Per sample: white noise of ``density · √(output rate)``. Per env (the plant's randomization):
  bias and a small misalignment rotation of the sensor in its mount.
* **Delivery.** Samples wait in the serial buffer and reach the host in bursts, every ``burst_period_ns``
  (uniform per burst), stamped on receipt ``in_burst_spacing_ns`` apart. A consumer that reads "the latest
  sample" sees it up to one burst period late.

Everything is batched over envs on the device; the step loop has a fixed trip count, so no host syncs. State is
updated in place and the step time may be a device tensor, so a step can be captured in a CUDA graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImuDeliveryCfg:
    sample_period_ns: int
    burst_period_ns: tuple[int, int]  # uniform per burst
    in_burst_spacing_ns: int
    gyro_noise_density: float  # rad/s/√Hz
    accel_noise_density: float  # m/s²/√Hz
    history: int = 16  # delivered samples kept per env
    pending: int = 16  # samples a burst can carry

    @classmethod
    def from_profile(cls, block: dict) -> ImuDeliveryCfg:
        return cls(sample_period_ns=int(block["sample_period_ns"]),
                   burst_period_ns=tuple(int(v) for v in block["burst_period_ns"]),
                   in_burst_spacing_ns=int(block["in_burst_spacing_ns"]),
                   gyro_noise_density=float(block["gyro_noise_density_rad_s_sqrt_hz"]),
                   accel_noise_density=float(block["accel_noise_density_m_s2_sqrt_hz"]))


@dataclass
class ImuSamples:
    """Delivered samples per env, newest first. Slots past ``count`` hold nothing (stamp = -1)."""

    gyro: torch.Tensor  # [E, H, 3] rad/s, sensor frame
    specific_force: torch.Tensor  # [E, H, 3] m/s², sensor frame
    sample_ns: torch.Tensor  # [E, H] measurement time
    stamp_ns: torch.Tensor  # [E, H] host receipt
    seq: torch.Tensor  # [E, H] per-env sequence number (from 1 after reset)
    count: torch.Tensor  # [E] delivered since reset


def rotvec_matrix(v: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` of rotation vectors ``[..., 3]`` (Rodrigues)."""
    theta = v.norm(dim=-1, keepdim=True).unsqueeze(-1)
    k = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    kx, ky, kz = k.unbind(-1)
    z = torch.zeros_like(kx)
    skew = torch.stack([torch.stack([z, -kz, ky], -1), torch.stack([kz, z, -kx], -1),
                        torch.stack([-ky, kx, z], -1)], -2)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(skew)
    return eye + torch.sin(theta) * skew + (1 - torch.cos(theta)) * skew @ skew


class ImuDelivery:
    def __init__(self, cfg: ImuDeliveryCfg, num_envs: int, *, device="cpu", dtype=torch.float32, seed: int = 0):
        if cfg.pending * cfg.sample_period_ns < cfg.burst_period_ns[1]:
            raise ValueError("pending buffer shorter than a burst period")
        self.cfg, self.num_envs = cfg, num_envs
        self.device, self.dtype = torch.device(device), dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        rate = 1e9 / cfg.sample_period_ns
        self.gyro_sigma = cfg.gyro_noise_density * math.sqrt(rate)
        self.accel_sigma = cfg.accel_noise_density * math.sqrt(rate)
        e, h, p = num_envs, cfg.history, cfg.pending
        f = dict(device=self.device, dtype=dtype)
        self._sigma = torch.tensor([self.gyro_sigma] * 3 + [self.accel_sigma] * 3, **f)
        self._nominal_sigma = self._sigma.clone()
        i64 = dict(device=self.device, dtype=torch.int64)
        # Errors (written by the plant's randomization).
        self.gyro_bias = torch.zeros(e, 3, **f)
        self.accel_bias = torch.zeros(e, 3, **f)
        self.misalignment = torch.zeros(e, 3, **f)
        # Sample clock and delivery schedule.
        self.next_sample = torch.zeros(e, **i64)
        self.next_burst = torch.zeros(e, **i64)
        self.seq = torch.zeros(e, **i64)
        # Measured, not yet delivered: [E, P, 6] values, [E, P] times, oldest first.
        self.pending_values = torch.zeros(e, p, 6, **f)
        self.pending_ns = torch.zeros(e, p, **i64)
        self.pending_seq = torch.zeros(e, p, **i64)
        self.pending_count = torch.zeros(e, **i64)
        self.samples = ImuSamples(torch.zeros(e, h, 3, **f), torch.zeros(e, h, 3, **f),
                                  torch.full((e, h), -1, **i64), torch.full((e, h), -1, **i64),
                                  torch.zeros(e, h, **i64), torch.zeros(e, **i64))
        self._prev = torch.zeros(e, 6, **f)
        self._prev_ns = torch.zeros((), **i64)  # the previous step boundary, on the device
        self._prev_host = 0  # and on the host, for steps given as ints
        self._slots = torch.arange(p, device=self.device)
        self._hslots = torch.arange(h, device=self.device)

    # -- schedule ---------------------------------------------------------------------------------

    def _uniform_ns(self, n: int, lo: int, hi: int) -> torch.Tensor:
        u = torch.rand(n, generator=self.generator, device=self.device, dtype=torch.float64)
        return (lo + (hi - lo) * u).round().to(torch.int64)

    def reset(self, env_ids: torch.Tensor, t_ns: int, value: torch.Tensor | None = None) -> None:
        """Restart ``env_ids`` at ``t_ns``: empty buffers, random sample phase and burst phase."""
        n = len(env_ids)
        self.next_sample[env_ids] = t_ns + self._uniform_ns(n, 0, self.cfg.sample_period_ns)
        self.next_burst[env_ids] = t_ns + self._uniform_ns(n, 0, self.cfg.burst_period_ns[1])
        self.seq.index_fill_(0, env_ids, 0)
        self.pending_count.index_fill_(0, env_ids, 0)
        s = self.samples
        s.stamp_ns.index_fill_(0, env_ids, -1)
        s.sample_ns.index_fill_(0, env_ids, -1)
        s.seq.index_fill_(0, env_ids, 0)
        s.count.index_fill_(0, env_ids, 0)
        if value is not None:
            self._prev[env_ids] = value

    # -- stepping ---------------------------------------------------------------------------------

    def set_noise(self, enabled: bool) -> None:
        """White noise on (the configured densities) or off (an ideal sensor), from the next sample. In place, so
        a captured step sees it."""
        self._sigma.copy_(self._nominal_sigma if enabled else torch.zeros_like(self._sigma))

    def step(self, t_ns: int | torch.Tensor, gyro: torch.Tensor, specific_force: torch.Tensor, *,
             period_ns: int | None = None) -> None:
        """Ideal sensor-frame values ``[E, 3]`` at step boundary ``t_ns``: measure every sample due in
        ``(previous boundary, t_ns]``, then deliver every burst due by ``t_ns``.

        ``t_ns`` is an int, or a 0-d int64 device tensor with ``period_ns`` (the step length, an int) giving the
        loops' trip counts."""
        value = torch.cat([gyro, specific_force], dim=1).to(self.dtype)
        on_device = isinstance(t_ns, torch.Tensor)
        if on_device and period_ns is None:
            raise ValueError("a device-tensor step time needs period_ns")
        t0, t1 = (self._prev_ns, t_ns) if on_device else (self._prev_host, t_ns)
        span = period_ns if on_device else t1 - t0
        if span > 0:
            dt = (t1 - t0).clamp(min=1).to(torch.float64) if on_device else float(span)
            rotation = rotvec_matrix(self.misalignment)
            for _ in range(-(-span // self.cfg.sample_period_ns) + 1):
                due = self.next_sample <= t1
                frac = ((self.next_sample - t0).clamp(min=0).to(torch.float64) / dt).to(self.dtype)
                v = self._prev + frac.unsqueeze(1) * (value - self._prev)
                self._measure(due, self.next_sample.clone(), v, rotation)
                self.next_sample.copy_(torch.where(due, self.next_sample + self.cfg.sample_period_ns,
                                                   self.next_sample))
            for _ in range(-(-span // self.cfg.burst_period_ns[0]) + 1):
                due = self.next_burst <= t1
                self._deliver(due, self.next_burst.clone())
                step = self._uniform_ns(self.num_envs, *self.cfg.burst_period_ns)
                self.next_burst.copy_(torch.where(due, self.next_burst + step, self.next_burst))
        self._prev.copy_(value)
        if on_device:
            self._prev_ns.copy_(t1)
        else:
            self._prev_host = t1
            self._prev_ns.fill_(t1)

    def _measure(self, due: torch.Tensor, at_ns: torch.Tensor, ideal: torch.Tensor, r: torch.Tensor) -> None:
        gyro = (r @ ideal[:, :3].unsqueeze(-1)).squeeze(-1) + self.gyro_bias
        acc = (r @ ideal[:, 3:].unsqueeze(-1)).squeeze(-1) + self.accel_bias
        noise = torch.randn(self.num_envs, 6, generator=self.generator, device=self.device, dtype=self.dtype)
        sample = torch.cat([gyro, acc], dim=1) + noise * self._sigma
        self.seq.add_(due.long())
        full = due & (self.pending_count >= self.cfg.pending)
        self._drop_oldest_pending(full)
        slot = self.pending_count.clamp(max=self.cfg.pending - 1)
        write = due.unsqueeze(1) & (self._slots.unsqueeze(0) == slot.unsqueeze(1))
        self.pending_values.copy_(torch.where(write.unsqueeze(-1), sample.unsqueeze(1), self.pending_values))
        self.pending_ns.copy_(torch.where(write, at_ns.unsqueeze(1), self.pending_ns))
        self.pending_seq.copy_(torch.where(write, self.seq.unsqueeze(1), self.pending_seq))
        self.pending_count.copy_((self.pending_count + due.long()).clamp(max=self.cfg.pending))

    def _drop_oldest_pending(self, mask: torch.Tensor) -> None:
        self._shift_pending(mask.long())

    def _shift_pending(self, n: torch.Tensor) -> None:
        """Drop the ``n[e]`` oldest pending samples of each env."""
        src = (self._slots.unsqueeze(0) + n.unsqueeze(1)).clamp(max=self.cfg.pending - 1)
        self.pending_values.copy_(self.pending_values.gather(1, src.unsqueeze(-1).expand(-1, -1, 6)))
        self.pending_ns.copy_(self.pending_ns.gather(1, src))
        self.pending_seq.copy_(self.pending_seq.gather(1, src))
        self.pending_count.sub_(n)

    def _deliver(self, due: torch.Tensor, at_ns: torch.Tensor) -> None:
        """Deliver, for the ``due`` envs, every pending sample measured by ``at_ns``: to the front of the
        history, newest first, stamped on receipt (the oldest at ``at_ns``)."""
        held = (self._slots.unsqueeze(0) < self.pending_count.unsqueeze(1)) & (self.pending_ns <= at_ns.unsqueeze(1))
        n = torch.where(due, held.sum(dim=1), torch.zeros_like(self.pending_count))  # [E]
        s, h = self.samples, self.cfg.history
        # New history slot j < n takes pending sample n-1-j; slot j >= n takes old history slot j-n.
        j = self._hslots.unsqueeze(0)
        fresh = j < n.unsqueeze(1)
        p_idx = (n.unsqueeze(1) - 1 - j).clamp(min=0, max=self.cfg.pending - 1)
        h_off = j - n.unsqueeze(1)
        h_idx = h_off.clamp(min=0, max=h - 1)

        def merge(hist, pend, empty):
            extra = (1,) * (hist.dim() - 2)
            pv = pend.gather(1, p_idx.reshape(p_idx.shape + extra).expand(-1, -1, *pend.shape[2:]))
            hv = hist.gather(1, h_idx.reshape(h_idx.shape + extra).expand(-1, -1, *hist.shape[2:]))
            hv = torch.where((h_off < h).reshape(h_off.shape + extra), hv, torch.full_like(hv, empty))
            return torch.where(fresh.reshape(fresh.shape + extra), pv, hv)

        stamps = at_ns.unsqueeze(1) + p_idx * self.cfg.in_burst_spacing_ns
        stamp_ns = torch.where(fresh, stamps, merge(s.stamp_ns, self.pending_ns, -1))
        s.gyro.copy_(merge(s.gyro, self.pending_values[..., :3], 0.0))
        s.specific_force.copy_(merge(s.specific_force, self.pending_values[..., 3:], 0.0))
        s.sample_ns.copy_(merge(s.sample_ns, self.pending_ns, -1))
        s.seq.copy_(merge(s.seq, self.pending_seq, 0))
        s.stamp_ns.copy_(stamp_ns)
        s.count.add_(n)
        self._shift_pending(n)
