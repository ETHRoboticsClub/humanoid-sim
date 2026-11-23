"""Feedback age: the drive feedback the controller holds is a sample, not the plant's present state.

The V1's master receives each drive's process data (PDO feedback) at whole bus cycles, but not every cycle: the
samples arrive 3 to 7.5 ms apart (p5 to p95), 5.08 ms on average, and the master holds the last one in between. The
plant otherwise reports the present state at every physics step. :class:`FeedbackSampler` holds a drive's reported
feedback between samples: a sample is taken when its time comes, and the next one ``k`` bus cycles later, ``k`` drawn
from the profile's measured distribution (``devices.feedback_sampling``, :meth:`FeedbackSampler.for_robot`).

Measured: pdo_feedback.csv of the V1 excitation captures exc-53f09771 and exc-5fd2bcca (2026-10-28), the
``t_source_ros_ns`` of each drive's sample: 35,814 spacings, every one a whole number of 1.5 ms bus cycles to within
0.1 ms. All 22 drives' samples of a frame carry the same stamp to within 17 us, so by default an env's drives are
sampled together (one draw per env); ``per_drive`` draws each drive's own.

Publish delay: a sample reaches the reader ``publish_delay_ns`` after it was taken (the profile's
``devices.feedback_sampling.publish_delay_ns``: 2.24 ms median on the V1, ``t_publish - t_source`` of the same
captures), so the reading's age is the held sample's age plus that. The delay is on the plant's step grid: a sample
is visible at the first step at or after its time. The first sample after a reset is visible at once.

Not modeled: the spacings are drawn independently, but the measured ones have a lag-1 autocorrelation of -0.40
(a 4 and a 6 ms spacing tend to follow each other, as for samples taken on a steady clock with jitter). The age of
a held sample at a random read time differs little from the independent draw's.

Batched ``[E, D]``; sample times are integer nanoseconds, ``t_ns`` an int or a 0-d int64 device tensor. Nothing here
syncs with the host. The draw uses the module's generator (like :mod:`~humanoid_sim.devices.reporting`'s jitter), so
a CUDA graph that contains :meth:`step` has to register it.
"""

from __future__ import annotations

import torch

from humanoid_sim.devices.feedback import DriveFeedback

_UNSET = -1  # next_ns of an env that has not sampled since its reset: due at the next step
_EMPTY = torch.iinfo(torch.int64).max  # ready time of a flight slot that holds no sample
_FIELDS = ("position_rad", "velocity_rad_s", "applied_torque_nm", "demanded_torque_nm")


class FeedbackSampler:
    def __init__(self, num_envs: int, num_drives: int, *, cycle_ns: int, cycles: list[int],
                 probability: list[float], per_drive: bool = False, publish_delay_ns: int = 0, device="cpu",
                 dtype=torch.float32, seed: int = 0):
        if not isinstance(cycle_ns, int) or cycle_ns <= 0:
            raise ValueError(f"cycle_ns {cycle_ns!r}: expected a positive integer of ns")
        if not isinstance(publish_delay_ns, int) or isinstance(publish_delay_ns, bool) or publish_delay_ns < 0:
            raise ValueError(f"publish_delay_ns {publish_delay_ns!r}: expected a non-negative integer of ns")
        if len(cycles) != len(probability) or not cycles:
            raise ValueError("feedback sampling needs one probability per cycle count")
        if any(not isinstance(c, int) or c < 1 for c in cycles):
            raise ValueError(f"cycle counts {cycles}: expected integers of at least 1")
        total = float(sum(probability))
        if any(p < 0 for p in probability) or total <= 0:
            raise ValueError(f"probabilities {probability}: expected non-negative with a positive sum")
        self.device = torch.device(device)
        self.num_envs, self.num_drives, self.dtype = num_envs, num_drives, dtype
        self.cycle_ns, self.per_drive, self.publish_delay_ns = cycle_ns, per_drive, publish_delay_ns
        self.cycles = torch.tensor(cycles, dtype=torch.int64, device=self.device)
        self._cdf = (torch.tensor(probability, dtype=torch.float64, device=self.device).cumsum(0) / total)[:-1]
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (num_envs, num_drives)
        self.next_ns = torch.full(shape, _UNSET, dtype=torch.int64, device=self.device)
        self.sample_ns = torch.full(shape, _UNSET, dtype=torch.int64, device=self.device)  # when the held one was taken
        z = torch.zeros(shape, dtype=dtype, device=self.device)
        self.held = DriveFeedback(z.clone(), z.clone(), z.clone(), z.clone())
        if publish_delay_ns:
            # Samples taken and not yet visible, per drive: at most one per step, so a few cycles' worth.
            self._slots = -(-publish_delay_ns // cycle_ns) + 3
            ring = (num_envs, num_drives, self._slots)
            self._flight = {name: torch.zeros(ring, dtype=dtype, device=self.device) for name in _FIELDS}
            self._ready_ns = torch.full(ring, _EMPTY, dtype=torch.int64, device=self.device)
            self._taken_ns = torch.zeros(ring, dtype=torch.int64, device=self.device)
            self._head = torch.zeros(shape, dtype=torch.int64, device=self.device)
            self._visible_ns = torch.full(shape, _UNSET, dtype=torch.int64, device=self.device)
            self._lanes = torch.arange(self._slots, device=self.device)

    @classmethod
    def for_robot(cls, desc, profile: dict, num_envs: int, num_drives: int, **kw) -> FeedbackSampler | None:
        """From the profile's ``devices.feedback_sampling`` (``cycles``, ``probability``) and the description's bus
        cycle; None when the profile has no measured sampling."""
        block = profile.get("devices", {}).get("feedback_sampling")
        if block is None:
            return None
        cycle_ns = desc.control_period_ns()
        if cycle_ns is None:
            raise ValueError(f"{desc.robot}: devices.feedback_sampling needs the description's backend control_period_s")
        kw.setdefault("publish_delay_ns", int(block.get("publish_delay_ns", 0)))
        return cls(num_envs, num_drives, cycle_ns=cycle_ns, cycles=[int(c) for c in block["cycles"]],
                   probability=[float(p) for p in block["probability"]], **kw)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Forget the samples of ``env_ids``: the next :meth:`step` takes a fresh one."""
        self.next_ns.index_fill_(0, env_ids, _UNSET)
        self.sample_ns.index_fill_(0, env_ids, _UNSET)
        if self.publish_delay_ns:
            self._ready_ns.index_fill_(0, env_ids, _EMPTY)
            self._head.index_fill_(0, env_ids, 0)
            self._visible_ns.index_fill_(0, env_ids, _UNSET)

    def step(self, t_ns: int | torch.Tensor, fb: DriveFeedback) -> DriveFeedback:
        """The feedback as held at ``t_ns``, given the present reading ``fb`` ``[E, D]``. A drive whose sample is
        due takes ``fb``; the others keep their last. Calling it again for the same ``t_ns`` changes nothing."""
        due = self.next_ns <= t_ns
        now = torch.zeros_like(self.next_ns) + t_ns
        if self.publish_delay_ns:
            self._take(due, now, fb)
            self._release(now)
        else:
            for name in _FIELDS:
                held = getattr(self.held, name)
                held.copy_(torch.where(due, getattr(fb, name).to(held.dtype), held))
            self.sample_ns.copy_(torch.where(due, now, self.sample_ns))
        # The next sample falls whole bus cycles after the one before it, not after this step: a sample taken a
        # little late (the step grid is not the bus's) does not push the later ones back.
        anchor = torch.where(self.next_ns == _UNSET, now, self.next_ns)
        following = torch.maximum(anchor + self._draw() * self.cycle_ns, now + 1)
        self.next_ns.copy_(torch.where(due, following, self.next_ns))
        h = self.held
        return DriveFeedback(h.position_rad.clone(), h.velocity_rad_s.clone(), h.applied_torque_nm.clone(),
                             h.demanded_torque_nm.clone())

    def _take(self, due: torch.Tensor, now: torch.Tensor, fb: DriveFeedback) -> None:
        """Put the samples ``due`` now in flight, visible ``publish_delay_ns`` on (at once for the first since a
        reset)."""
        write = due.unsqueeze(-1) & (self._lanes == (self._head % self._slots).unsqueeze(-1))
        for name in _FIELDS:
            flight = self._flight[name]
            flight.copy_(torch.where(write, getattr(fb, name).to(flight.dtype).unsqueeze(-1), flight))
        delay = torch.where(self.next_ns == _UNSET, torch.zeros_like(now), torch.full_like(now, self.publish_delay_ns))
        self._ready_ns.copy_(torch.where(write, (now + delay).unsqueeze(-1), self._ready_ns))
        self._taken_ns.copy_(torch.where(write, now.unsqueeze(-1), self._taken_ns))
        self._head.add_(due.long())

    def _release(self, now: torch.Tensor) -> None:
        """The newest sample whose delay has passed becomes the held one."""
        ready = torch.where(self._ready_ns <= now.unsqueeze(-1), self._ready_ns, torch.full_like(self._ready_ns, _UNSET))
        best, at = ready.max(dim=-1)
        newer = best > self._visible_ns
        pick = at.unsqueeze(-1)
        for name in _FIELDS:
            held = getattr(self.held, name)
            held.copy_(torch.where(newer, self._flight[name].gather(-1, pick).squeeze(-1), held))
        self.sample_ns.copy_(torch.where(newer, self._taken_ns.gather(-1, pick).squeeze(-1), self.sample_ns))
        self._visible_ns.copy_(torch.where(newer, best, self._visible_ns))

    def _draw(self) -> torch.Tensor:
        """Bus cycles to the next sample ``[E, D]``."""
        shape = (self.num_envs, self.num_drives if self.per_drive else 1)
        u = torch.rand(shape, generator=self.generator, device=self.device, dtype=torch.float64)
        k = self.cycles[torch.bucketize(u, self._cdf)]
        return k.expand(self.num_envs, self.num_drives)
