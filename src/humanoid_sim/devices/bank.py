"""Batched drive bank: demands in, drive-output torque out, in the physical drive frame.

Force law and clip follow Controls' MuJoCo plants (``ethrc_sim/plant.py`` ``Plant.step``, the Unitree board's
``board.py``), per the demand's law (:mod:`humanoid_sim.devices.demand`):

* PD: ``τ = kp (target − q) + kd (v_target − v) + ff``;
* FREEZE: the same about the position latched on entry, ``v_target = 0``, no ff;
* DISABLED: no torque;
* ``demanded`` = control-law output before any clip; ``applied`` = clipped by the torque-speed envelope and by
  the demand's torque limit.

Demands arrive through an :class:`~humanoid_sim.timing.ArrivalQueue` with one lane per drive, so each drive
has its own latency; the law is evaluated once per record in force during the step (at the step-start
state) and blended by covered time.

Per env and drive, the bank also carries the drive's deviations from what it is told (randomized in
training, identity otherwise): ``p_scale``/``d_scale`` (realized over demanded gain), ``torque_scale`` (on the
envelope and cap) and ``position_offset`` (the drive's position reading minus the truth, so it regulates the
offset reading).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from humanoid_sim.devices.demand import LAW_DISABLED, LAW_FREEZE, LAW_PD, DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.timing import ArrivalQueue

OFF, PD, FREEZE = LAW_DISABLED, LAW_PD, LAW_FREEZE


@dataclass
class DriveOutput:
    demanded: torch.Tensor  # [E, D] Nm, control-law output (what the drive's current demand reflects)
    applied: torch.Tensor  # [E, D] Nm, after envelope and cap
    enabled_fraction: torch.Tensor  # [E, D] fraction of the step the drive was not off
    target_velocity: torch.Tensor  # [E, D] time-weighted v_target (0 when off/frozen)


class DriveBank:
    def __init__(self, envelope: TorqueSpeedEnvelope, num_envs: int, num_drives: int, *,
                 queue_capacity: int = 4, device="cpu", dtype=torch.float32):
        self.envelope = envelope
        self.num_envs, self.num_drives = num_envs, num_drives
        self.device, self.dtype = torch.device(device), dtype
        initial = DriveDemand.disabled(num_envs, num_drives, device=device, dtype=dtype).as_dict()
        self.queue = ArrivalQueue(initial, queue_capacity, lanes=num_drives)
        self.frozen = torch.zeros(num_envs, num_drives, device=device, dtype=dtype)
        self.last_kind = torch.full((num_envs, num_drives), OFF, device=device, dtype=torch.int8)
        ones = torch.ones(num_envs, num_drives, device=device, dtype=dtype)
        self.p_scale, self.d_scale, self.torque_scale = ones.clone(), ones.clone(), ones.clone()
        self.position_offset = torch.zeros_like(ones)

    def reset(self, env_ids: torch.Tensor, demand: DriveDemand | None = None) -> None:
        if demand is None:
            demand = DriveDemand.disabled(len(env_ids), self.num_drives, device=self.device, dtype=self.dtype)
        self.queue.reset(env_ids, demand.as_dict())
        self.last_kind.index_fill_(0, env_ids, OFF)

    def submit(self, demand: DriveDemand, arrival_ns: torch.Tensor, env_mask: torch.Tensor | None = None):
        """Queue a demand taking effect at ``arrival_ns`` ``[E]`` (or ``[E, D]``: per drive)."""
        self.queue.submit(demand.as_dict(), arrival_ns, env_mask)

    def compute(self, t0_ns: int | torch.Tensor, t1_ns: int | torch.Tensor, q: torch.Tensor,
                v: torch.Tensor) -> DriveOutput:
        """Torques for the step ``[t0, t1)`` from drive-frame state ``q, v`` [E, D] at its start."""
        w = self.queue.weights(t0_ns, t1_ns).to(self.dtype)  # [E, R, D]
        rec = self.queue.records()  # [E, R, D]
        kind = rec["law"]

        # The drive acts on its own reading. Freeze latches it when a drive enters freeze: a freeze demand in
        # effect during this step (queued ones that haven't arrived, and spent slots, don't count).
        q = q + self.position_offset
        entering = ((kind == FREEZE) & (w > 0)).any(dim=1) & (self.last_kind != FREEZE)
        self.frozen.copy_(torch.where(entering, q, self.frozen))

        qe, ve = q.unsqueeze(1), v.unsqueeze(1)
        kp = rec["stiffness_nm_per_rad"] * self.p_scale.unsqueeze(1)
        kd = rec["damping_nm_s_per_rad"] * self.d_scale.unsqueeze(1)
        pd = kind == PD
        target = torch.where(pd, rec["target_position_rad"], self.frozen.unsqueeze(1))
        vel = torch.where(pd, rec["target_velocity_rad_s"], torch.zeros_like(kp))
        ff = torch.where(pd, rec["feedforward_torque_nm"], torch.zeros_like(kp))
        tau = kp * (target - qe) + kd * (vel - ve) + ff
        tau = torch.where(kind == OFF, torch.zeros_like(tau), tau)

        ts = self.torque_scale.unsqueeze(1)
        nonzero = torch.where(ts > 0, ts, torch.ones_like(ts))  # a scale of 0 (no power) applies no torque
        applied = self.envelope.clamp(tau / nonzero, ve.expand_as(tau)) * ts
        limit = rec["torque_limit_nm"].abs()
        cap = torch.where(torch.isinf(limit), limit, limit * ts)  # inf: no cap (and inf * 0 would be NaN)
        applied = torch.maximum(torch.minimum(applied, cap), -cap)
        applied = torch.where(kind == OFF, torch.zeros_like(applied), applied)

        wk = w
        out = DriveOutput(
            demanded=(tau * wk).sum(1), applied=(applied * wk).sum(1),
            enabled_fraction=((kind != OFF).to(self.dtype) * wk).sum(1),
            target_velocity=(vel * pd * wk).sum(1),
        )
        # The mode in force at the end of this step (the last record with a share of it), which the next step's
        # freeze entry compares against. Records are in arrival order.
        slots = torch.arange(w.shape[1], device=w.device).view(1, -1, 1).expand_as(w)
        last = torch.where(w > 0, slots, torch.zeros_like(slots)).amax(dim=1, keepdim=True)
        self.last_kind.copy_(torch.gather(kind, 1, last).squeeze(1))
        self.queue.advance(t1_ns)
        return out
