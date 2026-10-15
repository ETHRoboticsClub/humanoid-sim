"""Batched drive bank: demands in, drive-output torque out, in the physical drive frame.

Force law and clip follow Controls' MuJoCo plant (``ethrc_sim/plant.py`` ``Plant.step`` and
``plant_host.references_from_demands``), which runs behind Controls' fake EtherCAT bank:

* enabled and mode JVPT (-64): ``τ = kp (target − q) + kd (v_target − v) + ff``;
* enabled and mode freeze (-62) or homing (6): hold the position latched on entry, ``v_target = 0``, no ff;
* otherwise off: no torque;
* ``demanded`` = control-law output before any clip; ``applied`` = clipped by the torque–speed envelope
  (braking side bounded by the peak) and by 0x34C6:04 when set;
* a non-zero I-gain is refused (Controls raises and the plant faults): flagged in ``fault``, treated as off.

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

from humanoid_sim.devices.demand import JVPT_PER_SI_GAIN, MODE_FREEZE, MODE_HOMING, MODE_JVPT, DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.timing import ArrivalQueue

OFF, JVPT, FREEZE = 0, 1, 2


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
        self.fault = torch.zeros(num_envs, num_drives, device=device, dtype=torch.bool)
        ones = torch.ones(num_envs, num_drives, device=device, dtype=dtype)
        self.p_scale, self.d_scale, self.torque_scale = ones.clone(), ones.clone(), ones.clone()
        self.position_offset = torch.zeros_like(ones)

    def reset(self, env_ids: torch.Tensor, demand: DriveDemand | None = None) -> None:
        if demand is None:
            demand = DriveDemand.disabled(len(env_ids), self.num_drives, device=self.device, dtype=self.dtype)
        self.queue.reset(env_ids, demand.as_dict())
        self.last_kind[env_ids] = OFF
        self.fault[env_ids] = False

    def submit(self, demand: DriveDemand, arrival_ns: torch.Tensor, env_mask: torch.Tensor | None = None):
        """Queue a demand taking effect at ``arrival_ns`` ``[E]`` (or ``[E, D]``: per drive)."""
        self.queue.submit(demand.as_dict(), arrival_ns, env_mask)

    @staticmethod
    def _kind(rec: dict[str, torch.Tensor]) -> torch.Tensor:
        enabled = rec["torque_enabled"] & (rec["i_gain_raw"] == 0)
        mode = rec["mode"].to(torch.int64)
        kind = torch.where(enabled & (mode == MODE_JVPT), JVPT, OFF)
        kind = torch.where(enabled & ((mode == MODE_FREEZE) | (mode == MODE_HOMING)), FREEZE, kind)
        return kind.to(torch.int8)

    def compute(self, t0_ns: int, t1_ns: int, q: torch.Tensor, v: torch.Tensor) -> DriveOutput:
        """Torques for the step ``[t0, t1)`` from drive-frame state ``q, v`` [E, D] at its start."""
        w = self.queue.weights(t0_ns, t1_ns).to(self.dtype)  # [E, R, D]
        rec = self.queue.records()  # [E, R, D]
        kind = self._kind(rec)
        self.fault |= ((rec["i_gain_raw"] != 0) & rec["torque_enabled"] & (w > 0)).any(dim=1)

        # The drive acts on its own reading. Freeze latches it when a drive enters freeze.
        q = q + self.position_offset
        entering = (kind == FREEZE).any(dim=1) & (self.last_kind != FREEZE)
        self.frozen = torch.where(entering, q, self.frozen)

        qe, ve = q.unsqueeze(1), v.unsqueeze(1)
        kp = rec["p_gain_raw"].to(self.dtype) / JVPT_PER_SI_GAIN * self.p_scale.unsqueeze(1)
        kd = rec["d_gain_raw"].to(self.dtype) / JVPT_PER_SI_GAIN * self.d_scale.unsqueeze(1)
        jvpt = kind == JVPT
        target = torch.where(jvpt, rec["target_position_rad"], self.frozen.unsqueeze(1))
        vel = torch.where(jvpt, rec["target_velocity_rad_s"], torch.zeros_like(kp))
        ff = torch.where(jvpt, rec["feedforward_torque_nm"], torch.zeros_like(kp))
        tau = kp * (target - qe) + kd * (vel - ve) + ff
        tau = torch.where(kind == OFF, torch.zeros_like(tau), tau)

        ts = self.torque_scale.unsqueeze(1)
        applied = self.envelope.clamp(tau / ts, ve.expand_as(tau)) * ts
        cap = rec["max_torque_mnm"].to(self.dtype) / 1000.0 * ts
        applied = torch.where(cap > 0, torch.maximum(torch.minimum(applied, cap.abs()), -cap.abs()), applied)
        applied = torch.where(kind == OFF, torch.zeros_like(applied), applied)

        wk = w
        out = DriveOutput(
            demanded=(tau * wk).sum(1), applied=(applied * wk).sum(1),
            enabled_fraction=((kind != OFF).to(self.dtype) * wk).sum(1),
            target_velocity=(vel * (kind == JVPT) * wk).sum(1),
        )
        self.queue.advance(t1_ns)
        self.last_kind = self._kind(self.queue.active)
        return out
