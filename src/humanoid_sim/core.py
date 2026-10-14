"""PlantCore: everything between a drive demand and the physics solver, backend-independent.

One physics step ``[t0, t0 + period)``:

1. joint state → drive-frame state (1:1 signs, ankle crank transmission);
2. drive bank: demands in force during the step (arrival-blended) → demanded and applied drive torque;
3. drive torque → joint torque (``sign``, ankle ``Jᵀ``);
4. joint torque → position target for the constant solver PD (target shaping);
5. after the solver step: drive-frame feedback (position, velocity, applied and demanded torque).

The physics backend owns stepping; this class owns no solver state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from humanoid_sim.devices.bank import DriveBank, DriveOutput
from humanoid_sim.devices.demand import JVPT_PER_SI_GAIN, DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.devices.shaping import outside_stable_region, shaped_position_target
from humanoid_sim.model.description import Description
from humanoid_sim.timing.clock import NS_PER_MS, check_period
from humanoid_sim.transmissions.drive_map import DriveJointMap


@dataclass
class PlantCfg:
    robot: str = "ethrc_v1"
    physics_period_ns: int = 4 * NS_PER_MS
    queue_capacity: int = 4
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass
class DriveFeedback:
    """Physical drive-frame feedback ``[E, D]``: what Controls' fake bank accepts (``fake_bank_feedback``)."""

    position_rad: torch.Tensor
    velocity_rad_s: torch.Tensor
    applied_torque_nm: torch.Tensor
    demanded_torque_nm: torch.Tensor


class PlantCore:
    def __init__(self, desc: Description, joint_names: list[str], num_envs: int, cfg: PlantCfg | None = None):
        self.cfg = cfg = cfg or PlantCfg()
        check_period("physics_period_ns", cfg.physics_period_ns)
        self.desc, self.num_envs = desc, num_envs
        dev, dt = cfg.device, cfg.dtype
        self.map = DriveJointMap(desc, joint_names, device=dev, dtype=dt)
        drives = [desc.drives[n] for n in self.map.drive_names]
        self.envelope = TorqueSpeedEnvelope([desc.torque_speed(d.family) for d in drives], device=dev, dtype=dt)
        self.bank = DriveBank(self.envelope, num_envs, self.map.num_drives, queue_capacity=cfg.queue_capacity,
                              device=dev, dtype=dt)
        self.drive_stiffness = torch.tensor([d.baseline_stiffness for d in drives], device=dev, dtype=dt)
        self.drive_damping = torch.tensor([d.baseline_damping for d in drives], device=dev, dtype=dt)
        standing = torch.tensor([desc.joints[n].standing_offset_rad if n in desc.joints else 0.0
                                 for n in joint_names], device=dev, dtype=dt)
        k0, d0 = self.map.joint_space_gains(self.drive_stiffness, self.drive_damping, standing)
        # Constant per-episode solver gains [E, J]; randomization writes these at reset only.
        self.k0 = k0.expand(num_envs, -1).clone()
        self.d0 = d0.expand(num_envs, -1).clone()
        self.actuated = self.map.actuated_joints
        self._jacs: list[torch.Tensor] = []
        self.last: DriveOutput | None = None
        self.gain_warnings = torch.zeros(num_envs, self.map.num_drives, device=dev, dtype=torch.bool)

    # -- demands ----------------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor, demand: DriveDemand | None = None) -> None:
        self.bank.reset(env_ids, demand)
        self.gain_warnings[env_ids] = False

    def submit(self, demand: DriveDemand, arrival_ns: torch.Tensor, env_mask: torch.Tensor | None = None):
        # The solver gains are the baseline drive gains, so shaping is stable up to twice those.
        self.gain_warnings |= outside_stable_region(
            demand.p_gain_raw.to(self.cfg.dtype) / JVPT_PER_SI_GAIN,
            demand.d_gain_raw.to(self.cfg.dtype) / JVPT_PER_SI_GAIN,
            self.drive_stiffness, self.drive_damping) & demand.torque_enabled
        self.bank.submit(demand, arrival_ns, env_mask)

    # -- stepping ---------------------------------------------------------------------------------

    def pre_physics(self, t0_ns: int, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """Joint position targets ``[E, J]`` for the step starting at ``t0_ns`` (velocity target 0)."""
        qdrv, vdrv, self._jacs = self.map.drive_state(q, qd)
        out = self.bank.compute(t0_ns, t0_ns + self.cfg.physics_period_ns, qdrv, vdrv)
        self.last = out
        tau = self.map.joint_torque(out.applied, self._jacs)
        target = q.clone()
        a = self.actuated
        target[:, a] = shaped_position_target(q[:, a], qd[:, a], tau[:, a], self.k0[:, a], self.d0[:, a])
        return target

    def feedback(self, q: torch.Tensor, qd: torch.Tensor) -> DriveFeedback:
        """Drive-frame feedback after the step (state ``q, q̇``; torques of the step just taken)."""
        qdrv, vdrv, _ = self.map.drive_state(q, qd)
        z = torch.zeros_like(qdrv)
        last = self.last
        return DriveFeedback(qdrv, vdrv, last.applied if last else z, last.demanded if last else z)
