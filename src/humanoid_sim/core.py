"""PlantCore: everything between a drive demand and the physics solver, backend-independent.

One physics step ``[t0, t0 + period)``:

1. joint state → drive-frame state (1:1 signs, ankle crank transmission);
2. drive bank: demands in force during the step (arrival-blended) → demanded and applied drive torque;
3. drive torque → joint torque (``sign``, ankle ``Jᵀ``), plus any external joint torque (:attr:`external_torque`);
4. joint torque → position target for the constant solver PD (target shaping);
5. after the solver step: drive-frame feedback (position, velocity, applied and demanded torque).

The physics backend owns stepping; this class owns no solver state. Step 1 of a step converts the state step 5
of the previous step converted, so :meth:`PlantCore.feedback` keeps the drive-frame state and
:meth:`PlantCore.pre_physics` takes it (``state``) instead of running the transmission again. Times may be 0-d
int64 device tensors (CUDA graphs).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from humanoid_sim.devices.bank import DriveBank, DriveOutput
from humanoid_sim.devices.demand import JVPT_PER_SI_GAIN, DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.devices.feedback import DriveFeedback
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
        # Constant per-episode solver gains [E, J], written at reset only.
        self.k0 = k0.expand(num_envs, -1).clone()
        self.d0 = d0.expand(num_envs, -1).clone()
        self.actuated = self.map.actuated_joints
        self._jacs: list[torch.Tensor] = []
        self.post_state: tuple | None = None
        self.last: DriveOutput | None = None
        self.gain_warnings = torch.zeros(num_envs, self.map.num_drives, device=dev, dtype=torch.bool)
        # External torque on each joint [E, J] (a test's disturbance), on top of the drives' and outside their
        # feedback. Written in place.
        self.external_torque = torch.zeros(num_envs, len(joint_names), device=dev, dtype=dt)

    # -- demands ----------------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor, demand: DriveDemand | None = None) -> None:
        self.bank.reset(env_ids, demand)
        self.gain_warnings.index_fill_(0, env_ids, False)
        self.external_torque.index_fill_(0, env_ids, 0.0)

    def submit(self, demand: DriveDemand, arrival_ns: torch.Tensor, env_mask: torch.Tensor | None = None):
        # The solver gains are the baseline drive gains, so shaping is stable up to twice those. The check uses
        # the realized gains (randomized scales included) of the envs this demand reaches.
        bank = self.bank
        flagged = outside_stable_region(
            demand.p_gain_raw.to(self.cfg.dtype) / JVPT_PER_SI_GAIN * bank.p_scale,
            demand.d_gain_raw.to(self.cfg.dtype) / JVPT_PER_SI_GAIN * bank.d_scale,
            self.drive_stiffness, self.drive_damping) & demand.torque_enabled
        if env_mask is not None:
            flagged &= env_mask.unsqueeze(1)
        self.gain_warnings |= flagged
        self.bank.submit(demand, arrival_ns, env_mask)

    # -- stepping ---------------------------------------------------------------------------------

    def pre_physics(self, t0_ns: int | torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
                    state: tuple | None = None) -> torch.Tensor:
        """Joint position targets ``[E, J]`` for the step starting at ``t0_ns`` (velocity target 0). ``state``: the
        drive-frame state of ``q, qd`` if already computed (``DriveJointMap.drive_state``)."""
        qdrv, vdrv, self._jacs = state if state is not None else self.map.drive_state(q, qd)
        out = self._keep("last", self.bank.compute(t0_ns, t0_ns + self.cfg.physics_period_ns, qdrv, vdrv))
        tau = self.map.joint_torque(out.applied, self._jacs) + self.external_torque
        target = q.clone()
        a = self.actuated
        target[:, a] = shaped_position_target(q[:, a], qd[:, a], tau[:, a], self.k0[:, a], self.d0[:, a])
        return target

    def feedback(self, q: torch.Tensor, qd: torch.Tensor) -> DriveFeedback:
        """Drive-frame feedback after the step (state ``q, q̇``; torques of the step just taken). The drive-frame
        state is kept as :attr:`post_state` for the next :meth:`pre_physics`."""
        qdrv, vdrv, jacs = self.map.drive_state(q, qd)
        self._keep("post_state", (qdrv, vdrv, tuple(jacs)))
        qdrv, vdrv, _ = self.post_state
        z = torch.zeros_like(qdrv)
        last = self.last
        return DriveFeedback(qdrv, vdrv, last.applied if last else z, last.demanded if last else z)

    def _keep(self, name: str, value):
        """Store a step output in buffers that persist across steps (copied in place after the first), so a
        captured CUDA graph that reads it sees every later step's value."""
        kept = getattr(self, name)
        if kept is None:
            leaves = [t.clone() for t in _leaves(value)]
            kept = _rebuild(value, iter(leaves))
            setattr(self, name, kept)
        else:
            for dst, src in zip(_leaves(kept), _leaves(value), strict=True):
                dst.copy_(src)
        return kept


def _leaves(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _leaves(v)]
    return [t for f in fields(value) for t in _leaves(getattr(value, f.name))]


def _rebuild(template, leaves):
    if isinstance(template, torch.Tensor):
        return next(leaves)
    if isinstance(template, (tuple, list)):
        return type(template)(_rebuild(v, leaves) for v in template)
    return type(template)(**{f.name: _rebuild(getattr(template, f.name), leaves) for f in fields(template)})

