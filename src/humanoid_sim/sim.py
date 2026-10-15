"""SimCore: the plant core on Isaac Lab / MJWarp, stepped on an integer-ns clock.

One ``step()`` advances one physics period: plant → targets → solver → feedback. Isaac Lab with Newton's
MJWarp solver is the only physics in this repo (on CUDA, or on Warp's CPU device for tests).

With ``base_joints`` (the asset variant built with them), :meth:`SimCore.set_base` prescribes the base
orientation as yaw, pitch and roll (``R = Rz Ry Rx``) and their rates, held through the next steps.
"""

from __future__ import annotations

from pathlib import Path

import torch

from humanoid_sim.core import DriveFeedback, PlantCfg, PlantCore
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.joint_properties import joint_properties
from humanoid_sim.model.urdf import BASE_JOINTS
from humanoid_sim.timing.clock import NS_PER_S


class SimCore:
    def __init__(self, usd_path: Path, *, device: str, num_envs: int = 1, robot: str = "ethrc_v1",
                 period_ns: int | None = None, fixed_base: bool = False, base_height_m: float = 1.0,
                 integrator: str | None = None, rerun: bool = False, rerun_rrd: str | None = None,
                 self_collision: bool | None = None, base_joints: bool = False):
        self.desc = Description.load(robot)
        self.profile = load_profile(robot)
        phys = self.profile["physics"]
        self.period_ns = int(period_ns or phys["period_ns"])
        integrator = integrator or phys["integrator"]
        joints = list(self.desc.joints)
        # Description joint order for the plant; the articulation's order is mapped through names.
        self.plant = PlantCore(self.desc, joints, num_envs,
                               PlantCfg(robot=robot, physics_period_ns=self.period_ns, device=device))
        props = joint_properties(self.plant, joints)
        period_s = self.period_ns / NS_PER_S
        from humanoid_sim.physics.isaaclab import IsaacLabCfg, IsaacLabRobot

        self.backend = IsaacLabRobot(self.desc, usd_path, IsaacLabCfg(
            num_envs=num_envs, device=device, period_s=period_s, integrator=integrator,
            fixed_base=fixed_base, base_height_m=base_height_m,
            iterations=phys["iterations"], ls_iterations=phys["ls_iterations"], cone=phys["cone"],
            njmax=phys["njmax"], nconmax=phys["nconmax"],
            self_collision=phys["self_collision"] if self_collision is None else self_collision,
            base_joints=base_joints, ground_friction=self.profile["model"]["ground_friction"],
            rerun=rerun, rerun_rrd=rerun_rrd), props=props)
        order = [self.backend.joint_names.index(n) for n in joints]
        self._to_plant = torch.tensor(order, device=self.plant.cfg.device)
        self._base = (torch.tensor([self.backend.joint_names.index(n) for n, _ in BASE_JOINTS],
                                   device=self.plant.cfg.device) if base_joints else None)
        self._base_target: tuple[torch.Tensor, torch.Tensor] | None = None
        self.t_ns = 0
        self.feedback: DriveFeedback | None = None

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd = self.backend.joint_state()
        return q[:, self._to_plant].to(self.plant.cfg.dtype), qd[:, self._to_plant].to(self.plant.cfg.dtype)

    def write_joint_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        bq, bqd = self.backend.joint_state()
        bq, bqd = bq.clone(), bqd.clone()
        bq[:, self._to_plant] = q.to(bq.dtype)
        bqd[:, self._to_plant] = qd.to(bqd.dtype)
        self.backend.write_state(bq, bqd)

    def close(self) -> None:
        self.backend.close()

    def set_base(self, angles: torch.Tensor, rates: torch.Tensor, *, write_state: bool = False) -> None:
        """Base yaw, pitch, roll ``[E, 3]`` (rad) and their rates (rad/s) for the next steps; with ``write_state``
        also set them as the current state (start of a replay)."""
        if self._base is None:
            raise RuntimeError("the asset has no base joints (build it with base joints, open with base_joints=True)")
        self._base_target = (angles, rates)
        if write_state:
            bq, bqd = self.backend.joint_state()
            bq, bqd = bq.clone(), bqd.clone()
            bq[:, self._base] = angles.to(bq.dtype)
            bqd[:, self._base] = rates.to(bqd.dtype)
            self.backend.write_state(bq, bqd)

    def step(self) -> DriveFeedback:
        q, qd = self.joint_state()
        target = self.plant.pre_physics(self.t_ns, q, qd)
        bt, _ = self.backend.joint_state()
        bt = bt.clone()
        bt[:, self._to_plant] = target.to(bt.dtype)
        if self._base_target is not None:
            bv = torch.zeros_like(bt)
            bt[:, self._base] = self._base_target[0].to(bt.dtype)
            bv[:, self._base] = self._base_target[1].to(bt.dtype)
            self.backend.set_targets(bt, bv)
        else:
            self.backend.set_targets(bt)
        self.backend.step()
        self.t_ns += self.period_ns
        q, qd = self.joint_state()
        self.feedback = self.plant.feedback(q, qd)
        return self.feedback
