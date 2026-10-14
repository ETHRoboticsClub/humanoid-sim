"""SimCore: the plant core on a physics backend, stepped on an integer-ns clock.

One ``step()`` advances one physics period: plant → targets → solver → feedback. Backends implement
``joint_state()``, ``write_state(q, qd)``, ``set_targets(target)`` and ``step()``; joint tensors are in the
backend's joint order, which the plant core is built with.
"""

from __future__ import annotations

from pathlib import Path

import torch

from humanoid_sim.core import DriveFeedback, PlantCfg, PlantCore
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.joint_properties import joint_properties
from humanoid_sim.timing.clock import NS_PER_S

BACKENDS = ("newton", "isaaclab")


class SimCore:
    def __init__(self, backend: str, usd_path: Path, *, num_envs: int = 1, robot: str = "ethrc_v1",
                 period_ns: int | None = None, fixed_base: bool = False, base_height_m: float = 1.0,
                 integrator: str | None = None, device: str = "cpu", use_mujoco_cpu: bool | None = None,
                 rerun: bool = False, rerun_rrd: str | None = None, self_collision: bool | None = None):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose one of {BACKENDS}")
        if use_mujoco_cpu is None:
            use_mujoco_cpu = backend == "newton" and device == "cpu"
        # isaaclab + use_mujoco_cpu goes through physics/isaaclab_cpu.py (works around an Isaac Lab 3.0 gap).
        self.desc = Description.load(robot)
        self.profile = load_profile(robot)
        phys = self.profile["physics"]
        self.period_ns = int(period_ns or phys["period_ns"])
        integrator = integrator or phys["integrator"]
        joints = list(self.desc.joints)
        # Description joint order for the plant; backends map to their own order through names.
        self.plant = PlantCore(self.desc, joints, num_envs,
                               PlantCfg(robot=robot, physics_period_ns=self.period_ns, device=device))
        props = joint_properties(self.plant, joints)
        period_s = self.period_ns / NS_PER_S
        if backend == "newton":
            from humanoid_sim.physics.newton_direct import NewtonCfg, NewtonRobot

            self.backend = NewtonRobot(usd_path, props, NewtonCfg(
                num_envs=num_envs, device=device, period_s=period_s, integrator=integrator,
                use_mujoco_cpu=use_mujoco_cpu, fixed_base=fixed_base, base_height_m=base_height_m,
                iterations=phys["iterations"], ls_iterations=phys["ls_iterations"], cone=phys["cone"]))
        else:
            from humanoid_sim.physics.isaaclab import IsaacLabCfg, IsaacLabRobot

            self.backend = IsaacLabRobot(self.desc, usd_path, IsaacLabCfg(
                num_envs=num_envs, device=device, period_s=period_s, integrator=integrator,
                use_mujoco_cpu=use_mujoco_cpu, fixed_base=fixed_base, base_height_m=base_height_m,
                iterations=phys["iterations"], ls_iterations=phys["ls_iterations"], cone=phys["cone"],
                njmax=phys["njmax"], nconmax=phys["nconmax"],
                self_collision=phys["self_collision"] if self_collision is None else self_collision,
                rerun=rerun, rerun_rrd=rerun_rrd), props=props)
        order = [self.backend.joint_names.index(n) for n in joints]
        self._to_plant = torch.tensor(order, device=self.plant.cfg.device)
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

    def step(self) -> DriveFeedback:
        q, qd = self.joint_state()
        target = self.plant.pre_physics(self.t_ns, q, qd)
        bt, _ = self.backend.joint_state()
        bt = bt.clone()
        bt[:, self._to_plant] = target.to(bt.dtype)
        self.backend.set_targets(bt)
        self.backend.step()
        self.t_ns += self.period_ns
        q, qd = self.joint_state()
        self.feedback = self.plant.feedback(q, qd)
        return self.feedback
