"""Newton backend without Isaac Lab: the robot USD in a Newton model, stepped by ``SolverMuJoCo``.

This is the CPU runtime path (``use_mujoco_cpu``: MuJoCo C) and the macOS-native path. Isaac Lab 3.0's
MJWarp manager cannot run ``use_mujoco_cpu`` (it asks the solver for ``get_max_contact_count``, which the
CPU path does not implement), and MJWarp kernels on Warp's CPU device run far below real time.
Same USD, same per-joint properties (:mod:`humanoid_sim.model.joint_properties`) as the Isaac Lab backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.model.joint_properties import JointProperties


@dataclass
class NewtonCfg:
    num_envs: int = 1
    device: str = "cpu"
    period_s: float = 0.004
    integrator: str = "implicitfast"
    use_mujoco_cpu: bool = True
    fixed_base: bool = False
    base_height_m: float = 1.0
    iterations: int = 100
    ls_iterations: int = 50
    cone: str = "pyramidal"
    ground: bool = True
    spacing_m: tuple[float, float, float] = (3.0, 3.0, 0.0)
    kinematics_only: bool = False  # model + FK only (viewers); no solver


class NewtonRobot:
    def __init__(self, usd_path: Path, props: JointProperties, cfg: NewtonCfg):
        import newton
        import warp as wp

        self.cfg = cfg
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.add_usd(str(usd_path), xform=wp.transform((0.0, 0.0, cfg.base_height_m), wp.quat_identity()),
                      floating=not cfg.fixed_base)
        labels = [lbl.rsplit("/", 1)[-1] for lbl in robot.joint_label]
        qstart, dstart = list(robot.joint_q_start), list(robot.joint_qd_start)
        name_to_dof, name_to_q = {}, {}
        for j, name in enumerate(labels):
            name_to_dof[name], name_to_q[name] = dstart[j], qstart[j]
        missing = [n for n in props.names if n not in name_to_dof]
        if missing:
            raise RuntimeError(f"USD lacks joints {missing}")
        self.joint_names = list(props.names)
        self._dof = [name_to_dof[n] for n in props.names]
        self._qi = [name_to_q[n] for n in props.names]
        for i, n in enumerate(props.names):
            d = name_to_dof[n]
            robot.joint_target_ke[d] = props.stiffness[i]
            robot.joint_target_kd[d] = props.damping[i]
            robot.joint_armature[d] = props.armature[i]
            robot.joint_friction[d] = props.friction[i]
            robot.joint_damping[d] = props.passive_damping[i]
            robot.joint_effort_limit[d] = props.effort_limit[i]
            robot.joint_target_mode[d] = int(newton.JointTargetMode.POSITION_VELOCITY)
            robot.joint_q[name_to_q[n]] = props.standing[i]
            robot.joint_target_q[d] = props.standing[i]
        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.replicate(robot, cfg.num_envs, spacing=cfg.spacing_m)
        if cfg.ground:
            scene.add_ground_plane()
        self.model = scene.finalize(device=cfg.device)
        if cfg.use_mujoco_cpu and cfg.num_envs > 1 and not cfg.kinematics_only:
            raise ValueError("MuJoCo C (use_mujoco_cpu) simulates one world; use MJWarp for several envs")
        self.solver = None if cfg.kinematics_only else newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_cpu=cfg.use_mujoco_cpu, integrator=cfg.integrator, cone=cfg.cone,
            iterations=cfg.iterations, ls_iterations=cfg.ls_iterations)
        self.state0, self.state1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state0)
        self._newton, self._wp = newton, wp
        per_q, per_d = len(robot.joint_q), len(robot.joint_qd)
        e = np.arange(cfg.num_envs)[:, None]
        dev = torch.device(str(self.model.device))
        self._qcols = torch.as_tensor(e * per_q + np.asarray(self._qi)[None], device=dev)
        self._dcols = torch.as_tensor(e * per_d + np.asarray(self._dof)[None], device=dev)
        tstart = list(robot.joint_target_q_start) if hasattr(robot, "joint_target_q_start") else qstart
        per_t = len(robot.joint_target_q)
        self._tcols = torch.as_tensor(e * per_t + np.asarray([tstart[labels.index(n)] for n in props.names])[None],
                                      device=dev)

    # Torch views share memory with the Warp arrays (CPU or CUDA): no copies through the host.
    def _t(self, arr):
        return self._wp.to_torch(arr)

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._t(self.state0.joint_q)[self._qcols], self._t(self.state0.joint_qd)[self._dcols]

    def write_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        self._t(self.state0.joint_q)[self._qcols] = q.to(self._t(self.state0.joint_q).dtype)
        self._t(self.state0.joint_qd)[self._dcols] = qd.to(self._t(self.state0.joint_qd).dtype)
        self._newton.eval_fk(self.model, self.state0.joint_q, self.state0.joint_qd, self.state0)

    def set_targets(self, target: torch.Tensor) -> None:
        # Targets are indexed like joint_q (coordinates) and joint_qd (dofs); revolute joints have one each.
        tq = self._t(self.control.joint_target_q)
        tq[self._tcols] = target.to(tq.dtype)
        self._t(self.control.joint_target_qd)[self._dcols] = 0.0

    def step(self) -> None:
        self.state0.clear_forces()
        self.solver.step(self.state0, self.state1, self.control, None, self.cfg.period_s)
        self.state0, self.state1 = self.state1, self.state0

    def body_poses(self) -> np.ndarray:
        return self.state0.body_q.numpy()
