"""RobotPlant: the batched plant inside an Isaac Lab scene, for training.

The environment owns the scene (:func:`~humanoid_sim.train.scene.robot_scene_cfg`) and the stepping; the plant
sits between demands and the solver:

* :meth:`RobotPlant.submit` queues one SI ``DriveDemand`` per env and drive, sent at ``sent_ns``; it
  takes effect after the randomized ``latency`` before the drive (per env and drive) and then the drive's own
  delay (its model's ``drive_delay_ns``, :mod:`humanoid_sim.devices.models`);
* :meth:`RobotPlant.pre_physics`, once before every physics step (an action term's ``apply_actions``): samples
  the sensors at the step start, runs the drive bank over the step, writes the solver PD targets and applies
  :attr:`RobotPlant.gantry` (a :class:`~humanoid_sim.world.supports.Gantry`: no rope until a task hangs one, and
  resets keep it);
* :meth:`RobotPlant.measured` is what the robot's host would read: drive feedback as the drive reports it (its
  position offset included, in whole encoder counts with ±1 count of jitter, velocity per controller sample;
  :class:`~humanoid_sim.devices.reporting.DriveReporting`), a short per-step history of it, and the raw IMU
  samples received so far (sensor frame, receipt stamps). :meth:`RobotPlant.truth` is everything else, for
  critics and rewards;
* :meth:`RobotPlant.reset` writes the state of the envs being reset and redraws their device parameters;
  :meth:`RobotPlant.set_level` redraws everything at a new randomization level (solver writes: rare);
* :meth:`RobotPlant.log` writes named signals of a few envs into a Rerun recording at plant time
  (:class:`~humanoid_sim.signals.log.SignalLog`, attached with :meth:`RobotPlant.attach_log`): a host copy per
  call, so for inspection runs, not steady-state training.

Time is integer nanoseconds from construction, one physics period per :meth:`pre_physics`. Joint order is the
description's; drive order is the description's drive order. No host syncs in steady state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.devices.models import drive_models
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.randomization import PARAMS, ParamSpec, Randomizer, parse_specs
from humanoid_sim.sensors import rigs
from humanoid_sim.sensors.imu import ImuDelivery, ImuSamples
from humanoid_sim.timing.clock import NS_PER_S
from humanoid_sim.world.supports import Gantry


@dataclass
class TrainPlantCfg:
    robot: str = "ethrc_v1"
    randomization: list[ParamSpec | dict] = field(default_factory=list)
    level: float = 1.0
    seed: int = 0
    feedback_history: int = 16  # physics steps of drive feedback kept for :meth:`RobotPlant.measured`
    reporting: bool = True  # measured drive feedback quantized as the drive reports it
    queue_capacity: int = 8  # demands in flight per env: ≥ (max latency + drive delay) / demand period + 1


@dataclass
class Measured:
    drive: DriveFeedback  # [E, D] as reported: offset included, quantized (TrainPlantCfg.reporting)
    history: DriveFeedback  # [E, H, D], newest first, one per physics step
    history_ns: torch.Tensor  # [E, H] time of each history entry (-1: none yet)
    imu: ImuSamples | None


@dataclass
class Truth:
    joint_pos: torch.Tensor  # [E, J] URDF joint coordinates
    joint_vel: torch.Tensor
    root_pose_w: torch.Tensor  # [E, 7] position, quaternion xyzw (env frame origin not subtracted)
    root_lin_vel_w: torch.Tensor  # [E, 3]
    root_ang_vel_b: torch.Tensor  # [E, 3]
    drive: DriveFeedback  # [E, D] physical drive frame, no offsets
    contact_forces_w: torch.Tensor | None  # [E, B, 3] net force per body


class RobotPlant:
    def __init__(self, robot, cfg: TrainPlantCfg | None = None, *, imu=None, contacts=None, physics_manager=None):
        """``robot``: the scene's articulation; ``imu`` / ``contacts``: its sensors (optional);
        ``physics_manager``: ``env.sim.physics_manager`` (needed to randomize shape friction)."""
        self.cfg = cfg = cfg or TrainPlantCfg()
        self.robot, self.imu_sensor, self.contacts = robot, imu, contacts
        self.desc = Description.load(cfg.robot)
        self.profile = load_profile(cfg.robot)
        self.num_envs = e = robot.num_instances
        self.device = torch.device(robot.device)
        self.period_ns = int(self.profile["physics"]["period_ns"])
        self.joints = list(self.desc.joints)
        self.joint_ids = torch.tensor([robot.joint_names.index(n) for n in self.joints], device=self.device)
        self.native_state = None
        physical = None
        if self.profile["model"].get("ankle_representation") == "native":
            from humanoid_sim.model.native_state import NativeState
            from humanoid_sim.transmissions.drive_map import native_joint_names

            physical = native_joint_names(self.desc)
            self.native_state = NativeState(self.desc, device=self.device)
        self.core = PlantCore(self.desc, self.joints, e, PlantCfg(robot=cfg.robot, physics_period_ns=self.period_ns,
                                                                  queue_capacity=cfg.queue_capacity,
                                                                  device=str(self.device)), solver_joint_names=physical)
        self.solver_joint_ids = torch.tensor([robot.joint_names.index(n) for n in self.core.solver_map.joint_names],
                                            device=self.device)
        self.drives = self.core.map.drive_names
        d = len(self.drives)
        self.t_ns = 0
        self.latency_ns = torch.zeros(e, d, dtype=torch.int64, device=self.device)  # before the drive, randomized
        models = drive_models(self.desc)
        self.drive_models = [models[n] for n in self.drives]
        self.drive_delay_ns = torch.tensor([m.drive_delay_ns for m in self.drive_models], dtype=torch.int64,
                                           device=self.device)
        self._targets = robot.data.default_joint_pos.torch.clone()
        self._physics_manager = physics_manager
        harness = self.profile["support"]["harness"]
        self.gantry = Gantry(robot, harness=None if harness is None else harness["body"])  # no rope until hung
        # Every env starts at the articulation's default joint state, the pose the scene chose (standing or flat
        # feet). (At spawn the joints sit at the URDF zero, the calibration pose.)
        robot.write_joint_state_to_sim_index(position=self._targets.clone(), velocity=torch.zeros_like(self._targets))

        h = cfg.feedback_history
        z = torch.zeros(e, h, d, device=self.device)
        self._hist = DriveFeedback(z.clone(), z.clone(), z.clone(), z.clone())
        self._hist_ns = torch.full((e, h), -1, dtype=torch.int64, device=self.device)

        self.reporting = None
        if cfg.reporting and self.profile.get("devices", {}).get("encoders"):  # else exact values
            from humanoid_sim.devices.reporting import DriveReporting

            self.reporting = DriveReporting.for_drives(self.desc, self.drives, self.profile, device=self.device,
                                                       seed=cfg.seed + 2)
        self.imu = None
        if imu is not None:
            self.imu = ImuDelivery(rigs.imu(self.desc).delivery_cfg(), e,
                                   device=self.device, seed=cfg.seed + 1)
            self.imu.reset(torch.arange(e, device=self.device), 0)

        specs = parse_specs(cfg.randomization)
        self._nominal_physics = self._read_physics()
        nominal = {name: v for name, v in self._nominal_physics.items()} | self._device_nominal()
        self._ratio = nominal["joint_friction_ratio"]
        used = {s.param for s in specs}
        if {"friction", "shape_friction"} <= used:
            raise ValueError("choose per-env friction or per-shape friction, not both")
        self.randomizer = Randomizer(specs, {k: v for k, v in nominal.items() if k in used},
                                     {"body": list(robot.body_names), "joint": self.core.solver_map.joint_names,
                                      "drive": self.drives,
                                      "shape": [f"shape_{i}" for i in range(nominal.get("shape_friction", torch.empty(e, 0)).shape[1])]},
                                     num_envs=e, device=self.device, seed=cfg.seed, level=cfg.level)
        from humanoid_sim.physics.crank_stops import check_crank_stops
        from humanoid_sim.physics.isaaclab import (
            apply_heightfield_base,
            apply_joint_limits,
            apply_loop_closure,
            running_solver,
        )

        if used & {"friction", "shape_friction"} and physics_manager is None:
            raise ValueError("randomizing friction needs the physics manager (env.sim.physics_manager)")
        self._solver = running_solver()
        check_crank_stops(cfg.robot, self.profile["physics"])
        apply_joint_limits(self.profile["physics"])
        apply_loop_closure(self.profile["physics"])
        apply_heightfield_base(self.profile["physics"])
        self._write_physics()
        self._write_device(torch.arange(e, device=self.device))

    # -- randomization ----------------------------------------------------------------------------

    def _read_physics(self) -> dict[str, torch.Tensor]:
        data, ids = self.robot.data, self.solver_joint_ids
        out = {
            "body_mass": data.body_mass.torch.clone(),
            "body_com": torch.zeros_like(data.body_com_pos_b.torch),
            "body_inertia": torch.ones_like(data.body_mass.torch),
            "joint_armature": data.joint_armature.torch[:, ids].clone(),
            "joint_friction": data.joint_friction_coeff.torch[:, ids].clone(),
            "joint_damping": data.joint_viscous_friction_coeff.torch[:, ids].clone(),
        }
        self._com0 = data.body_com_pos_b.torch.clone()
        self._inertia0 = data.body_inertia.torch.clone()
        if self._physics_manager is not None:
            mu = self._shape_friction()
            out["friction"] = mu[:, :1].clone()
            out["shape_friction"] = mu.clone()
        return out

    def _device_nominal(self) -> dict[str, torch.Tensor]:
        e, d = self.num_envs, len(self.drives)
        ones = torch.ones(e, d, device=self.device)
        zeros3 = torch.zeros(e, 3, device=self.device)
        static = self._nominal_physics["joint_friction"][0]
        dynamic = torch.tensor([self.desc.joints[n].dynamic_friction if n in self.desc.joints else 0.0
                                for n in self.core.solver_map.joint_names], device=self.device)
        ratio = torch.where(static > 0, dynamic / static.clamp(min=1e-12), torch.ones_like(static))
        return {"drive_stiffness": ones, "drive_damping": ones.clone(), "torque_limit": ones.clone(),
                "latency": torch.zeros_like(ones), "encoder_offset": torch.zeros_like(ones),
                "joint_friction_ratio": ratio.expand(e, -1).clone(),
                "imu_gyro_bias": zeros3, "imu_accel_bias": zeros3.clone(), "imu_misalignment": zeros3.clone()}

    def _shape_friction(self) -> torch.Tensor:
        import warp as wp

        model = self._physics_manager.get_model()
        return wp.to_torch(self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0])

    def _write_physics(self) -> None:
        """Write every randomized solver parameter, all envs (startup and level changes only)."""
        v, r = self.randomizer.values, self.robot
        if self._solver == "kamino":
            # Kamino cannot switch a DoF's armature, damping or friction on or off without being rebuilt.
            for name in ("joint_armature", "joint_friction", "joint_damping"):
                if name in v and bool((v[name] <= 0).any()):
                    raise ValueError(f"Kamino needs {name} above zero; the randomization drew zero or less")
        if "body_mass" in v:
            r.set_masses_index(masses=v["body_mass"])
        if "body_mass" in v or "body_inertia" in v:
            scale = torch.ones_like(self._nominal_physics["body_mass"])
            if "body_mass" in v:
                scale = v["body_mass"] / self._nominal_physics["body_mass"].clamp(min=1e-9)
            if "body_inertia" in v:
                scale = scale * v["body_inertia"]
            r.set_inertias_index(inertias=self._inertia0 * scale.unsqueeze(-1))
        if "body_com" in v:
            r.set_coms_index(coms=self._com0 + v["body_com"])
        if "joint_armature" in v:
            r.write_joint_armature_to_sim_index(armature=v["joint_armature"], joint_ids=self.solver_joint_ids)
        if "joint_friction" in v:
            r.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=v["joint_friction"],
                                                            joint_ids=self.solver_joint_ids)
        if "joint_damping" in v:
            r.write_joint_viscous_friction_coefficient_to_sim_index(joint_viscous_friction_coeff=v["joint_damping"],
                                                                    joint_ids=self.solver_joint_ids)
        if "friction" in v or "shape_friction" in v:
            from newton import ModelFlags

            mu = self._shape_friction()
            mu[:] = v["shape_friction" if "shape_friction" in v else "friction"].to(mu.dtype)
            self._physics_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def _write_device(self, env_ids: torch.Tensor) -> None:
        v, bank = self.randomizer.values, self.core.bank
        for name, target in (("drive_stiffness", bank.p_scale), ("drive_damping", bank.d_scale),
                             ("torque_limit", bank.torque_scale), ("encoder_offset", bank.position_offset)):
            if name in v:
                target[env_ids] = v[name][env_ids].to(target.dtype)
        if "latency" in v:
            self.latency_ns[env_ids] = (v["latency"][env_ids] * NS_PER_S).round().to(torch.int64)
        if "joint_friction_ratio" in v or "joint_friction" in v:
            # The relief follows the static friction in the solver, randomized or not.
            static = v.get("joint_friction", self._nominal_physics["joint_friction"])[env_ids]
            ratio = v["joint_friction_ratio"][env_ids] if "joint_friction_ratio" in v else self._ratio[env_ids]
            relief = self.core.friction_relief
            relief[env_ids] = (static * (1 - ratio.clamp(0, 1))).to(relief.dtype)
        if self.imu is not None:
            for name, target in (("imu_gyro_bias", self.imu.gyro_bias), ("imu_accel_bias", self.imu.accel_bias),
                                 ("imu_misalignment", self.imu.misalignment)):
                if name in v:
                    target[env_ids] = v[name][env_ids].to(target.dtype)

    def attach_log(self, log) -> None:
        """Log signals into ``log`` (a :class:`~humanoid_sim.signals.log.SignalLog`) from now on."""
        self._log = log

    def log(self, name: str, value: torch.Tensor, *, envs=(0,), group: str = "signals") -> None:
        """Log ``value`` (``[E]`` or ``[E, ...]``) of ``envs`` as ``/sim/{env}/{group}/{name}`` at the plant time."""
        if getattr(self, "_log", None) is None:
            raise RuntimeError("no signal log attached (RobotPlant.attach_log)")
        self._log.at(self.t_ns)
        self._log.log(name, value.detach().float().cpu().numpy(), group=group, envs=envs)

    def params(self) -> dict[str, torch.Tensor]:
        """Realized randomized parameters per env (units: :data:`~humanoid_sim.randomization.PARAMS`)."""
        return self.randomizer.values

    def set_level(self, level: float) -> None:
        """Redraw everything at ``level`` and write it (solver parameters too: expensive, keep rare)."""
        self.randomizer.set_level(level)
        self._write_physics()
        self._write_device(torch.arange(self.num_envs, device=self.device))

    # -- demands and stepping ---------------------------------------------------------------------

    def submit(self, demand: DriveDemand, sent_ns: torch.Tensor, env_mask: torch.Tensor | None = None) -> None:
        """Demands ``[E, D]`` sent at ``sent_ns`` ``[E]``; each drive applies them after the latency before it and its
        own delay."""
        arrival = sent_ns.to(torch.int64).unsqueeze(1) + self.latency_ns + self.drive_delay_ns
        self.core.submit(demand, arrival, env_mask)

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        return data.joint_pos.torch[:, self.joint_ids], data.joint_vel.torch[:, self.joint_ids]

    def solver_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        return data.joint_pos.torch[:, self.solver_joint_ids], data.joint_vel.torch[:, self.solver_joint_ids]

    def write_joint_state(self, env_ids: torch.Tensor, q: torch.Tensor, qd: torch.Tensor) -> None:
        """Write semantic body state, closing every native rod before physics resumes."""
        if self.native_state is None:
            self.robot.write_joint_state_to_sim_index(position=q, velocity=qd, joint_ids=self.joint_ids,
                                                      env_ids=env_ids)
            return
        values, rates = self.native_state.dofs(q, qd)
        names = list(values)
        ids = torch.tensor([self.robot.joint_names.index(n) for n in names], device=self.device)
        self.robot.write_joint_state_to_sim_index(
            position=torch.stack([values[n] for n in names], -1).to(q.dtype),
            velocity=torch.stack([rates[n] for n in names], -1).to(qd.dtype), joint_ids=ids, env_ids=env_ids)

    def pre_physics(self) -> None:
        """Before each physics step: sample sensors at the step start, run the drives, write the targets, apply
        the gantry rope."""
        q, qd = self.solver_state()
        fb = self.core.feedback(q, qd)
        self._push_history(fb)
        if self.imu is not None:
            data = self.imu_sensor.data
            self.imu.step(self.t_ns, data.ang_vel_b.torch, data.lin_acc_b.torch)
        target = self.core.pre_physics(self.t_ns, q, qd, self.core.post_state)  # feedback converted q above
        self._targets[:, self.solver_joint_ids] = target.to(self._targets.dtype)
        tc = self.robot.actuators.target_command
        tc.set_position_index(value=self._targets)
        tc.set_velocity_index(value=torch.zeros_like(self._targets))
        effort = torch.zeros_like(self._targets)
        effort[:, self.solver_joint_ids] = self.core.passive_effort.to(effort.dtype)
        tc.set_effort_index(value=effort)
        self.gantry.apply(self.t_ns)
        self.t_ns += self.period_ns

    def _push_history(self, fb: DriveFeedback) -> None:
        h = self._hist
        for name in ("position_rad", "velocity_rad_s", "applied_torque_nm", "demanded_torque_nm"):
            buf = getattr(h, name)
            setattr(h, name, torch.cat([getattr(fb, name).unsqueeze(1), buf[:, :-1]], dim=1))
        self._hist_ns = torch.cat([torch.full_like(self._hist_ns[:, :1], self.t_ns), self._hist_ns[:, :-1]], dim=1)

    # -- outputs ----------------------------------------------------------------------------------

    def _with_offset(self, fb: DriveFeedback, offset: torch.Tensor) -> DriveFeedback:
        return DriveFeedback(fb.position_rad + offset, fb.velocity_rad_s, fb.applied_torque_nm, fb.demanded_torque_nm)

    def measured(self) -> Measured:
        q, qd = self.solver_state()
        offset = self.core.bank.position_offset
        drive = self._with_offset(self.core.feedback(q, qd), offset)
        history = self._with_offset(self._hist, offset.unsqueeze(1))
        if self.reporting is not None:
            drive, history = self.reporting.report(drive), self.reporting.report(history)
        return Measured(drive, history, self._hist_ns, self.imu.samples if self.imu is not None else None)

    def truth(self) -> Truth:
        q, qd = self.joint_state()
        data = self.robot.data
        forces = self.contacts.data.net_forces_w.torch if self.contacts is not None else None
        return Truth(q, qd, data.root_link_pose_w.torch, data.root_com_vel_w.torch[:, :3],
                     data.root_com_ang_vel_b.torch, self.core.feedback(*self.solver_state()), forces)

    # -- reset ------------------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor, *, joint_pos: torch.Tensor | None = None,
              joint_vel: torch.Tensor | None = None, root_pose: torch.Tensor | None = None,
              root_vel: torch.Tensor | None = None, demand: DriveDemand | None = None) -> None:
        """Reset ``env_ids``: state (description joint order; root pose xyzw, world), drives off (or
        ``demand`` in force), sensors empty, device parameters redrawn."""
        r = self.robot
        if root_pose is not None:
            r.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        if root_vel is not None:
            r.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        if joint_pos is not None or joint_vel is not None or self.native_state is not None:
            q0, qd0 = self.joint_state()
            q = q0[env_ids] if joint_pos is None else joint_pos
            qd = (qd0[env_ids] if joint_pos is None else torch.zeros_like(q)) if joint_vel is None else joint_vel
            self.write_joint_state(env_ids, q, qd)
            self._targets[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)] = q.to(self._targets.dtype)
        self.core.reset(env_ids, demand)
        self.randomizer.draw_reset(env_ids)
        self._write_device(env_ids)
        self._hist_ns.index_fill_(0, env_ids, -1)
        if self.imu is not None:
            self.imu.reset(env_ids, self.t_ns)


__all__ = ["PARAMS", "Measured", "RobotPlant", "TrainPlantCfg", "Truth"]
