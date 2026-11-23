"""SimCore: the plant core on Isaac Lab / Newton, stepped on an integer-ns clock.

One ``step()`` advances one physics period: plant → targets → solver → feedback. Isaac Lab on Newton is the only
physics in this repo (on CUDA, or on Warp's CPU device for tests); ``solver`` selects Newton's solver by name
(``mjwarp`` or ``kamino``), the profile's when not given.

With ``base_joints`` (the asset variant built with them), :meth:`SimCore.set_base` prescribes the base
orientation as yaw, pitch and roll (``R = Rz Ry Rx``) and their rates, held through the next steps.

``scene`` (:class:`~humanoid_sim.world.boxes.StaticBoxes`) puts static boxes around the robot, which starts at
the world origin (the scene's spawn point). ``collision`` (:class:`~humanoid_sim.world.hulls.StaticHulls`) gives the
room's collision as convex pieces fitted to its scan instead; the boxes then stay on the stage for rays and drawing.

``terrain`` (a :class:`~humanoid_sim.world.terrain_io.Terrain`, or the Isaac Lab ``TerrainImporterCfg`` that
``heightfield_terrain_cfg`` gives training) puts a heightfield in place of the flat ground, built the way training
builds it. Each env's robot starts at its tile's origin, ``base_height_m`` above it (:attr:`SimCore.env_origins`);
:attr:`SimCore.ground` gives the surface height under any (x, y). Not with ``scene``.

``objects`` (:class:`~humanoid_sim.world.objects.WorldObjects`) adds bodies on scripted paths and free bodies the robot
can push. Each step first puts the moving bodies where their paths are at the plant time since the last
:meth:`reset_objects` (the start of the session), so a replay and the runtime see the same scene.

``gantry`` (:class:`~humanoid_sim.world.supports.Gantry`) hangs the robot from a rope from the next step on;
``fixed_base`` suspends it (the base welded to the world).

With ``imu``, :attr:`SimCore.imu` produces the raw IMU samples as the robot's host receives them
(:class:`~humanoid_sim.sensors.imu.ImuDelivery`, sampled at every step start, the chip rotated in its mount as the
profile's ``devices.imu_mount`` measured it); with ``contacts``, the scene has a
contact sensor on every body (``backend.contacts``). With ``imu`` and ``extra_rig`` (a name of
``sensors.rigs.EXTRAS``), :attr:`SimCore.extra_imus` holds the same for each further IMU of that rig, by name.

With ``cuda_graph`` (CUDA only) the plant's work around the solver step runs as two captured CUDA graphs, one
before the step (IMU sampling, drive bank, target shaping) and one after (drive feedback): one launch each instead
of hundreds of small kernels, which is what bounds a single robot's step time. ``compile`` also fuses those
kernels with ``torch.compile`` first (about a minute at construction, then roughly half the GPU time). The
feedback it returns is then a set of static buffers, overwritten by the next step: copy what you keep.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.devices.winding import profile_current
from humanoid_sim.model.asset import check_asset_robot, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import BASE_JOINTS
from humanoid_sim.physics.joint_properties import joint_properties
from humanoid_sim.timing.clock import NS_PER_S
from humanoid_sim.world.boxes import StaticBoxes
from humanoid_sim.world.hulls import StaticHulls
from humanoid_sim.world.objects import WorldObjects
from humanoid_sim.world.terrain_io import Terrain, as_terrain


class SimCore:
    def __init__(self, usd_path: Path, *, device: str, num_envs: int = 1, robot: str = "ethrc_v1",
                 period_ns: int | None = None, fixed_base: bool = False, base_height_m: float = 1.0,
                 solver: str | None = None, integrator: str | None = None, rerun: bool = False, rerun_rrd: str | None = None,
                 self_collision: bool | None = None, base_joints: bool = False, imu: bool = False,
                 contacts: bool = False, seed: int = 0, cuda_graph: bool = False, compile: bool = False,
                 imu_history: int | None = None, extra_rig: str | None = None, scene: StaticBoxes | None = None,
                 objects: WorldObjects | None = None, entities: dict | None = None, render_each_step: bool = True,
                 queue_capacity: int | None = None, terrain: Terrain | object | None = None,
                 physics_overrides: dict | None = None, gravity_m_s2: tuple[float, float, float] | None = None,
                 collision: StaticHulls | None = None):
        check_asset_robot(usd_path, robot)
        self.terrain = as_terrain(terrain)
        self.desc = Description.load(robot)
        # The IMUs of ``extra_rig`` (sensors/rigs.py EXTRAS) besides the plant's own, each simulated where its frame
        # is: ``SimCore.extra_imus[name]``, stepped with the plant's IMU.
        extra_imus = {}
        if extra_rig:
            if not imu:
                raise ValueError("extra_rig needs imu=True")
            from humanoid_sim.sensors import rigs

            extra_imus = {n: s for n, s in rigs.load(robot, self.desc, extra_rig).sensors.items()
                          if s.kind == "imu" and n != "imu"}
        self.profile = load_profile(robot, physics_overrides=physics_overrides)
        phys = self.profile["physics"]
        self.period_ns = int(period_ns or phys["period_ns"])
        from humanoid_sim.physics.isaaclab import (
            IsaacLabCfg,
            IsaacLabRobot,
            backlash_widths,
            compliance_gains,
            solver_settings,
            spring_gains,
        )

        self.solver, settings = solver_settings(phys, solver)
        self.integrator = integrator or settings["integrator"]
        joints = list(self.desc.joints)
        self.native_state = None
        physical = joints
        if self.desc.usd_path is not None:
            from humanoid_sim.model.native_state import NativeState
            from humanoid_sim.transmissions.drive_map import native_joint_names

            self.native_state = NativeState(self.desc, device=device)
            physical = native_joint_names(self.desc)
        # Description joint order for the plant; the articulation's order is mapped through names.
        self.plant = PlantCore(self.desc, joints, num_envs,
                               PlantCfg(robot=robot, physics_period_ns=self.period_ns, device=device,
                                        **profile_current(self.profile),
                                        **({} if queue_capacity is None else {"queue_capacity": queue_capacity})),
                               solver_joint_names=physical if self.native_state is not None else None)
        props = joint_properties(self.plant, physical)
        self.plant.enabled_friction[:] = torch.tensor(props.friction, dtype=self.plant.cfg.dtype)
        period_s = self.period_ns / NS_PER_S
        springs = spring_gains(usd_path, phys)
        from humanoid_sim.model.asset import backlash_output_inertia

        flange = backlash_output_inertia(usd_path)
        if flange is not None and springs:  # before the simulation exists: an unstable flange fails here
            from humanoid_sim.transmissions.backlash import check_output_stability

            k, b = zip(*springs.values(), strict=True)
            check_output_stability(torch.tensor(k), torch.tensor(b), flange, period_s)
        self.backend = IsaacLabRobot(self.desc, usd_path, IsaacLabCfg(
            num_envs=num_envs, device=device, period_s=period_s, solver=self.solver, settings=settings,
            integrator=self.integrator, fixed_base=fixed_base, base_height_m=base_height_m,
            self_collision=phys["self_collision"] if self_collision is None else self_collision,
            base_joints=base_joints, ground_friction=self.profile["model"]["ground_friction"],
            imu=imu, extra_imus={n: s.frame for n, s in extra_imus.items()}, contacts=contacts, rerun=rerun,
            rerun_rrd=rerun_rrd, scene=scene, collision=collision, objects=objects, entities=entities,
            render_each_step=render_each_step, terrain=None if self.terrain is None else self.terrain.cfg,
            springs=compliance_gains(usd_path, phys, self.desc) or None,
            backlash=backlash_widths(usd_path, phys) or None, gravity_m_s2=gravity_m_s2),
            props=props)
        self.scene = scene
        self.collision = collision
        self.ground = self.backend.ground
        self.env_origins = self.backend.env_origins
        self.objects = objects
        self._objects_epoch_ns = 0
        order = [self.backend.joint_names.index(n) for n in joints]
        self._to_plant = torch.tensor(order, device=self.plant.cfg.device)
        self._to_solver = torch.tensor([self.backend.joint_names.index(n) for n in physical], device=device)
        # Crank springs (an asset built with ankle compliance) in the description's crank drive order, and joint give
        # (built with joint_give) in the description's joint order: held at a zero target, written at zero with every
        # state write.
        from humanoid_sim.model.asset import backlash_joints, compliance_springs, give_springs

        self.springs = compliance_springs(usd_path)
        self.gives = give_springs(usd_path)
        self._springs = torch.tensor([self.backend.joint_names.index(n)
                                      for n in [*self.springs.values(), *self.gives.values()]],
                                     device=device, dtype=torch.long)
        # Crank backlash (an asset built with ankle backlash), in the description's crank drive order: no gain,
        # written at zero (mid-play) with every state write; the plant reports the output shaft past it.
        self.backlash = backlash_joints(usd_path)
        self._backlash = torch.tensor([self.backend.joint_names.index(n) for n in self.backlash.values()],
                                      device=device, dtype=torch.long)
        self._held = torch.cat([self._springs, self._backlash])  # joints written at zero
        if self.backlash:
            from humanoid_sim.transmissions.backlash import loop_encoder

            self.plant.use_backlash(loop_encoder(phys))
        if self.native_state is not None:
            self.write_joint_state(*self.joint_state())
        self._base = (torch.tensor([self.backend.joint_names.index(n) for n, _ in BASE_JOINTS],
                                   device=self.plant.cfg.device) if base_joints else None)
        self._base_target: tuple[torch.Tensor, torch.Tensor] | None = None
        self._base_acceleration: torch.Tensor | None = None
        from humanoid_sim.world.harness import resolve_harness
        from humanoid_sim.world.supports import Gantry

        self.gantry = Gantry(self.backend.robot, harness=resolve_harness(self.desc, self.profile["support"]["harness"]))
        self.gantry.cuda_graph = cuda_graph
        # Kamino's soft crank stops, when the profile asks for them (physics.kamino.crank_stops.soft): torques on the
        # crank joints every step. A plant that randomizes the stops adds them later (kamino_crank_stops).
        self.crank_stops = None
        if self.solver == "kamino":
            from humanoid_sim.physics.crank_stops import kamino_crank_stops

            try:
                self.crank_stops = kamino_crank_stops(self.backend.robot, self.desc, phys, period_s)
            except Exception:
                self.backend.close()  # Isaac Lab allows one simulation per process
                raise
        self.t_ns = 0
        self._feedback: DriveFeedback | None = None
        self._state_written = True  # joint state changed outside a step: the plant's post-step state is stale
        self.imu = None
        if imu:
            from humanoid_sim.sensors import rigs
            from humanoid_sim.sensors.imu import ImuDelivery, mount_misalignment_rad

            cfg = rigs.imu(self.desc).delivery_cfg()
            if imu_history is not None:
                cfg = dataclasses.replace(cfg, history=imu_history)
            self.imu = ImuDelivery(cfg, num_envs, device=device, seed=seed)
            self.imu.reset(torch.arange(num_envs, device=device), 0)
            self.imu.misalignment.copy_(torch.tensor(mount_misalignment_rad(self.profile), device=device))
        self.extra_imus = {}
        for i, (name, sensor) in enumerate(extra_imus.items(), 1):
            from humanoid_sim.sensors.imu import ImuDelivery

            cfg = sensor.delivery_cfg()
            if imu_history is not None:
                cfg = dataclasses.replace(cfg, history=imu_history)
            self.extra_imus[name] = ImuDelivery(cfg, num_envs, device=device, seed=seed + i)
            self.extra_imus[name].reset(torch.arange(num_envs, device=device), 0)
        if cuda_graph and not str(device).startswith("cuda"):
            raise ValueError("cuda_graph needs a CUDA device")
        self._graphs = _StepGraphs(self, compile=compile) if cuda_graph else None
        self.write_friction()

    def write_friction(self) -> None:
        """Write the back-drive joints' solver friction (:meth:`PlantCore.joint_friction`) if it changed."""
        plant = self.plant
        if not plant.friction_changed:
            return
        plant.friction_changed = False
        j = plant.backdrive_joints
        if len(j):
            self.backend.robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=plant.joint_friction()[:, j].float(), joint_ids=self._to_solver[j])

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd = self.backend.joint_state()
        return q[:, self._to_plant].to(self.plant.cfg.dtype), qd[:, self._to_plant].to(self.plant.cfg.dtype)

    def solver_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd = self.backend.joint_state()
        return q[:, self._to_solver].to(self.plant.cfg.dtype), qd[:, self._to_solver].to(self.plant.cfg.dtype)

    @property
    def feedback(self) -> DriveFeedback:
        """Drive feedback at the current joint state, also after a write without a physics step."""
        if self._feedback is None:
            self._read_backlash()
            self._feedback = self.plant.feedback(*self.solver_state())
        return self._feedback

    def write_joint_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        bq, bqd = self.backend.joint_state()
        bq, bqd = bq.clone(), bqd.clone()
        bq[:, self._to_plant] = q.to(bq.dtype)
        bqd[:, self._to_plant] = qd.to(bqd.dtype)
        if self.native_state is not None:
            values, rates = self.native_state.dofs(q, qd)
            for name in values:
                i = self.backend.joint_names.index(name)
                bq[:, i], bqd[:, i] = values[name].to(bq.dtype), rates[name].to(bqd.dtype)
        bq[:, self._held] = 0.0
        bqd[:, self._held] = 0.0
        self.backend.write_state(bq, bqd)
        self._state_written = True
        self._feedback = None

    def reset_objects(self) -> None:
        """Start the moving objects' paths again from the current plant time and put the free bodies back at their
        start poses, at rest."""
        self._objects_epoch_ns = self.t_ns
        if self.backend.objects is not None:
            self.backend.objects.reset()

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
            self._state_written = True

    def base_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Actual base-joint angles/rates, for prescribed-motion diagnostics."""
        if self._base is None:
            raise ValueError("base_state requires base-joint asset")
        q, v = self.backend.joint_state()
        return q[:, self._base], v[:, self._base]

    def set_base_acceleration(self, acceleration_w: torch.Tensor | None) -> None:
        """Equivalent translational inertial loading, not root motion.

        Finite input must be validated before stepping, as rotational targets are.
        This setter checks layout without synchronizing CUDA for every sample.
        None disables the correction. Values persist until replaced or cleared.
        """
        if acceleration_w is not None:
            if not (self.backend.cfg.fixed_base or self.backend.cfg.base_joints):
                raise ValueError("equivalent base acceleration requires fixed translation")
            if (acceleration_w.shape != (self.plant.num_envs, 3)
                    or acceleration_w.device != torch.device(self.plant.cfg.device)):
                raise ValueError("base acceleration must be [envs, 3] on the plant device")
            if self.gantry.active:
                raise ValueError("base acceleration replay cannot use an active gantry")
        self._base_acceleration = acceleration_w

    def _read_backlash(self) -> None:
        """Hand the plant each crank's play before it computes feedback."""
        if self.backlash:
            self.plant.write_backlash(self.backend.joint_state()[0][:, self._backlash].to(self.plant.cfg.dtype))

    def step(self) -> DriveFeedback:
        if self._graphs is not None:
            # The state was written outside a step: convert it again first. The targets reach the solver's buffers
            # inside the graph; only a prescribed base motion (replays) writes them from here.
            target = self._graphs.pre(refresh=self._state_written)
            if self._base_target is not None:
                self._write_targets(target)
        else:
            if self.imu is not None:
                data = self.backend.imu.data
                self.imu.step(self.t_ns, data.ang_vel_b.torch, data.lin_acc_b.torch)
            for name, delivery in self.extra_imus.items():
                data = self.backend.extra_imus[name].data
                delivery.step(self.t_ns, data.ang_vel_b.torch, data.lin_acc_b.torch)
            q, qd = self.solver_state()
            self._read_backlash()  # a state written since the last step is converted again before it
            state = None if self._state_written else self.plant.post_state
            self._write_targets(self.plant.pre_physics(self.t_ns, q, qd, state))
        self.write_friction()
        self.gantry.apply(self.t_ns)
        stops = self.crank_stops or getattr(self.backend.robot, "_kamino_crank_stops", None)
        if stops is not None:
            stops.apply()
        if self._base_acceleration is not None:
            self.backend.apply_base_acceleration(self._base_acceleration)
        if self.backend.objects is not None:
            self.backend.objects.update(self.t_ns - self._objects_epoch_ns)
        self.backend.step()
        self.t_ns += self.period_ns
        if self._graphs is not None:
            self._feedback = self._graphs.post()
        else:
            q, qd = self.solver_state()
            self._read_backlash()
            self._feedback = self.plant.feedback(q, qd)
        self._state_written = False
        return self.feedback

    def _target_buffers(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The solver's joint targets (solver joint order) and efforts for the step: the plant's shaped targets on
        the plant's joints, the current position elsewhere, the springs' at zero, and the plant's passive efforts."""
        bt, _ = self.backend.joint_state()
        bt = bt.clone()
        bt[:, self._to_solver] = target.to(bt.dtype)
        if len(self._springs):
            bt[:, self._springs] = 0.0
        effort = torch.zeros_like(bt)
        effort[:, self._to_solver] = self.plant.passive_effort.to(effort.dtype)
        return bt, effort

    def _write_targets(self, target: torch.Tensor) -> None:
        bt, effort = self._target_buffers(target)
        if self._base_target is not None:
            bv = torch.zeros_like(bt)
            bt[:, self._base] = self._base_target[0].to(bt.dtype)
            bv[:, self._base] = self._base_target[1].to(bt.dtype)
            self.backend.set_targets(bt, bv, effort=effort)
        else:
            self.backend.set_targets(bt, effort=effort)


class _StepGraphs:
    """The plant's per-step work as two CUDA graphs over static buffers: ``pre`` (IMU sampling at the step start,
    drive bank, target shaping, the joint targets and efforts written into the solver's command buffers) and
    ``post`` (the joint state in the plant's joint order, drive feedback and the drive-frame state ``pre`` reuses).
    State lives in the plant's tensors, updated in place; the step time is a device scalar; the IMUs' generators
    (noise and sample dropout) are registered with the graph so each replay draws new noise, burst times and drops.
    The solver's joint state and command buffers are the same memory every step, which is what lets the graphs read
    and write them; the IMUs' outputs are not (their data is re-viewed on every read), so ``pre`` copies them in."""

    def __init__(self, sim: SimCore, warmup: int = 3, compile: bool = False):
        self.sim = sim
        self._post_core, self._pre_core = sim.plant.feedback, self._pre_math
        if compile:
            self._post_core = torch.compile(self._post_core, dynamic=False)
            self._pre_core = torch.compile(self._pre_core, dynamic=False)
        p = sim.plant
        q, qd = sim.solver_state()
        self.q, self.qd = q.clone(), qd.clone()
        self.t = torch.zeros((), dtype=torch.int64, device=q.device)
        e = q.shape[0]
        self.gyro = torch.zeros(e, 3, device=q.device)
        self.acc = torch.zeros(e, 3, device=q.device)
        self.extra = {name: (torch.zeros(e, 3, device=q.device), torch.zeros(e, 3, device=q.device))
                      for name in sim.extra_imus}
        self.command = sim.backend.command_buffers()
        self._ptrs = self._pointers()
        p.feedback(self.q, self.qd)  # a post-step state for pre to start from
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._post()
                self._pre()
        torch.cuda.current_stream().wait_stream(side)
        generators = [g for imu in (sim.imu, *sim.extra_imus.values()) if imu is not None for g in imu.generators()]
        self.g_post, self.g_pre = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        for g in (self.g_post, self.g_pre):
            for gen in generators:
                g.register_generator_state(gen)
        with torch.cuda.graph(self.g_post):
            self.feedback = self._post()
        with torch.cuda.graph(self.g_pre):
            self.target = self._pre()

    def _pointers(self) -> list[int]:
        """The addresses of every solver-side buffer the graphs read or write; they must not move after capture."""
        sim = self.sim
        return [t.data_ptr() for t in (*sim.backend.joint_state(), *self.command)]

    def stable(self) -> bool:
        """Whether those buffers are still where the graphs captured them."""
        return self._pointers() == self._ptrs

    def _post(self) -> DriveFeedback:
        q, qd = self.sim.solver_state()
        self.q.copy_(q)
        self.qd.copy_(qd)
        self.sim._read_backlash()  # in place into the plant's play, read by the feedback
        return self._post_core(self.q, self.qd)

    def _pre_math(self) -> torch.Tensor:
        sim = self.sim
        if sim.imu is not None:
            sim.imu.step(self.t, self.gyro, self.acc, period_ns=sim.period_ns)
        for name, delivery in sim.extra_imus.items():
            delivery.step(self.t, *self.extra[name], period_ns=sim.period_ns)
        return sim.plant.pre_physics(self.t, self.q, self.qd, sim.plant.post_state)

    def _pre(self) -> torch.Tensor:
        sim = self.sim
        target = self._pre_core()
        bt, effort = sim._target_buffers(target)
        position, velocity, force = self.command
        position.copy_(bt)
        velocity.zero_()
        force.copy_(effort)
        return target

    def pre(self, *, refresh: bool) -> torch.Tensor:
        sim = self.sim
        if refresh:  # the joint state was written: convert it again before the step
            self.g_post.replay()
        if sim.imu is not None:
            data = sim.backend.imu.data
            self.gyro.copy_(data.ang_vel_b.torch)
            self.acc.copy_(data.lin_acc_b.torch)
        for name, (gyro, acc) in self.extra.items():
            data = sim.backend.extra_imus[name].data
            gyro.copy_(data.ang_vel_b.torch)
            acc.copy_(data.lin_acc_b.torch)
        self.t.fill_(sim.t_ns)
        self.g_pre.replay()
        return self.target

    def post(self) -> DriveFeedback:
        self.g_post.replay()
        return self.feedback
