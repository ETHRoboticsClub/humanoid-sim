"""SimCore: the plant core on Isaac Lab / Newton, stepped on an integer-ns clock.

One ``step()`` advances one physics period: plant → targets → solver → feedback. Isaac Lab on Newton is the only
physics in this repo (on CUDA, or on Warp's CPU device for tests); ``solver`` selects Newton's solver by name
(``mjwarp`` or ``kamino``), the profile's when not given.

With ``base_joints`` (the asset variant built with them), :meth:`SimCore.set_base` prescribes the base
orientation as yaw, pitch and roll (``R = Rz Ry Rx``) and their rates, held through the next steps.

``scene`` (:class:`~humanoid_sim.world.boxes.StaticBoxes`) puts static boxes around the robot, which starts at
the world origin (the scene's spawn point).

``gantry`` (:class:`~humanoid_sim.world.supports.Gantry`) hangs the robot from a rope from the next step on;
``fixed_base`` suspends it (the base welded to the world).

With ``imu``, :attr:`SimCore.imu` produces the raw IMU samples as the robot's host receives them
(:class:`~humanoid_sim.sensors.imu.ImuDelivery`, sampled at every step start); with ``contacts``, the scene has a
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
from humanoid_sim.model.asset import check_asset_robot, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import BASE_JOINTS
from humanoid_sim.physics.joint_properties import joint_properties
from humanoid_sim.timing.clock import NS_PER_S
from humanoid_sim.world.boxes import StaticBoxes


class SimCore:
    def __init__(self, usd_path: Path, *, device: str, num_envs: int = 1, robot: str = "ethrc_v1",
                 period_ns: int | None = None, fixed_base: bool = False, base_height_m: float = 1.0,
                 solver: str | None = None, integrator: str | None = None, rerun: bool = False, rerun_rrd: str | None = None,
                 self_collision: bool | None = None, base_joints: bool = False, imu: bool = False,
                 contacts: bool = False, seed: int = 0, cuda_graph: bool = False, compile: bool = False,
                 imu_history: int | None = None, extra_rig: str | None = None, scene: StaticBoxes | None = None,
                 entities: dict | None = None, render_each_step: bool = True,
                 queue_capacity: int | None = None):
        check_asset_robot(usd_path, robot)
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
        self.profile = load_profile(robot)
        phys = self.profile["physics"]
        self.period_ns = int(period_ns or phys["period_ns"])
        from humanoid_sim.physics.isaaclab import IsaacLabCfg, IsaacLabRobot, solver_settings

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
                                        **({} if queue_capacity is None else {"queue_capacity": queue_capacity})),
                               solver_joint_names=physical if self.native_state is not None else None)
        props = joint_properties(self.plant, physical)
        period_s = self.period_ns / NS_PER_S
        self.backend = IsaacLabRobot(self.desc, usd_path, IsaacLabCfg(
            num_envs=num_envs, device=device, period_s=period_s, solver=self.solver, settings=settings,
            integrator=self.integrator, fixed_base=fixed_base, base_height_m=base_height_m,
            self_collision=phys["self_collision"] if self_collision is None else self_collision,
            base_joints=base_joints, ground_friction=self.profile["model"]["ground_friction"],
            imu=imu, extra_imus={n: s.frame for n, s in extra_imus.items()}, contacts=contacts, rerun=rerun,
            rerun_rrd=rerun_rrd, scene=scene, entities=entities,
            render_each_step=render_each_step),
            props=props)
        self.scene = scene
        order = [self.backend.joint_names.index(n) for n in joints]
        self._to_plant = torch.tensor(order, device=self.plant.cfg.device)
        self._to_solver = torch.tensor([self.backend.joint_names.index(n) for n in physical], device=device)
        if self.native_state is not None:
            self.write_joint_state(*self.joint_state())
        self._base = (torch.tensor([self.backend.joint_names.index(n) for n, _ in BASE_JOINTS],
                                   device=self.plant.cfg.device) if base_joints else None)
        self._base_target: tuple[torch.Tensor, torch.Tensor] | None = None
        from humanoid_sim.world.supports import Gantry

        harness = self.profile["support"]["harness"]
        self.gantry = Gantry(self.backend.robot, harness=None if harness is None else harness["body"])
        self.gantry.cuda_graph = cuda_graph
        self.t_ns = 0
        self.feedback: DriveFeedback | None = None
        self._state_written = True  # joint state changed outside a step: the plant's post-step state is stale
        self.imu = None
        if imu:
            from humanoid_sim.sensors import rigs
            from humanoid_sim.sensors.imu import ImuDelivery

            cfg = rigs.imu(self.desc).delivery_cfg()
            if imu_history is not None:
                cfg = dataclasses.replace(cfg, history=imu_history)
            self.imu = ImuDelivery(cfg, num_envs, device=device, seed=seed)
            self.imu.reset(torch.arange(num_envs, device=device), 0)
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

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd = self.backend.joint_state()
        return q[:, self._to_plant].to(self.plant.cfg.dtype), qd[:, self._to_plant].to(self.plant.cfg.dtype)

    def solver_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        q, qd = self.backend.joint_state()
        return q[:, self._to_solver].to(self.plant.cfg.dtype), qd[:, self._to_solver].to(self.plant.cfg.dtype)

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
        self.backend.write_state(bq, bqd)
        self._state_written = True

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

    def step(self) -> DriveFeedback:
        q, qd = self.solver_state()
        if self._graphs is not None:
            target = self._graphs.pre(q, qd, refresh=self._state_written)
        else:
            if self.imu is not None:
                data = self.backend.imu.data
                self.imu.step(self.t_ns, data.ang_vel_b.torch, data.lin_acc_b.torch)
            for name, delivery in self.extra_imus.items():
                data = self.backend.extra_imus[name].data
                delivery.step(self.t_ns, data.ang_vel_b.torch, data.lin_acc_b.torch)
            state = None if self._state_written else self.plant.post_state
            target = self.plant.pre_physics(self.t_ns, q, qd, state)
        bt, _ = self.backend.joint_state()
        bt = bt.clone()
        bt[:, self._to_solver] = target.to(bt.dtype)
        effort = torch.zeros_like(bt)
        effort[:, self._to_solver] = self.plant.passive_effort.to(effort.dtype)
        if self._base_target is not None:
            bv = torch.zeros_like(bt)
            bt[:, self._base] = self._base_target[0].to(bt.dtype)
            bv[:, self._base] = self._base_target[1].to(bt.dtype)
            self.backend.set_targets(bt, bv, effort=effort)
        else:
            self.backend.set_targets(bt, effort=effort)
        self.gantry.apply(self.t_ns)
        self.backend.step()
        self.t_ns += self.period_ns
        q, qd = self.solver_state()
        self.feedback = self._graphs.post(q, qd) if self._graphs is not None else self.plant.feedback(q, qd)
        self._state_written = False
        return self.feedback


class _StepGraphs:
    """The plant's per-step work as two CUDA graphs over static buffers: ``pre`` (IMU sampling at the step start,
    drive bank, target shaping) and ``post`` (drive feedback and the drive-frame state ``pre`` reuses). State lives
    in the plant's tensors, updated in place; the step time is a device scalar; the IMU's generator is registered
    with the graph so each replay draws new noise and burst times."""

    def __init__(self, sim: SimCore, warmup: int = 3, compile: bool = False):
        self.sim = sim
        if compile:
            self._post = torch.compile(self._post, dynamic=False)
            self._pre = torch.compile(self._pre, dynamic=False)
        p = sim.plant
        q, qd = sim.solver_state()
        self.q, self.qd = q.clone(), qd.clone()
        self.t = torch.zeros((), dtype=torch.int64, device=q.device)
        e = q.shape[0]
        self.gyro = torch.zeros(e, 3, device=q.device)
        self.acc = torch.zeros(e, 3, device=q.device)
        self.extra = {name: (torch.zeros(e, 3, device=q.device), torch.zeros(e, 3, device=q.device))
                      for name in sim.extra_imus}
        p.feedback(self.q, self.qd)  # a post-step state for pre to start from
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._post()
                self._pre()
        torch.cuda.current_stream().wait_stream(side)
        generators = ([sim.imu.generator] if sim.imu is not None else []) + [d.generator for d in sim.extra_imus.values()]
        self.g_post, self.g_pre = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        for g in (self.g_post, self.g_pre):
            for gen in generators:
                g.register_generator_state(gen)
        with torch.cuda.graph(self.g_post):
            self.feedback = self._post()
        with torch.cuda.graph(self.g_pre):
            self.target = self._pre()

    def _post(self) -> DriveFeedback:
        return self.sim.plant.feedback(self.q, self.qd)

    def _pre(self) -> torch.Tensor:
        sim = self.sim
        if sim.imu is not None:
            sim.imu.step(self.t, self.gyro, self.acc, period_ns=sim.period_ns)
        for name, delivery in sim.extra_imus.items():
            delivery.step(self.t, *self.extra[name], period_ns=sim.period_ns)
        return sim.plant.pre_physics(self.t, self.q, self.qd, sim.plant.post_state)

    def pre(self, q: torch.Tensor, qd: torch.Tensor, *, refresh: bool) -> torch.Tensor:
        sim = self.sim
        if refresh:  # the joint state was written: convert it again before the step
            self.q.copy_(q)
            self.qd.copy_(qd)
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

    def post(self, q: torch.Tensor, qd: torch.Tensor) -> DriveFeedback:
        self.q.copy_(q)
        self.qd.copy_(qd)
        self.g_post.replay()
        return self.feedback
