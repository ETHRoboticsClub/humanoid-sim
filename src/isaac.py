"""Effort-only Isaac Lab hooks; importing this module does not start Isaac.

The caller owns physics stepping. In an Isaac Lab environment, call ``pre``
at each physics substep, then the existing scene write/step/update, then
``post(update_articulation=False)``. Never add a second simulation loop.
"""

from __future__ import annotations

from typing import Any

import torch

from .clock import TickScheduler


def _tensor(value):
    """Current Isaac Lab ProxyArray and ordinary Torch stubs share this path."""
    return value if isinstance(value, torch.Tensor) else value.torch


class IsaacEffortAdapter:
    """Translate physical state to devices and firmware torque to joint effort.

    ``transmission`` optionally implements ``device_state(q, dq)`` and
    ``joint_effort(q, device_effort)``. Without it, selected joints and devices
    correspond one to one. Joint order must be explicit and match this mapping.
    """

    def __init__(
        self,
        articulation: Any,
        devices: Any,
        *,
        physics_period_ns=400_000,
        joint_ids=None,
        transmission=None,
        check_finite=False,
    ):
        self.articulation, self.devices = articulation, devices
        self.joint_ids = list(joint_ids) if joint_ids is not None else None
        self.transmission = transmission
        self.check_finite = check_finite
        self.clock = TickScheduler(devices.firmware_period_ns, physics_period_ns)
        self.dt = physics_period_ns * 1e-9
        self._pending = False
        self._held_effort = torch.zeros(devices.shape, device=devices.device, dtype=devices.dtype)
        q, dq = self._joint_state()
        motor_q, motor_dq = self._device_state(q, dq)
        devices.sensors.validate(motor_q, "mapped position")
        devices.sensors.validate(motor_dq, "mapped velocity")
        self._validate_effort_only()

    def _joint_state(self):
        q = _tensor(self.articulation.data.joint_pos)
        dq = _tensor(self.articulation.data.joint_vel)
        if self.joint_ids is not None:
            q, dq = q[:, self.joint_ids], dq[:, self.joint_ids]
        if q.device != self.devices.device or q.dtype != self.devices.dtype:
            raise ValueError("Isaac joint state and DeviceBank must share device and dtype")
        return q, dq

    def _device_state(self, q, dq):
        if self.transmission is None:
            return q, dq
        return self.transmission.device_state(q, dq)

    def _validate_effort_only(self):
        # Validate live solver properties, rather than trusting config defaults.
        for key in ("joint_stiffness", "joint_damping"):
            value = _tensor(getattr(self.articulation.data, key))
            if self.joint_ids is not None:
                value = value[:, self.joint_ids]
            if not torch.equal(value, torch.zeros_like(value)):
                raise ValueError(f"{key} must be zero: firmware owns the drive controller")
        for config in self.articulation.cfg.actuators.values():
            if not getattr(config.class_type, "is_implicit_model", False):
                raise ValueError("explicit Isaac actuators would recompute firmware effort")

    def pre(self, *, write_to_sim=True):
        """Sample state, advance only on firmware ticks, and stage held effort."""
        if self._pending:
            raise RuntimeError("pre called twice without a completed physics step")
        event = self.clock.current()
        q, dq = self._joint_state()
        motor_q, motor_dq = self._device_state(q, dq)
        if event.firmware:
            self._held_effort = self.devices.advance(event.now_ns, motor_q, motor_dq)
        tau = self._held_effort
        if self.transmission is not None:
            tau = self.transmission.joint_effort(q, tau)
        if tau.shape != q.shape or (self.check_finite and not bool(torch.isfinite(tau).all())):
            raise ValueError("mapped effort must be finite and match selected joint state")
        self.articulation.actuators.target_command.set_effort_index(value=tau, joint_ids=self.joint_ids)
        if write_to_sim:
            self.articulation.write_data_to_sim()
        self._pending = True
        return tau.clone()

    def post(self, *, update_articulation=True, measured_device_effort=None):
        """Refresh sensors after the caller's single physics step.

        Supply measured effort only from an actual backend sensor. Isaac actuator
        ``applied_effort`` telemetry is a command estimate, not a torque sensor.
        """
        if not self._pending:
            raise RuntimeError("post requires one preceding pre and caller-owned physics step")
        if update_articulation:
            self.articulation.update(self.dt)
        q, dq = self._joint_state()
        motor_q, motor_dq = self._device_state(q, dq)
        self.devices.refresh_feedback(
            motor_q,
            motor_dq,
            measured_device_effort,
            timestamp_ns=self.clock.now_ns + self.clock.physics_period_ns,
        )
        self.clock.step()
        self._pending = False
        return self.devices.feedback

    def reset(self, env_ids=None, *, now_ns=None):
        """Clear firmware state after the caller has reset physical state."""
        if self._pending:
            raise RuntimeError("reset must happen between complete physics steps")
        q, dq = self._joint_state()
        motor_q, motor_dq = self._device_state(q, dq)
        if env_ids is None:
            epoch = self.clock.now_ns if now_ns is None else now_ns
            self.clock.reset(epoch)
            self.devices.reset(position=motor_q, velocity=motor_dq, now_ns=epoch)
            self._held_effort.zero_()
        else:
            if now_ns is not None:
                raise ValueError("selective reset preserves the global clock")
            self.devices.reset(env_ids, position=motor_q, velocity=motor_dq)
            self._held_effort[env_ids] = 0.0
        zero = torch.zeros_like(q if env_ids is None else q[env_ids])
        self.articulation.actuators.target_command.set_effort_index(
            value=zero, joint_ids=self.joint_ids, env_ids=env_ids
        )


def author_single_joint(stage, *, num_envs=1):
    """Author a fixed-base revolute pendulum from USD primitives, without downloads."""
    from pxr import Gf, UsdGeom, UsdPhysics

    for index in range(num_envs):
        root = f"/World/env_{index}/Drive"
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
        prim = UsdGeom.Xform.Define(stage, root).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        x = index * 0.4
        for name, position, scale, mass in (
            ("base", (x, 0.0, 1.0), (0.1, 0.1, 0.1), 0.1),
            ("link", (x, 0.0, 0.75), (0.04, 0.04, 0.5), 1.0),
        ):
            body = UsdGeom.Xform.Define(stage, f"{root}/{name}")
            body.AddTranslateOp().Set(Gf.Vec3d(*position))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
            # Keep joint-body frames unscaled; geometry alone supplies dimensions.
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}/geometry")
            cube.CreateSizeAttr(1.0)
            cube.AddScaleOp().Set(Gf.Vec3d(*scale))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.6, 0.8)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{root}/root_joint")
        fixed.CreateBody1Rel().SetTargets([f"{root}/base"])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(x, 0, 1))
        hinge = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/hinge")
        hinge.CreateBody0Rel().SetTargets([f"{root}/base"])
        hinge.CreateBody1Rel().SetTargets([f"{root}/link"])
        hinge.CreateAxisAttr("X")
        hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0.25))
        hinge.CreateLowerLimitAttr(-170.0)
        hinge.CreateUpperLimitAttr(170.0)
        drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
        drive.CreateMaxForceAttr(1e6)


def make_newton_simulation_config(
    *,
    device="cuda:0",
    physics_period_ns=400_000,
    visualizer="none",
    viewer_host="127.0.0.1",
    viewer_port=8080,
):
    """Build current pinned Isaac Lab config; no PhysX fallback or native PD."""
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

    visualizers = []
    if visualizer == "viser":
        from isaaclab_visualizers.viser import ViserVisualizerCfg

        visualizers = [
            ViserVisualizerCfg(
                port=viewer_port,
                bind_address=viewer_host,
                share=False,
                eye=(2.0, 2.0, 1.5),
                lookat=(0.0, 0.0, 0.75),
            )
        ]
    return SimulationCfg(
        device=device,
        dt=physics_period_ns * 1e-9,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(
            solver_cfg=FeatherstoneSolverCfg(angular_damping=0.0),
            num_substeps=1,
            use_cuda_graph=device.startswith("cuda"),
        ),
        use_newton_actuators=False,
        create_stage_in_memory=True,
        visualizer_cfgs=visualizers,
        render_interval=40,
    )


def make_single_joint_articulation():
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    return Articulation(
        ArticulationCfg(
            prim_path="/World/env_.*/Drive",
            spawn=None,
            actuators={
                "effort": ImplicitActuatorCfg(
                    joint_names_expr=["hinge"],
                    stiffness=0.0,
                    damping=0.0,
                    friction=0.0,
                    viscous_friction=0.0,
                    joint_effort_limit=1e6,
                )
            },
        )
    )
