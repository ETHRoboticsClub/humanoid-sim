"""Isaac Lab 3.0 / Newton: the robot USD as an Isaac Lab articulation, stepped by MJWarp (the only physics).

``use_mujoco_cpu`` is never set: MuJoCo C does not step the robot in this repo. On ``device="cpu"`` MJWarp
runs on Warp's CPU device (slow, for tests); real-time and batched use is on CUDA.

The solver PD gains are the plant's constant ``k0, d0`` (joint space), written once at spawn. Each step the
plant writes position targets (velocity target 0); nothing else changes in the model while stepping.
Passive joint damping and Coulomb friction, armature and effort limits come from the description.

With base joints (an asset built with them), the root is fixed to the world and the base orientation follows
position and velocity targets on the three base joints, through a stiff PD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import BASE_JOINTS
from humanoid_sim.physics.joint_properties import JointProperties

# Base-joint PD. Each joint carries the whole robot (a few kg m² about yaw, ~10 about pitch and roll at the
# pelvis), so 1e5 Nm/rad holds a 30 Nm waist reaction to 0.3 mrad and stays below the explicit-stiffness
# limit 4 I / dt² at 4 ms; the damping is implicit (implicitfast) and near critical.
BASE_JOINT_STIFFNESS = 1e5
BASE_JOINT_DAMPING = 2e3

ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"


@dataclass
class IsaacLabCfg:
    num_envs: int = 1
    device: str = "cpu"
    period_s: float = 0.004
    integrator: str = "implicitfast"
    fixed_base: bool = False
    base_height_m: float = 1.0
    env_spacing_m: float = 3.0
    iterations: int = 100
    ls_iterations: int = 50
    cone: str = "pyramidal"
    njmax: int = 300
    nconmax: int | None = None
    self_collision: bool = True
    joint_limits: str = "mujoco"  # see apply_joint_limits
    base_joints: bool = False
    ground_friction: float = 1.0
    imu: bool = False  # the IMU sensor at the description's mount
    contacts: bool = False  # a contact sensor on every body
    rerun: bool = False
    rerun_rrd: str | None = None


def apply_joint_limits(kind: str) -> None:
    """The joint-limit constraint of every DOF in the scene. ``mujoco``: MuJoCo's own (solref 0.02 s, damping
    ratio 1, scaled by each DOF's inertia), as Controls' MJCF has it; Newton writes it when the force-space limit
    gains are zero. ``newton``: Newton's force-space default, 1e4 Nm/rad whatever the inertia, in which 40 Nm on
    the hip abduction sinks 11 mrad past its stop (under 2 mrad with MuJoCo's). Applied from the next step."""
    if kind == "newton":
        return
    if kind != "mujoco":
        raise ValueError(f"joint limits {kind!r}: mujoco or newton")
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelFlags

    model = NewtonManager.get_model()
    model.joint_limit_ke.fill_(0.0)
    model.joint_limit_kd.fill_(0.0)
    NewtonManager.add_model_change(ModelFlags.JOINT_DOF_PROPERTIES)


def _per_joint(names, values):
    return {f"^{n}$": float(v) for n, v in zip(names, values, strict=True)}


def articulation_cfg(usd_path: Path, props: JointProperties, *, fixed_base: bool = False, base_joints: bool = False,
                     self_collision: bool = True, base_height_m: float = 1.0, contact_sensors: bool = False,
                     joint_pos: list[float] | None = None):
    """The robot as an Isaac Lab articulation: constant solver PD ``k0, d0`` and the description's passive
    properties per joint, initially at ``joint_pos`` (``props.names`` order; default: the standing pose); base
    joints (if the asset has them) on a stiff PD."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg

    joints = props.names
    actuators = {"drives": ImplicitActuatorCfg(
        joint_names_expr=[f"^{n}$" for n in joints],
        stiffness=_per_joint(joints, props.stiffness), damping=_per_joint(joints, props.damping),
        armature=_per_joint(joints, props.armature), friction=_per_joint(joints, props.friction),
        viscous_friction=_per_joint(joints, props.passive_damping),
        joint_effort_limit=_per_joint(joints, props.effort_limit),
    )}
    joint_pos = _per_joint(joints, props.standing if joint_pos is None else joint_pos)
    if base_joints:
        names = [name for name, _ in BASE_JOINTS]
        actuators["base"] = ImplicitActuatorCfg(
            joint_names_expr=[f"^{n}$" for n in names], stiffness=BASE_JOINT_STIFFNESS,
            damping=BASE_JOINT_DAMPING, armature=0.0, friction=0.0)
        joint_pos |= {f"^{n}$": 0.0 for n in names}
    return ArticulationCfg(
        prim_path=ROBOT_PRIM,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path), fix_root_link=fixed_base or base_joints,
            activate_contact_sensors=contact_sensors,
            articulation_props=[NewtonArticulationCfg(self_collision_enabled=self_collision)]),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, base_height_m), joint_pos=joint_pos, joint_vel={".*": 0.0}),
        actuators=actuators,
    )


def _quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion of URDF rpy (``R = Rz(yaw) Ry(pitch) Rx(roll)``)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def imu_sensor_cfg(robot: str = "ethrc_v1"):
    """Isaac Lab's IMU sensor at the description's mount (``imu_joint``): raw sensor-frame values."""
    from isaaclab.sensors import ImuCfg

    parent, xyz, rpy = Description.load(robot).imu_mount()
    return ImuCfg(prim_path=f"{ROBOT_PRIM}/Geometry/{parent}",
                  offset=ImuCfg.OffsetCfg(pos=tuple(xyz), rot=_quat_xyzw(*rpy)))


def contact_sensor_cfg(history: int = 3, air_time: bool = True):
    """Contact sensor on every body: net forces with history, air time."""
    from isaaclab.sensors import ContactSensorCfg

    return ContactSensorCfg(prim_path=f"{ROBOT_PRIM}/Geometry/.*", history_length=history, track_air_time=air_time)


def ground_cfg(friction: float):
    """Flat ground at z = 0. MJWarp takes the larger friction of a contact's two shapes."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

    material = RigidBodyMaterialBaseCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    return AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(physics_material=material))


def simulation_cfg(physics: dict, *, device: str, period_s: float, integrator: str | None = None,
                   njmax: int | None = None, nconmax: int | None = None, visualizers=None):
    """Isaac Lab on Newton with the MJWarp solver, configured from the profile's ``physics`` block."""
    import isaaclab.sim as sim_utils
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    solver = MJWarpSolverCfg(integrator=integrator or physics["integrator"], use_mujoco_cpu=False,
                             iterations=physics["iterations"], ls_iterations=physics["ls_iterations"],
                             cone=physics["cone"], njmax=njmax or physics["njmax"],
                             nconmax=nconmax or physics["nconmax"])
    return sim_utils.SimulationCfg(dt=period_s, device=device, physics=NewtonCfg(solver_cfg=solver),
                                   visualizer_cfgs=visualizers or None)


class IsaacLabRobot:
    """Scene with ground and robot; exposes joint state, target writes and stepping as torch tensors."""

    def __init__(self, desc: Description, usd_path: Path, cfg: IsaacLabCfg, *, props: JointProperties):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass, instantiate

        self.desc, self.cfg = desc, cfg
        joints = props.names

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            pass

        scene_cfg = SceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing_m, replicate_physics=True)
        scene_cfg.ground = ground_cfg(cfg.ground_friction)
        scene_cfg.robot = articulation_cfg(usd_path, props, fixed_base=cfg.fixed_base, base_joints=cfg.base_joints,
                                           self_collision=cfg.self_collision, base_height_m=cfg.base_height_m,
                                           contact_sensors=cfg.contacts)
        if cfg.imu:
            scene_cfg.imu = imu_sensor_cfg(desc.robot)
        if cfg.contacts:
            scene_cfg.contacts = contact_sensor_cfg(history=1, air_time=False)  # net forces now: the truth
        visualizers = []
        if cfg.rerun:
            from isaaclab_visualizers.rerun import RerunVisualizerCfg

            visualizers.append(RerunVisualizerCfg(record_to_rrd=cfg.rerun_rrd, keep_historical_data=True))
        physics = {"integrator": cfg.integrator, "iterations": cfg.iterations, "ls_iterations": cfg.ls_iterations,
                   "cone": cfg.cone, "njmax": cfg.njmax, "nconmax": cfg.nconmax}
        self.sim = sim_utils.SimulationContext(simulation_cfg(physics, device=cfg.device, period_s=cfg.period_s,
                                                              visualizers=visualizers))
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        apply_joint_limits(cfg.joint_limits)
        self.robot = self.scene["robot"]
        self.imu = self.scene["imu"] if cfg.imu else None
        self.contacts = self.scene["contacts"] if cfg.contacts else None
        # Start every env at the articulation's default (standing) joint state.
        default = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_position_to_sim_index(position=default)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default))
        self.scene.write_data_to_sim()
        self.joint_names = list(self.robot.joint_names)
        missing = set(joints) - set(self.joint_names)
        if missing:
            raise RuntimeError(f"articulation lacks description joints {sorted(missing)}")

    # -- state ------------------------------------------------------------------------------------

    def joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch

    def write_state(self, q: torch.Tensor, qd: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)

    def set_targets(self, target: torch.Tensor, velocity: torch.Tensor | None = None) -> None:
        tc = self.robot.actuators.target_command
        tc.set_position_index(value=target)
        tc.set_velocity_index(value=torch.zeros_like(target) if velocity is None else velocity)

    def step(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.period_s)

    def close(self) -> None:
        """Release the simulation (Isaac Lab allows one per process)."""
        import isaaclab.sim as sim_utils

        sim_utils.SimulationContext.clear_instance()

    def model_summary(self) -> dict:
        """Masses and names straight from the Newton model, for model checks."""
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        mass = model.body_mass.numpy()
        return {"bodies": model.body_count // max(self.cfg.num_envs, 1),
                "total_mass_kg": float(mass.sum()) / max(self.cfg.num_envs, 1),
                "shapes": model.shape_count // max(self.cfg.num_envs, 1),
                "joint_names": self.joint_names}
