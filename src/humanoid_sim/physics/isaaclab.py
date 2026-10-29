"""Isaac Lab 3.0 / Newton: the robot USD as an Isaac Lab articulation, stepped by a Newton solver.

The solver is selected by name (:data:`SOLVERS`): ``mjwarp`` (MuJoCo Warp) or ``kamino`` (Newton's
maximal-coordinate solver, beta), each configured from its block of the profile's ``physics``. ``use_mujoco_cpu``
is never set: MuJoCo C does not step the robot in this repo. On ``device="cpu"`` the solver runs on Warp's CPU
device (slow, for tests); real-time and batched use is on CUDA.

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
from humanoid_sim.physics.crank_stops import check_crank_stops, crank_stop_hook
from humanoid_sim.physics.joint_properties import JointProperties
from humanoid_sim.world.boxes import StaticBoxes

# Base-joint PD. Each joint carries the whole robot (a few kg m² about yaw, ~10 about pitch and roll at the
# pelvis), so 1e5 Nm/rad holds a 30 Nm waist reaction to 0.3 mrad and stays below the explicit-stiffness
# limit 4 I / dt² at 4 ms; the damping is implicit (MJWarp implicitfast, Kamino's implicit PD) and near critical.
BASE_JOINT_STIFFNESS = 1e5
BASE_JOINT_DAMPING = 2e3

ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"

SOLVERS = ("mjwarp", "kamino")
# Where MJWarp's contacts come from: Newton's collision pipeline or MJWarp's own (the profile's ``contacts``).
CONTACTS = ("newton", "mujoco")
# Isaac Lab's ``solver_type`` of each solver config.
_SOLVER_TYPES = {"mujoco_warp": "mjwarp", "kamino": "kamino"}


@dataclass
class IsaacLabCfg:
    num_envs: int = 1
    device: str = "cpu"
    period_s: float = 0.004
    solver: str = "mjwarp"
    settings: dict | None = None  # the solver's block of the profile's physics
    integrator: str | None = None  # overrides the solver block's
    fixed_base: bool = False
    base_height_m: float = 1.0
    env_spacing_m: float = 3.0
    self_collision: bool = True
    base_joints: bool = False
    ground_friction: float = 1.0
    imu: bool = False  # the IMU sensor at the description's mount
    contacts: bool = False  # a contact sensor on every body
    scene: StaticBoxes | None = None  # static boxes around the robot (every env's robot then starts at the origin)
    rerun: bool = False
    rerun_rrd: str | None = None
    # More scene entities by name (Isaac Lab entity configs: cameras, assets), added before the scene is built, as
    # the RTX sensors and a room's splat need (sensors/rtx.py).
    entities: dict | None = None
    # Render (update the visualizers, run Kit's app) after every physics step. The RTX sensors turn it off and render
    # at their own rates (sensors/rtx.py KitRender): with Kit up a render per step costs far more than the step.
    render_each_step: bool = True


def solver_settings(physics: dict, solver: str | None = None) -> tuple[str, dict]:
    """The selected solver (``solver``, else the profile's) and its block of the profile's ``physics``."""
    name = physics["solver"] if solver is None else solver
    if name not in SOLVERS:
        raise ValueError(f"physics solver {name!r}: one of {', '.join(SOLVERS)}")
    if name not in physics:
        raise ValueError(f"the profile's physics has no {name!r} block")
    return name, physics[name]


def running_solver() -> str:
    """The solver of the current Isaac Lab simulation."""
    import isaaclab.sim as sim_utils

    sim = sim_utils.SimulationContext.instance()
    if sim is None:
        raise RuntimeError("no simulation context")
    kind = getattr(getattr(sim.cfg.physics, "solver_cfg", None), "solver_type", None)
    if kind not in _SOLVER_TYPES:
        raise RuntimeError(f"simulation runs solver {kind!r}, not one of {', '.join(SOLVERS)}")
    return _SOLVER_TYPES[kind]


def apply_joint_limits(physics: dict) -> None:
    """The joint-limit constraint of every DOF in the scene, for the running solver. Applied from the next step.

    MJWarp, the block's ``joint_limits``: ``mujoco``, MuJoCo's own (solref 0.02 s, damping ratio 1, scaled by
    each DOF's inertia), as Controls' MJCF has it; Newton writes it when the force-space limit gains are zero.
    ``newton``: Newton's force-space default, 1e4 Nm/rad whatever the inertia, in which 40 Nm on the hip
    abduction sinks 11 mrad past its stop (under 2 mrad with MuJoCo's). Kamino's stops are hard constraints and
    take no setting."""
    solver = running_solver()
    if solver == "kamino":
        return
    kind = physics[solver]["joint_limits"]
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


def apply_heightfield_base(physics: dict) -> None:
    """Give every heightfield of the running MJWarp model a solid base ``heightfield_base_m`` deep below its lowest
    point (the block's setting). Newton exports a heightfield to MuJoCo with a 0.1 mm base, and MuJoCo pushes a
    shape that sinks deeper into a cell's prism than the prism is thick out through its bottom: a G1 foot sphere
    (5 mm) landing on the lowest cells of a rough tile falls through the terrain. The base, the heightfield's
    bounding box and its bounding radius are rewritten together. Kamino collides heightfields in Newton's own
    pipeline and takes no setting."""
    if running_solver() != "mjwarp" or physics["mjwarp"]["contacts"] != "mujoco":
        return  # Newton's collision pipeline collides heightfields itself
    import mujoco_warp
    import numpy as np
    from isaaclab_newton.physics import NewtonManager

    solver = NewtonManager._solver
    if solver is None or getattr(solver, "mjw_model", None) is None:
        raise RuntimeError("no MJWarp model yet: apply the heightfield base after the simulation is reset")
    m = solver.mjw_model
    if m.hfield_size.shape[0] == 0:
        return
    base = float(physics["mjwarp"]["heightfield_base_m"])
    size = m.hfield_size.numpy()
    size[:, 3] = np.maximum(size[:, 3], base)
    m.hfield_size.assign(size)
    geoms = np.nonzero(m.geom_type.numpy() == int(mujoco_warp.GeomType.HFIELD))[0]
    dataid, aabb, rbound = m.geom_dataid.numpy(), m.geom_aabb.numpy(), m.geom_rbound.numpy()
    for g in geoms:
        hx, hy, top, bottom = size[dataid[0, g]]
        # MuJoCo's box around a heightfield in its frame: centre (0, 0, (top - bottom) / 2), half sizes
        # (hx, hy, (top + bottom) / 2); its bounding sphere encloses the box's corners.
        aabb[:, g, 0] = (0.0, 0.0, (top - bottom) / 2)
        aabb[:, g, 1] = (hx, hy, (top + bottom) / 2)
        rbound[:, g] = float(np.sqrt(hx**2 + hy**2 + max(top, bottom) ** 2))
    m.geom_aabb.assign(aabb)
    m.geom_rbound.assign(rbound)


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
    from humanoid_sim.model.asset import asset_meta

    try:
        meta = asset_meta(usd_path)
    except FileNotFoundError:
        meta = {}
    if meta.get("crank_joints"):
        from humanoid_sim.model.native_state import NativeState

        desc = Description.load(meta["robot"])
        semantic = list(desc.joints)
        positions = torch.tensor([[joint_pos[f"^{name}$"] for name in semantic]], dtype=torch.float64)
        native = NativeState(desc, semantic, dtype=torch.float64)
        joint_pos.update({f"^{name}$": float(position[0])
                          for name, (position, _) in native.state(positions, torch.zeros_like(positions)).items()})
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

    desc = Description.load(robot)
    parent, xyz, rpy = desc.imu_mount()
    parent_path = f"{ROBOT_PRIM}/Geometry/{parent}"
    if desc.usd_path is not None:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(desc.usd_path))
        matches = [prim for prim in stage.Traverse()
                   if prim.GetName() == parent and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(matches) != 1:
            raise ValueError(f"USD must contain one IMU parent body named {parent}")
        relative = matches[0].GetPath().MakeRelativePath(stage.GetDefaultPrim().GetPath())
        parent_path = f"{ROBOT_PRIM}/{relative}"
    return ImuCfg(prim_path=parent_path,
                  offset=ImuCfg.OffsetCfg(pos=tuple(xyz), rot=_quat_xyzw(*rpy)))


def contact_sensor_cfg(history: int = 3, air_time: bool = True):
    """Contact sensor on every body: net forces with history, air time."""
    from isaaclab.sensors import ContactSensorCfg

    return ContactSensorCfg(prim_path=f"{ROBOT_PRIM}/.*", history_length=history, track_air_time=air_time)


def ground_cfg(friction: float):
    """Flat ground at z = 0. Both solvers take the larger friction of a contact's two shapes."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

    material = RigidBodyMaterialBaseCfg(static_friction=friction, dynamic_friction=friction, restitution=0.0)
    return AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(physics_material=material))


def solver_cfg(solver: str, settings: dict, *, integrator: str | None = None):
    """Isaac Lab's config for ``solver`` from its block of the profile's ``physics``."""
    integrator = integrator or settings["integrator"]
    if solver == "mjwarp":
        from isaaclab_newton.physics import MJWarpSolverCfg

        contacts = settings["contacts"]
        if contacts not in CONTACTS:
            raise ValueError(f"MJWarp contacts {contacts!r}: one of {', '.join(CONTACTS)}")
        return MJWarpSolverCfg(integrator=integrator, use_mujoco_cpu=False, iterations=settings["iterations"],
                               ls_iterations=settings["ls_iterations"], cone=settings["cone"],
                               njmax=settings["njmax"], nconmax=settings["nconmax"],
                               use_mujoco_contacts=contacts == "mujoco")
    if solver == "kamino":
        from isaaclab_newton.physics import KaminoMaterialsCfg, KaminoPADMMCfg, KaminoPADMMSolverCfg

        if integrator != "euler":
            # Newton would fall back to euler: moreau needs Kamino's own collision detector, not used here.
            raise ValueError(f"Kamino integrator {integrator!r}: euler (contacts come from Newton's pipeline)")

        return KaminoPADMMSolverCfg(integrator=integrator,
                                    dynamics_solver_cfg=KaminoPADMMCfg(max_iterations=settings["iterations"]),
                                    max_contacts_per_world=settings["max_contacts_per_world"],
                                    materials=KaminoMaterialsCfg(friction_mix_mode="max"))
    raise ValueError(f"physics solver {solver!r}: one of {', '.join(SOLVERS)}")


def simulation_cfg(physics: dict, *, device: str, period_s: float, solver: str | None = None,
                   integrator: str | None = None, visualizers=None):
    """Isaac Lab on Newton with the selected solver (``solver``, else the profile's), configured from the profile's
    ``physics`` block."""
    import isaaclab.sim as sim_utils
    from isaaclab_newton.physics import NewtonCfg

    name, settings = solver_settings(physics, solver)
    return sim_utils.SimulationCfg(
        dt=period_s, device=device, physics=NewtonCfg(solver_cfg=solver_cfg(name, settings, integrator=integrator)),
        visualizer_cfgs=visualizers or None)


class IsaacLabRobot:
    """Scene with ground, robot and any static boxes; exposes joint state, target writes and stepping as torch
    tensors."""

    def __init__(self, desc: Description, usd_path: Path, cfg: IsaacLabCfg, *, props: JointProperties):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass, instantiate

        self.desc, self.cfg = desc, cfg
        joints = props.names

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            pass

        # The boxes are one set of shapes every env's world collides with, at the same coordinates.
        spacing = 0.0 if cfg.scene is not None else cfg.env_spacing_m
        scene_cfg = SceneCfg(num_envs=cfg.num_envs, env_spacing=spacing, replicate_physics=True)
        scene_cfg.ground = ground_cfg(cfg.ground_friction)
        if cfg.scene is not None:
            from humanoid_sim.world.boxes import static_boxes_cfg

            scene_cfg.static_boxes = static_boxes_cfg(cfg.scene.file, cfg.scene.spawn_m, friction=cfg.ground_friction)
        scene_cfg.robot = articulation_cfg(usd_path, props, fixed_base=cfg.fixed_base, base_joints=cfg.base_joints,
                                           self_collision=cfg.self_collision, base_height_m=cfg.base_height_m,
                                           contact_sensors=cfg.contacts)
        if cfg.imu:
            scene_cfg.imu = imu_sensor_cfg(desc.robot)
        if cfg.contacts:
            scene_cfg.contacts = contact_sensor_cfg(history=1, air_time=False)  # net forces now: the truth
        for name, entity in (cfg.entities or {}).items():
            setattr(scene_cfg, name, entity)
        visualizers = []
        if cfg.rerun:
            from isaaclab_visualizers.rerun import RerunVisualizerCfg

            visualizers.append(RerunVisualizerCfg(record_to_rrd=cfg.rerun_rrd, keep_historical_data=True))
        if cfg.settings is None:
            raise ValueError("IsaacLabCfg.settings: the solver's block of the profile's physics")
        physics = {"solver": cfg.solver, cfg.solver: cfg.settings}
        self.sim = sim_utils.SimulationContext(simulation_cfg(physics, device=cfg.device, period_s=cfg.period_s,
                                                              integrator=cfg.integrator, visualizers=visualizers))
        with crank_stop_hook(desc.robot, physics):
            self.scene = instantiate(scene_cfg)
            self.sim.reset()
        check_crank_stops(desc.robot, physics)
        apply_joint_limits(physics)
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

    def set_targets(self, target: torch.Tensor, velocity: torch.Tensor | None = None,
                    effort: torch.Tensor | None = None) -> None:
        tc = self.robot.actuators.target_command
        tc.set_position_index(value=target)
        tc.set_velocity_index(value=torch.zeros_like(target) if velocity is None else velocity)
        tc.set_effort_index(value=torch.zeros_like(target) if effort is None else effort)

    def step(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step(render=self.cfg.render_each_step)
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
