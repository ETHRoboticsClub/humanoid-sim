"""Isaac Lab 3.0 / Newton backend: the robot USD as an Isaac Lab articulation, stepped by MJWarp.

The solver PD gains are the plant's constant ``k0, d0`` (joint space), written once at spawn. Each step the
plant writes position targets (velocity target 0); nothing else changes in the model while stepping.
Passive joint damping and Coulomb friction, armature and effort limits come from the description.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from humanoid_sim.model.description import Description
from humanoid_sim.model.joint_properties import JointProperties


@dataclass
class IsaacLabCfg:
    num_envs: int = 1
    device: str = "cpu"
    period_s: float = 0.004
    integrator: str = "implicitfast"
    use_mujoco_cpu: bool = True
    fixed_base: bool = False
    base_height_m: float = 1.0
    env_spacing_m: float = 3.0
    iterations: int = 100
    ls_iterations: int = 50
    cone: str = "pyramidal"
    njmax: int = 300
    nconmax: int | None = None
    self_collision: bool = True
    rerun: bool = False
    rerun_rrd: str | None = None


def _per_joint(names, values):
    return {f"^{n}$": float(v) for n, v in zip(names, values, strict=True)}


class IsaacLabRobot:
    """Scene with ground and robot; exposes joint state, target writes and stepping as torch tensors."""

    def __init__(self, desc: Description, usd_path: Path, cfg: IsaacLabCfg, *, props: JointProperties):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass, instantiate
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
        from isaaclab_newton.sim.schemas import NewtonArticulationCfg

        self.desc, self.cfg = desc, cfg
        joints = props.names
        actuator = ImplicitActuatorCfg(
            joint_names_expr=[f"^{n}$" for n in joints],
            stiffness=_per_joint(joints, props.stiffness), damping=_per_joint(joints, props.damping),
            armature=_per_joint(joints, props.armature), friction=_per_joint(joints, props.friction),
            viscous_friction=_per_joint(joints, props.passive_damping),
            joint_effort_limit=_per_joint(joints, props.effort_limit),
        )
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path), fix_root_link=cfg.fixed_base,
                articulation_props=[NewtonArticulationCfg(self_collision_enabled=cfg.self_collision)]),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, cfg.base_height_m), joint_pos=_per_joint(joints, props.standing),
                joint_vel={".*": 0.0}),
            actuators={"drives": actuator},
        )

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())

        scene_cfg = SceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing_m, replicate_physics=True)
        scene_cfg.robot = robot
        solver = MJWarpSolverCfg(integrator=cfg.integrator, use_mujoco_cpu=cfg.use_mujoco_cpu,
                                 iterations=cfg.iterations, ls_iterations=cfg.ls_iterations, cone=cfg.cone,
                                 njmax=cfg.njmax, nconmax=cfg.nconmax)
        visualizers = []
        if cfg.rerun:
            from isaaclab_visualizers.rerun import RerunVisualizerCfg

            visualizers.append(RerunVisualizerCfg(record_to_rrd=cfg.rerun_rrd, keep_historical_data=True))
        sim_cfg = sim_utils.SimulationCfg(dt=cfg.period_s, device=cfg.device,
                                          physics=NewtonCfg(solver_cfg=solver),
                                          visualizer_cfgs=visualizers or None)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = instantiate(scene_cfg)
        self.sim.reset()
        self.robot = self.scene["robot"]
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

    def set_targets(self, target: torch.Tensor) -> None:
        tc = self.robot.actuators.target_command
        tc.set_position_index(value=target)
        tc.set_velocity_index(value=torch.zeros_like(target))

    def step(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.period_s)

    def model_summary(self) -> dict:
        """Masses and names straight from the Newton model, for model checks."""
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        mass = model.body_mass.numpy()
        return {"bodies": model.body_count // max(self.cfg.num_envs, 1),
                "total_mass_kg": float(mass.sum()) / max(self.cfg.num_envs, 1),
                "shapes": model.shape_count // max(self.cfg.num_envs, 1),
                "joint_names": self.joint_names}
