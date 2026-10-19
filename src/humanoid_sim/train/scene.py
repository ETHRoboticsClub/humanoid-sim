"""Isaac Lab scene and simulation configs for training with this robot.

:func:`robot_scene_cfg` gives an ``InteractiveSceneCfg`` with:

* ``robot``: the built asset as an articulation (constant solver PD, the description's passive properties);
* ``imu``: Isaac Lab's IMU sensor at the description's mount (``imu_joint``), raw sensor-frame values;
* ``contacts``: a contact sensor on every body (net forces with history, air time);
* ``ground``: the flat ground at the profile's friction, unless a ``terrain`` config is given.

The robot's default state is upright, its lowest sole corner on the ground
(:func:`~humanoid_sim.model.kinematics.standing_base_height`), so ``reset_scene_to_default`` puts it on its feet,
in one of two poses (``pose``):

* ``"home"`` (default): the description's home pose (:attr:`~humanoid_sim.model.description.Description.home_pose`),
  the pose the CAD is exported in: legs straight, both soles level and at the same height.
* ``"standing"``: the description's standing pose, the one captured on the robot. Under an upright base its soles
  are not level (pitched by 10° and 17°), so a robot holding it rigidly tips over within about a second.

The articulation's ``data.default_joint_pos`` holds the chosen pose, for a task's action offsets.

:func:`robot_sim_cfg` gives the ``SimulationCfg``: one physics step per period on Newton's MJWarp solver, from
the profile. humanoid-learning builds its environment config from these and steps the plant through
:class:`~humanoid_sim.train.plant.RobotPlant`. Isaac Lab is imported lazily.
"""

from __future__ import annotations

from pathlib import Path

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.joint_properties import JointProperties
from humanoid_sim.physics.isaaclab import contact_sensor_cfg, imu_sensor_cfg

STANDING_CLEARANCE_M = 0.001  # spawn the soles just above the ground


def plant_joint_properties(robot: str = "ethrc_v1") -> JointProperties:
    """The solver properties of the description's joints (constant PD ``k0, d0``, passive terms, limits)."""
    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.joint_properties import joint_properties

    desc = Description.load(robot)
    joints = list(desc.joints)
    return joint_properties(PlantCore(desc, joints, 1, PlantCfg(robot=robot)), joints)


POSES = ("home", "standing")


def initial_pose(usd: Path, robot: str = "ethrc_v1", pose: str = "home"):
    """Joint positions ``[J]`` (description order) of ``pose`` (see :data:`POSES`) and the upright base height at
    which its lowest sole corner touches the ground."""
    import torch

    from humanoid_sim.model.asset import asset_meta
    from humanoid_sim.model.kinematics import standing_base_height

    desc = Description.load(robot)
    soles = asset_meta(usd)["sole_boxes"]
    if pose == "home":
        q = torch.tensor([desc.home_pose[n] for n in desc.joints], dtype=torch.float64)
    elif pose == "standing":
        q = torch.tensor([desc.joints[n].standing_offset_rad for n in desc.joints], dtype=torch.float64)
    else:
        raise ValueError(f"unknown pose {pose!r}; known: {POSES}")
    return q, standing_base_height(usd, desc, soles, q)


def robot_scene_cfg(num_envs: int, usd: Path, *, robot: str = "ethrc_v1", env_spacing: float = 2.5,
                    terrain=None, ground_friction: float | None = None, imu: bool = True, contacts: bool = True,
                    self_collision: bool | None = None, pose: str = "home"):
    """Scene config with the robot, its sensors and the ground (or ``terrain``, an Isaac Lab
    ``TerrainImporterCfg``), the robot on its feet in ``pose`` (see :data:`POSES`). With robot friction
    randomized, pass ``ground_friction=0``: MJWarp takes the larger friction of the two shapes in contact."""
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    from humanoid_sim.physics.isaaclab import articulation_cfg, ground_cfg

    profile = load_profile(robot)
    q, height = initial_pose(usd, robot, pose)
    props = plant_joint_properties(robot)
    names = list(Description.load(robot).joints)

    @configclass
    class RobotSceneCfg(InteractiveSceneCfg):
        pass

    cfg = RobotSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
    cfg.robot = articulation_cfg(
        usd, props, base_height_m=height + STANDING_CLEARANCE_M, contact_sensors=contacts,
        self_collision=profile["physics"]["self_collision"] if self_collision is None else self_collision,
        joint_pos=[float(q[names.index(n)]) for n in props.names])
    if terrain is not None:
        cfg.terrain = terrain
    else:
        cfg.ground = ground_cfg(profile["model"]["ground_friction"] if ground_friction is None else ground_friction)
    if imu:
        cfg.imu = imu_sensor_cfg(robot)
    if contacts:
        cfg.contacts = contact_sensor_cfg()
    return cfg


def robot_sim_cfg(robot: str = "ethrc_v1", *, device: str = "cuda:0", visualizers=None):
    """``SimulationCfg`` with the profile's physics period and MJWarp solver settings."""
    from humanoid_sim.physics.isaaclab import simulation_cfg

    physics = load_profile(robot)["physics"]
    return simulation_cfg(physics, device=device, period_s=int(physics["period_ns"]) / 1e9, visualizers=visualizers)
