"""Isaac Lab scene and simulation configs for training with this robot.

:func:`robot_scene_cfg` gives an ``InteractiveSceneCfg`` with:

* ``robot``: the built asset as an articulation (constant solver PD, the description's passive properties);
* ``imu``: Isaac Lab's IMU sensor at the description's mount (``imu_joint``), raw sensor-frame values;
* ``contacts``: a contact sensor on every body (net forces with history, air time);
* ``ground``: the flat ground at the profile's friction, unless a ``terrain`` config is given.

The robot's default state is the standing pose, upright, its lowest sole corner on the ground
(:func:`~humanoid_sim.model.kinematics.standing_base_height`), so ``reset_scene_to_default`` puts it on its feet.

:func:`robot_sim_cfg` gives the ``SimulationCfg``: one physics step per period on a Newton solver (``mjwarp`` or
``kamino``), from the profile. humanoid-learning builds its environment config from these and steps the plant through
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


def robot_scene_cfg(num_envs: int, usd: Path, *, robot: str = "ethrc_v1", env_spacing: float = 2.5,
                    terrain=None, ground_friction: float | None = None, imu: bool = True, contacts: bool = True,
                    self_collision: bool | None = None):
    """Scene config with the robot, its sensors and the ground (or ``terrain``, an Isaac Lab
    ``TerrainImporterCfg``). With robot friction randomized, pass ``ground_friction=0``: the solver takes the larger
    friction of the two shapes in contact."""
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    from humanoid_sim.model.asset import asset_meta
    from humanoid_sim.model.kinematics import standing_base_height
    from humanoid_sim.physics.isaaclab import articulation_cfg, ground_cfg

    profile = load_profile(robot)
    height = standing_base_height(usd, Description.load(robot), asset_meta(usd)["sole_boxes"])

    @configclass
    class RobotSceneCfg(InteractiveSceneCfg):
        pass

    cfg = RobotSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
    cfg.robot = articulation_cfg(
        usd, plant_joint_properties(robot), base_height_m=height + STANDING_CLEARANCE_M, contact_sensors=contacts,
        self_collision=profile["physics"]["self_collision"] if self_collision is None else self_collision)
    if terrain is not None:
        cfg.terrain = terrain
    else:
        cfg.ground = ground_cfg(profile["model"]["ground_friction"] if ground_friction is None else ground_friction)
    if imu:
        cfg.imu = imu_sensor_cfg(robot)
    if contacts:
        cfg.contacts = contact_sensor_cfg()
    return cfg


def robot_sim_cfg(robot: str = "ethrc_v1", *, device: str = "cuda:0", solver: str | None = None, visualizers=None):
    """``SimulationCfg`` with the profile's physics period and the settings of ``solver`` (the profile's if None)."""
    from humanoid_sim.physics.isaaclab import simulation_cfg

    physics = load_profile(robot)["physics"]
    return simulation_cfg(physics, device=device, period_s=int(physics["period_ns"]) / 1e9, solver=solver,
                          visualizers=visualizers)
