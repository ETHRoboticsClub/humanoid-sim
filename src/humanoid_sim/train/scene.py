"""Isaac Lab scene and simulation configs for training with this robot.

:func:`robot_scene_cfg` gives an ``InteractiveSceneCfg`` with:

* ``robot``: the built asset as an articulation (constant solver PD, the description's passive properties);
* ``imu``: Isaac Lab's IMU sensor at the description's mount (``imu_joint``), raw sensor-frame values;
* ``contacts``: a contact sensor on every body (net forces with history, air time);
* ``ground``: the flat ground at the profile's friction, unless a ``terrain`` config is given;
* ``static_boxes``: with ``scene`` (:class:`~humanoid_sim.world.boxes.StaticBoxes`), a room's walls and furniture
  as static boxes that every env collides with. Every env's robot then starts at the world origin, the scene's
  spawn point (each env is its own solver world, so robots at one spot never touch).

The robot's default state is upright, its lowest sole corner on the ground
(:func:`~humanoid_sim.model.kinematics.standing_base_height`), so ``reset_scene_to_default`` puts it on its feet,
in one of two poses (``pose``):

* ``"home"`` (default): the description's home pose (:attr:`~humanoid_sim.model.description.Description.home_pose`),
  the pose the CAD is exported in: legs straight, both soles level and at the same height.
* ``"standing"``: the description's standing pose, every joint at its standing offset (the V1's are 0, which makes
  it the home pose).

The articulation's ``data.default_joint_pos`` holds the chosen pose, for a task's action offsets.

:func:`robot_sim_cfg` gives the ``SimulationCfg``: one physics step per period on a Newton solver (``mjwarp`` or
``kamino``), from the profile. humanoid-learning builds its environment config from these and steps the plant through
:class:`~humanoid_sim.train.plant.RobotPlant`. Isaac Lab is imported lazily.
"""

from __future__ import annotations

from pathlib import Path

from humanoid_sim.model.asset import check_asset_robot, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.poses import POSES, STANDING_CLEARANCE_M, initial_pose
from humanoid_sim.physics.isaaclab import contact_sensor_cfg, imu_sensor_cfg
from humanoid_sim.physics.joint_properties import JointProperties
from humanoid_sim.world.boxes import StaticBoxes

__all__ = ["POSES", "STANDING_CLEARANCE_M", "initial_pose", "plant_joint_properties", "robot_scene_cfg",
           "robot_sim_cfg"]


def plant_joint_properties(robot: str = "ethrc_v1") -> JointProperties:
    """The solver properties of the description's joints (constant PD ``k0, d0``, passive terms, limits)."""
    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.physics.joint_properties import joint_properties

    desc = Description.load(robot)
    joints = list(desc.joints)
    return joint_properties(PlantCore(desc, joints, 1, PlantCfg(robot=robot)), joints)


def robot_scene_cfg(num_envs: int, usd: Path, *, robot: str = "ethrc_v1", env_spacing: float | None = None,
                    terrain=None, ground_friction: float | None = None, imu: bool = True, contacts: bool = True,
                    self_collision: bool | None = None, pose: str = "home", scene: StaticBoxes | None = None):
    """Scene config with the robot, its sensors and the ground (or ``terrain``, an Isaac Lab
    ``TerrainImporterCfg``), the robot on its feet in ``pose`` (see :data:`POSES`), and ``scene``'s static boxes.
    With robot friction randomized, pass ``ground_friction=0``: the solver takes the larger friction of the two
    shapes in contact; the boxes take the ground's friction. ``env_spacing`` defaults to 2.5 m, and to 0 with a
    ``scene``, which allows no other and no ``terrain``."""
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    from humanoid_sim.physics.isaaclab import articulation_cfg, ground_cfg

    check_asset_robot(usd, robot)
    if scene is not None and terrain is not None:
        raise ValueError("a scene's boxes stand on the flat ground: no terrain with them")
    if scene is not None and env_spacing not in (None, 0.0):
        raise ValueError("a scene's robots start at its spawn point, the world origin: env_spacing 0")
    env_spacing = (0.0 if scene is not None else 2.5) if env_spacing is None else env_spacing
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
    friction = profile["model"]["ground_friction"] if ground_friction is None else ground_friction
    if terrain is not None:
        cfg.terrain = terrain
    else:
        cfg.ground = ground_cfg(friction)
    if scene is not None:
        from humanoid_sim.world.boxes import static_boxes_cfg

        cfg.static_boxes = static_boxes_cfg(scene.file, scene.spawn_m, friction=friction)
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
