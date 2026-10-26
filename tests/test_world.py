"""Terrain and pushes in the training scene (Warp CPU device). Runs in the Isaac Lab image."""

import pytest
import torch
from conftest import GRAVITY, PARTS, PHYSICS_ROBOTS, built_usd

from humanoid_sim.train.plant import TrainPlantCfg

pytestmark = [pytest.mark.physics, pytest.mark.parametrize("robot", PHYSICS_ROBOTS)]


def _scene(robot, n, **kw):
    pytest.importorskip("isaaclab_newton")
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg

    usd = built_usd(robot)
    sim = sim_utils.SimulationContext(robot_sim_cfg(robot, device="cpu"))
    with robot_build(robot):
        scene = instantiate(robot_scene_cfg(n, usd, robot=robot, **kw))
        sim.reset()
    return sim, scene


def _weight_n(art):
    """The articulation's weight (N), from the masses the solver holds."""
    return float(art.data.body_mass.torch[0].sum()) * GRAVITY


def _step(sim, scene, plant, n):
    for _ in range(n):
        plant.pre_physics()
        scene.write_data_to_sim()
        sim.step()
        scene.update(plant.period_ns / 1e9)


def test_heightfield_terrain_carries_the_feet(robot):
    """A robot dropped onto a bumpy heightfield lands on it and stays on top: the G1's 5 mm foot spheres too, which
    fell through the lowest cells while the heightfield's solid base was 0.1 mm thick
    (``physics.mjwarp.heightfield_base_m``)."""
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg

    from humanoid_sim.model.asset import asset_meta
    from humanoid_sim.model.kinematics import _sole_corners
    from humanoid_sim.train.plant import RobotPlant
    from humanoid_sim.world.terrain import heightfield_terrain_cfg

    bumpy = {"bumps": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, 0.025), noise_step=0.005)}
    terrain = heightfield_terrain_cfg(bumpy, rows=1, cols=2, size_m=4.0, border_m=2.0, curriculum=False, seed=0)
    sim, scene = _scene(robot, 2, terrain=terrain)
    try:
        art = scene["robot"]
        plant = RobotPlant(art, TrainPlantCfg(robot=robot), imu=scene["imu"], contacts=scene["contacts"])
        stand = art.data.default_root_pose.torch[0, 2].item()  # soles on the ground, upright
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, stand + 0.05]),
                          torch.tensor([[0.0, 0, 0, 1]] * 2)], 1)
        plant.reset(torch.arange(2), root_pose=root, root_vel=torch.zeros(2, 6))
        boxes = asset_meta(built_usd(robot))["sole_boxes"]
        soles = [art.body_names.index(b["body"]) for b in boxes]
        settled = []
        forces = torch.zeros(2, len(art.body_names))
        for k in range(85):  # drives off: it drops 5 cm onto the terrain and folds up
            _step(sim, scene, plant, 1)
            if k >= 60:  # over the last 0.1 s: the feet bounce on the bumps, so average
                poses = art.data.body_link_pose_w.torch.numpy()
                settled.append(min(float(_sole_corners(poses[e, b], box)[:, 2].min())
                                   for e in range(2) for b, box in zip(soles, boxes, strict=True)))
                forces += plant.truth().contact_forces_w[..., 2] / 25
        # Landed, the soles rest on the terrain, at most a few millimetres (soft contact) below its lowest point (0);
        # through the 0.1 mm base the V1's sank about 3 cm and the G1's about 1.5 cm.
        assert sum(settled) / len(settled) > -0.005
        assert (forces.sum(1) > 0.2 * _weight_n(art)).all()
        assert (forces[:, soles].sum(1) > 0).all()  # on its feet, at least partly
        z = art.data.root_link_pose_w.torch[:, 2] - scene.env_origins[:, 2]
        assert (z > 0.1).all() and (z < stand + 0.05).all()
    finally:
        sim_utils.SimulationContext.clear_instance()


def _deepest_terrain_contact_m():
    """The deepest contact with a heightfield in the last step of the running MJWarp data (signed distance, negative
    inside), over all worlds; 0 when there is none. Self-contacts are left out: convex hulls of neighbouring links
    overlap by construction."""
    import mujoco_warp
    from isaaclab_newton.physics import NewtonManager

    solver = NewtonManager._solver
    d = solver.mjw_data
    n = int(d.nacon.numpy()[0])
    if not n:
        return 0.0
    hfield = solver.mjw_model.geom_type.numpy() == int(mujoco_warp.GeomType.HFIELD)
    geom = d.contact.geom.numpy()[:n]
    on = hfield[geom[:, 0]] | hfield[geom[:, 1]]
    return min(0.0, float(d.contact.dist.numpy()[:n][on].min())) if on.any() else 0.0


def test_a_robot_falling_on_a_heightfield_gains_no_energy(robot):
    """Drives off, a robot tipped over onto a bumpy heightfield is a passive system: its mechanical energy can only
    fall, so its centre-of-mass kinetic energy plus potential energy (a lower bound of the total) never exceeds what
    it started with. And no contact with the terrain is deeper than a few centimetres. MJWarp's own collision pipeline broke both now
    and then: a sole box against a prism came back metres deep and the solver flung the robot (``contacts:
    mujoco`` in the profile's ``physics.mjwarp``)."""
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg

    from humanoid_sim.train.plant import RobotPlant
    from humanoid_sim.world.terrain import heightfield_terrain_cfg

    n = 8
    bumpy = {"bumps": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, 0.025), noise_step=0.005)}
    terrain = heightfield_terrain_cfg(bumpy, rows=2, cols=4, size_m=4.0, border_m=2.0, curriculum=False, seed=0)
    sim, scene = _scene(robot, n, terrain=terrain)
    try:
        art = scene["robot"]
        plant = RobotPlant(art, TrainPlantCfg(robot=robot), imu=None, contacts=None)
        stand = art.data.default_root_pose.torch[0, 2].item()
        gen = torch.Generator().manual_seed(0)
        tilt = (torch.rand(n, 2, generator=gen) - 0.5) * 0.6  # up to 0.3 rad about x and y: every robot falls
        half = 0.5 * tilt
        quat = torch.stack([torch.sin(half[:, 0]), torch.sin(half[:, 1]), torch.zeros(n),
                            torch.cos(half[:, 0]) * torch.cos(half[:, 1])], 1)
        quat = quat / quat.norm(dim=1, keepdim=True)
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, stand + 0.05]), quat], 1)
        plant.reset(torch.arange(n), root_pose=root, root_vel=torch.zeros(n, 6))
        mass = art.data.body_mass.torch  # [n, bodies]
        ground = scene.env_origins[:, 2:3]

        def energy():
            z = art.data.body_com_pos_w.torch[..., 2] - ground
            v = art.data.body_com_lin_vel_w.torch
            return (mass * (GRAVITY * z + 0.5 * (v**2).sum(-1))).sum(1)

        _step(sim, scene, plant, 1)
        start, peak, deepest = energy(), energy(), 0.0
        for _ in range(int(1.5e9 / plant.period_ns)):  # 1.5 s: it tips over and lands
            _step(sim, scene, plant, 1)
            peak = torch.maximum(peak, energy())
            deepest = min(deepest, _deepest_terrain_contact_m())
        weight = mass.sum(1) * GRAVITY
        # Allowance: the energy a contact stores at a few millimetres of soft penetration under the robot's weight.
        assert (peak <= start + weight * 0.01).all(), (peak - start) / weight
        assert deepest > -0.03, deepest
        z = art.data.root_link_pose_w.torch[:, 2] - ground[:, 0]
        assert (z > 0.0).all() and (z < stand).all()  # on the terrain, fallen
    finally:
        sim_utils.SimulationContext.clear_instance()


def test_force_push_accelerates_the_robot_horizontally(robot):
    import isaaclab.sim as sim_utils

    from humanoid_sim.train.plant import RobotPlant
    from humanoid_sim.world.disturbances import ForcePush

    sim, scene = _scene(robot, 2)
    try:
        art = scene["robot"]
        plant = RobotPlant(art, TrainPlantCfg(robot=robot), imu=None, contacts=None)
        # In the air, so only the push acts horizontally.
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, 3.0]), torch.tensor([[0.0, 0, 0, 1]] * 2)], 1)
        plant.reset(torch.arange(2), root_pose=root, root_vel=torch.zeros(2, 6))
        push = ForcePush(art, PARTS[robot]["torso"], interval_s=(0.0, 0.0), duration_s=(1.0, 1.0),
                         magnitude_n=(500.0, 500.0))
        push.level = torch.tensor([0.0, 1.0])  # env 0 unpushed
        push.reset(torch.arange(2), plant.t_ns)
        _step(sim, scene, plant, 1)
        com0 = art.data.root_com_pos_w.torch[:, :2].clone()
        for _ in range(25):  # 0.1 s
            push.step(plant.t_ns)
            _step(sim, scene, plant, 1)
        dx = art.data.root_com_pos_w.torch[:, :2] - com0
        mass = art.data.body_mass.torch.sum(1)
        expect = 0.5 * 500.0 / mass[1] * 0.1 ** 2
        heading = push.force[1, :2] / push.force[1, :2].norm()
        assert push.force[0].norm() == 0 and dx[0].norm() < 1e-3
        assert (dx[1] @ heading).item() == pytest.approx(expect.item(), rel=0.15)  # off-COM push: some turns
    finally:
        sim_utils.SimulationContext.clear_instance()


def _toppling_ratio(robot):
    """The ankles' joint-space pitch stiffness at the baseline gains (both legs) over the stiffness at which the
    whole body, an inverted pendulum about the ankle pitch axes, topples (m g h, h the centre of mass above those
    axes at home), from the solver properties and host kinematics."""
    import numpy as np

    from humanoid_sim.model.description import Description
    from humanoid_sim.model.kinematics import RobotKinematics, _rotate
    from humanoid_sim.model.poses import initial_pose
    from humanoid_sim.train.scene import plant_joint_properties

    usd = built_usd(robot)
    joints = list(Description.load(robot).joints)
    q, _ = initial_pose(usd, robot, "home")
    kin = RobotKinematics(usd, joints, base_height_m=1.0)
    model = kin.model
    poses = kin.body_poses(q.reshape(1, -1))
    mass = model.body_mass.numpy().astype(np.float64)
    com = np.array([_rotate(p[3:], c) + p[:3] for p, c in zip(poses, model.body_com.numpy(), strict=True)])
    labels = [lbl.rsplit("/", 1)[-1] for lbl in model.joint_label]
    child, x_c = model.joint_child.numpy(), model.joint_X_c.numpy()
    axes_z = []
    for name in PARTS[robot]["ankle_pitch"]:
        j = labels.index(name)
        body = poses[int(child[j])]
        axes_z.append((_rotate(body[3:], x_c[j][:3]) + body[:3])[2])
    height = float((mass * com[:, 2]).sum() / mass.sum()) - float(np.mean(axes_z))
    props = plant_joint_properties(robot)
    stiffness = sum(props.stiffness[props.names.index(n)] for n in PARTS[robot]["ankle_pitch"])
    return stiffness / (float(mass.sum()) * GRAVITY * height)


def test_holding_the_home_pose_stands_only_on_ankles_stiffer_than_toppling(robot):
    """A robot holding its home pose at the baseline gains is an inverted pendulum on its ankles: it stands
    (upright, at its standing height, carried by the ground) when their pitch stiffness exceeds m g h, and tips over
    otherwise. The V1's crank-driven ankles are about 4.6 times stiffer than that; the G1's deploy gains (40 Nm/rad
    per ankle) are about 0.4 times as stiff, so the G1 needs a policy to stand."""
    import math

    import isaaclab.sim as sim_utils

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.train.plant import RobotPlant

    ratio = _toppling_ratio(robot)
    if 0.8 <= ratio <= 1.25:
        pytest.skip(f"ankle stiffness {ratio:.2f} times the toppling stiffness: too close to call")
    sim, scene = _scene(robot, 1)  # the default pose: home
    try:
        art = scene["robot"]
        stand = art.data.default_root_pose.torch[0, 2].item()
        plant = RobotPlant(art, TrainPlantCfg(robot=robot), imu=scene["imu"], contacts=scene["contacts"])
        default = art.data.default_joint_pos.torch
        q = default[:, [list(art.joint_names).index(n) for n in plant.core.map.joint_names]]
        core = plant.core
        drive_q, _, _ = core.map.drive_state(q, torch.zeros_like(q))
        hold = DriveDemand.pd(drive_q, core.drive_stiffness.expand(1, -1), core.drive_damping.expand(1, -1))
        plant.reset(torch.arange(1), demand=hold)
        _step(sim, scene, plant, int(2.0e9 / plant.period_ns))
        pose = plant.truth().root_pose_w[0]
        x, y, z, w = pose[3:].tolist()
        tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(x, y))))  # of the base's vertical axis
        if ratio > 1.25:
            assert pose[2].item() > stand - 0.05  # sags at most a few cm under the baseline gains
            assert tilt < 5.0
            assert plant.truth().contact_forces_w[0, :, 2].sum().item() == pytest.approx(_weight_n(art), rel=0.1)
        else:
            assert tilt > 45.0 and pose[2].item() < 0.5 * stand  # toppled
    finally:
        sim_utils.SimulationContext.clear_instance()
