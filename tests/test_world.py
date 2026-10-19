"""Terrain and pushes in the training scene (Warp CPU device). Runs in the Isaac Lab image."""

import json
from pathlib import Path

import pytest
import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"
pytestmark = pytest.mark.physics


@pytest.fixture
def usd():
    pytest.importorskip("isaaclab_newton")
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def _scene(usd, n, **kw):
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg

    sim = sim_utils.SimulationContext(robot_sim_cfg(device="cpu"))
    scene = instantiate(robot_scene_cfg(n, usd, **kw))
    sim.reset()
    return sim, scene


def _step(sim, scene, plant, n):
    for _ in range(n):
        plant.pre_physics()
        scene.write_data_to_sim()
        sim.step()
        scene.update(plant.period_ns / 1e9)


def test_rough_terrain_carries_the_feet(usd):
    import isaaclab.sim as sim_utils

    from humanoid_sim.train.plant import RobotPlant
    from humanoid_sim.world.terrain import rough_terrain_cfg

    sim, scene = _scene(usd, 2, terrain=rough_terrain_cfg(rows=1, cols=2, size_m=4.0, border_m=2.0, curriculum=False))
    try:
        robot = scene["robot"]
        plant = RobotPlant(robot, imu=scene["imu"], contacts=scene["contacts"])
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, 1.0]), torch.tensor([[0.0, 0, 0, 1]] * 2)], 1)
        plant.reset(torch.arange(2), root_pose=root, root_vel=torch.zeros(2, 6))
        _step(sim, scene, plant, 60)  # drives off: it drops onto the terrain
        feet = [robot.body_names.index(n) for n in ("bottom_universal_joint_l_1", "bottom_universal_joint_l_Mirror__1")]
        assert (plant.truth().contact_forces_w[..., 2].sum(1) > 100).all()
        z = robot.data.root_link_pose_w.torch[:, 2] - scene.env_origins[:, 2]
        assert (z > 0.1).all() and (z < 1.0).all()
        assert len(feet) == 2
    finally:
        sim_utils.SimulationContext.clear_instance()


def test_force_push_accelerates_the_robot_horizontally(usd):
    import isaaclab.sim as sim_utils

    from humanoid_sim.train.plant import RobotPlant
    from humanoid_sim.world.disturbances import ForcePush

    sim, scene = _scene(usd, 2)
    try:
        robot = scene["robot"]
        plant = RobotPlant(robot, imu=None, contacts=None)
        # In the air, so only the push acts horizontally.
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, 3.0]), torch.tensor([[0.0, 0, 0, 1]] * 2)], 1)
        plant.reset(torch.arange(2), root_pose=root, root_vel=torch.zeros(2, 6))
        push = ForcePush(robot, "Component171_1", interval_s=(0.0, 0.0), duration_s=(1.0, 1.0),
                         magnitude_n=(500.0, 500.0))
        push.level = torch.tensor([0.0, 1.0])  # env 0 unpushed
        push.reset(torch.arange(2), plant.t_ns)
        _step(sim, scene, plant, 1)
        com0 = robot.data.root_com_pos_w.torch[:, :2].clone()
        for _ in range(25):  # 0.1 s
            push.step(plant.t_ns)
            _step(sim, scene, plant, 1)
        dx = robot.data.root_com_pos_w.torch[:, :2] - com0
        mass = robot.data.body_mass.torch.sum(1)
        expect = 0.5 * 500.0 / mass[1] * 0.1 ** 2
        heading = push.force[1, :2] / push.force[1, :2].norm()
        assert push.force[0].norm() == 0 and dx[0].norm() < 1e-3
        assert (dx[1] @ heading).item() == pytest.approx(expect.item(), rel=0.15)  # off-COM push: some turns
    finally:
        sim_utils.SimulationContext.clear_instance()


def test_a_robot_holding_the_flat_foot_pose_stands(usd):
    """The flat-foot pose at the baseline gains stands on flat ground; the description's standing pose, whose
    soles are not level, does not (it tips over within about a second)."""
    import isaaclab.sim as sim_utils

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.train.plant import RobotPlant

    sim, scene = _scene(usd, 1, pose="flat_feet")
    try:
        robot = scene["robot"]
        plant = RobotPlant(robot, imu=scene["imu"], contacts=scene["contacts"])
        default = robot.data.default_joint_pos.torch
        q = default[:, [list(robot.joint_names).index(n) for n in plant.core.map.joint_names]]
        core = plant.core
        drive_q, _, _ = core.map.drive_state(q, torch.zeros_like(q))
        hold = DriveDemand.jvpt(drive_q, core.drive_stiffness.expand(1, -1), core.drive_damping.expand(1, -1))
        plant.reset(torch.arange(1), demand=hold)
        _step(sim, scene, plant, int(1.5e9 / plant.period_ns))
        pose = plant.truth().root_pose_w[0]
        assert pose[2].item() > 0.85
        assert pose[3:6].abs().max().item() < 0.05  # upright: small quaternion x, y, z
        assert (plant.truth().contact_forces_w[0, :, 2].sum() > 400).item()
    finally:
        sim_utils.SimulationContext.clear_instance()
