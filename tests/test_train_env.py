"""The Train-Time API in a manager-based Isaac Lab env (Warp CPU device). Runs in the Isaac Lab image."""

import json
from pathlib import Path

import pytest
import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"
pytestmark = pytest.mark.physics


def _usd():
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def _env(num_envs=2):
    pytest.importorskip("isaaclab_newton")
    import isaaclab.envs.mdp as mdp
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.managers import (
        EventTermCfg,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
    )
    from isaaclab.utils import configclass

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.train.isaaclab import PlantAction, PlantActionCfg
    from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg

    class HoldStanding(PlantAction):
        """Demand the standing pose at the baseline gains, offset by the action (rad, drive frame)."""

        def demands(self, actions):
            p = self.plant.core
            q = torch.tensor([p.desc.joints[n].standing_offset_rad for n in p.map.joint_names], device=self.device)
            q = q.expand(self.num_envs, -1)
            stand, _, _ = p.map.drive_state(q, torch.zeros_like(q))
            n = self.num_envs
            return (DriveDemand.pd(stand + actions, p.drive_stiffness.expand(n, -1), p.drive_damping.expand(n, -1)),
                    torch.full((n,), self.plant.t_ns, device=self.device))

    def drive_position(env):
        return env.action_manager.get_term("drives").plant.measured().drive.position_rad

    def imu_latest(env):
        s = env.action_manager.get_term("drives").plant.measured().imu
        return torch.cat([s.gyro[:, 0], s.specific_force[:, 0]], dim=1)

    @configclass
    class Actions:
        drives = PlantActionCfg(class_type=HoldStanding, action_dim=22, plant=None)

    @configclass
    class Observations:
        @configclass
        class Policy(ObservationGroupCfg):
            drives = ObservationTermCfg(func=drive_position)
            imu = ObservationTermCfg(func=imu_latest)

        policy = Policy()

    @configclass
    class Rewards:
        alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)

    @configclass
    class Terminations:
        time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    @configclass
    class Events:
        reset = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        decimation = 5
        episode_length_s = 3.0
        actions = Actions()
        observations = Observations()
        rewards = Rewards()
        terminations = Terminations()
        events = Events()

    from humanoid_sim.train.plant import TrainPlantCfg

    cfg = EnvCfg()
    cfg.scene = robot_scene_cfg(num_envs, _usd(), ground_friction=0.0)
    cfg.sim = robot_sim_cfg(device="cpu")
    cfg.actions.drives.plant = TrainPlantCfg(randomization=[
        {"param": "friction", "dist": "uniform", "range": [0.5, 1.0], "op": "set"},
        {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
        {"param": "body_inertia", "dist": "uniform", "range": [0.8, 1.2], "op": "scale"},
    ])
    return ManagerBasedRLEnv(cfg)


def _rotation(q_xyzw):
    x, y, z, w = q_xyzw.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def test_manager_based_env_steps_the_plant():
    from humanoid_sim.physics.isaaclab import _quat_xyzw

    env = _env()
    try:
        term = env.action_manager.get_term("drives")
        plant = term.plant
        obs, _ = env.reset()
        assert obs["policy"].shape == (2, 22 + 6)
        t0 = plant.t_ns
        for _ in range(140):  # 2.8 s: the rigid standing pose tips over and comes to rest
            obs, _, _, truncated, _ = env.step(torch.zeros(2, 22))
            assert not truncated.any()
        assert plant.t_ns - t0 == 140 * 5 * plant.period_ns
        assert torch.isfinite(obs["policy"]).all()
        # IMU frame: at rest the sensor reads gravity's reaction in the chip frame of the description's mount.
        _, _, rpy = plant.desc.imu_mount()
        r_chip = _rotation(torch.tensor(_quat_xyzw(*rpy)))
        r_base = _rotation(plant.truth().root_pose_w[:, 3:])
        up_chip = torch.einsum("ji,ejk,k->ei", r_chip, r_base.transpose(1, 2), torch.tensor([0.0, 0.0, 9.81]))
        sensor = env.scene["imu"].data.lin_acc_b.torch
        rest = env.scene["imu"].data.ang_vel_b.torch.norm(dim=1) < 0.05
        assert rest.any()
        assert torch.allclose(sensor[rest], up_chip[rest], atol=0.3)
        # The plant delivers what the sensor measured: noise and the MTi-610's accelerometer drift (0.15 mm/s²),
        # no randomized bias or misalignment.
        latest = plant.measured().imu.specific_force[:, 0]
        assert torch.allclose(latest[rest], sensor[rest], atol=0.05)
        lat = plant.params()["latency"]
        assert ((lat >= 0.010) & (lat <= 0.020)).all()
        scale = plant.params()["body_inertia"]
        diagonal = [0, 4, 8]  # of the row-major 3x3 inertia; near-zero products of inertia get revalidated
        written = plant.robot.data.body_inertia.torch[..., diagonal] / plant._inertia0[..., diagonal]
        assert torch.allclose(written, scale.unsqueeze(-1).expand_as(written), rtol=1e-4)
        # Episode end at 3 s: the reset envs start again, IMU delivery restarts from zero.
        dones = 0
        for _ in range(11):
            _, _, _, truncated, _ = env.step(torch.zeros(2, 22))
            dones += int(truncated.sum())
        assert dones == 2
        assert (plant.imu.samples.count < 365 * 0.1).all()
    finally:
        env.close()
