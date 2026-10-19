"""A training scene on the Train-Time API: many robots on rough terrain, each holding the flat-foot pose at the
baseline gains while it waves its arms, with randomized friction, base mass and drive latency, and random pushes.
Optionally recorded to Rerun with `humanoid_sim.signals.scene.SceneRecorder`.

    python examples/train_scene.py --num-envs 64 --seconds 6 --device cuda:0 --rrd outputs/train_scene.rrd

There's no policy here: the demands are scripted, so a robot falls when a push or the terrain beats a rigid
pose. A task in a training repo replaces `ScriptedDrives.demands` with its policy's action path. Runs in
the Isaac Lab image (Warp's CPU device works for a few envs, slowly).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
WAVE = {"shoulder_flexion_l": -0.7, "shoulder_flexion_r": 0.7, "elbow_flexion_l": 0.6, "elbow_flexion_r": 0.6,
        "shoulder_abduction_l": 0.25, "shoulder_abduction_r": -0.25}


def make_env(num_envs: int, device: str, *, flat: bool):
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
    from humanoid_sim.train import TrainPlantCfg
    from humanoid_sim.train.isaaclab import PlantAction, PlantActionCfg
    from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg
    from humanoid_sim.world.terrain import rough_terrain_cfg

    class ScriptedDrives(PlantAction):
        """The scene's default pose (the flat-foot pose) at the baseline gains, plus an arm wave with a phase per
        env (rad, joint space)."""

        def demands(self, actions):
            core, n = self.plant.core, self.num_envs
            names = core.map.joint_names
            robot = self._env.scene["robot"]
            cols = [list(robot.joint_names).index(j) for j in names]
            stand = robot.data.default_joint_pos.torch[:, cols].to(self.device)
            amp = torch.tensor([WAVE.get(j, 0.0) for j in names], device=self.device)
            t = self.plant.t_ns / 1e9
            phase = torch.arange(n, device=self.device, dtype=torch.float32) * 0.7
            wave = 0.5 * (1 - torch.cos(2 * math.pi * 0.5 * t + phase)).unsqueeze(1)
            q = stand + amp * wave
            drive_q, _, _ = core.map.drive_state(q, torch.zeros_like(q))
            demand = DriveDemand.jvpt(drive_q, core.drive_stiffness.expand(n, -1), core.drive_damping.expand(n, -1))
            return demand, torch.full((n,), self.plant.t_ns, device=self.device)

    def base_height(env):
        return env.action_manager.get_term("drives").plant.truth().root_pose_w[:, 2:3]

    @configclass
    class Actions:
        drives = PlantActionCfg(class_type=ScriptedDrives, action_dim=22, plant=TrainPlantCfg(randomization=[
            {"param": "friction", "dist": "uniform", "range": [0.4, 1.0], "op": "set"},
            {"param": "body_mass", "bodies": "base_link", "dist": "uniform", "range": [0.0, 4.5], "op": "add"},
            {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
        ]))

    @configclass
    class Observations:
        @configclass
        class Policy(ObservationGroupCfg):
            height = ObservationTermCfg(func=base_height)

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
    class SceneCfg(ManagerBasedRLEnvCfg):
        decimation = 5
        episode_length_s = 60.0
        actions = Actions()
        observations = Observations()
        rewards = Rewards()
        terminations = Terminations()
        events = Events()

    cfg = SceneCfg()
    # One row of 2.5 m tiles, one per robot: Isaac Lab spawns envs that share a tile on the same spot.
    terrain = None if flat else rough_terrain_cfg(friction=0.0, rows=1, cols=num_envs, size_m=2.5, border_m=3.0,
                                                  curriculum=False)
    cfg.scene = robot_scene_cfg(num_envs, robot_usd(), terrain=terrain, ground_friction=0.0 if flat else None,
                                env_spacing=2.0, pose="flat_feet")
    cfg.sim = robot_sim_cfg(device=device)
    return ManagerBasedRLEnv(cfg)


def robot_usd() -> Path:
    """The USD `humanoid-sim build-asset` wrote for this description and profile."""
    from humanoid_sim.model.asset import asset_dir, load_profile
    from humanoid_sim.model.description import Description

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rrd", default=None, help="record the scene to this Rerun file")
    p.add_argument("--record-every", type=int, default=2, help="record every this many policy steps")
    p.add_argument("--flat", action="store_true", help="a flat ground plane instead of rough terrain")
    args = p.parse_args(argv)

    from humanoid_sim.world.disturbances import ForcePush

    env = make_env(args.num_envs, args.device, flat=args.flat)
    recorder = None
    if args.rrd:
        from humanoid_sim.signals.scene import SceneRecorder

        recorder = SceneRecorder(env, robot_usd(), rrd=args.rrd)
    try:
        plant = env.action_manager.get_term("drives").plant
        if recorder is not None:
            from humanoid_sim.signals.log import SignalLog

            plant.attach_log(SignalLog(init=False))  # robot 0's signals join the scene recording
            knee = plant.drives.index(plant.core.desc.joints["knee_l"].device)
        push = ForcePush(env.scene["robot"], "base_link", interval_s=(1.5, 3.0), duration_s=(0.1, 0.2),
                         magnitude_n=(40.0, 120.0))
        env.reset()
        push.reset(torch.arange(args.num_envs, device=env.device), plant.t_ns)
        actions = torch.zeros(args.num_envs, 22, device=env.device)
        steps = int(args.seconds / env.step_dt)
        for k in range(steps):
            push.step(plant.t_ns)
            env.step(actions)
            if recorder is not None and k % args.record_every == 0:
                recorder.frame(plant.t_ns)
                m, t = plant.measured().drive, plant.truth().drive
                plant.log("knee_l/position_reported", m.position_rad[:, knee], group="drives")
                plant.log("knee_l/position_true", t.position_rad[:, knee], group="drives")
                plant.log("knee_l/applied_torque", t.applied_torque_nm[:, knee], group="drives")
                plant.log("base_height", plant.truth().root_pose_w[:, 2], group="base")
        up = (plant.truth().root_pose_w[:, 2] > 0.6).sum().item()
        print(f"{args.num_envs} robots, {steps * env.step_dt:.1f} s simulated, {up} still standing")
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
