"""A training scene on the Train-Time API: many robots on bumpy heightfield terrain, each holding the home pose at the
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
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
    from isaaclab.utils import configclass

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.train import TrainPlantCfg
    from humanoid_sim.train.isaaclab import PlantAction, PlantActionCfg
    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg
    from humanoid_sim.world.terrain import heightfield_terrain_cfg

    class ScriptedDrives(PlantAction):
        """The scene's default pose (the home pose) at the baseline gains, plus an arm wave with a phase per
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
            demand = DriveDemand.pd(drive_q, core.drive_stiffness.expand(n, -1), core.drive_damping.expand(n, -1))
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
    # One row of 2.5 m tiles, one per robot: Isaac Lab spawns envs that share a tile on the same spot. The tile mix
    # is this example's; a task takes its terrain from its own generator (humanoid-learning).
    bumps = {"bumps": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, 0.025), noise_step=0.005)}
    terrain = None if flat else heightfield_terrain_cfg(bumps, friction=0.0, rows=1, cols=num_envs, size_m=2.5,
                                                        border_m=3.0, curriculum=False)
    cfg.scene = robot_scene_cfg(num_envs, robot_usd(), terrain=terrain, ground_friction=0.0 if flat else None,
                                env_spacing=2.0)
    cfg.sim = robot_sim_cfg(device=device)
    with robot_build():
        return ManagerBasedRLEnv(cfg)


def robot_usd() -> Path:
    """The USD `humanoid-sim build-asset` wrote for this description and profile."""
    from humanoid_sim.model.asset import asset_dir, load_profile
    from humanoid_sim.model.description import Description

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    return out / json.loads((out / "asset.json").read_text())["usd"]


# What `--gantry` hangs, by env index modulo 4: a rope carrying 80% of the weight, a slack rope, no rope, and a rope
# whose anchor is off to the side by `OFF_ANCHOR_M` (the mistake of building the anchor from the wrong robot position).
LOAD_SHARE, SLACK_M, OFF_ANCHOR_M = 0.8, 0.05, 0.6


def hang_ropes(env, plant) -> None:
    """Hang the four kinds of rope above from the harness hook of each robot where it stands now."""
    from humanoid_sim.world.supports import GRAVITY_M_S2

    n, dev, robot = env.num_envs, env.device, env.scene["robot"]
    support = plant.profile["support"]
    mass = robot.data.body_mass.torch.sum(dim=1)
    k = mass * GRAVITY_M_S2 / support["full_weight_stretch_m"]
    point = torch.tensor([0.0, 0.0, 0.2], device=dev)  # base frame; the harness hook of a real task is on the torso
    pose = robot.data.body_link_pose_w.torch[:, 0]
    hook = pose[:, :3] + point
    anchor = hook + torch.tensor([0.0, 0.0, support["overhead_m"]], device=dev)
    kind = torch.arange(n, device=dev) % 4
    reach = (anchor - hook).norm(dim=1)  # the rope is cut for the right anchor, then hung from a wrong one
    anchor[kind == 3, 0] += OFF_ANCHOR_M
    carried = torch.where(kind == 1, -SLACK_M, LOAD_SHARE * mass * GRAVITY_M_S2 / k)  # stretch under the rope, m
    rest = (reach - carried).clamp(min=0.05)[:, None]
    ids = torch.nonzero(kind != 2).squeeze(1)
    plant.gantry.hang_cables(ids, plant.t_ns, point_b=point, anchor_w=anchor[ids],
                             times_ns=torch.zeros(len(ids), 1, dtype=torch.int64, device=dev), lengths_m=rest[ids],
                             stiffness_n_m=k[ids], damping_n_s_m=2.0 * (k[ids] * mass[ids]).sqrt(),
                             swing_damping_n_s_m=torch.zeros(len(ids), device=dev))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rrd", default=None, help="record the scene to this Rerun file")
    p.add_argument("--record-every", type=int, default=2, help="record every this many policy steps")
    p.add_argument("--flat", action="store_true", help="a flat ground plane instead of the bumpy terrain")
    p.add_argument("--gantry", action="store_true", help="hang the robots from gantry ropes, one of four kinds per env "
                                                         "(see hang_ropes), to show how a recording draws them")
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
        if args.gantry:
            env.step(actions)  # body poses are stale right after a reset
            hang_ropes(env, plant)
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
