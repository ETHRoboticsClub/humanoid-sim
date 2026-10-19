# Training on the Train-Time API

`humanoid_sim.train` puts the robot's [plant](glossary.md#plant) inside a manager-based Isaac Lab environment. The plant is everything between a drive demand and the physics solver: the 22 EtherCAT drives (18 that turn one joint each and 4 that turn the [ankle cranks](glossary.md#ankle-crank), the levers that move each foot in pitch and roll through a linkage), the crank transmission, the drives' feedback reporting and the IMU's sample delivery. A task turns policy actions into drive demands and reads back what the robot's computer would read. Physics is Isaac Lab 3.0 on Newton's MJWarp solver, with every env (one simulated robot) batched on the GPU.

This page is for someone writing a task. The drive model itself is in [drives.md](drives.md), the IMU and the drive reporting in [sensors.md](sensors.md), terrain, supports and pushes in [world.md](world.md).

## Start from the example

`examples/train_scene.py` is a complete scene on this API and the place to start a task from. It has no policy: its action term, `ScriptedDrives`, demands the scene's default pose at the baseline drive gains plus an arm wave (shoulders and elbows, 0.5 Hz, a different phase per env). A task replaces `ScriptedDrives.demands` with its policy's action path. The rest of the scene is what most tasks need:

- The robot starts in the home pose (the default of `robot_scene_cfg`, see [The initial pose](#the-initial-pose)), and the demands are taken from the articulation's `data.default_joint_pos`.
- The ground is a heightfield of random bumps up to 25 mm (`heightfield_terrain_cfg` with one `HfRandomUniformTerrainCfg` tile type) laid out as one row of 2.5 m tiles, one tile per robot, without a curriculum. The tile mix is the example's own; a task takes its terrain from its generator in humanoid-learning. Isaac Lab puts every env assigned to a tile at that tile's origin, so envs that share a tile spawn on the same spot. `--flat` uses the flat ground instead.
- Robot friction (0.4 to 1.0), a payload of 0 to 4.5 kg on `base_link` and one drive latency per env (10 to 20 ms) are randomized. The terrain and the flat ground have friction 0 so the robot's value decides.
- A `ForcePush` pushes `base_link` with 40 to 120 N for 0.1 to 0.2 s, with 1.5 to 3 s between pushes. The loop steps it once per policy step.
- With `--rrd`, a `SceneRecorder` records every robot to a Rerun file, and `plant.log` adds robot 0's left knee and base height to the same recording, every `--record-every` policy steps (default 2). See [Recording a scene](#recording-a-scene).

Run it in the Isaac Lab image (`hsim`, see [getting-started.md](getting-started.md#run-commands-in-the-image)) or on a GPU node. Warp's CPU device works for a few envs, slowly:

```bash
hsim python examples/train_scene.py --num-envs 4 --seconds 2 --device cpu
python examples/train_scene.py --num-envs 64 --seconds 6 --device cuda:0 --rrd outputs/train_scene.rrd
```

It prints how many robots still have their base above 0.6 m at the end. A rigid pose does not survive pushes and terrain for long, so most robots fall within a few seconds. Once they are down, MJWarp floods the console with `height field collision overflow` messages; [world.md](world.md#heightfield-collision-overflow) explains them.

## The pieces

| Name | Module | Role |
|---|---|---|
| `TrainPlantCfg` | `src/humanoid_sim/train/plant.py` | Plant settings: robot, randomization list, level, seed, history length |
| `RobotPlant` | `src/humanoid_sim/train/plant.py` | The batched plant for every env of a scene: `submit`, `pre_physics`, `measured`, `truth`, `params`, `reset`, `set_level`, `attach_log`, `log` |
| `Measured`, `Truth` | `src/humanoid_sim/train/plant.py` | What `measured()` and `truth()` return |
| `PlantAction`, `PlantActionCfg` | `src/humanoid_sim/train/isaaclab.py` | An Isaac Lab action term that owns a `RobotPlant`; you subclass it and implement `demands()` |
| `robot_scene_cfg`, `robot_sim_cfg` | `src/humanoid_sim/train/scene.py` | The `InteractiveSceneCfg` (robot, IMU, contact sensor, ground or terrain) and the `SimulationCfg` (one 4 ms step on MJWarp) |
| `initial_pose`, `POSES` | `src/humanoid_sim/model/poses.py`, re-exported by `humanoid_sim.train.scene` | The joint positions and base height of a named initial pose |
| `heightfield_terrain_cfg` | `src/humanoid_sim/world/terrain.py` | The terrain-importer config for a heightfield terrain the caller lays out, described in [world.md](world.md#heightfield-terrain) |
| `ForcePush`, `Gantry` | `src/humanoid_sim/world/` | Pushes and gantry ropes a task applies, described in [world.md](world.md) |
| `DriveDemand` | `src/humanoid_sim/devices/demand.py` | One SI demand per env and drive: law, targets, gains, torque limit |
| `SceneRecorder` | `src/humanoid_sim/signals/scene.py` | Records many envs of a scene to one Rerun recording |

`TrainPlantCfg` fields:

| Field | Default | Meaning |
|---|---|---|
| `robot` | `"ethrc_v1"` | Profile (`profiles/ethrc_v1.yaml`) and description to load |
| `randomization` | `[]` | Randomization specs, as dicts or `ParamSpec`; see [Randomization](#randomization) |
| `level` | `1.0` | Initial randomization level |
| `seed` | `0` | Seed of the randomizer; the IMU model uses `seed + 1` and the encoder jitter `seed + 2` |
| `feedback_history` | `16` | Physics steps of drive feedback kept in `Measured.history` |
| `reporting` | `True` | Quantize the measured drive feedback the way the drives report it |
| `queue_capacity` | `8` | Demands in flight per env; needs at least max latency / demand period + 1 |

`PlantActionCfg` extends Isaac Lab's `ActionTermCfg`:

| Field | Default | Meaning |
|---|---|---|
| `class_type` | `PlantAction` | Your subclass |
| `action_dim` | required | Size of the policy action (22 for one value per drive) |
| `plant` | `TrainPlantCfg()` | The plant config |
| `asset_name` | `"robot"` | Scene entry of the robot articulation |
| `imu_name` | `"imu"` | Scene entry of the IMU sensor; `None` for no IMU, and then `Measured.imu` is `None` |
| `contacts_name` | `"contacts"` | Scene entry of the contact sensor; `None` for none, and then `Truth.contact_forces_w` is `None` |

### How an env step runs the plant

Isaac Lab's `ManagerBasedRLEnv.step` calls each action term's `process_actions` once per env step and `apply_actions` before every physics step. `PlantAction` uses them like this:

1. `process_actions(actions)` calls your `demands(actions)` and hands the result to `RobotPlant.submit`. Each drive queues the demand to take effect at `sent_ns` plus that drive's [latency](glossary.md#latency).
2. `apply_actions()` calls `RobotPlant.pre_physics()`. With the state at the start of the physics step it appends drive feedback to the history and steps the IMU delivery model; then it runs the drive bank over the step (every demand in force during the step, weighted by the time it covers), maps drive torque to joint torque through the ankle transmission, and writes the position target that makes the solver's constant PD produce that torque. `plant.t_ns` then advances by one physics period.
3. When envs finish an episode, Isaac Lab runs the reset events, which write the new state, and then the action manager's reset, which calls `RobotPlant.reset(env_ids)`.

The solver's PD gains are constant per joint and written once at spawn. The drives' own gains, the torque-speed envelope and the torque cap are all realized by moving the position target ([target shaping](glossary.md#target-shaping), see [drives.md](drives.md)), so nothing in the solver model changes while training steps.

## Build the robot USD first

`robot_scene_cfg` reads the built asset: `asset.json` holds the sole boxes, from which it computes the initial pose and the base height with Newton's forward kinematics on the host. The USD must exist before you build the env config. Build it once in the Isaac Lab image:

```bash
humanoid-sim build-asset
```

The asset goes to `outputs/assets/ethrc_v1/<key>/`, where the key is a digest of the description and the profile; changing either needs a rebuild. In code, `build_asset` (`src/humanoid_sim/model/asset.py`) returns the USD path of an asset that is already built, and builds it otherwise (which needs the image):

```python
from pathlib import Path

from humanoid_sim.model.asset import build_asset, load_profile
from humanoid_sim.model.description import Description

usd = build_asset(Description.load("ethrc_v1"), load_profile("ethrc_v1"), Path("outputs/assets"))
```

Train on the plain asset. The `--base-joints` variant is for replays ([replay-and-sysid.md](replay-and-sysid.md)). Setup of the image is in [getting-started.md](getting-started.md).

## The initial pose

`robot_scene_cfg(..., pose=...)` picks the joint positions every env spawns and resets to. The names are in `POSES`; any other name raises a `ValueError`.

| `pose` | Joint positions | Held rigidly on flat ground |
|---|---|---|
| `"home"` (default) | The description's [home pose](glossary.md#home-pose) (`Description.home_pose`): the V1's CAD export pose, legs straight; the G1's deploy standing pose, knees bent 0.3 rad; both soles level at one height | At the baseline gains the V1 stands still on flat ground (60 s measured, the base within 0.6° of upright); pushes and rough terrain make it fall unless a policy balances it. The G1 tips over within about 1.3 s: an inverted pendulum on its ankles, it stands only on ankles stiffer in pitch than its weight times its centre of mass's height above them (about 210 Nm/rad), and Unitree's deploy gains give 80 Nm/rad for both. A G1 task needs its policy to balance from the first step (`tests/test_world.py::test_holding_the_home_pose_stands_only_on_ankles_stiffer_than_toppling`). |
| `"standing"` | The description's [standing pose](glossary.md#standing-pose), captured on the robot: every joint at its `standing_offset_rad` | Under an upright base the soles are pitched about 10° (left) and 17° (right). The robot lands on its sole edges, hops and tips over within about a second. |

In both poses the base is upright and the lowest sole corner is 1 mm above the ground (`STANDING_CLEARANCE_M` in `src/humanoid_sim/model/poses.py`); on the current description the base starts 0.924 m up in the home pose and 0.945 m up in the standing pose. Where the home pose comes from, and why the standing pose does not stand, is in [robot-model.md](robot-model.md#standing-on-the-ground).

The articulation's `data.default_joint_pos` holds the chosen pose (articulation joint order), so `mdp.reset_scene_to_default` puts the robot back into it. Take action offsets from `default_joint_pos`, as `examples/train_scene.py` does, instead of from `standing_offset_rad`; then the task follows whichever pose the scene was built with.

`initial_pose(usd, robot="ethrc_v1", pose="home")` returns the pose's joint positions `[J]` (description order, float64) and the base height at which its lowest sole corner touches the ground, without the clearance. To build your own scene config, `articulation_cfg(usd, props, ..., joint_pos=...)` in `src/humanoid_sim/physics/isaaclab.py` takes the initial joint positions in `props.names` order; without `joint_pos` it uses the standing pose.

## A minimal environment

This env holds the home pose with policy actions as offsets, observes measured signals in the actor group and truth in the critic group, and randomizes friction, latency and inertia. It follows `tests/test_train_env.py` and `examples/train_scene.py`, and runs as written in the Isaac Lab image (change the device to `"cpu"` and the env count to a few on a machine without CUDA).

```python
from pathlib import Path

import torch

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
from humanoid_sim.model.asset import build_asset, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.train import TrainPlantCfg
from humanoid_sim.train.isaaclab import PlantAction, PlantActionCfg
from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg


class PoseOffsets(PlantAction):
    """Actions are offsets (rad, drive frame) from the scene's default pose, at the baseline drive gains."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        core, n = self.plant.core, self.num_envs
        q = self._asset.data.default_joint_pos.torch[:, self.plant.joint_ids]  # [E, 22], description order
        self.pose, _, _ = core.map.drive_state(q, torch.zeros_like(q))  # [E, 22], drive frame
        self.kp = core.drive_stiffness.expand(n, -1)  # baseline gains, Nm/rad and Nm s/rad
        self.kd = core.drive_damping.expand(n, -1)

    def demands(self, actions):
        demand = DriveDemand.pd(self.pose + 0.1 * actions, self.kp, self.kd)
        sent_ns = torch.full((self.num_envs,), self.plant.t_ns, dtype=torch.int64, device=self.device)
        return demand, sent_ns


def plant(env):
    return env.action_manager.get_term("drives").plant


def actor_obs(env):
    m = plant(env).measured()  # one call: the encoder jitter is drawn per call
    return torch.cat([m.drive.position_rad, m.drive.velocity_rad_s, m.imu.gyro[:, 0], m.imu.specific_force[:, 0]],
                     dim=1)


def base_height(env):
    return plant(env).truth().root_pose_w[:, 2] - env.scene.env_origins[:, 2]


def critic_obs(env):
    p = plant(env)
    t = p.truth()
    return torch.cat([t.joint_pos, t.joint_vel, base_height(env).unsqueeze(1), t.root_pose_w[:, 3:],
                      t.root_lin_vel_w, t.root_ang_vel_b, p.params()["latency"]], dim=1)


def fallen(env):
    return base_height(env) < 0.6  # the base starts 0.924 m up


@configclass
class Actions:
    drives = PlantActionCfg(class_type=PoseOffsets, action_dim=22, plant=TrainPlantCfg(randomization=[
        {"param": "friction", "dist": "uniform", "range": [0.4, 1.0], "op": "set"},
        {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
        {"param": "body_inertia", "dist": "uniform", "range": [0.8, 1.2], "op": "scale"},
    ]))


@configclass
class Observations:
    @configclass
    class Policy(ObservationGroupCfg):
        measured = ObservationTermCfg(func=actor_obs)

    @configclass
    class Critic(ObservationGroupCfg):
        truth = ObservationTermCfg(func=critic_obs)

    policy = Policy()
    critic = Critic()


@configclass
class Rewards:
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    height = RewardTermCfg(func=base_height, weight=1.0)


@configclass
class Terminations:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fallen = TerminationTermCfg(func=fallen)


@configclass
class Events:
    reset = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class StandCfg(ManagerBasedRLEnvCfg):
    decimation = 5  # 20 ms policy period at the plant's 4 ms physics period
    episode_length_s = 10.0
    actions = Actions()
    observations = Observations()
    rewards = Rewards()
    terminations = Terminations()
    events = Events()


usd = build_asset(Description.load("ethrc_v1"), load_profile("ethrc_v1"), Path("outputs/assets"))
cfg = StandCfg()
cfg.scene = robot_scene_cfg(4096, usd, ground_friction=0.0)  # home pose; friction is randomized on the robot
cfg.sim = robot_sim_cfg(device="cuda:0")
env = ManagerBasedRLEnv(cfg)
```

The observation groups come out as `policy` `[E, 50]` and `critic` `[E, 77]`.

The [physics period](glossary.md#physics-period) comes from the profile (`physics.period_ns: 4000000`, 4 ms or 250 Hz), and `robot_sim_cfg` sets the simulation `dt` from it. With `decimation = 5` the policy runs every 20 ms (50 Hz). The plant does not depend on the decimation: it runs once per physics step and stamps everything on its own clock.

Other terms reach the plant through the action term, `env.action_manager.get_term("drives").plant`; the term can have any name.

For the rsl-rl side, `tests/test_train_smoke.py` in [humanoid-learning](https://github.com/ETHRoboticsClub/humanoid-learning) is the working example: it builds an env on this API, wraps it in `RslRlVecEnvWrapper` and trains one PPO iteration on the plant. With separate actor and critic groups as above, the runner config maps them with `obs_groups={"actor": ["policy"], "critic": ["critic"]}`.

### What `demands()` returns

`demands(actions)` returns a pair `(demand, sent_ns)`.

`demand` is a `DriveDemand` with every field `[E, D]` (D is 22 for the ETHRC V1, 29 for the G1), in the plant's drive order (`plant.drives`, the description's drive order). It is in SI units and names no vendor:

| Field | Type | Meaning |
|---|---|---|
| `law` | int8 | `LAW_DISABLED` (0), `LAW_PD` (1) or `LAW_FREEZE` (2) |
| `target_position_rad` | float | Target in the physical drive-output frame (rad) |
| `target_velocity_rad_s` | float | Velocity target (rad/s) |
| `feedforward_torque_nm` | float | Feed-forward torque (Nm) |
| `stiffness_nm_per_rad`, `damping_nm_s_per_rad` | float | P and D gains |
| `torque_limit_nm` | float | Cap on the applied torque, `inf` for none |

Under the PD law a drive applies `kp (target - q) + kd (v_target - v) + ff`, clipped by its torque-speed envelope and the limit. `DriveDemand.pd(position, stiffness, damping, *, velocity=None, feedforward=None, torque_limit_nm=None)` builds a PD demand; `DriveDemand.disabled(num_envs, num_drives, device=...)` disables every drive. The laws are in [drives.md](drives.md#the-demand).

Positions are in the [drive frame](glossary.md#drive-frame). For a 1:1 drive that is the joint angle times a sign; for an ankle drive it is the crank angle, which is neither the foot pitch nor the foot roll. `plant.core.map.drive_state(q, qd)` maps joint positions in description order to drive positions through the ankle transmission, which is how `PoseOffsets` turns the default pose into drive targets. A policy that acts in foot pitch and roll converts them the same way inside `demands()`.

`sent_ns` is an `[E]` int64 tensor: when the demand leaves the EtherCAT master, in nanoseconds on the plant clock. `self.plant.t_ns` is now, the start of the first physics step of this env step. Drive `d` of env `e` applies the demand from `sent_ns[e] + latency[e, d]`, where `latency` comes from randomization and is zero when it isn't randomized. A demand that arrives inside a physics step is in force for the part of the step it covers. Demands never overtake each other on a drive: an arrival earlier than the last queued one is moved to it. When a queue is full, its oldest pending demand is dropped, so size `queue_capacity` for your submit rate and maximum latency.

The [baseline gains](glossary.md#baseline-gains) (`plant.core.drive_stiffness`, `plant.core.drive_damping`, from humanoid-urdf) are also the solver gains. Target shaping is guaranteed stable for demanded gains up to twice the baseline; `plant.core.gain_warnings` (`[E, 22]` bool, cleared at reset) marks drives that received an enabled demand above that.

## Reading the plant

### `measured()`

`measured()` returns what the robot's computer reads:

| Field | Shape | Content |
|---|---|---|
| `drive` | `DriveFeedback`, each field `[E, 22]` | Feedback as the drives report it now: `position_rad`, `velocity_rad_s`, `applied_torque_nm`, `demanded_torque_nm` |
| `history` | `DriveFeedback`, each field `[E, H, 22]` | The same, one entry per physics step, newest first (`H = feedback_history`) |
| `history_ns` | `[E, H]` int64 | Plant time of each history entry; -1 where nothing was recorded since the last reset |
| `imu` | `ImuSamples` or `None` | Raw IMU samples received so far, newest first |

With `reporting=True` the drive values are quantized the way Controls' fake EtherCAT drive (`Epos4Slave.cpp`) reports them, in `src/humanoid_sim/devices/reporting.py`. The position is the drive's own reading in whole encoder counts, so it includes the drive's `encoder_offset`, plus a random -1, 0 or +1 count on every read. The velocity is whole counts per controller sample at 2.5 kHz, converted back to rad/s. The applied torque is in whole mNm; the demanded torque passes through. Values stay in rad, rad/s and Nm. With `reporting=False` only the offset is added.

| Drive family | Drives | Counts per rev | Position step | Velocity step |
|---|---|---|---|---|
| HEJ90-48-140 | central hip rotation, waist yaw, hips, knees | 4096 | 1.53 mrad | 3.83 rad/s |
| HEJ70-48-50 | shoulder flexion and abduction | 16384 | 0.38 mrad | 0.96 rad/s |
| HEJ50-48-30 | shoulder rotation, elbows, ankle cranks | 16384 | 0.38 mrad | 0.96 rad/s |

The velocity steps are coarse; this is what the real drives report.

`ImuSamples` (`src/humanoid_sim/sensors/imu.py`) keeps the 16 most recent delivered samples:

| Field | Shape | Content |
|---|---|---|
| `gyro` | `[E, 16, 3]` | Angular velocity, rad/s, sensor frame |
| `specific_force` | `[E, 16, 3]` | Accelerometer reading, m/s², sensor frame (reads +9.81 along "up" at rest) |
| `sample_ns` | `[E, 16]` | Measurement time on the plant clock; -1 for an empty slot |
| `stamp_ns` | `[E, 16]` | Time the host received the sample; -1 for an empty slot |
| `seq` | `[E, 16]` | Sequence number, from 1 after a reset; 0 for an empty slot |
| `count` | `[E]` | Samples delivered since the last reset |

The sensor frame is the chip frame at the description's IMU mount (`imu_joint` on `base_link`); nothing rotates it into the base frame. The sensor samples every 2.74 ms (about 365 samples/s) and the host receives them in [bursts](glossary.md#imu-burst) every 13.9 to 17.6 ms, so the newest sample can be one burst period old. Each sample carries white noise at the MTi-610 datasheet densities, plus any bias and misalignment the randomization drew. Details are in [sensors.md](sensors.md).

Actor observations come from `measured()` only. That is the information a deployed policy has on the robot; `truth()` and `params()` are for critics, rewards, terminations and diagnostics. The plant hands out raw drive feedback and raw IMU samples. What Controls computes from them before a policy sees them (an orientation estimate, the standing-frame joint offsets, the observation layout of its policy contract) is Controls behaviour and is not modeled here; a task that trains for Controls reproduces it from measured signals ([controls-integration.md](controls-integration.md)).

### `truth()`

| Field | Shape | Content |
|---|---|---|
| `joint_pos`, `joint_vel` | `[E, 22]` | URDF joint coordinates (rad, rad/s), description joint order |
| `root_pose_w` | `[E, 7]` | Base position (m) and quaternion (x, y, z, w) in the world frame; the env origin is not subtracted |
| `root_lin_vel_w` | `[E, 3]` | Linear velocity of the base's centre of mass, world frame (m/s) |
| `root_ang_vel_b` | `[E, 3]` | Base angular velocity, base frame (rad/s) |
| `drive` | `DriveFeedback`, each field `[E, 22]` | Physical drive-frame feedback, without offsets or quantization |
| `contact_forces_w` | `[E, B, 3]` or `None` | Net contact force on each body (N, world frame) |

The contact sensor covers every body and also tracks force history (3 steps) and air time; read those from `env.scene["contacts"].data`. Its body order is the sensor's own (`sensor_names`), so look bodies up by name there, as the runtime host does, instead of assuming the articulation's `body_names` order.

### `params()`

`params()` returns the realized randomized parameters per env as a dict keyed by parameter name. Only the parameters in the spec list appear. Units are those of the [parameter table](#parameters); shapes are `[E, N]` over the parameter's targets, `[E, 23, 3]` for `body_com`, `[E, 1]` for `friction` and `[E, 3]` for the IMU parameters. They suit privileged critic inputs (latency, payload, friction). The tensors are the randomizer's own; clone what you keep.

## Time

`plant.t_ns` is a Python int: nanoseconds since the plant was constructed, advanced by one physics period in every `pre_physics()`. There is one clock for all envs, and resets do not restart it. `sent_ns`, `history_ns` and the IMU's `sample_ns` and `stamp_ns` are all on this clock, so the age of the newest IMU sample is `plant.t_ns - m.imu.stamp_ns[:, 0]`. For time within an episode use Isaac Lab's `env.episode_length_buf` or record `plant.t_ns` at reset.

## Resets

`RobotPlant.reset(env_ids, *, joint_pos=None, joint_vel=None, root_pose=None, root_vel=None, demand=None)` resets the given envs:

- The given state is written: joint state in description order (a missing `joint_vel` means zero), root pose as position plus quaternion (x, y, z, w) in the world frame (add `env.scene.env_origins`), root velocity `[n, 6]`.
- Queued demands are dropped and every drive is off, unless `demand` (rows for `env_ids`) is passed; it is then in force from the next physics step, with no latency.
- Gain warnings and external joint torques (`plant.core.external_torque`) are cleared.
- The reset-scope device parameters are redrawn for these envs and written. Solver parameters are not redrawn.
- The feedback history is marked empty: every `history_ns` entry is -1. The history values themselves still hold the previous episode's feedback until new physics steps push it out.
- IMU delivery restarts at `plant.t_ns` with a new random sample phase and burst phase. `count` is 0, every slot has `sample_ns` and `stamp_ns` -1 and `seq` 0, and `gyro` and `specific_force` keep the previous episode's samples until new ones are delivered.

`plant.t_ns` keeps running. `PlantAction.reset` calls `plant.reset(env_ids)` with no arguments after Isaac Lab's reset events, so the state comes from your events and the drives come back off. The first demand after the reset is submitted in the next env step and acts one latency later (10 to 20 ms with the example's randomization); a standing robot is unpowered until then. To have the drives hold from the first physics step, pass a demand from your own `reset`:

```python
class HeldPoseOffsets(PoseOffsets):  # PoseOffsets from the minimal environment
    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.raw_actions[env_ids] = 0.0
        n = len(env_ids)
        self.plant.reset(env_ids, demand=DriveDemand.pd(self.pose[env_ids], self.kp[:n], self.kd[:n]))
```

## Randomization

The plant draws its own [randomization](glossary.md#domain-randomization) (`src/humanoid_sim/randomization/`), batched on the device and seeded by `TrainPlantCfg.seed`. Each entry of `TrainPlantCfg.randomization` is a dict (parsed by `ParamSpec.from_dict`) or a `ParamSpec`:

| Key | Values | Meaning |
|---|---|---|
| `param` | a name from the [parameter table](#parameters) | What to draw |
| `dist` | `uniform`, `log_uniform`, `normal` | Distribution |
| `range` | two numbers | `[low, high]` for `uniform` and `log_uniform` (positive for `log_uniform`), `[mean, std]` for `normal` |
| `op` | `scale`, `add`, `set` | Multiply the current value, add to it, or replace it |
| `bodies`, `joints`, `drives` | regex, default `.*` | Which targets, by full match on their names; the key must match the parameter's target |
| `select` | regex over `x`, `y`, `z`, default `.*` | Which axes, for the three `imu_*` parameters |
| `scope` | `startup`, `reset`, unset | When to draw; see below |
| `levels` | `true` (default), `false` | Whether the level shrinks this draw |
| `per_target` | `true` (default), `false` | One draw per selected target, or one draw shared by all of an env's selected targets |

Unknown names, distributions, ops or scopes raise, as does a selector key that doesn't fit the parameter (a `bodies` key on a drive parameter) or a regex that selects nothing. Body names are the merged URDF link names (`Component171_1` is the upper body above the waist yaw joint, `base_link` the root body that carries the IMU; see [robot-model.md](robot-model.md)), joint names are the description's, drive names the EtherCAT drive names such as `maxonHEJ90L4`. `friction` has no selector. The IMU parameters apply to all three axes unless `select` picks some, for example `{"param": "imu_gyro_bias", "select": "z", "dist": "normal", "range": [0.0, 0.003], "op": "add"}` for a yaw-rate bias only; `ParamSpec(..., select="z")` is the same.

Every parameter starts from its nominal value. Startup draws apply first, in list order; reset draws then apply in list order on top of the startup value, for the envs being reset. So `body_mass` can be scaled on every body and then get a payload added on one, and a reset-scope `encoder_offset` can add to a startup-scope one.

When a spec is drawn depends on the parameter's kind:

- Physics parameters live in the solver model. Writing them on MJWarp updates every world and syncs with the host, so they are drawn at construction and redrawn only by `set_level`. `scope: reset` on one raises.
- Device parameters live in this repo's tensors and are redrawn for each env at its reset (and at construction). With `scope: startup` they are drawn at construction and on `set_level` only.

The level, in [0, 1], pulls every draw toward its identity: `draw(level) = identity + level * (draw - identity)`. The identity is 1 for `scale`, 0 for `add`, and for `set` the centre of the distribution (the midpoint for `uniform`, the geometric mean for `log_uniform`, the mean for `normal`). `levels: false` exempts a spec. At level 0 every scale and add spec gives the nominal value.

### Parameters

The full table is `PARAMS` in `src/humanoid_sim/randomization/spec.py`. E is the number of envs; the robot has 23 bodies, 22 joints and 22 drives.

| Parameter | Kind | Shape | Unit | Nominal | Meaning |
|---|---|---|---|---|---|
| `body_mass` | physics | `[E, 23]` | kg | model masses | Body mass; the inertia scales with it |
| `body_com` | physics | `[E, 23, 3]` | m | 0 | Centre-of-mass offset in the body frame, added to the model's |
| `body_inertia` | physics | `[E, 23]` | ratio | 1 | Scale on the rotational inertia beyond the mass scaling |
| `joint_armature` | physics | `[E, 22]` | kg m² | model values | Reflected rotor inertia |
| `joint_friction` | physics | `[E, 22]` | N m | model values | Coulomb friction (MJWarp frictionloss) |
| `joint_damping` | physics | `[E, 22]` | N m s/rad | model values | Passive viscous damping |
| `friction` | physics | `[E, 1]` | coefficient | robot shape friction | Friction coefficient of every robot shape |
| `drive_stiffness` | device | `[E, 22]` | ratio | 1 | Realized P gain over the demanded one |
| `drive_damping` | device | `[E, 22]` | ratio | 1 | Realized D gain over the demanded one |
| `torque_limit` | device | `[E, 22]` | ratio | 1 | Scale on the torque-speed envelope and the torque cap |
| `latency` | device | `[E, 22]` | s | 0 | Bus and drive latency from demand sent to demand in force |
| `encoder_offset` | device | `[E, 22]` | rad | 0 | The drive's position reading minus the true position |
| `imu_gyro_bias` | device | `[E, 3]` | rad/s | 0 | Gyro bias, sensor frame |
| `imu_accel_bias` | device | `[E, 3]` | m/s² | 0 | Accelerometer bias, sensor frame |
| `imu_misalignment` | device | `[E, 3]` | rad | 0 | Rotation of the sensor in its mount, as a rotation vector |

A drive acts on its own reading, so with an `encoder_offset` it regulates the offset position: the drive output settles `encoder_offset` away from the demanded target, and `measured()` shows the offset reading while `truth()` shows the physical one. Randomizing `friction` needs the physics manager, which `PlantAction` passes in; see the ground-friction pitfall below. The IMU parameters only take effect when the scene has the IMU.

A list like the one the train benchmark uses (`src/humanoid_sim/bench/train.py`), with a few additions:

```python
RANDOMIZATION = [
    {"param": "body_mass", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
    {"param": "body_mass", "bodies": "Component171_1", "dist": "uniform", "range": [0.0, 4.5], "op": "add"},
    {"param": "body_com", "bodies": "Component171_1", "dist": "uniform", "range": [-0.02, 0.02], "op": "add"},
    {"param": "joint_armature", "dist": "uniform", "range": [0.8, 1.2], "op": "scale"},
    {"param": "joint_friction", "joints": "shoulder_.*|elbow_.*", "dist": "log_uniform", "range": [0.5, 2.0],
     "op": "scale"},
    {"param": "friction", "dist": "uniform", "range": [0.3, 1.0], "op": "set", "levels": False},
    {"param": "drive_stiffness", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
    {"param": "drive_damping", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
    {"param": "torque_limit", "dist": "uniform", "range": [0.85, 1.0], "op": "scale"},
    {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
    {"param": "latency", "drives": ".*ANK.*", "dist": "uniform", "range": [0.010, 0.020], "op": "add",
     "per_target": False},
    {"param": "encoder_offset", "dist": "uniform", "range": [-0.026, 0.026], "op": "add"},
    {"param": "encoder_offset", "drives": "maxonHEJ50._ANK.*", "dist": "uniform", "range": [-0.14, 0.14],
     "op": "add"},
    {"param": "imu_gyro_bias", "dist": "normal", "range": [0.0, 0.0015], "op": "add"},
    {"param": "imu_gyro_bias", "select": "z", "dist": "normal", "range": [0.0, 0.003], "op": "add"},
    {"param": "imu_misalignment", "dist": "normal", "range": [0.0, 0.01], "op": "add", "scope": "startup"},
]
```

This scales every body's mass and adds up to 4.5 kg of payload on the upper body, shares one latency across an env's drives with extra latency on the four ankle drives, adds a second, wider offset on the ankle cranks, adds a larger gyro bias on the z axis, and keeps one IMU misalignment per env until the level changes.

### Levels

`plant.set_level(level)` redraws everything at the new level, startup and reset draws for every env, and writes it all, solver parameters included. Running envs get new device parameters immediately. Because of the solver write, call it rarely and only when the level changes, for example from a curriculum term that steps through a few levels. This extends the minimal environment above:

```python
from isaaclab.managers import CurriculumTermCfg


def randomization_level(env, env_ids, ramp_steps: int = 100_000):
    p = env.action_manager.get_term("drives").plant
    level = round(min(1.0, env.common_step_counter / ramp_steps), 1)  # eleven distinct levels
    if level != p.randomizer.level:
        p.set_level(level)
    return level


@configclass
class Curriculum:
    randomization = CurriculumTermCfg(func=randomization_level)
```

Add `curriculum = Curriculum()` to the env config and start the plant at `TrainPlantCfg(level=0.0)`. Isaac Lab calls curriculum terms on every reset, which is why the function compares before it writes.

## Signal logging

`plant.attach_log(log)` attaches a `SignalLog` (`src/humanoid_sim/signals/log.py`), and `plant.log(name, value, *, envs=(0,), group="signals")` writes `value` (`[E]` or `[E, ...]`) for the listed envs as `/sim/{env}/{group}/{name}` at the plant time, on the same `time` timeline as the 3D views. With `env` from the minimal environment:

```python
from humanoid_sim.signals.log import SignalLog

p = env.action_manager.get_term("drives").plant
p.attach_log(SignalLog(rrd="outputs/train_debug.rrd"))
p.log("base_height", p.truth().root_pose_w[:, 2], envs=(0, 1), group="base")
```

Every `log` call copies to the host, so keep it to inspection runs. Opening and reading the recordings is in [visualization.md](visualization.md).

### Recording a scene

`SceneRecorder(env, usd, *, rrd=None, connect=None, envs=None)` (`src/humanoid_sim/signals/scene.py`) records the robots of an Isaac Lab scene into one [Rerun](glossary.md#rerun) recording. It logs the ground once (the generated terrain read from the USD stage, or a 40 m square for the flat ground) and each visual mesh of the robot once. Every `frame(t_ns)` call then logs the body poses of the recorded envs as instance poses of those meshes. `envs` lists the env indices to record (default all), `rrd` writes a file and `connect` streams to a running viewer; the recording's application id is `humanoid-sim scene`. `close()` ends the recording.

A frame copies `[envs, bodies, 7]` poses from the GPU, so record at a low rate (every few policy steps) and use it for debugging and showcases. Leave it out of a training loop that has to stay free of host syncs.

A `SignalLog(init=False)` joins the recording the recorder opened, so one robot's signals appear next to the scene. This fragment is the recording part of `examples/train_scene.py` (`env`, `usd`, `steps` and `actions` come from the surrounding script):

```python
from humanoid_sim.signals.log import SignalLog
from humanoid_sim.signals.scene import SceneRecorder

p = env.action_manager.get_term("drives").plant
recorder = SceneRecorder(env, usd, rrd="outputs/train_scene.rrd")
p.attach_log(SignalLog(init=False))
for k in range(steps):
    env.step(actions)
    if k % 2 == 0:
        recorder.frame(p.t_ns)
        p.log("base_height", p.truth().root_pose_w[:, 2], group="base")
recorder.close()
```

Screenshots of such a recording are in [visualization.md](visualization.md#training-scenes).

## Performance

The plant's steady-state work (`submit`, `pre_physics`, `measured`, `truth`, `reset`) is batched torch on the device with no host syncs. `humanoid-sim bench --train` checks this by running its loop under `torch.cuda.set_sync_debug_mode("error")` ([benchmarks.md](benchmarks.md)). Task code should follow the same rules:

- Keep observation, reward and termination terms free of host syncs: no `.item()`, `.cpu()`, `.tolist()` or printing of device tensors, no Python branches on tensor values, no boolean-mask indexing.
- Compute constants once in `__init__`, as `PoseOffsets` does with its targets and gains.
- Write the solver model only at startup and on level changes. The plant writes physics parameters at construction and in `set_level`. Isaac Lab events in reset mode that change masses, materials or joint properties write the same solver model on every reset; randomize those through the plant instead.
- Solver gains stay fixed. Gain changes go in the demand and are realized by target shaping at no solver cost.
- `plant.log`, `SceneRecorder.frame` and `set_level` sync with the host by design: use them in inspection runs and for rare curriculum steps only.

## Pitfalls

- The solver uses the larger friction of the two shapes in contact (MJWarp always, Kamino as the profile configures it). When `friction` is randomized on the robot, give the ground zero friction (`robot_scene_cfg(..., ground_friction=0.0)` or `heightfield_terrain_cfg(..., friction=0.0)`); with the profile's ground friction of 1.0 every robot value below 1 has no effect.
- Take targets and action offsets from `data.default_joint_pos`, not from `standing_offset_rad`: the scene spawns in the home pose, and the standing pose, which a rigid robot cannot hold on flat ground, is only used when a scene asks for it. See [The initial pose](#the-initial-pose).
- The encoder jitter is drawn on every `measured()` call, for `drive` and for `history`. Two observation terms that each call `measured()` see different jitter in the same step; read it once in one term, as `actor_obs` does.
- `history[:, 0]` is one physics step older than `drive`. The history and the IMU are sampled at the start of each physics step, so after an env step their newest entries are from `plant.t_ns - 4 ms`, while `drive` is the state now. Use `history_ns` for the times.
- After a reset, the history and IMU buffers keep the previous episode's values until they are overwritten; only `history_ns`, `sample_ns`, `stamp_ns`, `seq` and `count` mark them empty. The observation computed right after a reset, in the same env step, always has `count == 0`, and the first burst can take more than one policy period to arrive. Mask on `history_ns >= 0` and `stamp_ns >= 0` where it matters.
- `measured().imu` is the delivery model's buffer, updated in place every physics step. Clone it to keep it.
- Joint order differs. `truth()`, `reset(joint_pos=...)` and the plant use the description's joint order; `env.scene["robot"].data.joint_pos` and `data.default_joint_pos` use the articulation's order. `plant.joint_ids` maps one to the other.
- Build the USD before creating the env config; `robot_scene_cfg` fails without the asset.
- Use one `PlantAction` per robot and no other action term that writes the robot's joint targets; the plant overwrites them before every physics step.
- Keep the articulation's actuators implicit, as `robot_scene_cfg` builds them. If explicit actuator configs are added, Isaac Lab's Newton backend can run the whole decimation loop in one physics call, and `apply_actions` (the plant) would run once per policy step instead of once per physics step.
- Drives are off after a reset until a demand arrives; see [Resets](#resets).
- Training keeps each joint's Coulomb friction at its enabled-drive value from the description while a drive is off. The runtime host switches the joints of disabled 1:1 drives to the family's [back-drive](glossary.md#back-drive) torque (`src/humanoid_sim/runtime/host.py`), which is larger: 1.5 Nm on an HEJ 90 joint where the knee's enabled friction is 0.31 Nm. An unpowered robot in training therefore folds more easily than the same robot in the runtime. The ankle joints keep their friction in both.
