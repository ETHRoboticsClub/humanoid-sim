# humanoid-sim

The ETHRC humanoid below Controls: drives, ankle mechanism and body, on Isaac Lab 3.0 /
Newton (MJWarp). ROS-free. See `AGENTS.md` for the rules and the layout.

```bash
git submodule update --init --recursive
uv sync --extra dev --extra newton   # contract and kinematics tests: uv run pytest -m "not physics"
docker buildx build --platform linux/arm64 -f docker/Dockerfile -t humanoid-sim:dev .   # Isaac Lab image
```

## Use

Commands run in the Isaac Lab image (`docker run --rm -v "$PWD":/workspace/humanoid-sim -w
/workspace/humanoid-sim humanoid-sim:dev <command>`) or on a GPU node; `sky/` has SkyPilot tasks for the GPU
runs (`bench`, `replay`, `sysid`).

```bash
humanoid-sim build-asset                  # robot USD for the profile, under outputs/assets/
humanoid-sim build-asset --base-joints    # the variant replays use: yaw/pitch/roll joints above the base
humanoid-sim replay <exc-dir> [--demand-delay-ms -7]   # drive-level replay, the base turning with the IMU yaw
humanoid-sim view-replay <replay-dir> --save replay.rrd
humanoid-sim bench --num-envs 4096 [--train]   # or --runtime: the real-time host's step
humanoid-sim fit-delay <exc-dir>...       # when each recorded demand acts, relative to its stamp
humanoid-sim fit-passive <exc-dir>... --holdout <exc-dir> --num-envs 512   # upper-body passive parameters
humanoid-sim serve --segment /run/hsim/plant --support suspended   # one robot in real time, for Controls
```

`serve` publishes the runtime segment, a shared-memory file a client (Controls' plant host) attaches to: drive
demands in, physical drive feedback and raw IMU samples out, a mailbox for reset, pause, supports, disturbances,
faults and prescribed poses, and a truth region for tests. A session ends when the client detaches or goes silent;
with `--persist` the robot is reset and the next client gets a new session without restarting. The layout and
protocol are in `src/humanoid_sim/runtime/abi/plant_abi.py` and `plant_abi.h`, which a client vendors unchanged.
On CUDA the host captures and compiles the plant's step at startup (about a minute) and then runs about 2.6x real
time on an A10G (`humanoid-sim bench --runtime`); on Warp's CPU device it runs below real time.

Train-Time API, from a manager-based Isaac Lab environment config:

```python
from humanoid_sim.train import TrainPlantCfg
from humanoid_sim.train.isaaclab import PlantAction, PlantActionCfg
from humanoid_sim.train.scene import robot_scene_cfg, robot_sim_cfg
from humanoid_sim.world.terrain import rough_terrain_cfg

class Drives(PlantAction):
    def demands(self, actions):  # → DriveDemand [E, 22] and the time [E] (ns) it leaves the master
        ...

cfg.scene = robot_scene_cfg(4096, usd, terrain=rough_terrain_cfg(friction=0.0))
cfg.sim = robot_sim_cfg(device="cuda:0")
cfg.decimation = 5  # 20 ms policy period, 4 ms physics
cfg.actions.drives = PlantActionCfg(class_type=Drives, action_dim=22, plant=TrainPlantCfg(randomization=[
    {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
    {"param": "friction", "dist": "uniform", "range": [0.3, 1.0], "op": "set"},
]))
# in terms: plant = env.action_manager.get_term("drives").plant
#   plant.measured(): drive feedback as the drives report it (encoder counts, mNm), its per-step history, and
#     the raw IMU samples received so far
#   plant.truth(): joint state, root pose and twist, contact forces
#   plant.attach_log(SignalLog(...)); plant.log("name", tensor): named signals into the Rerun log
```
