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
humanoid-sim replay <exc-dir>             # drive-level replay, the base turning with the capture's IMU yaw
humanoid-sim view-replay <replay-dir> --save replay.rrd
humanoid-sim bench --num-envs 4096 [--train]
humanoid-sim fit-passive <exc-dir>... --holdout <exc-dir> --num-envs 512   # upper-body passive parameters
```

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
#   plant.measured(): drive feedback (and per-step history) and the raw IMU samples received so far
#   plant.truth(): joint state, root pose and twist, contact forces
```
