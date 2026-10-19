# Benchmarks

`humanoid-sim bench` measures how fast the simulation runs. It has three workloads:

| Command | Workload | Main figure |
|---|---|---|
| `humanoid-sim bench --num-envs N` | The plant and MJWarp physics for N robots standing on flat ground | `sim_seconds_per_wall_second` |
| `humanoid-sim bench --train --num-envs N` | The Train-Time API scene: plant, IMU, contact sensors, randomization and resets | `sim_seconds_per_wall_second` |
| `humanoid-sim bench --runtime` | The real-time host's step behind the runtime segment, unpaced | `ms_per_step`, `real_time_factor` |

The benchmarks need Isaac Lab, so they run in the Docker image or on a GPU node, and they load the plain robot asset (`humanoid-sim build-asset`; see [getting-started.md](getting-started.md#build-the-robot-asset)). They also run on `--device cpu`, which shows that they work but gives numbers far below the GPU's.

## Options

| Option | Default | Applies to | Meaning |
|---|---|---|---|
| `--num-envs` | 1 | default, `--train` | Number of robots simulated in parallel |
| `--steps` | 1000 | all | Physics steps timed after the warm-up; each step is 4 ms of simulated time |
| `--device` | `cuda:0` | all | `cuda:N`, or `cpu` for Warp's CPU device |
| `--integrator` | the profile's (`implicitfast`) | default | MJWarp integrator |
| `--no-self-collision` | off | default, `--train` | Turn off contacts between the robot's own bodies |
| `--train` | off | | Run the training workload |
| `--runtime` | off | | Run the runtime workload; it takes precedence over `--train` |

Options the workload does not use are ignored without a warning. The global options `--robot`, `--assets`, `--usd` and `--solver` go before `bench`, for example `humanoid-sim --usd path/to/robot.usda bench --num-envs 4096`.

## Output

The command prints Isaac Lab's and MJWarp's log lines, then one indented JSON object. To keep only the JSON, do what the SkyPilot tasks do:

```bash
humanoid-sim bench --num-envs 4096 --device cuda:0 --steps 500 > bench.log 2>&1
sed -n '/^{/,/^}/p' bench.log
grep -c overflow bench.log
```

MJWarp prints a warning containing "overflow" when a step needs more contacts or constraint rows than the profile's `nconmax` and `njmax` allow. The bench task counts these warnings for each run; a run that has any does not measure the normal workload and should not be compared.

The default and training workloads time their steps with torch's sync debug mode set to `error` on CUDA, so any host synchronization in the steady-state loop stops the run. The benchmark then prints `{"error": "host sync in steady state", "detail": "..."}` instead of the figures, with torch's error message in `detail` (and, for the training workload, the call stack that led to it).

## Physics benchmark (default)

The default workload builds the simulation core with N robots, each with a free base, spawned upright on flat ground in the home pose ([robot-model.md](robot-model.md#standing-on-the-ground)), and every drive holds that pose at its baseline gains, so the robots stand. Every 20 ms (the policy period) each robot gets a new JVPT position demand ([drives.md](drives.md)) carrying a normally distributed offset with a standard deviation of 0.02 rad. The first 50 steps compile the kernels and are not timed; the next `--steps` steps are. The timing covers the whole step: the plant's drive and transmission math and the MJWarp solve.

Isaac Lab captures its solver step as a CUDA graph (a recorded sequence of GPU kernels replayed with one launch); the plant's own work runs as ordinary kernel launches here, unlike in the runtime host, which captures it too.

| Field | Meaning |
|---|---|
| `physics` | `isaaclab/<solver>`: `isaaclab/mjwarp` or `isaaclab/kamino` |
| `device` | The device the run used |
| `num_envs` | Number of robots |
| `period_ns` | Physics period: 4,000,000 ns |
| `steps` | Number of timed steps |
| `wall_s` | Wall time of the timed steps, in seconds |
| `rtf_per_env` | Simulated seconds per wall second for one robot: each robot's real-time factor |
| `env_steps_per_s` | Robot-steps per wall second: `num_envs` x `steps` / `wall_s` |
| `sim_seconds_per_wall_second` | Simulated seconds of all robots together per wall second: `num_envs` x `rtf_per_env` |
| `fraction_envs_near_pose` | Share of robots whose every joint ended within 0.5 rad of where it started. It checks the joints only. |
| `fraction_envs_up` | Share of robots whose base is above 0.7 m at the end. A fallen robot has more contacts and steps at a different cost, so compare runs only when this is 1.0. |
| `self_collision` | `false` with `--no-self-collision`, otherwise `true` |
| `host_syncs` | `none (torch sync debug mode)` on CUDA, `n/a` on the CPU |

## Training benchmark (`--train`)

The training workload builds the scene a training environment uses ([training.md](training.md)): the robot, its IMU, a contact sensor on every body and flat ground. The ground's friction is 0 so the randomized robot friction decides the contact friction. The plant runs through `RobotPlant` with a randomization like a locomotion policy's. Every draw is uniform except the gyro bias, which is normal with mean 0.

| Parameter | Draw |
|---|---|
| Body masses | Scaled by 0.9 to 1.1 |
| Payload on the body `Component171_1` | 0 to 4.5 kg added |
| Joint armature | Scaled by 0.8 to 1.2 |
| Friction coefficient of every robot shape | Set between 0.3 and 1.0 |
| Drive stiffness and damping | Scaled by 0.9 to 1.1 |
| Torque limits | Scaled by 0.85 to 1.0 |
| Latency from demand sent to demand in force | 10 to 20 ms, plus another 10 to 20 ms on the four ankle drives |
| Encoder offsets | Between -0.026 and 0.026 rad on every drive, plus between -0.14 and 0.14 rad on the drives matching `maxonHEJ50L_ANK.*` (the left ankle cranks) |
| IMU gyro bias | Standard deviation 0.0015 rad/s |

The robots start in the home pose (the scene's default) and hold it. Every 20 ms the benchmark reads `measured()` and `truth()`, resets about one in fifty robots as if their episodes had ended (`RobotPlant.reset` back to the default state: joints, base pose and velocity, drives off, sensors emptied, device parameters drawn again), and submits a demand for the home pose for every robot. Every 4 ms it runs the plant, writes the scene to the simulation, steps physics and updates the scene. As in the default workload, 50 warm-up steps are not timed.

| Field | Meaning |
|---|---|
| `workload` | `train` |
| `device`, `num_envs`, `steps`, `wall_s` | As in the default workload |
| `env_steps_per_s` | Robot physics steps per wall second |
| `sim_seconds_per_wall_second` | Simulated seconds of all robots together per wall second |
| `self_collision` | `false` with `--no-self-collision`, otherwise `true` |
| `host_syncs` | `none (torch sync debug mode)` on CUDA, `n/a` on the CPU |
| `checksum_finite` | `true` when every value read from `measured()` and `truth()` stayed finite (no NaN or infinity) |
| `fraction_envs_up` | Share of robots whose base is above 0.7 m at the end |

## Runtime benchmark (`--runtime`)

The runtime workload starts the real-time host ([runtime.md](runtime.md)) with its segment in a temporary directory and a free robot on the ground (`support="free"`), reset to the home pose. A client in the same process writes a new demand record before every step, holding that pose at the baseline gains. A new record every step is what the host sees when the client's bus cycle is shorter than the 4 ms physics period, as Controls' 1.5 ms cycle is. Each timed step covers the client's write and the host's whole step: mailbox commands, demand intake, the plant and physics, publishing feedback, and publishing the truth every fifth step.

The robot stands still (in a 60 s run its base stayed at 0.923 m), and `base_height_end_m` in the output confirms that a standing robot was timed.

The steps are not paced: they run back to back as fast as the host can go, so the real-time factor is the headroom the host has when it runs paced. On CUDA the host captures its step as CUDA graphs and compiles it at startup, which takes about a minute and is not timed. After 50 warm-up steps, `--steps` steps are timed; the `e2e` task uses 2000. The workload always simulates one robot, so `--num-envs`, `--integrator` and `--no-self-collision` have no effect.

| Field | Meaning |
|---|---|
| `workload` | `runtime host, a new demand every step` |
| `device`, `steps` | As in the default workload |
| `ms_per_step` | Mean wall time of one step, in milliseconds |
| `ms_p99` | 99th percentile of the per-step wall time: one step in a hundred took longer |
| `ms_max` | Longest step |
| `steps_over_period` | Number of steps that took longer than the 4 ms physics period |
| `real_time_factor` | Physics period divided by the mean step time; above 1.0 the host keeps up with real time |
| `cuda_graph` | `true` when the host's step runs as CUDA graphs (on CUDA) |

When paced, the host sleeps until 0.1 ms before each deadline and then spins, and it waits for the GPU on a blocking event, so it does not hold a whole CPU core while it waits; Controls shares the machine. The benchmark does not measure CPU use.

## Reference numbers

These are measurements to compare your own runs against. They depend on the GPU, the driver and the node's CPU, and they move when the description or the profile changes (contact buffers, self-collision sets, solver iterations). Compare runs on the same GPU type.

The latest `e2e` run, on an NVIDIA L4 on 2026-10-19 (`sky/e2e.sky.yaml` takes an A10G or an L4, whichever has capacity), wrote these (`runtime.json` and `train_4096.json` in `outputs/e2e/` on the node), both from the home pose on MJWarp:

| Runtime host, one robot, a new demand every step, 2000 steps | Value |
|---|---|
| `ms_per_step` | 1.492 ms |
| `ms_p99` | 2.476 ms |
| `ms_max` | 3.740 ms |
| `steps_over_period` | 0 of 2000 |
| `real_time_factor` | 2.68 |
| `cuda_graph` | `true` |
| `base_height_end_m` | 0.923 m (still standing) |

| Training scene, 4096 robots, 500 steps | Value |
|---|---|
| `sim_seconds_per_wall_second` | 1,216 |
| `env_steps_per_s` | 303,998 |
| `wall_s` | 6.74 s |
| `host_syncs` | `none (torch sync debug mode)` |
| `checksum_finite` | `true` |
| `fraction_envs_up` | 0.95 |

What the runtime figures mean for a client:

- A real-time factor of 2.68 means the host can step about 2.7 times as fast as the wall clock. Paced to real time, it spends about 1.5 ms of each 4 ms period stepping and the rest waiting for the next deadline.
- The p99 of 2.48 ms and the longest step, 3.74 ms, are inside the 4 ms period, so no step ran late. A paced step that does end a little late does not count in `late_steps`: the next steps run back to back until the host is on schedule again, and only a step that starts more than a full period late is counted ([runtime.md](runtime.md#the-clock)). These figures leave a paced host on an L4 with margin.

What the training figures mean for a training run:

- 1,216 simulated seconds per wall second, over 4096 robots, is about 1,200 hours of robot experience per hour of wall time. Each robot advances at about 0.3 times real time (1,216 / 4096).
- 303,998 physics steps of 4 ms per second across all robots is about 60,000 policy steps per second at a 20 ms policy period.
- No host syncs: the timed loop never made the CPU wait for the GPU (torch's sync debug mode was set to `error`), so Python round trips do not limit the rate.
- 5% of the robots were down at the end. The randomization includes encoder offsets of up to 0.14 rad on the left ankle cranks, so some robots hold their feet at the wrong angle and tip over; a fallen robot is reset only when its turn in the 1-in-50 resets comes.

The same run replayed the three captures at the replay defaults, one robot each: 30.7 s of capture in 59.3 s, 45.7 s in 94.1 s and 45.8 s in 85.3 s, about 0.5 times real time. One robot leaves most of the GPU idle, which is why the `replay` task runs one process per capture in parallel.

Earlier measurements with the physics workload, self-collision on, on an L40S on 2026-10-15. The robots then held the description's standing pose and fell over in the first second, so these time mostly fallen robots and are not directly comparable with a run from the home pose:

| Robots | `sim_seconds_per_wall_second` | Command |
|---|---|---|
| 4096 | 1,826 | `humanoid-sim bench --num-envs 4096 --device cuda:0 --steps 500` |
| 16384 | 5,770 | `humanoid-sim bench --num-envs 16384 --device cuda:0 --steps 500` |

The paced host's CPU use was checked once by hand with `top` on an A10G node, at about 41% of one core. No recorded output backs that figure; measure it yourself as described below before relying on it.

The SkyPilot tasks run these commands ([testing-and-gpu.md](testing-and-gpu.md#tasks)). The `bench` task runs the physics benchmark at 4096 and 16384 robots on an L40S, among other counts. The `e2e` task runs the training benchmark at 4096 robots and the runtime benchmark on an A10G or an L4, whichever has capacity. To get an A10G for comparison with the tables, pass `--gpus A10G:1`:

```bash
sky launch -y -d -c hsim-e2e -i 10 --gpus A10G:1 sky/e2e.sky.yaml
sky logs hsim-e2e
rsync -avz hsim-e2e:sky_workdir/outputs/e2e/ outputs/e2e-gpu/
```

The results are then in `outputs/e2e-gpu/runtime.json` and `outputs/e2e-gpu/train_4096.json`, and in `outputs/bench-gpu/` for the bench task. The setup log, and Warp's start-up lines in each benchmark log, name the GPU the node got.

To run a single benchmark on a running cluster, log in with `ssh <cluster>` and run it from `~/sky_workdir` after setting the path as the tasks do ([testing-and-gpu.md](testing-and-gpu.md#node-setup-skysetupsh)).

To check the paced host's CPU use, serve a client ([runtime.md](runtime.md)) on the GPU node and watch the `humanoid-sim serve` process with `top` (the `%CPU` column) or `pidstat -p <pid> 1`. 100% is one full core.
