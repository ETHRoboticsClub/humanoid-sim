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
| `--steps` | 1000 | all | Physics steps timed after the warm-up; each step is the profile's physics period of simulated time (1 ms for the native V1, 4 ms for the G1) |
| `--device` | `cuda:0` | all | `cuda:N`, or `cpu` for Warp's CPU device |
| `--integrator` | the profile's | default, `--train` | Integrator of the selected solver |
| `--period-ms` | the profile's | `--train` | Physics period; it must divide the 20 ms policy period. Compare runs at the same simulated time (`--steps` x period) |
| `--no-self-collision` | off | default, `--train` | Turn off contacts between the robot's own bodies |
| `--train` | off | | Run the training workload |
| `--runtime` | off | | Run the runtime workload; it takes precedence over `--train` |
| `--scene PATH`, `--spawn X Y` | none, `0 0` | all | Stand the robots among the static boxes in `PATH`, every robot at the spawn point (see [world.md](world.md#static-boxes)); the JSON gets a `scene` entry |

| `--objects PATH` | none | all | Add the moving and free objects in `PATH` (JSON, see [world.md](world.md#moving-and-free-objects)), in every env; the JSON gets an `objects` entry |

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

The default workload builds the simulation core with N robots, each with a free base, spawned upright on flat ground in the home pose ([robot-model.md](robot-model.md#standing-on-the-ground)), and every drive holds that pose at its baseline gains, so the robots stand. Every 20 ms (the policy period) each robot gets a new PD position demand ([drives.md](drives.md)) carrying a normally distributed offset with a standard deviation of 0.02 rad. The first 50 steps compile the kernels and are not timed; the next `--steps` steps are. The timing covers the whole step: the plant's drive and transmission math and the MJWarp solve.

Isaac Lab captures its solver step as a CUDA graph (a recorded sequence of GPU kernels replayed with one launch); the plant's own work runs as ordinary kernel launches here, unlike in the runtime host, which captures it too.

| Field | Meaning |
|---|---|
| `physics` | `isaaclab/<solver>`: `isaaclab/mjwarp` or `isaaclab/kamino` |
| `device` | The device the run used |
| `num_envs` | Number of robots |
| `period_ns` | Physics period: 1,000,000 ns for the native V1, 4,000,000 ns for the G1 |
| `steps` | Number of timed steps |
| `wall_s` | Wall time of the timed steps, in seconds |
| `rtf_per_env` | Simulated seconds per wall second for one robot: each robot's real-time factor |
| `env_steps_per_s` | Robot-steps per wall second: `num_envs` x `steps` / `wall_s` |
| `sim_seconds_per_wall_second` | Simulated seconds of all robots together per wall second: `num_envs` x `rtf_per_env` |
| `fraction_envs_near_pose` | Share of robots whose every joint ended within 0.5 rad of where it started. It checks the joints only. |
| `fraction_envs_up` | Share of robots whose base is above 3/4 of its standing height at the end. A fallen robot has more contacts and steps at a different cost, so compare runs only when this is 1.0. |
| `robot` | The robot benchmarked (`--robot`) |
| `self_collision` | `false` with `--no-self-collision`, otherwise the profile's `physics.self_collision` |
| `host_syncs` | `none (torch sync debug mode)` on CUDA, `n/a` on the CPU |

## Training benchmark (`--train`)

The training workload builds the scene a training environment uses ([training.md](training.md)): the robot, its IMU, a contact sensor on every body and flat ground. The ground's friction is 0 so the randomized robot friction decides the contact friction. The plant runs through `RobotPlant` with a randomization like a locomotion policy's. Every draw is uniform except the gyro bias, which is normal with mean 0.

| Parameter | Draw |
|---|---|
| Body masses | Scaled by 0.9 to 1.1 |
| Payload on the heaviest body (the V1's `Component171_1`, the G1's `torso_link`) | 0 to 4.5 kg added |
| Joint armature | Scaled by 0.8 to 1.2 |
| Friction coefficient of every robot shape | Set between 0.3 and 1.0 |
| Drive stiffness and damping | Scaled by 0.9 to 1.1 |
| Torque limits | Scaled by 0.85 to 1.0 |
| Latency before the drives, from demand sent to the controller staging it | 10 to 20 ms, plus another 10 to 20 ms on a robot's ankle crank drives (the V1's four); each drive's own delay (5 ms on the V1) follows |
| Encoder offsets | Between -0.026 and 0.026 rad on every drive, plus between -0.14 and 0.14 rad on the left ankle's crank drives (the V1's) |
| IMU gyro bias | Standard deviation 0.0015 rad/s |

The plant runs compiled and captured as CUDA graphs on CUDA, as training runs it. The robots start in the home pose (the scene's default) and hold it. The V1 stands; the G1 tips over within about 1.3 s of every reset ([training.md](training.md#the-initial-pose)), so its robots spend much of the run on the ground and `fraction_envs_up` stays low. Every 20 ms the benchmark reads `measured()` and `truth()`, resets about one in fifty robots as if their episodes had ended (`RobotPlant.reset` back to the default state: joints, base pose and velocity, drives off, sensors emptied, device parameters drawn again), and submits a demand for the home pose for every robot. Every physics period it runs the plant, writes the scene to the simulation, steps physics and updates the scene. As in the default workload, 50 warm-up steps are not timed.

| Field | Meaning |
|---|---|
| `workload` | `train` |
| `robot`, `device`, `num_envs`, `steps`, `wall_s` | As in the default workload |
| `env_steps_per_s` | Robot physics steps per wall second |
| `sim_seconds_per_wall_second` | Simulated seconds of all robots together per wall second |
| `self_collision` | `false` with `--no-self-collision`, otherwise the profile's `physics.self_collision` |
| `host_syncs` | `none (torch sync debug mode)` on CUDA, `n/a` on the CPU |
| `checksum_finite` | `true` when every value read from `measured()` and `truth()` stayed finite (no NaN or infinity) |
| `fraction_envs_up` | Share of robots whose base is above 3/4 of its standing height at the end |
| `solver`, `period_ns`, `integrator` | The physics settings the run used |
| `peak_body_ang_vel_rad_s` | Fastest angular speed of any body over the timed steps; a light body spinning up (or `NaN`) shows a setting is unstable |

## Runtime benchmark (`--runtime`)

The runtime workload starts the real-time host ([runtime.md](runtime.md)) with its segment in a temporary directory and a free robot on the ground (`support="free"`), reset to the home pose. A client in the same process writes a new demand record before every step, holding that pose at the baseline gains. A new record every step is what the host sees when the client's bus cycle is shorter than the physics period, as Controls' 1.5 ms cycle is for a 4 ms period (at the native V1's 1 ms period a record arrives about every second step). Each timed step covers the client's write and the host's whole step: mailbox commands, demand intake, the plant and physics, publishing feedback, and publishing the truth every fifth step.

The robot stands still (in a 60 s run its base stayed at 0.923 m), and `base_height_end_m` in the output confirms that a standing robot was timed.

The steps are not paced: they run back to back as fast as the host can go, so the real-time factor is the headroom the host has when it runs paced. On CUDA the host captures its step as CUDA graphs and compiles it at startup, which takes about a minute and is not timed. After 50 warm-up steps, `--steps` steps are timed; the `e2e` task uses 2000. The workload always simulates one robot, so `--num-envs`, `--integrator` and `--no-self-collision` have no effect.

| Field | Meaning |
|---|---|
| `workload` | `runtime host, a new demand every step` |
| `robot`, `device`, `steps` | As in the default workload |
| `ms_per_step` | Mean wall time of one step, in milliseconds |
| `ms_p99` | 99th percentile of the per-step wall time: one step in a hundred took longer |
| `ms_max` | Longest step |
| `steps_over_period` | Number of steps that took longer than the physics period |
| `real_time_factor` | Physics period divided by the mean step time; above 1.0 the host keeps up with real time |
| `cuda_graph` | `true` when the host's step runs as CUDA graphs (on CUDA) |

When paced, the host sleeps until 0.1 ms before each deadline and then spins, and it waits for the GPU on a blocking event, so it does not hold a whole CPU core while it waits; Controls shares the machine. The benchmark does not measure CPU use.

## Reference numbers

These are measurements to compare your own runs against. They depend on the GPU, the driver and the node's CPU, and they move when the description or the profile changes (contact buffers, self-collision sets, solver iterations). Compare runs on the same GPU type.

### Native V1 at 1 ms, 2026-11-17

`humanoid-sim serve` on the native V1 (MJWarp, `rk4`, the profile's defaults) does not keep up at the 1 ms physics period: one step costs 2.3 to 2.4 ms on an L40S, so the plant runs at about 0.4 times real time and every paced step is late. The figures below are from an NVIDIA L40S (AWS g6e.2xlarge, driver 580, torch 2.11 with CUDA 13) with Isaac Lab at the pinned commit, `master` at f9245f4 against the change that added `tests/test_runtime_keeping_up.py`.

| Native V1, 1 ms, one robot, 2000 steps | master | this change |
|---|---|---|
| `serve`'s step, suspended, a demand every bus cycle (1.5 ms): mean (`ms_per_step`) | 2.81 ms | 2.28 ms |
| the same, p99 | 3.01 ms | 2.62 ms |
| the same, real-time factor | 0.36 | 0.44 |
| `bench --runtime` (free robot, a demand every step), two runs: `ms_per_step` | 2.55, 2.50 ms | 2.14, 2.37 ms |
| `bench --runtime`: `real_time_factor` | 0.39, 0.40 | 0.47, 0.42 |
| paced `serve` seen by a client (25 s): plant time over wall time | 0.35 (the #142 probe) | 0.41 |

Where one `serve` step goes (suspended; wall time on the host thread, in ms; the GPU runs asynchronously, so the wait for it lands in the publish row):

| Part of the step | master | this change |
|---|---|---|
| intake of the client's demands (`_take_demands`) | 0.12 | 0.12 |
| plant graph before the solver (`graphs.pre`) | 0.14 | 0.15 |
| eager joint-target assembly and write into the solver's buffers | about 0.50 | inside that graph |
| gantry rope forces (`gantry.apply`) | 0.28 | 0.22 |
| Isaac Lab `write_data_to_sim`, solver launch, `scene.update` | 0.51 | 0.52 |
| plant graph after the solver (`graphs.post`, with the joint-state gather) | 0.03 | 0.01 |
| feedback packing (`_feedback_flat`) | 0.16 | 0.04 |
| waiting for the GPU (one blocking event), copies, writes into the segment | 0.84 | about 1.1 including the GPU's remaining work |
| truth, every fifth step (0.57 ms each) | 0.11 | 0.11 |
| total | 2.81 | 2.28 |

What limits it. The solver's own CUDA graph (Isaac Lab's captured `NewtonManager` step: Newton's collision pipeline, MJWarp's `rk4` with its four forward-dynamics evaluations, constraints and the solve) takes 1.35 ms of GPU time by itself, measured replaying it back to back. It is about 850 kernels of 1 to 20 microseconds each (the tree-depth loops of `crb`, `cfrc` and `subtree_com`, the loop-closure `equality_connect`, one 130 us narrow-phase kernel), so it is latency-bound, not bandwidth-bound. Alone it caps the host at 0.74 times real time. The host's own work (the plant and drive bank, the Isaac scene's per-step Python and kernel launches, the feedback and the truth) is about 0.9 ms of launch time on the host thread that cannot start the solver graph before it has finished the launches ahead of it, and the host then waits for the whole GPU chain before it publishes.

Solver settings that would change the step cost, measured as the solver graph alone (a free robot that tips over, 2000 steps) and, for `implicitfast`, in the whole host (changing any of them changes the physics, so none is applied):

| MJWarp setting | solver graph | whole `serve` step (this change) | Accuracy cost |
|---|---|---|---|
| `integrator: rk4` (the profile's) | 1.35 ms | 2.28 ms (0.44x) | reference |
| `integrator: implicitfast` | 0.52 ms | 1.60 ms (0.62x) | first-order: at 4 ms on the V1 the angular momentum drift of a free-falling robot is 5.4 % against 0.017 % for `rk4` (`profiles/defaults.yaml`; not re-measured at 1 ms); it also treats the explicit actuator and compliance springs differently |
| `integrator: euler` | 0.51 ms | not run | as `implicitfast`, and explicit in the damping |
| `iterations` 100 to 30, or 10 with `ls_iterations` 10 | 1.35 ms, 1.36 ms | not run | none seen (the cost did not move, the trajectories were not compared) |
| `ls_iterations` 50 to 20 | 1.35 ms | not run | none seen at this load |

Even with `implicitfast` the host sits at 1.60 ms, because its own launch chain, not the GPU, is then the limit (on master it is 2.10 ms). Reaching 1.0 times real time needs both a cheaper solver step and a much shorter host chain. What would shorten the host chain, none of it measured here: capturing Isaac Lab's `write_data_to_sim` and `scene.update` kernels in the step graph (they are launched eagerly from Python today, about 0.4 ms), batching the gantry's per-point wrench calls into one, and publishing a step's feedback after the next step's launches (the copy is already queued; this adds up to one launch chain of latency to the published feedback, so it would be used only while the host is behind). The truth and the IMU's data cannot simply be captured: Isaac Lab derives their views lazily on read, and a captured read returned stale values (a truth graph and a captured gantry rope read both changed results and were dropped).

The run that produced these numbers checked the physics against `master` with the same demands and the same schedule over 1500 steps (`humanoid-sim` host, suspended): the drive feedback and the truth differ from `master` by no more than two `master` runs differ from each other (truth at most 1e-6, the drive columns' mean difference 4.7e-5 for both; one nondeterministic demanded-torque spike of 0.96 N m appears in both comparisons).

`tests/test_runtime_keeping_up.py` runs the host unpaced on an L40S and fails when the step's mean or p99 cost goes over its budget (2.6 and 3.2 ms); a second test asserts the step is under the period and is expected to fail (`xfail`, strict) until the solver and the host chain shrink. The host reports the step cost and `overrun_steps` (steps that cost more than the period) in its status line, and `late_steps` and `keeping_up` in every feedback record ([runtime.md](runtime.md#the-clock)).

The 4 ms figures below were measured on URDF_6 before the native V1 moved to 1 ms and are kept for the G1. The latest `e2e` run, on an NVIDIA L40S on 2026-10-20 (`sky launch --gpus L40S:1` with `sky/e2e.sky.yaml`), wrote these (`runtime*.json` and `train_4096*.json` in `outputs/e2e/` on the node), from the home pose on MJWarp, for the V1 and the G1 Controls runs (`g1_29dof_with_hand_rev_1_0`):

| Runtime host, one robot, a new demand every step, 2000 steps | V1 | G1 |
|---|---|---|
| `ms_per_step` | 1.350 ms | 1.355 ms |
| `ms_p99` | 2.348 ms | 2.519 ms |
| `ms_max` | 2.842 ms | 2.719 ms |
| `steps_over_period` | 0 of 2000 | 0 of 2000 |
| `real_time_factor` | 2.96 | 2.95 |
| `cuda_graph` | `true` | `true` |
| `base_height_end_m` | 0.923 m (still standing) | 0.116 m (fallen: the G1 cannot hold its home pose at its deploy gains) |

| Training scene, 4096 robots, 500 steps | V1 | G1 |
|---|---|---|
| `sim_seconds_per_wall_second` | 1,260 | 1,691 |
| `env_steps_per_s` | 315,005 | 422,708 |
| `wall_s` | 6.50 s | 4.84 s |
| `self_collision` | `true` | `false` (its profile at the time) |
| `host_syncs` | `none (torch sync debug mode)` | `none (torch sync debug mode)` |
| `checksum_finite` | `true` | `true` |
| `fraction_envs_up` | 0.95 | 0.63 |

What the runtime figures mean for a client:

- A real-time factor of about 2.95 means the host can step about three times as fast as the wall clock, for either robot. Paced to real time, at the 4 ms period it spent about 1.4 ms of each period stepping and the rest waiting for the next deadline.
- The p99 of 2.3 to 2.5 ms and the longest steps, under 2.9 ms, are inside the 4 ms period of that run, so no step ran late. A paced step that does end a little late does not count in `late_steps`: the next steps run back to back until the host is on schedule again, and only a step that starts more than a full period late is counted ([runtime.md](runtime.md#the-clock)). These figures leave a paced host on an L40S with margin.

What the training figures mean for a training run:

- 1,260 simulated seconds per wall second for the V1, over 4096 robots, is about 1,260 hours of robot experience per hour of wall time; each robot advances at about 0.3 times real time. The G1's 1,691 is a third more: no self-collision (its profile at the time; it now collides arms with legs), fewer contacts (four small spheres per foot) and a lighter model, against 7 more joints.
- 315,005 physics steps of 4 ms per second (URDF_6 run) across all V1 robots is about 63,000 policy steps per second at a 20 ms policy period; for the G1, about 85,000.
- No host syncs: the timed loop never made the CPU wait for the GPU (torch's sync debug mode was set to `error`), so Python round trips do not limit the rate.
- 5% of the V1 robots were down at the end. The randomization includes encoder offsets of up to 0.14 rad on the left ankle cranks, so some robots hold their feet at the wrong angle and tip over; a fallen robot is reset only when its turn in the 1-in-50 resets comes. 37% of the G1s were down: they tip over within about 1.3 s of a reset ([training.md](training.md#the-initial-pose)), so the G1 figure times a mix of standing and fallen robots.

The same run replayed the three captures at the replay defaults, one robot each: 30.7 s of capture in 54.2 s, 45.7 s in 86.5 s and 45.8 s in 78.7 s, about 0.55 times real time. One robot leaves most of the GPU idle, which is why the `replay` task runs one process per capture in parallel.

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
