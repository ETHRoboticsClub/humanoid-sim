# Testing and GPU runs

The test suite has three tiers. Contract tests run natively on any machine. Physics tests step Isaac Lab and
MJWarp, and run in the Docker image on Warp's CPU device. A few tests need CUDA and run only on a GPU, which in
this repository means a SkyPilot node. This page describes the suite, the CI jobs, and how to run work on a GPU.
[getting-started.md](getting-started.md) covers installing uv, building the image and the robot asset.

## Test layout

Every test lives in `tests/`. The last column lists the markers in each file; files without one run natively.

| File | What it checks | Markers |
|---|---|---|
| `test_ankle_transmission.py` | The torch ankle transmission (cranks, Jacobian, inverse) against `ankle_kinematics` at Controls' pin and against humanoid-urdf's ankle model | |
| `test_arrival.py` | The arrival queue: a demand acts from its arrival time, time-weighted inside a step, in order, with overflow counted | |
| `test_base_motion.py` | Base motion from an IMU trace: Euler angles and rates, burst sample times | |
| `test_delay.py` | Demand-delay scan: drive regions, and a replay with each env's demands shifted by its own delay | 1 test `physics` and `data` |
| `test_devices.py` | The drive bank: the JVPT law against Controls' plant, envelope clipping, freeze (the held position latched when the freeze demand arrives), refused I gain, target shaping, arrival | |
| `test_envelope.py` | Torque-speed envelope interpolation in all four quadrants | |
| `test_exc_reader.py` | The capture reader: sampling helpers, captures that match the description, the bursty base-frame IMU stream | 2 tests `data` |
| `test_imu_delivery.py` | IMU sampling on the sensor clock, burst delivery, bias, misalignment and noise, per-env reset | |
| `test_mirror.py` | The left/right mirror map against URDF forward kinematics | |
| `test_model_build.py` | The prepared URDF: inventory, welds, limits, sole boxes, digest, body tree, collision meshes, self-collision sets, base joints | |
| `test_physics.py` | The built asset: mass and bodies and the flat-foot pose (both soles level and at one height) with host kinematics; on MJWarp, holding a standing pose, base joints following a prescribed yaw, raw IMU samples, CUDA-graph step against eager, a joint pushed into its stop staying there | 5 tests `physics`, 2 unmarked (see below) |
| `test_plant_log.py` | `RobotPlant.log` writing signals at plant time | |
| `test_randomization.py` | Randomization specs: parsing, selection (bodies, drives, an IMU axis), composition, scopes, levels, reproducibility | |
| `test_reporting.py` | Drive reporting in encoder counts, velocity and torque, against values Controls' fake drive tests pin | |
| `test_runtime_abi.py` | The runtime segment: C header against the Python layout (needs a C compiler), identity checks, torn-record rejection, mailbox and demand records, concurrent reads | |
| `test_runtime_host.py` | `humanoid-sim serve` end to end with the test as client: round trip, prescribed pose, detach and next session, demand intake, joint disturbance, back-driven disabled drive | `physics` |
| `test_runtime_view.py` | `view-runtime` recording frames from a segment that a stand-in host writes through the ABI (no physics; needs Newton, Rerun and the built asset) | |
| `test_supports.py` | Gantry programs, and rope tension and cable length on the robot | 2 tests `physics` |
| `test_sysid.py` | The passive-fit search space and CMA-ES | |
| `test_train_env.py` | The Train-Time API in a manager-based Isaac Lab environment with two robots | `physics` |
| `test_world.py` | Rough terrain under the feet, force pushes, and a robot holding the flat-foot pose standing on flat ground | `physics` |

A test skips when something it needs is missing: Isaac Lab (`isaaclab_newton`), Newton, the Rerun SDK, scipy, the
built robot asset, the captures, CUDA or a C compiler. The skip reason names it, and pytest prints every reason at
the end of the run (`-ra` is set in `pyproject.toml`). A physics run without the built assets therefore passes with
most tests skipped; read the skip summary after a physics run.

Three tests need the built plain asset but carry no marker, because they use host kinematics (Newton, with the
`newton` extra) and step no physics: `test_kinematic_model_mass_and_bodies` and
`test_flat_foot_pose_levels_both_soles_at_one_height` in `test_physics.py`, and
`test_view_runtime_records_frames_from_a_segment` in `test_runtime_view.py`, which also needs the Rerun SDK (the
`viz` extra). They run natively once the asset exists in `outputs/assets/`, and skip without it. None of them runs
in CI: the `cpu-contracts` job has no asset, and the `isaaclab-image` job selects `-m physics`. All three run under
a plain `pytest` in the image and in the `e2e` task.

## Markers

The markers are declared in `pyproject.toml`.

| Marker | Meaning | Where it runs |
|---|---|---|
| `physics` | Steps physics through Isaac Lab and MJWarp | The image, or a GPU node set up by `sky/setup.sh` |
| `data` | Reads the `exc-*` captures | Anywhere the captures are; one of the three is also `physics` |

The `data` tests look for captures in the directory named by `HUMANOID_SIM_DATASETS`. When it is unset they use
`datasets/humanoid-v1` next to the checkout (`../datasets/humanoid-v1` from the repository root). Without
captures the `data` tests skip.

## What runs where

| Where | Command | What runs |
|---|---|---|
| Natively, Mac or Linux | `uv run pytest -m "not physics and not data"` | The contract tests, as in CI |
| Natively, with captures | `HUMANOID_SIM_DATASETS=<path>/humanoid-v1 uv run pytest -m "data and not physics"` | The capture reader tests |
| Image, Warp's CPU device | see below, `pytest -m physics` | The physics tests; the CUDA tests skip |
| Image, everything | see below, `pytest` | Every test the machine can run |
| GPU node | the `e2e` SkyPilot task | Everything, including the CUDA and capture tests |

Natively you need the `dev` and `newton` extras (`uv sync --extra dev --extra newton`); without `newton`, the
tests that need scipy or Newton skip. On an Apple silicon Mac with the `newton` and `viz` extras and the plain
asset built, the native command reports 75 passed and 20 deselected (the `physics` and `data` tests) in about 20 s.

The image includes pytest (`docker/Dockerfile` installs `pytest>=9,<10`). Build both robot assets first, or most
physics tests skip. This is the `isaaclab-image` CI job's sequence:

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev build-asset
docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev build-asset --base-joints
docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev \
  pytest -m physics
```

For the whole suite, mount the captures where the tests look by default and drop the marker filter:

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -v "$(dirname "$PWD")/datasets":/workspace/datasets \
  -w /workspace/humanoid-sim humanoid-sim:dev pytest
```

To run one file or one test, name it: `pytest tests/test_world.py` or `pytest -k cable`. On Warp's CPU device the
physics tests are slow. In the arm64 image on an Apple silicon Mac, with both assets built and Warp's kernel cache
kept on a volume (`-v hsim-warp-cache:/cache`), the whole suite gives 91 passed and 4 skipped in about 3 minutes:
the CUDA-graph test, and three capture tests that need the captures mounted. Without the cache volume every run compiles the kernels again and takes
longer.

Two tests exercise CUDA graphs, the captured GPU step the runtime host uses:
`test_physics.py::test_cuda_graph_step_matches_eager` skips without CUDA, and
`test_runtime_host.py::test_demand_intake_queues_what_the_record_says` checks the graphed demand intake only
when CUDA is available. The `e2e` task runs both on a GPU.

The linter runs natively:

```bash
uv run ruff check src tests spikes examples docs
```

## CI

`.github/workflows/ci.yaml` runs three jobs on every push and pull request.

| Job | Runner | Steps |
|---|---|---|
| `cpu-contracts` | `ubuntu-24.04` (x86) and `ubuntu-24.04-arm`, each its own job | Check out with submodules, install uv 0.13.1, `uv sync --python 3.12 --extra dev --extra newton`, `uv run ruff check src tests spikes examples docs`, `uv run pytest -m "not physics and not data"` |
| `isaaclab-image` | `ubuntu-24.04` (x86) | Check out with submodules, `docker build -f docker/Dockerfile -t humanoid-sim:ci .` (amd64), build both assets, run `pytest -m physics` in the image on Warp's CPU device |
| `controls-pins` | `ubuntu-24.04` | Compare the submodule commits with the ones Controls pins |

No CI runner has a GPU or the captures. The `data` tests run wherever the captures are (natively, in the image or
in the `e2e` task), and the CUDA tests only on a GPU node, through the `e2e` task.

### Controls pins

humanoid-urdf and ankle_kinematics must be at the same commits as in Controls. The `controls-pins` job compares
the commit each submodule points to here with the commit that Controls' repository
(`ETHRoboticsClub/ETHRC-Humanoid-Controls`) pins for the same repository, and fails if they differ:

| Submodule here | Path in Controls |
|---|---|
| `third_party/humanoid-urdf` | `ros2_ws/src/description/URDF_5_description` |
| `third_party/ankle_kinematics` | `ros2_ws/src/drivers/ankle_kinematics` |

Two repository settings control the job:

- `SUBMODULES_TOKEN` is a repository secret: a GitHub token that can read Controls' repository and the submodule
  repositories. The job uses it to read Controls' pins through the GitHub API. Without it, the job prints a
  warning and passes without checking anything. The checkout steps of `cpu-contracts` and `isaaclab-image`
  also use it to fetch the submodules, falling back to the workflow's own token when it is unset.
- `CONTROLS_REF` is a repository variable: the Controls branch, tag or commit whose pins are compared. It
  defaults to `master`.

Set them with the GitHub CLI (`gh`), or under Settings, Secrets and variables, Actions:

```bash
gh secret set SUBMODULES_TOKEN
gh variable set CONTROLS_REF --body master
```

To compare the pins yourself:

```bash
git ls-tree HEAD third_party/humanoid-urdf third_party/ankle_kinematics
gh api "repos/ETHRoboticsClub/ETHRC-Humanoid-Controls/contents/ros2_ws/src/description/URDF_5_description?ref=master" --jq .sha
gh api "repos/ETHRoboticsClub/ETHRC-Humanoid-Controls/contents/ros2_ws/src/drivers/ankle_kinematics?ref=master" --jq .sha
```

To move a submodule to Controls' pin, check out that commit inside it and commit the new pointer:

```bash
git -C third_party/humanoid-urdf fetch
git -C third_party/humanoid-urdf checkout <commit>
git add third_party/humanoid-urdf
```

## GPU runs through SkyPilot

[SkyPilot](https://docs.skypilot.co/) starts cloud machines from a task file. It copies the working directory to
the machine, runs the task's `setup` and then its `run` script, and can stop the machine once it has been idle
for a set time. The tasks in `sky/` run on AWS, on one GPU each, without Docker: `sky/setup.sh` installs Isaac Lab on
the node directly.

### Install and check SkyPilot

Install the SkyPilot client with AWS support, for example with `uv tool install "skypilot[aws]"`, and check
that it can reach AWS with `sky check aws`. If you use a shared SkyPilot API server, log in to it with
`sky api login` instead of configuring AWS credentials locally.

Run every `sky` command from the repository root. Each task sets `workdir: .`, so the checkout is copied to
`~/sky_workdir` on the node. Git-ignored files such as `.venv/` and `outputs/` are not copied, so every task
builds the robot assets it needs. The tasks also copy captures from `../datasets/humanoid-v1` (see
[getting-started.md](getting-started.md#clone-the-repository)).

### Node setup: `sky/setup.sh`

Every task's `setup` runs `sky/setup.sh`:

1. It prints the GPU name and driver version.
2. It installs the system libraries Isaac Lab needs, and uv at the task's `UV_VERSION`.
3. It fetches Isaac Lab at the task's `ISAACLAB_COMMIT` into `~/isaaclab`, creates `~/il-venv` with Python 3.12
   and installs Isaac Lab from its own frozen `uv.lock` with the `importers`, `rerun` and `viser` extras.
4. If the CUDA 13.0 build of torch cannot use the node's driver (it needs driver 580 or newer), it installs the
   same torch version built for CUDA 12.8.
5. It prints the torch version and the number of CUDA devices Warp sees.
6. It installs the checkout into `~/il-venv` as an editable install, without dependencies.

The Isaac Lab commit and uv version in each task's `envs` are the same ones `docker/Dockerfile` pins, so the GPU
node and the image install the same Isaac Lab. Change them together.

To run your own commands on a node, log in with `ssh <cluster>` and set up the shell the way the tasks do:

```bash
export PATH="$HOME/.local/bin:$HOME/il-venv/bin:$PATH"
cd ~/sky_workdir
humanoid-sim --help
```

The node's copy of the checkout is updated by `sky launch` and `sky exec`. Local edits do not reach it until you
run one of them.

### Tasks

| Task | GPU | Cluster | What it does | Outputs on the node |
|---|---|---|---|---|
| `sky/bench.sky.yaml` | L40S | `hsim-bench` | Builds both assets. Runs the physics benchmark at 1, 4096, 8192 and 16384 envs with self-collision and at 4096 and 16384 without, the training benchmark at 1, 4096 and 16384 envs, all at 500 steps, and one replay. | `outputs/bench/` |
| `sky/replay.sky.yaml` | L40S | `hsim-bench` | Builds both assets. Replays three captures at 4 ms with `implicitfast`, 4 ms with `euler` and 1 ms with `implicitfast`, one process per capture in parallel. | `outputs/replay-mjwarp/` |
| `sky/sysid.sky.yaml` | L40S | `hsim-bench` | Builds the `--base-joints` asset. Runs `fit-passive` with 512 envs and 20 generations on the low-band capture and scores the fit on the two high-band ones, then fits on a high-band capture and scores on the low-band one. | `outputs/sysid/fit-on-low/`, `outputs/sysid/fit-on-high/` |
| `sky/e2e.sky.yaml` | A10G or L4, whichever has capacity | `hsim-e2e` | Builds both assets. Runs the whole test suite with the captures, the runtime benchmark for 2000 steps and the training benchmark at 4096 envs, replays every capture, and runs the examples. See [The end-to-end check](#the-end-to-end-check-skye2eskyyaml). | `outputs/e2e/` (cleared at the start of each run) |

Each task's `envs` block holds its settings, for example `STEPS`, `ENV_COUNTS` and `REPLAY` in the bench task,
`RUNS` in the replay task, and `NUM_ENVS`, `GENERATIONS`, `DURATION` and `BOUND` in the sysid task. Override one
at launch with `--env`, for example `--env STEPS=1000`. The bench task's `SPIKE=1` also runs the closed-chain
ankle spike in `spikes/closed_chain/`.

The launch commands are in a comment at the top of each task file:

```bash
# bench, then fetch the results
sky launch -y -c hsim-bench -i 10 sky/bench.sky.yaml
rsync -avz hsim-bench:sky_workdir/outputs/bench/ outputs/bench-gpu/

# run the bench task again on the same cluster, skipping setup
sky exec hsim-bench sky/bench.sky.yaml

# replays
sky launch -y -c hsim-bench -i 10 sky/replay.sky.yaml
rsync -avz hsim-bench:sky_workdir/outputs/replay-mjwarp/ outputs/replay-mjwarp/

# passive-parameter fits
sky launch -y -c hsim-bench -i 10 sky/sysid.sky.yaml
rsync -avz hsim-bench:sky_workdir/outputs/sysid/ outputs/sysid/

# end-to-end check, detached
sky launch -y -d -c hsim-e2e -i 10 sky/e2e.sky.yaml
sky logs hsim-e2e
rsync -avz hsim-e2e:sky_workdir/outputs/e2e/ outputs/e2e-gpu/
```

`-y` skips the confirmation prompt, `-c` names the cluster, `-i 10` sets autostop after 10 idle minutes and `-d`
returns once the job is submitted. The bench, replay and sysid tasks share the `hsim-bench` cluster, so they
reuse its Isaac Lab install, and a job launched while another runs waits in the cluster's queue.

### The end-to-end check: `sky/e2e.sky.yaml`

The `e2e` task exercises the whole repository on one GPU: everything CI cannot run, plus the benchmarks and the
examples. Its setup runs `sky/setup.sh` and installs pytest into `~/il-venv`; the captures are mounted at
`/data/humanoid-v1` and `HUMANOID_SIM_DATASETS` points there. The run script clears `outputs/e2e/`, builds both
assets, and then runs, in order:

| Step | Command | Outputs in `outputs/e2e/` |
|---|---|---|
| Whole test suite, no marker filter, so the CUDA and capture tests run too | `pytest -q` | `pytest.log` |
| Runtime benchmark | `humanoid-sim bench --runtime --device cuda:0 --steps 2000` | `runtime.log`, `runtime.json` |
| Training benchmark | `humanoid-sim bench --train --num-envs 4096 --device cuda:0 --steps 500` | `train_4096.log`, `train_4096.json` |
| A replay of every capture, at the replay defaults | `humanoid-sim replay <capture> --device cuda:0` | `replay/<capture>/`, `replay_<capture>.log` |
| The examples: training scenes recorded to Rerun, and a runtime session driven by `examples/runtime_client.py` and recorded by `view-runtime` | see the task file | `shots/*.rrd`, `serve.log`, `runtime_client.log`, `view_runtime.log` |

The script runs with `set -euo pipefail`, so the first step that fails ends the job and the later outputs are
missing; the pytest summary line is printed last. [benchmarks.md](benchmarks.md#reference-numbers) explains the
benchmark figures and lists those of the latest run. In the run of 2026-10-18 on an A10G every test passed and none
skipped (90 tests at that revision, in about 4 minutes).

Launch it detached and follow the log, as in the commands above. The outputs stay on the node: fetch them with
`rsync` within 10 minutes of the job ending, before autostop stops the machine, or restart the stopped machine with
`sky start hsim-e2e` and fetch them then. Launching the task again instead clears `outputs/e2e/` first. The `.rrd`
recordings open locally with `uv run rerun <file>.rrd` and the `viz` extra.

### Autostop

Launch every cluster with `-i 10`. After ten idle minutes SkyPilot stops the machine: the GPU is released and
billing drops to the disk, which keeps the Isaac Lab install, the assets and the outputs. The next `sky launch`
on the same cluster name starts it again, and its setup reuses the install already on the disk. Do not add
`--down` to these launches; that deletes the machine and its disk when idle.

`sky status` lists your clusters, their state and their autostop setting. When you are done with a cluster for
good, fetch its outputs and delete it with `sky down <cluster>`.

### Detached launches and logs

With `-d`, `sky launch` returns as soon as the job is submitted and the job runs on without your terminal.
`sky logs <cluster>` follows the output of the latest job; if the connection drops, the job keeps running and
you can run `sky logs` again. `sky queue <cluster>` lists the cluster's jobs, and `sky logs <cluster> <job id>`
shows an earlier one. Use `-d` for long tasks such as `e2e`.

### Fetching outputs

Outputs stay on the node's disk under `~/sky_workdir/outputs/`. SkyPilot adds an SSH host entry for every
cluster, so `ssh hsim-bench` and `rsync` reach it by cluster name. Copy results back before you run `sky down`:

```bash
rsync -avz hsim-bench:sky_workdir/outputs/bench/ outputs/bench-gpu/
```

A replay fetched this way can be viewed locally with `humanoid-sim view-replay`
([visualization.md](visualization.md)).

### GPU types

Use one A10G, L4 or L40S. The `e2e` task asks for an A10G or an L4; the other tasks ask for an L40S. To choose
the GPU at launch, pass `--gpus`, for example `sky launch --gpus A10G:1 ...`. Anything larger, such as an A100 or
H100, or more than one GPU, needs sign-off before you launch it.
