# Getting started

This page goes from a fresh clone to each kind of run: the CPU tests, the Isaac Lab image, the robot asset, a replay of a recorded capture, a training environment and the real-time runtime. Run every command from the repository root unless a step says otherwise.

## Which machine do I need

humanoid-sim runs in three places. The contract tests and the viewers run natively, in a Python environment managed by uv. Physics runs only through Isaac Lab and MJWarp, which are installed in the Docker image built from `docker/Dockerfile` (Linux, amd64 or arm64) and on the GPU nodes that SkyPilot sets up. On a machine without an NVIDIA GPU the image runs the same solver on Warp's CPU device, well below real time; that is meant for tests and short trial runs.

| Task | Mac (Apple silicon) | Linux, CPU only | Linux with an NVIDIA GPU | SkyPilot GPU node |
|---|---|---|---|---|
| CPU contract tests | Native | Native | Native | Not needed |
| Build the robot asset | arm64 image | Image | Image | Every task builds it |
| Physics tests | arm64 image, CPU device | Image, CPU device | Image | `e2e` task |
| CUDA-graph tests | No | No | Image | `e2e` task |
| Replay a capture | arm64 image, short runs | Image, short runs | Image | `replay` and `e2e` tasks |
| View a replay in Rerun | Native | Native | Native | Fetch the output and view it locally |
| `fit-passive`, `fit-delay` | Small runs only | Small runs only | Image | `sysid` task (`fit-passive`) |
| `check-bag` | Native | Native | Native | Not needed |
| `fit-pace`, `pace-selftest` | Small runs only | Small runs only | Image | `pace` task |
| Training environment | arm64 image, a few envs | Image, a few envs | Image | `bench` and `e2e` tasks benchmark it |
| Benchmarks | Runs; the numbers mean little | Runs; the numbers mean little | Image | `bench` and `e2e` tasks |
| Serve the runtime to Controls | arm64 image, below real time | Image, below real time | Image, real time | Only with the client on the same node |
| Watch the runtime in Rerun (`view-runtime`) | arm64 image, writing a recording | Next to the host: image, or native | Next to the host: image, or native | `e2e` task records one |

"Image" means the Docker image built for that machine's architecture. In the GPU column, start the container with `--gpus all` (see [Run commands in the image](#run-commands-in-the-image)). The repository's own GPU runs go through SkyPilot, described in [testing-and-gpu.md](testing-and-gpu.md#gpu-runs-through-skypilot).

## Clone the repository

You need read access to the ETHRoboticsClub repositories on GitHub.

```bash
git clone --recurse-submodules https://github.com/ETHRoboticsClub/humanoid-sim.git
cd humanoid-sim
```

If you cloned without `--recurse-submodules`, or pulled a change that moves a submodule, update them:

```bash
git submodule update --init --recursive
```

The submodule lives in `third_party/`:

| Submodule | Contents | Used for |
|---|---|---|
| `humanoid-urdf` | The robot description: URDF, meshes and `actuators.yaml` (drives, signs, gains, envelopes, ankle geometry) | Every model build and the drive inventory |

It must stay at the commit Controls pins. CI checks this ([testing-and-gpu.md](testing-and-gpu.md#controls-pins)).

Replays, fits and the `data` tests read excitation captures recorded on the robot: directories named `exc-*`. They are not part of the repository. Put them in `datasets/humanoid-v1/` next to the checkout, which is where the tests look by default and where the SkyPilot tasks upload them from:

```text
ETHRC/
  humanoid-sim/                      this repository
  datasets/
    humanoid-v1/
      exc-503fc128aaf14405a0c2418d63ff9a60/
      exc-3a498000fc114f2db2b0655d5d87280c/
      ...
```

[replay-and-sysid.md](replay-and-sysid.md) describes what a capture contains.

## Set up the Python environment

[uv](https://docs.astral.sh/uv/) is a Python project manager. It installs the Python version the project needs (3.12), creates a virtual environment in `.venv/` and installs the dependencies locked in `uv.lock`. Install it with `brew install uv` on a Mac, or with `curl -LsSf https://astral.sh/uv/install.sh | sh` on Linux. CI uses uv 0.13.1.

```bash
uv sync --extra dev --extra newton --extra viz
```

The core dependencies are numpy, torch and PyYAML. Each extra adds a group of optional packages:

| Extra | Installs | Needed for |
|---|---|---|
| `dev` | pytest, ruff | Running the tests and the linter |
| `newton` | Newton, Warp, newton-usd-schemas, usd-core, trimesh, scipy, pycollada and resolve-robotics-uri-py | Host kinematics: the replay and runtime viewers, model checks and the kinematics tests. Newton here never steps physics. |
| `viz` | rerun-sdk, pyarrow | `humanoid-sim view-replay`, `humanoid-sim view-runtime` and Rerun signal logs |
| `bag` | rosbags | Reading Controls' rosbag2 recordings (`check-bag` and `fit-pace` on a bag, `tests/test_bag_reader.py`). The image and the GPU nodes install it. |
| `kinematics-check` | MuJoCo | Cross-checking kinematics against Controls' MJCF by hand. No test or command needs it. |

`uv sync` installs exactly the extras you name and removes the others, so name every extra you want in one command. `uv run <command>` runs a command in that environment and leaves the installed extras in place.

None of the extras installs Isaac Lab. The commands that need it (`build-asset`, `replay`, `bench`, `serve`, `fit-passive`, `fit-delay`, `fit-pace`, `pace-selftest`) fail natively and must run in the image or on a GPU node. `view-replay` and `view-runtime` run natively with the `newton` and `viz` extras. The example runtime client, `examples/runtime_client.py`, needs only the core dependencies; like `view-runtime`, it must run on the machine the host runs on.

## Run the CPU contract tests

```bash
uv run pytest -m "not physics and not data"
uv run ruff check src tests spikes examples docs
```

These are the commands the `cpu-contracts` CI job runs. They cover the drive layer, the ankle transmission, the model build, the capture readers, the runtime segment layout and host kinematics. Tests marked `physics` or `data` are deselected. On an Apple silicon Mac with the three extras above and the plain robot asset built, pytest reports 77 passed and 22 deselected in about 20 s. Before the asset is built (see below), the three tests that load it with host kinematics are reported as skipped; pytest prints each skip reason. The full test layout and what runs where are in [testing-and-gpu.md](testing-and-gpu.md).

## Build the Isaac Lab image

Docker runs programs in containers: isolated Linux environments started from an image. On a Mac, install Docker Desktop. The image contains Isaac Lab 3.0, Newton, MJWarp and Warp at the versions pinned in `docker/Dockerfile`, plus this package. It takes about 13 GB of disk.

Build it for your machine's architecture:

```bash
# Mac with Apple silicon, or arm64 Linux
docker buildx build --platform linux/arm64 -f docker/Dockerfile -t humanoid-sim:dev .

# x86 Linux
docker buildx build --platform linux/amd64 -f docker/Dockerfile -t humanoid-sim:dev .
```

A build for the other architecture runs under emulation and is very slow. The first build downloads several gigabytes. The container runs the code in your checkout (next section), so code changes do not need a rebuild; rebuild when `docker/Dockerfile` changes.

## Run commands in the image

Mount the checkout into the container at `/workspace/humanoid-sim` and work there:

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev --help
```

`--rm` deletes the container when the command ends, `-v` mounts the current directory into the container, and `-w` sets the working directory. When the checkout is mounted there, the image's entrypoint (`docker/entrypoint.sh`) does two things before running your command:

1. It installs the mounted checkout into the image's Python environment as an editable install, so the container runs your working tree.
2. It sets `HUMANOID_SIM_DESCRIPTION_ROOT` to `/workspace/humanoid-sim/third_party/humanoid-urdf`, the robot description in your checkout. To use another humanoid-urdf checkout, mount it and pass `-e HUMANOID_SIM_DESCRIPTION_ROOT=<path in the container>`.

Always mount the checkout. The copy of the package baked into the image has neither `profiles/` nor the submodules, so it cannot load the robot.

The first argument decides what runs. `python`, `pytest` and `bash` run as given; anything else is passed to `humanoid-sim`, so you write `... humanoid-sim:dev build-asset`, without the word `humanoid-sim`. The image includes pytest, so `... humanoid-sim:dev pytest -m physics` runs the physics tests (see [testing-and-gpu.md](testing-and-gpu.md#what-runs-where)). For an interactive shell, add `-it` and run `bash`; inside it the `humanoid-sim` command is on the path.

A shell function saves typing. Define it in your shell (or your `~/.zshrc`) and call it from the repository root:

```bash
hsim() { docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev "$@"; }
hsim --help
```

The rest of this page uses `hsim`. Three additions are useful:

- To make the captures visible, add `-v "$(dirname "$PWD")/datasets":/workspace/datasets`. The tests then find them at their default location, `/workspace/datasets/humanoid-v1`.
- To keep Warp's compiled kernels between runs, add `-v hsim-warp-cache:/cache`. The image sets `WARP_CACHE_PATH=/cache/warp`; without the volume every container compiles the kernels again.
- On a Linux machine with an NVIDIA GPU, add `--gpus all`. This needs the NVIDIA Container Toolkit and a driver recent enough for the image's CUDA 13.0 build of torch (driver 580 or newer).

On Linux, files the container writes under `outputs/` belong to root.

## Build the robot asset

Isaac Lab loads robots only from USD, so every physics run starts from a USD asset built from the description and the profile (`profiles/ethrc_v1.yaml`). Build both variants once:

```bash
hsim build-asset
hsim build-asset --base-joints
```

This runs on the CPU and needs no GPU. Each command prints the path of the USD it built under `outputs/assets/ethrc_v1/<key>/`, next to an `asset.json` that records what went into it. The key is a digest of the description, the profile's `model` block and the prepared URDF built from them: after a change to any of them, the next build goes to a new directory, and an unchanged asset is reused (`--force` rebuilds it).

| Variant | Used by |
|---|---|
| Plain (`build-asset`) | Training, benchmarks, the runtime, most physics tests, `view-replay`, `view-runtime` |
| `--base-joints` | `replay` and learning's fitting experiments with a moving base (the default, `--base yaw`) |

The `--base-joints` variant adds yaw, pitch and roll joints above the base so a replay can turn the robot the way the capture's IMU says it turned. [robot-model.md](robot-model.md) explains the build.

The commands above build the ETHRC V1. Another robot of the description is selected with `--robot` before the subcommand, for every command, for example the Unitree G1 that Controls runs:

```bash
hsim --robot g1_29dof_with_hand_rev_1_0 build-asset
```

## A first replay

A replay feeds the drive demands recorded in a capture to the simulated drives and compares the simulated drive feedback with the recorded one. With the captures mounted, replay the first five seconds of one capture on Warp's CPU device:

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -v "$(dirname "$PWD")/datasets":/workspace/datasets \
  -w /workspace/humanoid-sim humanoid-sim:dev \
  replay /workspace/datasets/humanoid-v1/exc-503fc128aaf14405a0c2418d63ff9a60 --device cpu --duration 5
```

`replay` defaults to `--device cuda:0`, so on a machine without a GPU pass `--device cpu`. `--duration` keeps the run short; on the CPU device it runs well below real time. The command prints a JSON line with the wall and simulated time, then the position error of each drive against the capture. It writes `trace.npz` and `summary.json` to `outputs/replay/exc-503fc128aaf14405a0c2418d63ff9a60_mjwarp_implicitfast_4000us_base-yaw/`.

View the result natively, with the `newton` and `viz` extras installed:

```bash
uv run humanoid-sim view-replay \
  outputs/replay/exc-503fc128aaf14405a0c2418d63ff9a60_mjwarp_implicitfast_4000us_base-yaw --save replay.rrd
uv run rerun replay.rrd
```

The recording shows the simulated robot with the real one as a ghost, and every drive signal. [visualization.md](visualization.md) covers the viewer; [replay-and-sysid.md](replay-and-sysid.md) covers replay options and demand delays; fitting is documented in humanoid-learning. Full-length replays of every capture run on a GPU through the `replay` SkyPilot task.

## A first training environment

The Train-Time API puts the plant into a manager-based Isaac Lab environment: the policy's actions become drive demands, and observations read drive feedback as the drives report it. `tests/test_train_env.py` builds a complete environment with two robots on Warp's CPU device and steps it for about three simulated seconds. Run it in the image after building the plain asset:

```bash
hsim pytest tests/test_train_env.py -q
```

`examples/train_scene.py` is a fuller scene to start from: robots on rough terrain (or flat ground with `--flat`), each holding the home pose at the baseline gains while it waves its arms, with randomized friction, base mass and drive latency, and random pushes. Its drives are scripted, so there is no policy and a robot falls when a push or the terrain beats its rigid pose. With `--rrd` it records the scene to Rerun. A short run on the CPU:

```bash
hsim python examples/train_scene.py --num-envs 2 --seconds 2 --device cpu --rrd outputs/train_scene.rrd
```

It ends by printing how many robots are still standing. Open the recording natively with `uv run rerun outputs/train_scene.rrd`.

To time the training scene with its sensors and randomization, use the benchmark, here on the CPU with two robots:

```bash
hsim bench --train --num-envs 2 --steps 100 --device cpu
```

[training.md](training.md) explains how to write an environment config against the API. [benchmarks.md](benchmarks.md) covers the benchmark and the GPU reference numbers.

## Serve the runtime

`humanoid-sim serve` runs one robot against the wall clock behind the runtime segment, a shared-memory file that a client (Controls' plant host) attaches to. The segment file must be on a volume that the client's container also mounts. Create a named volume and serve on it:

```bash
docker volume create hsim-run
docker run --rm -v "$PWD":/workspace/humanoid-sim -v hsim-run:/run/hsim -w /workspace/humanoid-sim \
  humanoid-sim:dev serve --segment /run/hsim/plant --support suspended --device cpu
```

When the host is ready it prints a JSON line with `"ready": true` and the session id. `--support suspended` hangs the robot from the lab gantry's harness cable; `--support free` stands the robot on the ground. On the CPU device the host runs below real time, which is enough to check the protocol against a client: in the arm64 image on an Apple silicon Mac it was ready 25 to 40 s after start, with Warp's kernel cache kept on a volume. On a GPU, drop `--device cpu` and add `--gpus all`: the host then captures and compiles its step at startup (about a minute) and runs faster than real time. Ctrl-C stops the host.

The `justfile` wraps these commands: `just image` builds the image, `just asset [robot]` the robot asset, and `just serve [robot] [support]` serves on the `hsim-run` volume with `--persist`, on `cuda:0` when `nvidia-smi` is present and on the CPU otherwise (`HSIM_DEVICE` overrides it). Controls' simulation stack attaches to that volume. `just view` records the running host to `outputs/runtime.rrd`.

The smallest client is `examples/runtime_client.py`. It holds the robot where it starts at the baseline gains, waves the arms, pushes the left forearm after 8 s and detaches. In a second terminal, run it in another container on the same volume:

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -v hsim-run:/run/hsim -w /workspace/humanoid-sim \
  humanoid-sim:dev python examples/runtime_client.py /run/hsim/plant --seconds 20
```

It prints the plant time, the host's late steps and the largest applied torque once a second. When it detaches, `serve` ends the session and, without `--persist`, exits. Containers on one machine share the clock the protocol uses, and both run as root, so the client can open the file `serve` created.

To watch the robot, start `view-runtime` in a third container before the client, writing a recording to the checkout, and open it natively once it ends (it stops 10 s after the last session ends):

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -v hsim-run:/run/hsim -w /workspace/humanoid-sim \
  humanoid-sim:dev view-runtime /run/hsim/plant --save outputs/runtime.rrd
uv run rerun outputs/runtime.rrd
```

[runtime.md](runtime.md) describes the segment, the session, the commands and the viewer. [visualization.md](visualization.md) shows the viewers. [controls-integration.md](controls-integration.md) describes how Controls connects to the runtime.
