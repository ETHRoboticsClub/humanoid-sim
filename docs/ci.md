# CI and the self-hosted runner

CI runs on one self-hosted runner, an NVIDIA DGX Spark (linux, arm64, GPU), not on GitHub-hosted runners. GitHub-hosted runners run out of quota and cannot clone the private submodules. The same setup is used by humanoid-learning and humanoid-urdf, which link to this page.

## Runner

Labels: `self-hosted`, `linux`, `arm64`, `spark`. Every job uses `runs-on: [self-hosted, linux, arm64, spark]`.

The runner service runs as an unprivileged user. That user needs:

- `uv` 0.13.1 on its `PATH`, with the cache in `~/.cache/uv`. uv installs Python 3.12 itself.
- Docker, with the user in the `docker` group, and the NVIDIA container toolkit (`docker run --gpus all` works).
- Read-only deploy keys for humanoid-urdf and humanoid-sim, kept on the machine and never in GitHub secrets, with `github.com` in `~/.ssh/known_hosts`.
- `gh` on its `PATH`, only for the `controls-pins` job.

Expected `~/.ssh/config`, one alias per deploy key:

```
Host github-humanoid-urdf
  HostName github.com
  User git
  IdentityFile ~/.ssh/deploy_humanoid_urdf
  IdentitiesOnly yes

Host github-humanoid-sim
  HostName github.com
  User git
  IdentityFile ~/.ssh/deploy_humanoid_sim
  IdentitiesOnly yes
```

Each workflow checks out with `submodules: false`, then runs `git submodule update --init --recursive`. The job's environment sets `GIT_CONFIG_*` variables that rewrite the submodule URLs (`git@github.com:ETHRoboticsClub/<repo>.git`) to `<alias>:ETHRoboticsClub/<repo>.git`. The rewrite applies to that job only, including nested submodules, and nothing is written to a global git config.

## Workflows

| Workflow | Triggers | What runs | Timeout |
|---|---|---|---|
| `CI` (`ci.yaml`) | pull request, push to master; skipped for changes to Markdown and `docs/images` only | `cpu-contracts`: ruff and `pytest -m "not physics and not data"`. `controls-pins`: humanoid-urdf against Controls' pin | 20 min |
| `Physics` (`physics.yaml`) | the `physics` label on a pull request, manual dispatch, nightly at 02:00 UTC | Builds `docker/Dockerfile`, builds the robot assets, runs the `physics` tests on the GPU, one pytest process per test file | 90 min |

Pull requests from forks never run on the runner: every job skips unless the head repository is this repository. A newer push to the same ref cancels the run in progress. The workflows have `contents: read` and none uses `pull_request_target`.

## Trigger the physics lane

- On a pull request: add the `physics` label. Pushing to the pull request runs it again while the label stays.
- By hand: Actions, Physics, Run workflow, or `gh workflow run physics.yaml --ref <branch>`.
- Nightly: automatic, on master.

The `rtx` tests are not run: they need the Kit image (`docker/Dockerfile.kit`) and an RTX GPU. The `data` tests need the captures, which the runner does not have.

## Run the same commands locally

CPU lane, from a clone with submodules (`git submodule update --init --recursive`):

```
uv sync --python 3.12 --extra dev --extra newton --extra kinematics-check --extra viz --extra bag
uv run ruff check src tests spikes examples docs
uv run --extra dev pytest -m "not physics and not data"
```

Physics lane, on a Linux machine with an NVIDIA GPU (drop `--gpus all` on a Mac to use Warp's CPU device, where the CUDA tests skip):

```
docker build -f docker/Dockerfile -t humanoid-sim:ci .
run() { docker run --rm --gpus all -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:ci "$@"; }
run build-asset
run build-asset --base-joints
run --robot g1_29dof_rev_1_0 build-asset
run --robot g1_29dof_with_hand_rev_1_0 build-asset
run pytest -m "physics and not rtx" tests/test_physics.py
```

### humanoid-learning

Same runner, same labels, same `physics` label and dispatch. Its `Physics` workflow runs at 03:00 UTC. From a clone with submodules:

```
uv sync --python 3.12 --extra dev
uv run ruff check src tests
uv run pytest -m "not physics"
```

Physics lane:

```
docker build -f third_party/humanoid-sim/docker/Dockerfile -t humanoid-sim:ci third_party/humanoid-sim
docker build -f docker/Dockerfile --build-arg HUMANOID_SIM_IMAGE=humanoid-sim:ci -t humanoid-learning:ci .
run() { docker run --rm --gpus all -v "$PWD":/workspace/humanoid-learning -w /workspace/humanoid-learning humanoid-learning:ci "$@"; }
run humanoid-sim build-asset
run pytest -m physics --device cuda:0 tests/test_walk_task_physics.py
```

### humanoid-urdf

It has one workflow, with no submodules and no physics lane. From a clone:

```
uv run --no-project --python 3.12 --with pytest --with numpy --with pyyaml --with usd-core \
  env PYTHONPATH=ethrc_v1/v1_description pytest ethrc_v1/v1_description/test
```

See [testing-and-gpu.md](testing-and-gpu.md) for the test tiers and the GPU tasks.
