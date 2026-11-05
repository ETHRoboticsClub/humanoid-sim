# CI and the self-hosted runner

CI runs on a self-hosted runner, not on GitHub-hosted runners. GitHub-hosted runners run out of quota and cannot clone the private submodules. The same setup is used by humanoid-learning and humanoid-urdf, which link to this page.

## Runner

The runner is the SkyPilot cluster `ci-runner`, a long-lived Kubernetes pod on the `ssh/spark` pool (a DGX Spark: aarch64, Ubuntu 22.04, one GB10 GPU). It runs as the unprivileged user `sky`, never as root. There is no Docker in the pod. It runs five GitHub Actions runners: one each for humanoid-sim, humanoid-learning and humanoid-urdf, and a second for humanoid-sim and for humanoid-learning so that a long physics run does not block their CI, and has Isaac Lab in `~/il-venv` for the physics lane. There is no volume: the deploy keys, `~/.ssh` (a link to `~/ci/ssh`), the runner registrations and the work directories live in `~/ci` in the pod.

The launch task is `~/ETHRC/tools/ci-runner/ci-runner.sky.yaml` on the maintainer's machine; it is not in a repository. Its header has the commands. To restart the runners while the pod is up (they show offline under Settings, Actions, Runners), run `sky exec -d ci-runner --secret GH_TOKEN="$(gh auth token)" ~/ETHRC/tools/ci-runner/ci-runner.sky.yaml`: the keys and registrations in `~/ci` are kept. If the pod itself was restarted, `~/ci` is gone: relaunch with `sky launch` as in the task header. That reruns setup, generates new deploy keys and registers the runners again with `--replace`; delete the stale deploy keys titled "ci-runner spark (read-only)" in the three repositories afterwards. A supervisor loop restarts a runner whose process dies. The cluster has no autostop.

Labels: `self-hosted`, `linux`, `arm64`, `spark`. Every job uses `runs-on: [self-hosted, linux, arm64, spark]`.

The runner user needs:

- `git` and `uv` 0.13.1 on its `PATH` (`/home/sky/.local/bin/uv`), with its cache in `~/.cache/uv`. `uv sync --python 3.12` fetches Python itself.
- Read-only deploy keys for humanoid-urdf, humanoid-sim and ETHRC-Humanoid-Controls, generated on the cluster, kept in `~/ci/ssh` and never in GitHub secrets, with `github.com` in `~/.ssh/known_hosts`.

Expected `~/.ssh/config`, one alias per deploy key (the key files sit in `~/ci/ssh`, and the launch task writes this file):

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

Host github-controls
  HostName github.com
  User git
  IdentityFile ~/.ssh/deploy_controls
  IdentitiesOnly yes
```

Each workflow checks out with `submodules: false`, then runs `git submodule update --init --recursive`. The job's environment sets `GIT_CONFIG_*` variables that rewrite the submodule URLs (`git@github.com:ETHRoboticsClub/<repo>.git`) to `<alias>:ETHRoboticsClub/<repo>.git`. The rewrite applies to that job only, including nested submodules, and nothing is written to a global git config.

## Workflows

| Workflow | Triggers | What runs | Timeout |
|---|---|---|---|
| `CI` (`ci.yaml`) | pull request, push to master | `changes` decides whether anything besides Markdown and `docs/images` changed. If so: `cpu-contracts` (ruff, `pytest -m "not physics and not data"`) and `controls-pins` (humanoid-urdf against Controls' pin, read over the Controls deploy key). `ci-ok` always runs and passes when those passed or were skipped | 20 min |
| `Physics` (`physics.yaml`) | the `physics` label on a pull request, manual dispatch, nightly at 02:00 UTC | The `physics` tests (not `rtx`, not `data`) in `~/il-venv` on the GPU, one pytest process per test file, a per-file table in the job summary | 6 h |

`ci-ok` is the check to require on master. It is not affected by the path filter, which is why the filter is a job and not a workflow `paths` setting. The `Physics` job has `continue-on-error` and is not a dependency of `ci-ok`: Isaac Lab on the Spark is slow, so physics is informational and never a required check. A pull request from a fork gets no `ci-ok`, so it cannot merge on a green check.

Pull requests from forks never run on the runner: every job skips unless the head repository is this repository. A newer push to the same ref cancels the run in progress. The workflows have `contents: read` and none uses `pull_request_target`.

## Physics

The `Physics` workflow runs on the runner's GPU in `~/il-venv`, the environment that `sky/setup.sh` builds. Start it by adding the `physics` label to a pull request, from Actions, Physics, Run workflow, or with `gh workflow run physics.yaml --ref <branch>`. It also runs nightly. It can take hours; the job summary lists each test file with its result and time.

The same tests can run on a separate SkyPilot GPU node, with the cluster stopped afterwards (`-i 10`):

```
sky launch -y -d -c hsim-e2e -i 10 sky/e2e.sky.yaml
sky logs hsim-e2e
```

The `e2e` task also runs the benchmarks, the replays and the examples. See [testing-and-gpu.md](testing-and-gpu.md) for the tiers.

## Run the CI commands locally

From a clone with submodules (`git submodule update --init --recursive`):

```
uv sync --python 3.12 --extra dev --extra newton --extra kinematics-check --extra viz --extra bag
uv run ruff check src tests spikes examples docs
uv run --extra dev pytest -m "not physics and not data"
```

### humanoid-learning

```
uv sync --python 3.12 --extra dev
uv run ruff check src tests
uv run pytest -m "not physics"
```

Its `Physics` workflow runs `pytest -m "physics and not data" --device cuda:0` per file in `~/il-venv`. On a separate node: `sky launch -y -d -c hl-smoke -i 10 sky/smoke.sky.yaml`.

### humanoid-urdf

It has one workflow, with no submodules and no physics lane:

```
uv run --no-project --python 3.12 --with pytest --with numpy --with pyyaml --with usd-core \
  env PYTHONPATH=ethrc_v1/v1_description pytest ethrc_v1/v1_description/test
```
