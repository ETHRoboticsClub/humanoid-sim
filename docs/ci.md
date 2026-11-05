# CI and the self-hosted runner

CI runs on a self-hosted runner, not on GitHub-hosted runners. GitHub-hosted runners run out of quota and cannot clone the private submodules. The same setup is used by humanoid-learning and humanoid-urdf, which link to this page.

## Runner

The runner is a long-lived SkyPilot cluster on the `ssh/spark` pool, which is a Kubernetes pod on a DGX Spark (aarch64, Ubuntu 22.04). It runs as the unprivileged user `sky`. There is no Docker in the pod. The runner's state and the deploy keys live on a SkyPilot volume mounted in the pod, so they survive a restart of the pod.

Labels: `self-hosted`, `linux`, `arm64`, `spark`. Every job uses `runs-on: [self-hosted, linux, arm64, spark]`.

The runner user needs:

- `git` and `uv` 0.13.1 on its `PATH` (`/home/sky/.local/bin/uv`), with the cache on the volume. `uv sync --python 3.12` fetches Python itself.
- Read-only deploy keys for humanoid-urdf and humanoid-sim on the volume, never in GitHub secrets, with `github.com` in `~/.ssh/known_hosts`.
- `gh`, only for the `controls-pins` job.

Expected `~/.ssh/config`, one alias per deploy key (the key files sit on the volume):

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
| `Physics` (`physics.yaml`) | the `physics` label on a pull request, manual dispatch | A placeholder: prints how to run physics on SkyPilot and exits 0 | 5 min |

Pull requests from forks never run on the runner: every job skips unless the head repository is this repository. A newer push to the same ref cancels the run in progress. The workflows have `contents: read` and none uses `pull_request_target`.

## Physics on SkyPilot

Physics needs Docker or a full Isaac Lab environment and a datacenter GPU. The Sparks are only the CPU runner: Isaac Lab on their GB10 is far too slow to be usable, so physics never runs there. Run it by hand on a cloud GPU node (L40S first, then A10G or L4), with the cluster stopped afterwards (`-i 10` stops it after 10 idle minutes):

```
sky launch -y -d -c hsim-e2e -i 10 sky/e2e.sky.yaml
sky logs hsim-e2e
```

The `e2e` task builds the V1 and G1 assets, runs the whole test suite (physics and CUDA tests included), the benchmarks, the replays and the examples. See [testing-and-gpu.md](testing-and-gpu.md) for the other tasks and for the test tiers. The `physics` label and the manual dispatch of the `Physics` workflow only print this reminder.

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

Its physics tests run on SkyPilot with `sky launch -y -d -c hl-smoke -i 10 sky/smoke.sky.yaml`.

### humanoid-urdf

It has one workflow, with no submodules and no physics lane:

```
uv run --no-project --python 3.12 --with pytest --with numpy --with pyyaml --with usd-core \
  env PYTHONPATH=ethrc_v1/v1_description pytest ethrc_v1/v1_description/test
```
