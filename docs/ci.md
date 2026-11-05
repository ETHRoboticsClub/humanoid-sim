# CI

humanoid-sim is a public repository, so CI runs on GitHub-hosted runners only. No self-hosted runner may serve it, and none is registered. The private repositories humanoid-learning and humanoid-urdf have their own CI.

## Workflows

| Workflow | Triggers | What runs | Timeout |
|---|---|---|---|
| `CI` (`ci.yaml`) | pull request, push to master | `changes` decides whether anything besides Markdown and `docs/images` changed and whether the run has secrets. If it changed code and has secrets: `cpu-contracts` (ruff, `pytest -m "not physics and not data"` on `ubuntu-24.04`) and `controls-pins` (humanoid-urdf against Controls' pin). `ci-ok` always runs and passes when those passed or were skipped by the path filter | 20 min |
| `Physics` (`physics.yaml`) | the `physics` label on a pull request from this repository, manual dispatch, nightly at 02:00 UTC | `isaaclab-image`: builds the Isaac Lab image, the V1's assets and both G1s', and runs `pytest -m physics` on the CPU, on `ubuntu-24.04` | 3 h |

`ci-ok` is the check to require on master. It is not affected by the path filter, which is why the filter is a job and not a workflow `paths` setting. `Physics` is informational and not a dependency of `ci-ok`. A newer push to the same ref cancels the run in progress.

## Access to private repositories

The humanoid-urdf submodule and Controls are private. The jobs that need them mint a short-lived installation token with `actions/create-github-app-token` from the "ETHRC CI read" GitHub App:

- The app has Contents read on humanoid-urdf and ETHRC-Humanoid-Controls only. Its ID is the repository variable `CI_READ_APP_ID` and its private key is the repository secret `CI_READ_PRIVATE_KEY`.
- Each job narrows the token with `repositories:` to the one repository it needs: humanoid-urdf for `cpu-contracts` and `isaaclab-image`, ETHRC-Humanoid-Controls for `controls-pins`. The token expires with the job.
- Checkouts use `persist-credentials: false`, so the token is not left in the workspace's git config.
- The key is rotated when someone with write access leaves the organisation.

## Pull requests from forks

Pull requests from forks, and from Dependabot, get no repository secrets, so they cannot mint the token and cannot read humanoid-urdf. The jobs that need it are skipped for them, and `ci-ok` fails with a message saying so. `changes` runs for everyone, so docs-only pull requests from forks still pass.

To get a fork's change through CI, a maintainer reads the diff, then pushes the branch to `ETHRoboticsClub/humanoid-sim` and opens a pull request from it. That runs the full CI as a same-repository pull request. Never push a fork's branch unread: CI executes whatever the branch contains.

## Security

- No self-hosted runner for a public repository: any GitHub user can open a pull request whose workflow file runs on it. Keep every `runs-on` GitHub-hosted. The organisation's runner groups must keep "Allow public repositories" off.
- Never use `pull_request_target` or `workflow_run` with code from a pull request, and never put pull request text (title, branch name, label) into a `run:` script; pass values through `env:`.
- Workflows start with `permissions: {}`, and each job asks for `contents: read` only.
- Every action is pinned to a full commit SHA with its version in a comment, and `.github/dependabot.yml` proposes updates weekly. Python dependencies come from `uv.lock` (`uv sync --locked`), with uv at a fixed version.
- Branch protection on master should require a pull request and the `ci-ok` check.

## Physics

The `Physics` workflow runs the `physics` tests in the Isaac Lab image on the CPU. Start it by adding the `physics` label to a pull request from this repository, from Actions, Physics, Run workflow, or with `gh workflow run physics.yaml --ref <branch>`. It also runs nightly.

GPU physics for a branch runs in humanoid-learning's `Physics` workflow: its `sim-physics` job takes a `sim_ref` input (default `master`), checks out that branch of humanoid-sim and runs the `physics and not rtx and not data` tests on a GPU runner. The results appear in humanoid-learning's Actions, not on the pull request here. Run it for a branch before merging when GPU physics matters.

The same tests can run on a separate SkyPilot GPU node, with the cluster stopped afterwards (`-i 10`):

```
sky launch -y -d -c hsim-e2e -i 10 sky/e2e.sky.yaml
sky logs hsim-e2e
```

The `e2e` task also runs the benchmarks, the replays and the examples. See [testing-and-gpu.md](testing-and-gpu.md) for the tiers.

## The private repositories

humanoid-urdf runs on GitHub-hosted runners. humanoid-learning runs its GPU physics lane on ephemeral, single-job runners, each a fresh unprivileged container that is deleted when the job ends. Neither repository has a self-hosted runner that serves humanoid-sim.

## Run the CI commands locally

From a clone with submodules (`git submodule update --init --recursive`):

```
uv sync --locked --python 3.12 --extra dev --extra newton --extra kinematics-check --extra viz --extra bag
uv run ruff check src tests spikes examples docs
uv run pytest -m "not physics and not data"
```

### humanoid-learning

```
uv sync --locked --python 3.12 --extra dev
uv run ruff check src tests
uv run pytest -m "not physics"
```

On a SkyPilot node: `sky launch -y -d -c hl-smoke -i 10 sky/smoke.sky.yaml`.

### humanoid-urdf

It has one workflow, with no submodules and no physics lane:

```
uv venv --python 3.12
uv pip install --require-hashes -r ci/requirements-ci.txt
PYTHONPATH=ethrc_v1/v1_description .venv/bin/python -m pytest ethrc_v1/v1_description/test
```
