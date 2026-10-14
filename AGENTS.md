# humanoid-sim

The robot below Controls: drives, mechanism, body, sensors and world, on Isaac Lab 3.0 / Newton.
Design: `docs/architecture.md` (copy of the accepted proposal). Read it before adding a module.

## Boundaries

- **No ROS.** Nothing here imports rclpy, ament or colcon. Controls reaches the sim through the
  runtime segment (`runtime/abi`), never the other way round.
- **Drive boundary.** Input is a `FakeDriveDemand` mirror per EtherCAT drive (ankle cranks included);
  output is physical drive-output position, velocity, applied torque and demanded torque. Controller
  behaviour (standing frame, ankle targets, Madgwick, slew/blend, trip rule, policy contract) belongs
  to Controls and to humanoid-learning's `controls_model`, not here.
- **MuJoCo** is not a physics backend here. Newton's MJWarp solver is (and its `use_mujoco_cpu` path
  for the CPU runtime). MuJoCo is used only to cross-check kinematics against Controls' MJCF.

## Rules

- **Quantities:** keep requested, applied and measured quantities separate. Never synthesize a sensor
  the robot does not have.
- **Time:** integer nanoseconds everywhere, including delays and windows. Never count physics steps.
- **Imports:** Isaac Lab, Newton and Warp are imported lazily. `pytest -m "not physics"` must pass with
  only the core dependencies installed.
- **Backends:** selected explicitly; no silent fallback. Unknown names raise.
- **Modeled vs verified:** anything inferred from code or datasheets is labelled `modeled` until traces
  verify it. The description is loaded hash-verified.
- **Description-driven:** the model build and device inventory come from `third_party/humanoid-urdf`.
  No robot constant is typed into this repo; if a fact is missing, add it to humanoid-urdf.
- **Batched:** device and transmission math is torch over `[envs, ...]`. No host syncs in steady state.
- **Solver gains** are written at reset only. Gain changes and envelopes are target shaping.

## Layout

`src/humanoid_sim/`: `model/` (description → prepared URDF → USD asset), `timing/`, `devices/`,
`transmissions/`, `core.py` (plant), `physics/` (`isaaclab`, `newton` backends), `sim.py` (SimCore),
`replay/`, `signals/` (Rerun), `bench/`, `cli.py`. `sensors/` and `world/` arrive with their first feature.
Spikes live in `spikes/` and are not imported by the package. M0 results: `docs/m0-report.md`.

## Workflow

- `uv sync --extra dev` then `uv run pytest -m "not physics"`. Physics tests need `--extra physics` and a built
  asset: `humanoid-sim build-asset` runs only in the Isaac Lab image (`docker/Dockerfile`).
- Submodules: `git submodule update --init --recursive`. humanoid-urdf and ankle_kinematics must stay at
  the commits Controls pins.
- Preserve unrelated work. Commit completed, validated work in cohesive commits by concern; push only
  when authorized; never push directly to master.
- Commit messages say what changed and why. No `Co-Authored-By` trailers or AI/model attribution in
  commits, PRs or any committed file.
