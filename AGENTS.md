# humanoid-sim

The robot below Controls: drives, ankle mechanism and body, on Isaac Lab 3.0 / Newton (MJWarp).

## Boundaries

- **No ROS.** Nothing here imports rclpy, ament or colcon, or any Controls code. Controls behaviour that the
  plant depends on is mirrored here with its source cited.
- **Drive boundary.** Input is a `FakeDriveDemand` mirror per EtherCAT drive (ankle cranks included);
  output is physical drive-output position, velocity, applied torque and demanded torque. Controller
  behaviour (standing frame, ankle targets, Madgwick, slew/blend, trip rule, policy contract) belongs
  to Controls, not here.
- **Physics is Isaac Lab on Newton's MJWarp solver, only.** Never set `use_mujoco_cpu`: MuJoCo C does not
  step the robot here, on any device. Newton on the host is for kinematics (viewers, model checks), and
  MuJoCo only for cross-checking kinematics against Controls' MJCF. Physics runs on CUDA; Warp's CPU
  device is for tests only.

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

`src/humanoid_sim/`: `model/` (description → prepared URDF → USD asset; host kinematics), `timing/`, `devices/`,
`transmissions/`, `core.py` (plant), `sensors/` (IMU delivery), `randomization/`, `world/` (terrains, pushes, gantry),
`physics/isaaclab.py` (MJWarp), `sim.py` (SimCore: replays, benchmarks), `train/` (Train-Time API: the plant in
an Isaac Lab scene and action term), `replay/`, `sysid/` (fits to captures), `signals/` (Rerun), `bench/`, `cli.py`.
Spikes live in `spikes/` and are not imported by the package.

## Workflow

- `uv sync --extra dev` then `uv run pytest -m "not physics"`. Kinematics tests need `--extra newton`; physics
  tests and `humanoid-sim build-asset` run only in the Isaac Lab image (`docker/Dockerfile`).
- Submodules: `git submodule update --init --recursive`. humanoid-urdf and ankle_kinematics must stay at
  the commits Controls pins.
- Docs explain how the repo works and how to use it. Design proposals, ADRs, milestone reports and
  other records of how it was built stay out of the repo, as do references to them.
- Preserve unrelated work. Commit completed, validated work in cohesive commits by concern; push only
  when authorized; never push directly to master.
- Commit messages say what changed and why. No `Co-Authored-By` trailers or AI/model attribution in
  commits, PRs or any committed file.
