# humanoid-sim

The robot below Controls: drives, ankle mechanism and body, on Isaac Lab 3.0 / Newton (MJWarp or Kamino).

## Boundaries

- **No ROS.** Nothing here imports rclpy, ament or colcon, or any Controls code. Controls behaviour that the plant depends on is mirrored here with its source cited.
- **Drive boundary.** Input is a vendor-neutral SI `DriveDemand` per drive (law, targets, gains, torque limit; ankle cranks included), as in Controls' `docs/sim-learning-api-plan.md` §1.2; output is physical drive-output position, velocity, applied torque and demanded torque. Vendor commands (EPOS4 modes and raw gains, Unitree `LowCmd`) are converted on the client side. Controller behaviour (standing frame, ankle targets, Madgwick, slew/blend, trip rule, policy contract) belongs to Controls, not here.
- **Repo split.** Tasks, observations, rewards, terrain generators, curricula, policies and every policy evaluation (scenarios, robustness sweeps, sensor ablations) belong to humanoid-learning; this repo provides the robot, world, actuator and sensor models and the knobs a task turns (a terrain from the caller's tiles, a push process, a randomization level), never a stock choice of them. Its own replay, physics and benchmark checks validate the sim; system-identification fitting and its experiments belong to humanoid-learning. Controls reaches the sim only through the runtime adapter (its plant host, this repo's `runtime/` and the segment ABI).
- **Layering.** Front ends (`train/`, `runtime/`, `replay/`) use the plant and the models, never each other; models never import a front end (the segment ABI's description digest is the one exception).
- **Physics is Isaac Lab on a Newton solver, only:** MJWarp (the profile's default) or Kamino (beta), selected by name. Never set `use_mujoco_cpu`: MuJoCo C does not step the robot here, on any device. Newton on the host is for kinematics (viewers, model checks), and MuJoCo only for cross-checking kinematics against Controls' MJCF. Physics runs on CUDA; Warp's CPU device is for tests only.

## Rules

- **Quantities:** keep requested, applied and measured quantities separate. Never synthesize a sensor the robot does not have.
- **Time:** integer nanoseconds everywhere, including delays and windows. Never count physics steps.
- **Imports:** Isaac Lab, Newton and Warp are imported lazily. `pytest -m "not physics"` must pass with only the core dependencies installed.
- **Backends:** selected explicitly; no silent fallback. Unknown names raise.
- **Modeled vs verified:** anything inferred from code or datasheets is labelled `modeled` until traces verify it. The description is loaded hash-verified.
- **Description-driven:** the model build and device inventory come from `third_party/humanoid-urdf`. No robot constant is typed into this repo; if a fact is missing, add it to humanoid-urdf.
- **Tests test the code, not the description:** a legitimate edit to humanoid-urdf's values (limits, masses, gains, delays, poses, joint counts) must not fail a test. Use fixture values or expectations read from the description at runtime. Replays and sysid against robot captures are the exception: they validate the sim against the robot.
- **Batched:** device and transmission math is torch over `[envs, ...]`. No host syncs in steady state.
- **Solver gains** are written at reset only. Gain changes and envelopes are target shaping.

## Layout

`src/humanoid_sim/`: `model/` (description → prepared URDF → USD asset; host kinematics; named poses; left/right mirror map), `timing/`, `devices/`, `transmissions/`, `core.py` (plant), `sensors/` (named sensor models, rigs, IMU delivery), `randomization/`, `world/` (terrains, pushes, gantry), `physics/` (Isaac Lab on MJWarp or Kamino, solver joint properties), `sim.py` (SimCore: replays, benchmarks, the runtime), `train/` (Train-Time API: the plant in an Isaac Lab scene and action term), `runtime/` (real-time host and the segment ABI), `replay/` (capture readers, parameter application and replay mechanisms), `signals/` (Rerun), `bench/`, `cli.py`. Spikes live in `spikes/` and are not imported by the package.

## Workflow

- `uv sync --extra dev` then `uv run pytest -m "not physics"`. Kinematics tests need `--extra newton`; physics tests and `humanoid-sim build-asset` run only in the Isaac Lab image (`docker/Dockerfile`).
- Submodules: `git submodule update --init --recursive`. humanoid-urdf must stay at the commit Controls pins; CI checks it against Controls' `CONTROLS_REF` (a repository variable, default master).
- Docs explain how the repo works and how to use it. Design proposals, ADRs, milestone reports and other records of how it was built stay out of the repo, as do references to them.
- Preserve unrelated work. Commit completed, validated work in cohesive commits by concern; push only when authorized; never push directly to master.
- Commit messages say what changed and why. No `Co-Authored-By` trailers or AI/model attribution in commits, PRs or any committed file.
