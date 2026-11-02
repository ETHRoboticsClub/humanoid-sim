# Replay and system identification

humanoid-sim owns physical replay and capture readers. Fitting, acceptance, conversion commands, reports and fitting SkyPilot tasks live in humanoid-learning (`python -m humanoid_learning.sysid --help`). Its fitter supplies resolved drive targets through `DriveCommands`; replay does not choose a controller command clock.

`humanoid-sim replay` steps the plant in the Isaac Lab image on CUDA and compares recorded and simulated drive feedback. The capture reader and coordinate/conversion helpers are usable without physics. See [getting-started.md](getting-started.md) and [testing-and-gpu.md](testing-and-gpu.md).

## Excitation captures

An excitation capture is a dataset the Controls robot API records while it swings the robot's joints through a frequency sweep (a chirp) with the robot hanging on the [gantry](glossary.md#gantry). Each capture is a directory named after its `dataset_id`, `exc-<id>`. They are kept outside this repository, in `datasets/humanoid-v1/` beside the humanoid-sim checkout. `ExcitationCapture` reads format `ethrc-excitation/v8` only and raises on any other.

The SkyPilot tasks and the examples below use these three captures:

| Capture | Profile | Sweep |
|---|---|---|
| `exc-503fc128aaf14405a0c2418d63ff9a60` | `whole-body` (low band) | 0.2 to 3 Hz over 30 s |
| `exc-3a498000fc114f2db2b0655d5d87280c` | `whole-body-high` (high band) | 3 to 10 Hz over 45 s |
| `exc-ece46c33aacc4ec5bbdf538aaa7f297a` | `whole-body-high` (high band) | 3 to 10 Hz over 45 s |

### Files

| File | Contents | Used by humanoid-sim |
|---|---|---|
| `manifest.json` | Format, `dataset_id`, `display_name`, the `complete` flag, the robot's identity and calibration revisions, the resolved excitation (profile, sweep, per-joint amplitude and center), per-joint signal limits, stream schemas with units, row counts, file digests and a tracking-error summary | `format`, `complete`, `dataset_id`, `display_name` |
| `commands.csv` | The joint-space targets the robot API published, one frame every 20 ms (`q_nominal_rad` and `q_mirror_rad` per joint) | Read by the client to supply resolved drive targets |
| `acks.csv` | The receivers' acknowledgements of each command frame | No |
| `pdo_feedback.csv` | Per drive at about 200 Hz: measured position `native_position_rad`, `velocity_rad_s`, `torque_nm`, `current_a`, `motor_position_rad`, `status_word`, the drive's position demand `position_demand_rad` and its current demand `current_demand_a` | Position demand as the input; measured position, velocity and torque for scoring |
| `joints.csv` | Joint-space positions and velocities from Controls' `/rt/joint_states` | No |
| `imu.csv` | Controls' `/rt/imu`: orientation from the Madgwick filter, angular velocity and specific force, already rotated into the `base_link` frame, and the relay's sample counter `source_seq` | Orientation, angular velocity and `source_seq`, for moving-base modes; specific force only for opt-in acceleration loading |
| `drives.csv` | Controls' drive status diagnostic aggregate, including winding temperature | No |
| `gains.json` | The gains in force on each drive at the start and at the end of the capture, as SI values and as raw drive counts | Raw P and D gains; actuators listed without SI gains (the hands) are left out |

Every time humanoid-sim reads comes from a `t_source_ros_ns` column: integer nanoseconds on the robot's ROS clock. For the IMU that stamp is the time the host received the sample (see the base modes below). The `t_recv_runtime_ns` columns are on the host's monotonic clock and are never mixed in.

### What the replay uses

Each drive is fed its own position demand, object 0x6062 of the drive's CiA 402 object dictionary ("position demand value"), which the drive reports in every PDO sample (process data object: the cyclic data a drive exchanges with the EtherCAT master). It is the setpoint the drive's position loop worked on, after the master's slew and blend and the drive's own interpolation, in the drive's native frame: the joint angle for the 18 direct-drive joints and the crank angle for the four ankle cranks. Starting from 0x6062 means the replay does not need to model Controls' command path or the ankle inverse kinematics, which a replay of `commands.csv` would.

`ExcitationCapture.pdo()` groups `pdo_feedback.csv` by drive, sorts each drive's rows by `t_source_ros_ns` and drops rows with a repeated stamp. The replay interpolates linearly between the 200 Hz samples (`sample_linear`).

`ExcitationCapture.gains()` reads the P and D gains from `gains.json`, in SI (the raw JVPT counts it also holds are exactly 1000 times these). The replay runs every drive on the PD law at those gains, with zero target velocity and zero feed-forward torque, which is how the capture's commands were sent in [JVPT](glossary.md#jvpt) mode. [drives.md](drives.md) describes the drive model the demands go through.

`imu.csv` provides the base motion for the `yaw` and `orientation` base modes.

### Optional inputs and diagnostics

Ordinary replay uses recorded PDO demands. A fitting client can instead supply a resolved `DriveCommands` stream with an explicit frame/APPLIED clock and latency. ACK absence does not prove a command was discarded. Choosing clocks and converting semantic controller targets belong to the client.

Actual and requested currents remain separate raw telemetry. `--effort-mode raw` records coverage; calibrated comparison requires an explicit sourced current-to-torque artifact and never invents conversion constants. Reported torque is not independent current calibration.

`--base-acceleration imu` optionally applies the equivalent translational inertial load to each body's centre of mass, while the base position remains fixed. It requires full orientation mode, inactive simulated gantry and contact monitoring. It is off by default; this mechanism does not validate measured IMU frames, timing, bias or attitude reconstruction.

Identified incompatible captures require an explicit digest-bound coordinate artifact, or a hash-bound legacy-to-native conversion. These alternatives cannot both be applied. Original capture files stay unchanged. Windowed conversions retain parent provenance and hash their own transformed commands; copied source manifest digests describe the source, not the derived CSVs.

## Recording a capture for the sim

Captures come from the robot API's actuator identification: the `RUN_EXCITATION` action, run under the body lease, and the `IdentificationService` that stores the dataset and serves its archive. The procedure, the profiles, the amplitude limits and the safety checks are described in the Controls repository's `api/actuator-identification.md`. The archive is a ZIP with `<dataset_id>/` at its root; unpack it into `datasets/humanoid-v1/`.

This is what the code here needs from a capture:

- The robot hangs on the gantry with its feet unloaded. The identification profiles require gantry support. The replay keeps the base's position fixed and at most turns it as the IMU recorded, so it assumes that nothing but the gantry holds the robot.
- The gains stay the same for the whole capture. `ExcitationCapture.gains()` refuses a capture whose raw gains at the end differ from those at the start. The excitation never changes gains itself; set non-baseline gains with `SET_ACTUATOR_GAINS` before the run. Every replay runs the solver PD at the capture's own gains (`solver_gains_of` in `src/humanoid_sim/replay/replay.py`), so the target shaping, which is only guaranteed stable up to twice the solver gains (`outside_stable_region` in `src/humanoid_sim/devices/shaping.py`), is not stretched by a capture that runs the drives above their baseline.
- The capture is complete. A run the robot API ramped down early is stored with `complete: false`, and `ExcitationCapture` refuses it; explicit `--partial` fits completed sweeps only.
- The IMU stream has no gaps in `source_seq`. The moving-base modes rebuild the IMU sample times from the sequence and raise on a gap. A fixed-base replay does not read the IMU.
- Every drive in the description has rows in `pdo_feedback.csv` with `position_demand_rad` filled in.
- The drives move. Every score here is a position error, so a drive whose demand hardly changes contributes little but noise. On the present captures two ankle cranks, `maxonHEJ50L_ANK1` and `maxonHEJ50R_ANK2`, move by only 1 to 4 mrad peak to peak, so their numbers say little. Each body region has a low-band profile (`<region>`, 0.2 to 3 Hz over 30 s) and a high-band one (`<region>-high`, 3 to 10 Hz over 45 s); record both. The large excursions at low frequency expose friction, and the high band is where inertia (armature) dominates. `fit-passive` uses a capture from one band to check a fit made on the other.

Tests marked `data` read captures from `$HUMANOID_SIM_DATASETS`, which defaults to `../datasets/humanoid-v1` relative to the repository root, and skip when there are none (see [testing-and-gpu.md](testing-and-gpu.md)).

## Replaying a capture

The moving-base modes run on the asset variant with base joints, a fixed-base replay on the plain asset. Build the one you need first:

```bash
humanoid-sim build-asset --base-joints     # for --base yaw (the default) and --base orientation
humanoid-sim build-asset                   # for --base fixed
humanoid-sim replay ../datasets/humanoid-v1/exc-503fc128aaf14405a0c2418d63ff9a60
humanoid-sim replay <exc-dir> --base fixed --period-ms 1 --duration 10
```

In the Isaac Lab image, run these through `hsim` with the datasets mounted ([getting-started.md](getting-started.md#run-commands-in-the-image)), so that `../datasets/humanoid-v1` resolves inside the container. Physics requires CUDA; CPU use is limited to readers and offline analysis.

The top-level options `--robot` (default `ethrc_v1`), `--assets` (default `outputs/assets`), `--usd` (an explicit robot USD) and `--solver` (`mjwarp` or `kamino`, default the profile's) go before the subcommand. [robot-model.md](robot-model.md) describes the assets.

A replay (`replay()`, `_prepare()` and `_run()` in `src/humanoid_sim/replay/replay.py`) goes like this:

1. The time span runs from the latest first PDO sample of any drive to the earliest last one (`ExcitationCapture.time_span_ns`), cut to `--duration` seconds if given, in steps of `--period-ms`.
2. Every joint starts at the position its drive measured at the start, the ankle joints solved from their two crank angles (`initial_joint_state`), with all velocities zero. With a moving base, the base starts at the first IMU orientation and rate.
3. At each step every drive gets the demand sampled at the step's start (shifted by `--demand-delay-ms`), arriving at that instant, at the recorded raw gains. The base gets its target angles and rates. Physics advances one period.
4. After each step the sim's drive-frame position, velocity, applied torque and demanded torque are recorded and compared with the drive's measurements interpolated to the step's end.

| Option | Default | Meaning |
|---|---|---|
| `<exc-dir>` | required | The capture directory |
| `--period-ms` | 4 | Physics period and replay step, in ms. The profile's `physics.period_ns` is 4 ms. |
| `--integrator` | the profile's, `implicitfast` | MJWarp integrator, for example `euler` |
| `--device` | `cuda:0` | `cuda:N`, or `cpu` for Warp's CPU device, which is slow |
| `--duration` | the whole capture | Replay only the first this many seconds |
| `--base` | `yaw` | `yaw`, `orientation` or `fixed`; see below |
| `--demand-delay-ms` | 0 | Apply each recorded demand this many ms after its stamp; a negative value applies it before. A delay estimate can be obtained through humanoid-learning. |
| `--out` | `outputs/replay` | Parent directory of the result |

### Base modes

On the gantry the base swings, mostly in yaw. On the three present captures the IMU yaw spans 13 to 34 degrees peak to peak and the base turns at up to about 1.8 rad/s.

| `--base` | Asset | Base motion |
|---|---|---|
| `yaw` | built with `--base-joints` | Yaw from the IMU orientation; pitch and roll held at zero |
| `orientation` | built with `--base-joints` | Yaw, pitch and roll from the IMU orientation |
| `fixed` | plain | None: the base link is welded to the world (a fixed base) |

The base-joints asset hangs the robot from a world-fixed root through three revolute joints, `base_yaw`, `base_pitch` and `base_roll` (`R = Rz Ry Rx`). Stiff PD drives on them (`BASE_JOINT_STIFFNESS` and `BASE_JOINT_DAMPING` in `src/humanoid_sim/physics/isaaclab.py`) follow the prescribed angles and rates (`SimCore.set_base`). The base does not translate.

`BaseMotion.from_imu` in `src/humanoid_sim/replay/base.py` derives the motion from `imu.csv`:

- The IMU reaches the host in bursts of five or six samples about every 15 ms, each sample stamped on receipt, so a stamp lags its measurement by up to a burst period. `sample_times_ns` spreads each burst's samples evenly back to the previous burst, the last one at its own stamp.
- The orientation becomes yaw, pitch and roll, and the rates are the Euler-angle rates of the measured angular velocity. The filter has no heading reference, so the first sample's yaw is subtracted.
- The filter's tilt estimate leans with linear acceleration. This is why `yaw`, which keeps the base upright, is the default.

[sensors.md](sensors.md) covers the IMU itself.

### Output

The result directory is `<out>/<dataset_id>_mjwarp_<integrator>_<period in us>us_base-<base>`, with `_demand<delay>ms` appended when a delay is set (for example `exc-503fc128aaf14405a0c2418d63ff9a60_mjwarp_implicitfast_4000us_base-yaw_demand+2ms`). It holds `summary.json` and `trace.npz`. All values are SI.

`summary.json`:

| Key | Meaning |
|---|---|
| `dataset_id`, `name` | The capture's id and display name |
| `period_ns`, `integrator`, `device`, `physics`, `base`, `demand_delay_ns` | The run's settings |
| `wall_s`, `sim_s` | Wall-clock time of the replay and simulated time |
| `drives` | Drive names, in the order of every per-drive array |
| `rms_position_rad` | Per drive: RMS of sim minus measured position |
| `rms_velocity_rad_s` | Per drive: RMS of sim minus measured velocity |
| `rms_torque_nm` | Per drive: RMS of sim applied torque minus the drive's reported torque |
| `tracking_rms_rad` | Per drive: RMS of measured position minus demand |
| `robot`, `description_digest`, `description_revision` | The robot description the sim was built from |

`trace.npz` (read with `numpy.load`), with T samples, one per step:

| Array | Shape | Meaning |
|---|---|---|
| `t_ns` | [T] | Capture time (`t_source_ros_ns`) at the end of each step |
| `demand` | [T, 22] | Position demand applied during each step, rad, drive frame |
| `sim_position`, `sim_velocity` | [T, 22] | Sim drive-frame position (rad) and velocity (rad/s) after each step |
| `sim_applied`, `sim_demanded` | [T, 22] | Sim drive torque after and before the drive's torque limits, Nm |
| `sim_joint_q` | [T, 22] | Sim URDF joint positions, in the description's joint order |
| `sim_base_angles` | [T, 3] | Prescribed base yaw, pitch and roll (zeros on a fixed base) |
| `real_position`, `real_velocity`, `real_torque` | [T, 22] | The drive's measurements interpolated to `t_ns` |

The command also prints one JSON line (`dataset_id`, `period_ns`, `wall_s`, `sim_s`) and, per drive, the position RMS and the tracking RMS in mrad.

`humanoid-sim view-replay <result-dir> --save <file>.rrd` turns a result directory into a [Rerun](glossary.md#rerun) recording with the simulated robot, the real one as a ghost and every drive signal. `--connect <url>` streams it to a running viewer instead; with neither option it spawns a viewer. `--fps` (default 50) sets the frame rate of the 3D views. Every replay recording has the application id `humanoid-sim replay` and the recording name `<display name> (<dataset_id>)`, so a layout saved in the viewer applies to every replay. Screenshots and the views are in [visualization.md](visualization.md#replays-the-sim-next-to-the-real-robot).

### Reading the numbers

Each RMS (root mean square: the square root of the mean squared difference over time) is per drive and leaves out the first second after the start (`ReplayResult.rms`, `skip_s=1.0`), which keeps the start-up transient from the written initial state out of the score. Missing samples are skipped.

- Positions are in each drive's frame. The four `*_ANK*` entries are crank angles; the foot pitch and roll angles are in `sim_joint_q`.
- `tracking_rms_rad` is the error a model would have if every drive followed its demand exactly, which makes it the scale for `rms_position_rad`. A sim error well below it means the sim reproduces how the drive lags and overshoots its demand. A sim error close to it means the sim does little better than assuming perfect tracking. The tracking RMS compares the measurement at each step's end with the demand at its start, so it includes one period of lag.
- `rms_torque_nm` compares the sim's applied torque with the torque the drive estimates from its current. Controls' capture format notes give that estimate a steady-state error below 2 Nm, so differences of that order are within the measurement.
- A drive that barely moved has small numbers on both sides. Compare sim and tracking error only for drives that moved.
- To tell model error from solver error, replay again with `--period-ms 1` or `--integrator euler`. The SkyPilot replay task runs those variants for every capture.

## Fitting tools

Capture acceptance, delay/passive/PACE fitting, held-out reports and fitting Sky
jobs are in [humanoid-learning](https://github.com/ETHRoboticsClub/humanoid-learning),
under `humanoid_learning.sysid`. Run `python -m humanoid_learning.sysid --help`.
This repository keeps capture readers, explicit coordinate conversion, raw window
mechanisms, physical parameter application and drive replay. It never imports
learning or selects fitting bounds/objectives. Original capture archives stay
immutable; converted/windowed data must retain source hashes and derivation identity.

## Optional replay diagnostics

Replay accepts `--capture-coordinates` for explicitly sourced historical zero
translations, `--effort-comparison raw|current` for post-step current inspection,
and `--base orientation --base-acceleration imu --imu-window-ms WINDOW` for
optional equivalent inertial loading. Acceleration defaults off; it is not root
trajectory reconstruction and its real IMU input requires separate validation.

Current mode requires an explicit `ethrc-effort-calibration/v1` artifact bound to
the description digest, current_a, drive-output-equivalent frame, source/status and
loss assumptions, with gain_nm_per_a and zero_a per selected drive. It compares
modeled output-equivalent torque with sim applied torque, never creates measured
sim current or a current-loop model. Raw mode requires no guessed conversion.
Actual and requested currents, timing/coverage and missing samples stay distinct.
Missing telemetry is not filled by torque or a manufactured zero.

Acceleration loading applies -mass*acceleration at physical-body CoMs without
integrating a translation. It requires full orientation, valid synchronized
specific force/gyro, a supported rigid IMU mount, inactive gantry and contact
monitoring. It rejects contacts; it is not a substitute for unknown harness loads.
Signals and conversions retain source provenance. Stepping success does not prove
physical measurement validity.

Numerical replay PD matches the capture's gains at each reset (`solver_gains_of`),
including native crank shafts. Recorded gains are checked for finite positive P,
nonnegative D and coverage of the simulated drives. There is no fixed-envelope
selection option. These are numerical references, not newly fitted robot gains.
