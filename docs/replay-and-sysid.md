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
| `--period-ms` | the profile's | Physics period and replay step, in ms. The profile's `physics.period_ns` is 1 ms for the native V1 and 4 ms for the G1. |
| `--integrator` | the profile's, `rk4` | MJWarp integrator, for example `implicitfast` |
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

The result directory is `<out>/<dataset_id>_mjwarp_<integrator>_<period in us>us_base-<base>`, with `_demand<delay>ms` appended when a delay is set (for example `exc-503fc128aaf14405a0c2418d63ff9a60_mjwarp_rk4_4000us_base-yaw_demand+2ms`). It holds `summary.json` and `trace.npz`. All values are SI.

`summary.json`:

| Key | Meaning |
|---|---|
| `dataset_id`, `name` | The capture's id and display name |
| `period_ns`, `integrator`, `device`, `physics`, `base`, `demand_delay_ns` | The run's settings |
| `physics_overrides` | The `--physics-overrides` mapping merged over the profile's `physics` block, or null |
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
| `sim_motor_position` | [T, 22] | Sim motor side: `sim_position` less each crank's play on an asset built with `--ankle-backlash`, `sim_position` otherwise |
| `real_position`, `real_velocity`, `real_torque` | [T, 22] | The drive's measurements interpolated to `t_ns` |
| `real_motor_position` | [T, 22] | The drive's motor encoder (`motor_position_rad`, its own zero), when the capture has it |

`--physics-overrides` merges a YAML or JSON mapping, inline or from a file, over the profile's `physics` block. It is how a replay opens an asset whose knobs the profile leaves empty. For the [crank backlash](drives.md#crank-backlash) variant, pass `--usd` with that asset and the play's width, a number for every crank or a mapping by crank drive:

```bash
humanoid-sim --solver kamino --usd outputs/assets/ethrc_v1/<key>/robot.usda replay <exc-dir> --base fixed \
    --physics-overrides '{ankle_backlash: {width_rad: 0.008}}'
```

Motor minus output (`*_motor_position - *_position`) against the torque then shows the play the same way on both sides: a step of the play's width where the torque changes sign. The capture's motor encoder has its own zero, so compare steps, not levels.

The command also prints one JSON line (`dataset_id`, `period_ns`, `wall_s`, `sim_s`) and, per drive, the position RMS and the tracking RMS in mrad.

`humanoid-sim view-replay <result-dir> --save <file>.rrd` turns a result directory into a [Rerun](glossary.md#rerun) recording with the simulated robot, the real one as a ghost and every drive signal. `--connect <url>` streams it to a running viewer instead; with neither option it spawns a viewer. `--fps` (default 50) sets the frame rate of the 3D views. Every replay recording has the application id `humanoid-sim replay` and the recording name `<display name> (<dataset_id>)`, so a layout saved in the viewer applies to every replay. Screenshots and the views are in [visualization.md](visualization.md#replays-the-sim-next-to-the-real-robot).

### Reading the numbers

Each RMS (root mean square: the square root of the mean squared difference over time) is per drive and leaves out the first second after the start (`ReplayResult.rms`, `skip_s=1.0`), which keeps the start-up transient from the written initial state out of the score. Missing samples are skipped.

- Positions are in each drive's frame. The four `*_ANK*` entries are crank angles; the foot pitch and roll angles are in `sim_joint_q`.
- `tracking_rms_rad` is the error a model would have if every drive followed its demand exactly, which makes it the scale for `rms_position_rad`. A sim error well below it means the sim reproduces how the drive lags and overshoots its demand. A sim error close to it means the sim does little better than assuming perfect tracking. The tracking RMS compares the measurement at each step's end with the demand at its start, so it includes one period of lag.
- `rms_torque_nm` compares the sim's applied torque with the torque the drive estimates from its current. Controls' capture format notes give that estimate a steady-state error below 2 Nm, so differences of that order are within the measurement.
- A drive that barely moved has small numbers on both sides. Compare sim and tracking error only for drives that moved.
- To tell model error from solver error, replay again with a different `--period-ms` or `--integrator`. The SkyPilot replay task runs those variants for every capture.

## Rosbags

`BagCapture` reads rosbag2 recordings from Controls' robot API recorder (`ros2 bag record --all`; sqlite3 on ROS Humble, MCAP too), in `src/humanoid_sim/replay/bag.py`. It needs the `bag` extra (`rosbags`), which reads bags without ROS. It serves the same interface as `ExcitationCapture`:

| Topic | Type | Used as |
|---|---|---|
| `/ethrc/drive_feedback` | `msg_ethrc/DriveFeedback` | Per drive: `source_stamp_ns`, measured position, velocity and torque, and the 0x6062 position demand (the `pdo_feedback.csv` columns) |
| `/drive_gains/state` | `msg_ethrc/DriveGains` | The gains in force: the latest state at or before the feedback starts and every state during it, all applied, none ramping or failed, all equal |
| `/rt/body_control` | `msg_ethrc/BodyControlFrame` | COMMAND frames to the Joint Host, stamped when the loop host sent them (the `commands.csv` rows) |
| `/rt/imu` | `sensor_msgs/Imu` | The base motion for the moving-base modes |

Every time comes from the messages' own stamps, never from the bag's receipt times. Humble's sqlite3 bags carry no message definitions, so the reader uses the copies in `src/humanoid_sim/replay/msgs/`, kept byte-identical to Controls' `ros2_ws/src/contracts/msg_ethrc/msg`, unless the bag holds its own. A bag has no `complete` flag and no record of the gantry: that the robot hung on the gantry and ran an excitation is for whoever passes the bag to know. `sensor_msgs/Imu` has no sample counter, so a dropped IMU sample goes unnoticed. The capture id is `bag-` and 16 hex digits of the sha256 over the bag's files.

## Robot regression

`humanoid-sim regress` replays recorded robot bags on the plant and checks the sim against the robot, metric by metric, against pass bands that are versioned with the code. It is the check that the plant behaves like the robot on the signals that matter for control: a policy that passes every training eval can still shake on the robot, and this is where that should show first.

```bash
humanoid-sim build-asset --base-joints
humanoid-sim regress --recordings ../../data/recordings                 # every case, exit 1 if any fails
humanoid-sim regress --case c10-1.4hz-limit-cycle --out outputs/regress # one case
humanoid-sim regress --propose --recordings ../../data/recordings       # reset the bands from the robot (no sim)
humanoid-sim replay <bag-dir> --window 63 99                            # one bag window, the same metrics, no bands
```

`regress` needs the GPU image like `replay`; `--propose` and the tests need only the `bag` extra. `--recordings` (default `$HUMANOID_SIM_RECORDINGS`, else `../../data/recordings`) is the directory the cases' bag paths are under; each bag's sha256 is checked before it is read. The task `sky/regress.sky.yaml` runs it on an L40S, reading the bags from S3 through the `/ckpt` mount.

### What a case holds

`regression/cases.yaml` (format `humanoid-sim-regression/v1`) holds one entry per case:

| Key | Meaning |
|---|---|
| `id`, `description` | Name and what the robot was doing |
| `bag` | `s3`: where the bag lives (`s3://ethrc-rc-humanoid-ckpts/amoec/recordings/<path>`, the same path as under `data/recordings/`); `sha256`: the bag's contents (`BagCapture.sha256`); `path`: under the recordings directory |
| `support` | How the robot was carried: `floor, rope slack`, `gantry hang`, ... |
| `scoring` | `open_loop` where a held or hung base represents the recording (a gantry hang, an excitation); `closed_loop` where it does not (the robot stood on the floor with the rope slack): such a case is not scored by a replay with a prescribed base and reports `NOT SCORED` |
| `window_s` | Start and end in seconds after the bag's start (`metadata.yaml` `starting_time`, the zero the recording logs use) |
| `warmup_s` | Seconds replayed before the window so the start-up transient is not scored (0 when the gains were applied only just before the window) |
| `bands` | Per metric `{ref, lo, hi}`: `ref` is the robot's value when the band was set, `lo` and `hi` bound the sim's value |
| `sources` | Where each family of bands comes from: the robot's mean over the case's windows, with the tolerance |
| `tolerances` | The file-wide tolerances the bands were set with |

A band applies to the mean over the case's 2 s windows of the sim's value. Phase bands are circular, in degrees. The bands are set from what the robot shows, with these tolerances: tracking error 25 % plus 0.5 mrad; for open-loop cases only, the sim-minus-robot position error at most half the robot's own tracking error (plus 0.5 mrad), so the sim explains at least 75 % of its tracking-error power; response gain 30 %, response phase 20 degrees (the pitch fit found the sim's phase 80 to 90 degrees off, and timing errors of 20 to 27 degrees are what the alignment below removes); IMU sway 30 % plus 0.05 degrees; pitch-rate RMS 30 % plus 0.005 rad/s. The tolerances are provisional and say how far the sim may be from the robot's own window-to-window spread before it counts as a different plant; tighten them as the plant improves. `regress --propose` rewrites the bands from the bag and the tolerances, so a change to a tolerance, a window or a metric is one command and one reviewed diff. A robot value that no longer equals its `ref` (another bag, window or metric definition) makes the case report `stale` and fail rather than judge the sim against bands that describe something else. A metric the robot gives on fewer than three windows gets no band.

The first cases are three robot recordings, all made with the robot standing on the floor and the gantry rope slack, so all three are `closed_loop`: `v11-11400-quiet-stand` (the v11 policy standing quietly, 2026-11-16 gantry walk rep-4), `v15-12900-growing-shake` (the v15 policy's 3 to 6 Hz shake growing until the drives stopped, 2026-11-17 rep-3) and `c10-1.4hz-limit-cycle` (the c10 policy's steady 1.4 Hz sagittal limit cycle, 2026-11-17 rep-4). Their bands are the robot's values a closed-loop run is held to (the recorded policy on the recorded command timeline, free floor, no rope, the 5.3 ms command path, run by humanoid-learning's closed-loop runner); a replay with a prescribed base only reports the sim's values beside them. Add a case by adding its bag, window and description and running `regress --propose`, and upload the bag to the S3 path it names.

### How a bag is replayed

`replay` of a rosbag2 directory (`BagCapture`) feeds each drive the bag's COMMAND frames from `/rt/body_control`, converted to drive-frame targets (the targets in semantic radians plus the standing offsets, through the ankle transmission for the cranks; they agree with the drives' own 0x6062 demand to 1 to 9 mrad RMS), each at its frame stamp and taking effect `devices.command_path.frame_to_drive_ns` later (5.3 ms for the V1, measured on the 2026-11-16 captures: the controller path plus the drive's own 5 ms). The robot's recorded drive demand is not used. The gains come from `/drive_gains/state`, judged over the replayed span on the description's drives. The base follows the IMU's orientation (`--base orientation`, the default for a bag) on the asset built with base joints. `--window` and `--warmup` choose the span; without a window the whole bag after its first complete frame is replayed.

Signals are compared on one 200 Hz grid by their own source stamps (`replay/grid.py`). Drive position and torque and the IMU are linearly interpolated between their samples (the IMU on its receipt stamps, so up to a 15 ms burst late; the same for every bag), and a target is held from its frame's stamp, the instant the loop host sent it, until the next frame. Targets are never moved by the command latency: that belongs to the plant that answers them. The sim's step-end states are interpolated on the same grid and answer the same held target. Pairing each policy tick's new target with a gyro sample taken 7 ms before it skews phase by 20 to 27 degrees at 3 to 12 Hz, which is the size of the effect the suite is looking for.

### The metrics

Every metric is computed per 2 s window (every 1 s) and reported as mean, std, min and max over the windows, for the robot and the sim on the windows both have a value for (`summary.json`, `metrics`):

| Metric | Meaning |
|---|---|
| `tracking_rms.<drive>` | RMS of the drive's position minus the target in force, with the window's mean removed, rad |
| `tracking_offset.<drive>` | That mean, rad (information, no band) |
| `sim_real_rms.<drive>` | RMS of sim minus robot position, mean removed, rad (sim only) |
| `hip_q_over_target.<band>.gain`, `.phase_deg` | Symmetric hip flexion position over its target |
| `hip_tau_over_target.<band>.gain`, `.phase_deg` | Symmetric hip flexion torque over its target, N m per rad |
| `pitch_rate_over_hip_target.<band>`, `pitch_rate_over_hip_tau.<band>` | Torso pitch rate (IMU gyro y) over the symmetric hip flexion target and torque |
| `imu.pitch_sway_deg`, `imu.roll_sway_deg` | Std of the IMU pitch and roll angle in the window |
| `imu.pitch_rate_rms.<band>` | RMS of the pitch rate in the band, about its mean, rad/s |

The response bands are 1 to 3, 3 to 6 and 6 to 12 Hz; the pitch-rate bands 0.5 to 3, 3 to 6 and 6 to 12 Hz. The symmetric hip flexion is the left joint minus the right one, over two (the joints are mirrored). A response is estimated in each window from Hann-windowed spectra pooled over the band, `H = sum(conj(U) Y) / sum(|U|^2)`, phase as the angle of output over input (negative when the output lags). A window whose input is below 0.2 mrad (0.05 N m for torque) in the band, or whose coherence is below 0.5, is left out, and a band's phase statistics are circular. The metrics are in `replay/metrics.py`.

### What it can and cannot show yet

The base is prescribed from the robot's IMU and does not translate, so the replay is open loop with the base held. What it can show: how each drive follows the robot's own targets (lag, gain, overshoot, the command path), and how the hip drive and the leg answer a target with the torso given. That represents a robot hanging from the gantry or an excitation capture, and no other. What it cannot show:

- A robot standing on the floor. The sim robot hangs from its base and carries only its own legs, while the floor robot carries its weight: its static deflections (tens of mrad at the hips and ankles, 20 N m of hip abduction torque) are absent, its legs swing free instead of rocking the whole body, and an open-loop replay on a free floor falls within 1.4 to 2.3 s. A replay on the old default rope was no better, since that rope carried 46 to 54 % of the weight. For the same reason a floor case has no bands on the held-base metrics: it needs a closed-loop run, which is a later slice.
- The torso's own motion. With the base prescribed the sim's pitch rate is the robot's, so the IMU and pitch-rate families have only the robot's value. The hip-command-to-pitch-rate response that failed on the robot on 2026-11-17 needs the closed loop (or at least a free base); until then those metrics are reported `SKIP`, not `PASS`.
- Foot contact, the policy's observations and what a changed plant does to a policy. Nothing here runs the policy.

A `closed_loop` case is not scored by `regress` as it stands: it reports `NOT SCORED`, with the sim's held-base values beside the robot's for information, and does not fail the run. The closed-loop runner lives in humanoid-learning; the next slice calls it on these cases rather than building a second one. Open-loop cases (a gantry hang, an excitation, the hanging condition of the tap test) are scored as above; none is pinned yet.

The same cases are meant to be replayed on the Controls MuJoCo plant so that both plants are judged on the same data; that is not wired yet.

## Fitting

Fitting the demand delay, passive joint parameters and PACE's actuator model, and the capture checks that go with them, live in humanoid-learning (`python -m humanoid_learning.sysid --help`). humanoid-sim has no fitting commands, no `sysid` package and no fitting SkyPilot tasks; it supplies the plant, the replay and the capture and bag readers the fitters use.

## Running on a GPU with SkyPilot

`sky/replay.sky.yaml` runs the replays on one L40S on AWS (8 or more CPUs, 32 GB or more of memory, 200 GB of disk). `sky/setup.sh` installs Isaac Lab release 3.0.0 at `ISAACLAB_COMMIT` and humanoid-sim in editable mode. The repository is synced to `~/sky_workdir` on the node, and the three captures are mounted from `../datasets/humanoid-v1/` (beside the repository) at `/data/exc-<id>`. To use another capture, add it to `file_mounts`.

`replay.sky.yaml` builds both assets and replays every mounted capture once for each `period_ms:integrator` pair in `RUNS` (`4:rk4 4:implicitfast 1:rk4`), with the default `yaw` base and no demand delay, one process per capture in parallel. Results go to `outputs/replay-mjwarp/`, with a log per capture and run; the task ends by printing each run's JSON summary line.

```bash
sky launch -y -c hsim-bench -i 10 sky/replay.sky.yaml
rsync -avz hsim-bench:sky_workdir/outputs/replay-mjwarp/ outputs/replay-mjwarp/
```

`-i 10` stops the cluster after 10 idle minutes without tearing it down. The task uses the cluster name `hsim-bench`, as the benchmark task does.
