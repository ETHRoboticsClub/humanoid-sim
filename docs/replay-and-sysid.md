# Replay and system identification

humanoid-sim can replay an excitation capture recorded on the robot at the drive level and fit model parameters
to it. During a replay each of the 22 simulated drives receives the position demand the real drive acted on, and
the simulated drive feedback is compared with what the real drive measured. Three commands are built on this:

| Command | What it does | Code |
|---|---|---|
| `humanoid-sim replay` | Replays one capture and scores the sim against it | `src/humanoid_sim/replay/replay.py` |
| `humanoid-sim fit-delay` | Finds the timing offset between the recorded demands and the drives' response | `src/humanoid_sim/sysid/delay.py` |
| `humanoid-sim fit-passive` | Fits upper-body joint damping, friction and armature | `src/humanoid_sim/sysid/passive.py` |

All three step physics on Isaac Lab / MJWarp, so they run in the Isaac Lab image or on a GPU node (see
[getting-started.md](getting-started.md) and [testing-and-gpu.md](testing-and-gpu.md)). The capture reader,
`src/humanoid_sim/replay/exc.py`, needs only NumPy and can be used anywhere.

## Excitation captures

An excitation capture is a dataset the Controls robot API records while it swings the robot's joints through a
frequency sweep (a chirp) with the robot hanging on the [gantry](glossary.md#gantry). Each capture is a directory
named after its `dataset_id`, `exc-<id>`. They are kept outside this repository, in `datasets/humanoid-v1/` beside
the humanoid-sim checkout. `ExcitationCapture` reads format `ethrc-excitation/v8` only and raises on any other.

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
| `commands.csv` | The joint-space targets the robot API published, one frame every 20 ms (`q_nominal_rad` and `q_mirror_rad` per joint) | No |
| `acks.csv` | The receivers' acknowledgements of each command frame | No |
| `pdo_feedback.csv` | Per drive at about 200 Hz: measured position `native_position_rad`, `velocity_rad_s`, `torque_nm`, `current_a`, `motor_position_rad`, `status_word`, the drive's position demand `position_demand_rad` and its current demand `current_demand_a` | Position demand as the input; measured position, velocity and torque for scoring |
| `joints.csv` | Joint-space positions and velocities from Controls' `/rt/joint_states` | No |
| `imu.csv` | Controls' `/rt/imu`: orientation from the Madgwick filter, angular velocity and specific force, already rotated into the `base_link` frame, and the relay's sample counter `source_seq` | Orientation, angular velocity and `source_seq`, for the moving-base modes |
| `drives.csv` | Controls' drive status diagnostic aggregate, including winding temperature | No |
| `gains.json` | The gains in force on each drive at the start and at the end of the capture, as SI values and as raw drive counts | Raw P and D gains |

Every time humanoid-sim reads comes from a `t_source_ros_ns` column: integer nanoseconds on the robot's ROS clock.
For the IMU that stamp is the time the host received the sample (see the base modes below). The
`t_recv_runtime_ns` columns are on the host's monotonic clock and are never mixed in.

### What the replay uses

Each drive is fed its own position demand, object 0x6062 of the drive's CiA 402 object dictionary ("position
demand value"), which the drive reports in every PDO sample (process data object: the cyclic data a drive
exchanges with the EtherCAT master). It is the setpoint the drive's position loop worked on, after the master's
slew and blend and the drive's own interpolation, in the drive's native frame: the joint angle for the 18
direct-drive joints and the crank angle for the four ankle cranks. Starting from 0x6062 means the replay does not
need to model Controls' command path or the ankle inverse kinematics, which a replay of `commands.csv` would.

`ExcitationCapture.pdo()` groups `pdo_feedback.csv` by drive, sorts each drive's rows by `t_source_ros_ns` and
drops rows with a repeated stamp. The replay interpolates linearly between the 200 Hz samples (`sample_linear`).

`ExcitationCapture.gains()` reads the raw P and D gains from `gains.json`. The replay runs every drive in JVPT
mode, the drive's PD position mode (operating mode -64, `MODE_JVPT` in `src/humanoid_sim/devices/demand.py`), at
those raw gains, with zero target velocity and zero feed-forward torque, which is how the capture's commands were
sent. [drives.md](drives.md) describes the drive model the demands go through.

`imu.csv` provides the base motion for the `yaw` and `orientation` base modes.

### What the replay ignores

`commands.csv`, `acks.csv`, `joints.csv` and `drives.csv` are never read. From `pdo_feedback.csv`, `current_a`,
`current_demand_a` and `motor_position_rad` are loaded but not used, and `status_word` and the publish and receipt
stamps are not read. The IMU's specific force is not used, so the base's linear motion is not replayed. The reader
does not verify the manifest's digests, and nothing compares the capture's identity and calibration revisions with
the robot description in `third_party/humanoid-urdf`.

## Recording a capture for the sim

Captures come from the robot API's actuator identification: the `RUN_EXCITATION` action, run under the body lease,
and the `IdentificationService` that stores the dataset and serves its archive. The procedure, the profiles, the
amplitude limits and the safety checks are described in the Controls repository's `api/actuator-identification.md`.
The archive is a ZIP with `<dataset_id>/` at its root; unpack it into `datasets/humanoid-v1/`.

This is what the code here needs from a capture:

- The robot hangs on the gantry with its feet unloaded. The identification profiles require gantry support. The
  replay keeps the base's position fixed and at most turns it as the IMU recorded, so it assumes that nothing but
  the gantry holds the robot.
- The gains stay the same for the whole capture. `ExcitationCapture.gains()` refuses a capture whose raw gains at
  the end differ from those at the start. The excitation never changes gains itself; set non-baseline gains with
  `SET_ACTUATOR_GAINS` before the run. The sim's target shaping is only guaranteed stable up to twice the
  description's baseline gains (`outside_stable_region` in `src/humanoid_sim/devices/shaping.py`), and the replay
  does not report a capture that goes beyond that. The present captures run the baseline gains, which
  `tests/test_exc_reader.py` checks.
- The capture is complete. A run the robot API ramped down early is stored with `complete: false`, and
  `ExcitationCapture` refuses it.
- The IMU stream has no gaps in `source_seq`. The moving-base modes rebuild the IMU sample times from the sequence
  and raise on a gap. A fixed-base replay does not read the IMU.
- Every drive in the description has rows in `pdo_feedback.csv` with `position_demand_rad` filled in.
- The drives move. Every score here is a position error, so a drive whose demand hardly changes contributes little
  but noise. On the present captures two ankle cranks, `maxonHEJ50L_ANK1` and `maxonHEJ50R_ANK2`, move by only 1
  to 4 mrad peak to peak, so their numbers say little. Each body region has a low-band profile (`<region>`, 0.2 to
  3 Hz over 30 s) and a high-band one (`<region>-high`, 3 to 10 Hz over 45 s); record both. The large excursions at
  low frequency expose friction, and the high band is where inertia (armature) dominates. `fit-passive` uses a
  capture from one band to check a fit made on the other.

Tests marked `data` read captures from `$HUMANOID_SIM_DATASETS`, which defaults to `../datasets/humanoid-v1`
relative to the repository root, and skip when there are none (see [testing-and-gpu.md](testing-and-gpu.md)).

## Replaying a capture

The moving-base modes run on the asset variant with base joints, a fixed-base replay on the plain asset. Build the
one you need first:

```bash
humanoid-sim build-asset --base-joints     # for --base yaw (the default) and --base orientation
humanoid-sim build-asset                   # for --base fixed
humanoid-sim replay ../datasets/humanoid-v1/exc-503fc128aaf14405a0c2418d63ff9a60
humanoid-sim replay <exc-dir> --base fixed --period-ms 1 --duration 10
```

In the Isaac Lab image, run these through `hsim` with the datasets mounted
([getting-started.md](getting-started.md#run-commands-in-the-image)), so that `../datasets/humanoid-v1` resolves
inside the container. On a machine without CUDA add `--device cpu`.

The top-level options `--robot` (default `ethrc_v1`), `--assets` (default `outputs/assets`) and `--usd` (an
explicit robot USD) go before the subcommand. [robot-model.md](robot-model.md) describes the assets.

A replay (`replay()`, `_prepare()` and `_run()` in `src/humanoid_sim/replay/replay.py`) goes like this:

1. The time span runs from the latest first PDO sample of any drive to the earliest last one
   (`ExcitationCapture.time_span_ns`), cut to `--duration` seconds if given, in steps of `--period-ms`.
2. Every joint starts at the position its drive measured at the start, the ankle joints solved from their two crank
   angles (`initial_joint_state`), with all velocities zero. With a moving base, the base starts at the first IMU
   orientation and rate.
3. At each step every drive gets the demand sampled at the step's start (shifted by `--demand-delay-ms`), arriving
   at that instant, at the recorded raw gains. The base gets its target angles and rates. Physics advances one
   period.
4. After each step the sim's drive-frame position, velocity, applied torque and demanded torque are recorded and
   compared with the drive's measurements interpolated to the step's end.

| Option | Default | Meaning |
|---|---|---|
| `<exc-dir>` | required | The capture directory |
| `--period-ms` | 4 | Physics period and replay step, in ms. The profile's `physics.period_ns` is 4 ms. |
| `--integrator` | the profile's, `implicitfast` | MJWarp integrator, for example `euler` |
| `--device` | `cuda:0` | `cuda:N`, or `cpu` for Warp's CPU device, which is slow |
| `--duration` | the whole capture | Replay only the first this many seconds |
| `--base` | `yaw` | `yaw`, `orientation` or `fixed`; see below |
| `--demand-delay-ms` | 0 | Apply each recorded demand this many ms after its stamp; a negative value applies it before. Take it from `fit-delay`. |
| `--out` | `outputs/replay` | Parent directory of the result |

### Base modes

On the gantry the base swings, mostly in yaw. On the three present captures the IMU yaw spans 13 to 34 degrees
peak to peak and the base turns at up to about 1.8 rad/s.

| `--base` | Asset | Base motion |
|---|---|---|
| `yaw` | built with `--base-joints` | Yaw from the IMU orientation; pitch and roll held at zero |
| `orientation` | built with `--base-joints` | Yaw, pitch and roll from the IMU orientation |
| `fixed` | plain | None: the base link is welded to the world (a fixed base) |

The base-joints asset hangs the robot from a world-fixed root through three revolute joints, `base_yaw`,
`base_pitch` and `base_roll` (`R = Rz Ry Rx`). Stiff PD drives on them (`BASE_JOINT_STIFFNESS` and
`BASE_JOINT_DAMPING` in `src/humanoid_sim/physics/isaaclab.py`) follow the prescribed angles and rates
(`SimCore.set_base`). The base does not translate.

`BaseMotion.from_imu` in `src/humanoid_sim/replay/base.py` derives the motion from `imu.csv`:

- The IMU reaches the host in bursts of five or six samples about every 15 ms, each sample stamped on receipt, so a
  stamp lags its measurement by up to a burst period. `sample_times_ns` spreads each burst's samples evenly back to
  the previous burst, the last one at its own stamp.
- The orientation becomes yaw, pitch and roll, and the rates are the Euler-angle rates of the measured angular
  velocity. The filter has no heading reference, so the first sample's yaw is subtracted.
- The filter's tilt estimate leans with linear acceleration. This is why `yaw`, which keeps the base upright, is
  the default.

[sensors.md](sensors.md) covers the IMU itself.

### Output

The result directory is `<out>/<dataset_id>_mjwarp_<integrator>_<period in us>us_base-<base>`, with
`_demand<delay>ms` appended when a delay is set (for example
`exc-503fc128aaf14405a0c2418d63ff9a60_mjwarp_implicitfast_4000us_base-yaw_demand+2ms`). It holds `summary.json`
and `trace.npz`. All values are SI.

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

The command also prints one JSON line (`dataset_id`, `period_ns`, `wall_s`, `sim_s`) and, per drive, the position
RMS and the tracking RMS in mrad.

`humanoid-sim view-replay <result-dir> --save <file>.rrd` turns a result directory into a
[Rerun](glossary.md#rerun) recording with the simulated robot, the real one as a ghost and every drive signal.
`--connect <url>` streams it to a running viewer instead; with neither option it spawns a viewer. `--fps` (default 50) sets the frame rate of the 3D views. Every replay recording has the application id
`humanoid-sim replay` and the recording name `<display name> (<dataset_id>)`, so a layout saved in the viewer
applies to every replay. Screenshots and the views are in
[visualization.md](visualization.md#replays-the-sim-next-to-the-real-robot).

### Reading the numbers

Each RMS (root mean square: the square root of the mean squared difference over time) is per drive and leaves out
the first second after the start (`ReplayResult.rms`, `skip_s=1.0`), which keeps the start-up transient from the
written initial state out of the score. Missing samples are skipped.

- Positions are in each drive's frame. The four `*_ANK*` entries are crank angles; the foot pitch and roll angles
  are in `sim_joint_q`.
- `tracking_rms_rad` is the error a model would have if every drive followed its demand exactly, which makes it the
  scale for `rms_position_rad`. A sim error well below it means the sim reproduces how the drive lags and overshoots
  its demand. A sim error close to it means the sim does little better than assuming perfect tracking. The tracking
  RMS compares the measurement at each step's end with the demand at its start, so it includes one period of lag.
- `rms_torque_nm` compares the sim's applied torque with the torque the drive estimates from its current. Controls'
  capture format notes give that estimate a steady-state error below 2 Nm, so differences of that order are
  within the measurement.
- A drive that barely moved has small numbers on both sides. Compare sim and tracking error only for drives that
  moved.
- To tell model error from solver error, replay again with `--period-ms 1` or `--integrator euler`. The SkyPilot
  replay task runs those variants for every capture.

## Fitting the demand delay

```bash
humanoid-sim fit-delay <exc-dir>... [--delays-ms FROM TO STEP]
```

The demand and the measured position in a PDO sample share one stamp, but the drive may act on the demand earlier
or later than that stamp, by its PDO, bus and interpolation timing. `scan_delay` in `src/humanoid_sim/sysid/delay.py`
replays the captures as one batch with an environment per candidate delay: the environment for delay d applies the
demand recorded at t at t + d, the same shift `--demand-delay-ms` applies (`replay_position_rms`). It computes every
drive's position RMS in every environment after the first second and averages it over the captures.

| Option | Default | Meaning |
|---|---|---|
| `<exc-dir>...` | required | One or more captures |
| `--delays-ms FROM TO STEP` | `-10 10 0.5` | The delays scanned, in ms, both ends included. The default is 41 delays, so 41 environments. |
| `--duration` | the whole capture | Replay only the first this many seconds of each capture |
| `--base` | `yaw` | As for `replay` |
| `--device` | `cuda:0` | As for `replay` |
| `--out` | `outputs/sysid` | Directory for `fit-delay.json` |

It runs at the profile's period and integrator. The result, `<out>/fit-delay.json`:

| Key | Meaning |
|---|---|
| `delays_ns`, `drives` | The scanned delays and the drive order |
| `rms_rad` | [delay][drive] position RMS, averaged over the captures |
| `per_capture_rms_rad` | [capture][delay][drive] position RMS |
| `best_delay_ns` | Per drive, the delay with the lowest RMS |
| `regions` | Per region: `best_delay_ns`, `rms_best_rad` and `rms_at_zero_rad` (at the scanned delay closest to zero), from the region's mean RMS over its drives |
| `captures`, `base`, `wall_s`, `robot`, `description_digest`, `description_revision` | Run settings and the description |

The regions come from `drive_regions`: the description's `region` of the drive's joint (`leg` or `upper_body`;
`hip_central_rotation` counts as `upper_body`), and `ankle` for the four cranks. The command prints `regions`.

`replay` and `fit-passive` take a single delay for all drives, `--demand-delay-ms`, in ms (`best_delay_ns` divided
by 1e6). Use the region you are working on, for `fit-passive` the upper body. Compare `rms_best_rad` with
`rms_at_zero_rad` to see whether the delay matters at all, and check that the best delay is not at either end of
the scan; if it is, widen `--delays-ms`. The scan minimizes position error, so it also absorbs phase errors of the
model, such as wrong damping. Run it again after the passive parameters in the description change.

## Fitting passive joint parameters

```bash
humanoid-sim fit-passive <exc-dir>... --holdout <exc-dir>... [--num-envs 256] [--generations 15]
```

`fit-passive` fits viscous damping, Coulomb friction and armature of the upper-body joints by batched replay. The
joint groups are `UPPER_BODY_GROUPS` in `src/humanoid_sim/cli.py`; the joints of a group share one value, so left
and right are fitted together:

| Group | Joints |
|---|---|
| `waist` | `waist_upperbody_yaw` |
| `shoulder_flexion` | `shoulder_flexion_l`, `shoulder_flexion_r` |
| `shoulder_abduction` | `shoulder_abduction_l`, `shoulder_abduction_r` |
| `shoulder_rotation` | `shoulder_rotation_l`, `shoulder_rotation_r` |
| `elbow` | `elbow_flexion_l`, `elbow_flexion_r` |

Legs, ankles and `hip_central_rotation` are not fitted. The parameters, written to the solver per environment by
`_write` in `src/humanoid_sim/sysid/passive.py`:

| Parameter | Meaning | Written with |
|---|---|---|
| `damping` | Viscous joint damping, Nm s/rad | `write_joint_viscous_friction_coefficient_to_sim_index` |
| `friction` | Coulomb joint friction, Nm | `write_joint_friction_coefficient_to_sim_index` |
| `armature` | Reflected rotor inertia at the joint, kg m² | `write_joint_armature_to_sim_index` |

### How the search works

A candidate is a vector of log multipliers, one per group and fitted parameter. A joint's value is its start value
times `exp(x)`, with the multiplier clipped to `[1/bound, bound]`. The start value is the description's
(`third_party/humanoid-urdf/ethrc_v1/actuators.yaml`) unless `--start` sets one. Parameters left out of `--params`
stay at their start values.

The optimizer is CMA-ES (covariance matrix adaptation evolution strategy, the `Cmaes` class), a derivative-free
method: each generation draws a population of candidates from a multivariate normal distribution, ranks them by the
objective, moves the distribution's mean toward the better half and adapts its shape and step size. Here every
candidate is one environment of a single batched replay. The population is `--num-envs` minus one, because
environment 0 re-runs the best candidate found so far. The search starts at the start values (all
multipliers 1) with a step size of 0.7 in log units. A candidate outside the bounds is evaluated at the bound and
penalized by 1e-3 rad per squared log unit beyond it (`BOUND_PENALTY`), so the search does not drift where the
objective is flat.

The objective (`evaluate`) is the position RMS after the first second, averaged over the fitted captures and the
scored drives, which are the drives of the fitted joints. Generation 0 evaluates the start values; each generation
replays every fitted capture once.

After the search, each `--holdout` capture is replayed in full (`--duration` does not apply) once with the
start values and once with the fit. Use holdout captures from a different excitation profile than the
fitted ones.

### Options

| Option | Default | Meaning |
|---|---|---|
| `<exc-dir>...` | required | Captures to fit on |
| `--holdout` | none | Captures to score the fit on |
| `--num-envs` | 256 | Environments: the CMA-ES population plus one |
| `--generations` | 15 | CMA-ES generations |
| `--duration` | the whole capture | Replay only the first this many seconds of each fitted capture |
| `--groups` | all five | Groups to fit |
| `--params` | `damping friction armature` | Parameters to fit; the others stay at their start values |
| `--start` | none | `GROUP.PARAM=VALUE` start values for every joint of a group, instead of the description's, for example `elbow.armature=0.0055` |
| `--bound` | 10 | Multipliers stay within `[1/bound, bound]` |
| `--base` | `yaw` | As for `replay` |
| `--demand-delay-ms` | 0 | As for `replay` |
| `--device` | `cuda:0` | As for `replay` |
| `--seed` | 0 | CMA-ES random seed |
| `--out` | `outputs/sysid` | Directory for `fit-passive.json`; a later run with the same `--out` overwrites it |

Like `fit-delay`, it runs at the profile's period and integrator.

### Output

`<out>/fit-passive.json`:

| Key | Meaning |
|---|---|
| `names`, `x`, `multipliers` | Per group parameter (`<group>.<param>`): the best candidate's log multiplier and multiplier |
| `values` | Per joint: the resulting `damping`, `friction` and `armature` (start value times multiplier; the start value for parameters not fitted) |
| `objective`, `objective_description` | Best objective, and the objective at the start values, rad. The start values are the description's unless `--start` is given; the `_description` keys keep that name either way. |
| `history` | Per generation: the best objective so far, the population's median and the CMA-ES step size (generation 0, the start values, has no step size) |
| `rms_description`, `rms_best` | [capture][drive] position RMS on the fitted captures at the start values and at the best values |
| `holdout` | The holdout captures, `objective_description`, `objective_fit`, and `rms_description` and `rms_fit` per capture and drive |
| `captures`, `scored`, `drives`, `params`, `start`, `bound`, `base`, `num_envs`, `demand_delay_ns`, `wall_s` | Run settings |
| `robot`, `description_digest`, `description_revision` | The description the fit started from |

The command prints `gen 0 description values: ... mrad` for the start values, then a line per generation
(`gen N: best ... mrad, median ..., sigma ...`), a JSON line with the objectives, names and multipliers, and, with
holdouts, the holdout objective at the start values and with the fit, in rad.

A fit is worth keeping when `holdout.objective_fit` is clearly below `holdout.objective_description`. An improvement
on the fitted capture alone can mean the parameters absorbed something specific to that capture. A multiplier at
`1/bound` or `bound` means the optimum lies outside the search range or that the parameter is compensating for
something the model lacks; treat that value as unverified. Compare `rms_fit` with `rms_description` per drive to
see which joints improved.

### From a fit to the robot description

`fit-passive` writes nothing back. The passive parameters are robot facts, and robot facts live in humanoid-urdf:
the per-joint `damping`, `friction` and `armature` in
`third_party/humanoid-urdf/ethrc_v1/actuators.yaml`, each with its provenance in `damping_friction_source` or
`armature_source`. To adopt a fit:

1. Change the values in humanoid-urdf's `actuators.yaml` to those under `values` in `fit-passive.json`, and set the
   source fields the way the existing entries do (tags of the form `sysid-<date>`). Record which captures were
   fitted and held out, and the `description_revision` the fit started from.
2. humanoid-sim keeps `third_party/humanoid-urdf` at the commit Controls pins, so the change reaches humanoid-sim
   when Controls moves its pin and the submodule follows.
3. Rebuild the assets. Assets are keyed by the description's digest, and the CLI refuses an asset built from a
   different description.

`--start` values are inputs to a single run. They are not a place to keep robot constants, which this repository
never holds.

## Running on a GPU with SkyPilot

`sky/replay.sky.yaml` and `sky/sysid.sky.yaml` run the replays and the passive fit on one L40S on AWS (8 or more
CPUs, 32 GB or more of memory, 200 GB of disk). `sky/setup.sh` installs Isaac Lab release 3.0.0 at
`ISAACLAB_COMMIT` and humanoid-sim in editable mode. The repository is synced to `~/sky_workdir` on the node, and
the three captures are mounted from `../datasets/humanoid-v1/` (beside the repository) at `/data/exc-<id>`. To use
another capture, add it to `file_mounts`.

`replay.sky.yaml` builds both assets and replays every mounted capture once for each `period_ms:integrator` pair in
`RUNS` (`4:implicitfast 4:euler 1:implicitfast`), with the default `yaw` base and no demand delay, one process per
capture in parallel. Results go to `outputs/replay-mjwarp/`, with a log per capture and run; the task ends by
printing each run's JSON summary line.

```bash
sky launch -y -c hsim-bench -i 10 sky/replay.sky.yaml
rsync -avz hsim-bench:sky_workdir/outputs/replay-mjwarp/ outputs/replay-mjwarp/
```

`sysid.sky.yaml` builds the base-joints asset and runs two fits across the frequency bands, with `NUM_ENVS` 512,
`GENERATIONS` 20, `DURATION` 20 s on the fitted capture and `BOUND` 100:

| Run | Fitted capture | Holdout |
|---|---|---|
| `fit-on-low` | `exc-503fc128aaf14405a0c2418d63ff9a60` (low band) | both high-band captures |
| `fit-on-high` | `exc-3a498000fc114f2db2b0655d5d87280c` (high band) | the low-band capture |

The two high-band captures run the same profile, so one would not validate a fit on the other. Each run writes
`outputs/sysid/<run>/fit-passive.json` and `fit-passive.log`.

```bash
sky launch -y -c hsim-bench -i 10 sky/sysid.sky.yaml
rsync -avz hsim-bench:sky_workdir/outputs/sysid/ outputs/sysid/
```

`-i 10` stops the cluster after 10 idle minutes without tearing it down. Both tasks use the cluster name
`hsim-bench`, as the benchmark task does. `fit-delay` has no task of its own; run it in the Isaac Lab image or on a
GPU node.
