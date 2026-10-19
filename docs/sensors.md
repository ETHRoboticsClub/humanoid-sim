# Sensors

The plant reports three kinds of signal: [drive feedback](glossary.md#feedback) per drive, in the physical drive-output frame; raw IMU samples, as the robot's host receives them; and [truth](glossary.md#truth), which covers joint state, base pose and twist, contacts, and in the runtime the sole plates.

Drive feedback and IMU samples are measured signals, the ones the real robot has. Truth exists only in simulation. A policy's inputs are built from measured signals only: `RobotPlant.measured()` in training and the feedback region in the runtime. Truth is for critics, rewards, terminations and tests. The plant has no sensor the robot lacks, and processing on top of the raw signals, such as orientation filtering (Madgwick) and the standing frame, belongs to the consumer, which on the robot is Controls.

Frames and units are as in [drives.md](drives.md#frames): drive quantities in rad, rad/s and Nm at the drive output (crank angle for the ankle cranks), joint quantities in URDF joint coordinates, all times in integer nanoseconds on the plant clock.

## Drive feedback

`DriveFeedback` in `src/humanoid_sim/devices/feedback.py` is the physical feedback of every drive, `[E, D]`. It is what Controls' fake bank accepts (`fake_bank_feedback`).

| Field | Unit | Meaning |
|---|---|---|
| `position_rad` | rad | Drive-output position after the step (crank angle for the cranks) |
| `velocity_rad_s` | rad/s | Drive-output velocity after the step |
| `applied_torque_nm` | Nm | Torque the drive applied during the step, after the envelope and cap, time-weighted |
| `demanded_torque_nm` | Nm | Control-law output during the step, before any clip, time-weighted |

`PlantCore.feedback(q, qd)` builds it from the joint state after a solver step: position and velocity through the drive map, torques from the step just taken (`PlantCore.last`). Physical feedback has no encoder offset and no quantization.

### Reporting in training

A real drive does not report its physical state exactly. It reports whole encoder counts, a velocity measured over its controller sample, and a torque in whole mNm. In the runtime, Controls' `Epos4Slave` (its simulated EPOS4 drive) turns the plant's physical feedback into the drive's process data, so the runtime publishes physical feedback and leaves reporting to Controls; applying it in both places would quantize twice. Training has no Controls in the loop, so `DriveReporting` in `src/humanoid_sim/devices/reporting.py` mirrors `Epos4Slave` (`setFeedback`, `reportedPosition`, `reportedVelocityMilliRpm`, `torqueEstimate_mNm`):

- Position: whole encoder counts of the drive's reading (the truth plus its `encoder_offset`), `round(position · cpr / 2π)`, plus a jitter of -1, 0 or +1 count drawn uniformly on every read, converted back with `counts · 2π / cpr`. `cpr` is the encoder's counts per revolution.
- Velocity: whole counts per controller sample, `round(velocity · cpr / 2π · T)` with `T = 1 / controller_rate_hz`, converted back to rad/s.
- Applied torque: rounded to whole mNm. The demanded torque passes through unchanged.

Rounding is half away from zero, as C's `lround` does. The resulting resolution (controller rate 2500 Hz for every family, so `T` = 400 µs):

| Family | Encoder (counts/rev) | Position step | Velocity step |
|---|---|---|---|
| HEJ90-48-140 | 4096 | 1.534 mrad | 3.835 rad/s |
| HEJ70-48-50 | 16384 | 0.383 mrad | 0.959 rad/s |
| HEJ50-48-30 (arms and ankle cranks) | 16384 | 0.383 mrad | 0.959 rad/s |

The velocity is coarse: a HEJ90 joint moving slower than half a step, 1.92 rad/s, reports 0.

```python
import torch
from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.devices.reporting import DriveReporting
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description

desc = Description.load("ethrc_v1")
drives = list(desc.drives)
reporting = DriveReporting.for_drives(desc, drives, load_profile("ethrc_v1"), jitter=False)
fb = DriveFeedback(*(torch.zeros(1, len(drives)) for _ in range(4)))
knee = drives.index("maxonHEJ90L4")
fb.position_rad[0, knee], fb.velocity_rad_s[0, knee], fb.applied_torque_nm[0, knee] = 2.3074, 1.0, 12.3456
out = reporting.report(fb)
# knee: 2.30711 rad (1504 counts), 0.0 rad/s, 12.346 Nm
```

### Measured feedback in training

`RobotPlant.measured()` in `src/humanoid_sim/train/plant.py` returns a `Measured`:

| Field | Shape | Meaning |
|---|---|---|
| `drive` | `DriveFeedback [E, D]` | Feedback now, as reported: offset included, quantized when `TrainPlantCfg.reporting` is true (the default) |
| `history` | `DriveFeedback [E, H, D]` | One entry per physics step, newest first, reported the same way. H is `TrainPlantCfg.feedback_history` (16) |
| `history_ns` | `[E, H]` | Plant time of each history entry; -1 for none yet |
| `imu` | `ImuSamples` or None | The raw IMU samples received so far (below) |

A history entry is taken at the start of each physics step: the state at that time and the torques of the step that just ended. `history[:, 0]` is therefore one physics period older than `drive`. The jitter is drawn on every call, for the history too, so two calls in the same step give different jitter. With `reporting=False` the feedback is unquantized but still includes the encoder offset.

```python
plant = env.action_manager.get_term("drives").plant
m = plant.measured()
q = m.drive.position_rad          # [E, 22] rad, whole encoder counts, offset included
v = m.drive.velocity_rad_s        # [E, 22] rad/s, whole counts per 400 us sample
tau = m.drive.applied_torque_nm   # [E, 22] Nm, whole mNm
gyro = m.imu.gyro[:, 0]           # [E, 3] rad/s, newest delivered sample
```

## IMU

The robot's IMU is an MTi-610. `ImuDelivery` in `src/humanoid_sim/sensors/imu.py` turns the ideal readings of Isaac Lab's IMU sensor into raw samples as the robot's host receives them.

### Mount and ideal input

The sensor sits at the URDF's `imu_joint` (`Description.imu_mount`, `imu_sensor_cfg` in `src/humanoid_sim/physics/isaaclab.py`): parent `base_link`, offset (0, 0.04, 0.07) m, rotation rpy (π/2, π/2, 0) rad. Its frame is the chip frame given by that joint. Isaac Lab's IMU sensor provides, at each step boundary, the angular velocity (rad/s) and the specific force (m/s²) in the sensor frame. Specific force is what an accelerometer measures: at rest it reads gravity's reaction, 9.81 m/s² pointing up.

The plant advances the IMU at the start of every physics step, to that step's start time (`SimCore.step`, `RobotPlant.pre_physics`). What a consumer reads after a step therefore holds the samples measured and delivered up to the start of that step; a sample or burst that falls inside the step shows up one step later.

### Sampling and errors

The sensor takes a sample every `sample_period_ns` = 2.74 ms (about 365 Hz) on its own clock, with a random phase per env drawn at reset. Each sample's value is linearly interpolated between the two step-boundary readings around it.

Each sample and axis gets Gaussian white noise with `σ = density · √(sample rate)`. The densities are the MTi-610 datasheet figures (from Controls' `ethrc_ssot/imu_sensors.py`), set in `profiles/ethrc_v1.yaml` under `sensors.imu`:

| Channel | Density | σ per sample |
|---|---|---|
| Gyro | 1.2217e-4 rad/s/√Hz (0.007 °/s/√Hz) | 2.33e-3 rad/s |
| Accelerometer | 5.884e-4 m/s²/√Hz (60 µg/√Hz) | 0.0112 m/s² |

Each env also has a gyro bias `gyro_bias` (rad/s), an accelerometer bias `accel_bias` (m/s²) and a `misalignment` (a rotation vector in rad: the sensor's rotation in its mount). A sample is `R(misalignment) · ideal + bias + noise`. In training the three come from the `imu_gyro_bias`, `imu_accel_bias` and `imu_misalignment` [randomization](training.md#randomization) parameters and stay constant through an episode; there is no bias drift. In the runtime they are zero unless an `imu_bias` or `imu_spike` fault is injected (see [drives.md](drives.md#physical-faults)).

Each of the three is a vector over the sensor axes `x`, `y` and `z`. A spec draws every axis unless its `select` key, a regular expression over the axis names, picks some. The dict form takes `select` as `ParamSpec` does:

```python
from humanoid_sim.train.plant import TrainPlantCfg

cfg = TrainPlantCfg(randomization=[
    {"param": "imu_gyro_bias", "dist": "normal", "range": [0.0, 0.0015], "op": "add"},                  # x, y and z
    {"param": "imu_gyro_bias", "select": "z", "dist": "uniform", "range": [-0.01, 0.01], "op": "add"},  # z only
])
```

Each selected axis gets its own draw; `"per_target": False` shares one draw across them.

### Delivery

On the robot the samples wait in a serial buffer and reach the host in [bursts](glossary.md#imu-burst). The plant reproduces the delivery seen in the robot's captures:

- Measured samples wait in a pending buffer of 16 samples (`ImuDeliveryCfg.pending`); if it fills, the oldest pending sample is dropped.
- A burst arrives every `burst_period_ns`, drawn uniformly from 13.9 to 17.6 ms for each burst, starting from a random phase per env. It delivers every pending sample measured by the burst time.
- Delivered samples get host-receipt stamps: the oldest sample of a burst is stamped at the burst time and each newer one `in_burst_spacing_ns` = 13 µs later.

A burst carries 5 to 7 samples, and a consumer that reads "the latest sample" sees it up to one burst period late.

### `ImuSamples`

The last H delivered samples per env, newest first. H is `ImuDeliveryCfg.history`: 16 by default, which is what training keeps, and 32 in the runtime, which builds its `SimCore` with `imu_history=abi.IMU_SAMPLES` so the feedback region's 32 slots can fill. Slots not yet filled have a `sample_ns` and `stamp_ns` of -1.

| Field | Shape | Unit | Meaning |
|---|---|---|---|
| `gyro` | `[E, H, 3]` | rad/s | Angular velocity, sensor frame |
| `specific_force` | `[E, H, 3]` | m/s² | Specific force, sensor frame |
| `sample_ns` | `[E, H]` | ns | Measurement time, plant clock |
| `stamp_ns` | `[E, H]` | ns | Host receipt time, plant clock |
| `seq` | `[E, H]` | count | Per-env sequence number of the sample (see below) |
| `count` | `[E]` | count | Samples delivered since the last reset |

The tensors are updated in place every step; clone what you keep. A reset (`ImuDelivery.reset`) empties the buffers, sets `count` to 0 and draws new sample and burst phases. What happens to `seq` depends on the front end:

- Training: every env reset restarts `seq`, so the first sample after it is number 1.
- Runtime: `RESET` and a new session call `ImuDelivery.reset(..., keep_seq=True)`, and `PRESCRIBE_POSE` does not reset the IMU at all, so sequence numbers keep rising for as long as the host runs. A client that remembers the last `seq` it saw and takes only newer samples (as Controls' `new_imu_samples` does) keeps receiving samples across a reset.

### Ideal or real IMU

`ImuDelivery.set_noise(enabled)` turns the white noise off (an ideal sensor) or back on, in place, so it also works inside a captured CUDA graph. The sample clock, interpolation, burst delivery and any bias stay as they are. The runtime exposes it as the `SET_FIDELITY` mailbox command; every session starts with `"real"`:

```python
from humanoid_sim.runtime.abi import plant_abi as abi

seg = abi.Segment.open("/run/hsim/plant", wait_s=5)
seq = seg.send(abi.CMD_SET_FIDELITY, {"imu": "ideal"})  # or "real"
print(seg.wait_ack(seq))  # (0, 'imu ideal')
```

## What the runtime publishes

Every physics period, paused or not, the runtime host writes the feedback region of the segment (`RuntimeHost._publish_feedback` in `src/humanoid_sim/runtime/host.py`; layout in `src/humanoid_sim/runtime/abi/plant_abi.py`):

| Field | Meaning |
|---|---|
| `drive[:num_drives]` | Physical `DriveFeedback` per drive (`position_rad`, `velocity_rad_s`, `applied_torque_nm`, `demanded_torque_nm`), with no offset or quantization |
| `imu[:imu_count]` | The latest delivered IMU samples, newest first (`seq`, `sample_ns`, `stamp_ns`, `gyro_rad_s`, `specific_force_m_s2`). The region has 32 slots and the runtime keeps 32 delivered samples, so `imu_count` reaches 32 about 0.1 s after a reset. `seq` keeps rising across resets. |
| `imu_total` | Samples delivered since the IMU was last reset (host start, `RESET` or a new session) |
| `sim_time_ns` | Plant time after the step |
| `written_mono_ns` | The host's `CLOCK_MONOTONIC` when the record was written |

IMU times are plant time. A plant time `t` maps to the host clock as `written_mono_ns - (sim_time_ns - t)`:

```python
from humanoid_sim.runtime.abi import plant_abi as abi

seg = abi.Segment.open("/run/hsim/plant", wait_s=5)
_, fb = seg.regions["feedback"].read()
n, k = int(fb["num_drives"]), int(fb["imu_count"])
position = fb["drive"]["position_rad"][:n]   # rad, physical drive-output frame
gyro = fb["imu"]["gyro_rad_s"][:k]           # rad/s, sensor frame, newest first
offset = int(fb["written_mono_ns"]) - int(fb["sim_time_ns"])
stamp_mono_ns = fb["imu"]["stamp_ns"][:k] + offset  # receipt stamps on CLOCK_MONOTONIC
```

The other feedback fields (lateness, pause) and the session protocol are in [runtime.md](runtime.md).

## Truth

### Training

`RobotPlant.truth()` returns a `Truth`:

| Field | Shape | Unit | Meaning |
|---|---|---|---|
| `joint_pos` | `[E, J]` | rad | Joint positions, URDF joint coordinates, description joint order |
| `joint_vel` | `[E, J]` | rad/s | Joint velocities |
| `root_pose_w` | `[E, 7]` | m, quaternion xyzw | Base pose in the world (the env's origin is not subtracted) |
| `root_lin_vel_w` | `[E, 3]` | m/s | Linear velocity of the base's centre of mass, world frame |
| `root_ang_vel_b` | `[E, 3]` | rad/s | Base angular velocity, base frame |
| `drive` | `DriveFeedback [E, D]` | | Physical drive feedback: no offset, no quantization |
| `contact_forces_w` | `[E, B, 3]` or None | N | Net contact force on each body, world frame (None without the contact sensor) |

The base frame is that of `base_link`. With the base upright its x axis is lateral (the left foot at -x), y points forward and z up; see [robot-model.md](robot-model.md#base-frame-and-forward-direction).

### Runtime truth region

The runtime host writes the truth region every `truth_every` steps (default 5, so every 20 ms; `serve --truth-every`) and after every mailbox command it applies (`ACK_OK`); its `sim_time_ns` says when (`RuntimeHost._publish_truth`). It is the oracle for tests; a controller must not read it.

| Field | Unit | Meaning |
|---|---|---|
| `base_position_m`, `base_quat_xyzw` | m, quaternion | Base pose, world |
| `base_lin_vel_m_s`, `base_ang_vel_rad_s` | m/s, rad/s | Base link twist, world frame |
| `com_m` | m | Whole-robot centre of mass, world |
| `joint_pos_rad[:num_joints]`, `joint_vel_rad_s[:num_joints]` | rad, rad/s | Joint state, URDF joint coordinates |
| `gantry_tension_n` | N | Gantry rope tension (0 without a gantry) |
| `mass_kg` | kg | Total robot mass |
| `sole[2]` | | The two sole plates, left then right |

The sole plates are the [sole boxes](glossary.md#sole-box), the box contact primitives fitted under each foot (on the body carrying each foot_roll joint; see [robot-model.md](robot-model.md#the-prepared-urdf)). Per plate:

| Field | Unit | Meaning |
|---|---|---|
| `position_m`, `quat_xyzw` | m, quaternion | Pose of the sole body's frame, world |
| `lowest_z_m` | m | Height of the plate's lowest corner |
| `normal_force_n` | N | Vertical component of the net contact force on the sole body |
| `plate_normal_w` | unit vector | The plate's z axis in the world, pointing up into the plate: (0, 0, 1) when it lies flat |
| `plate_face_height_span_m` | m | Height difference across the plate's bottom face (0 when it lies flat) |
| `face_xy_m` | m, `[4, 2]` | The plate's bottom-face corners, world x and y |

In the description's standing pose under an upright base the plates are tilted (about 10° on the left, 17° on the right); in the home pose they lie flat. See [robot-model.md](robot-model.md#standing-on-the-ground).
