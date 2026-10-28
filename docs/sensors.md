# Sensors

The plant reports three kinds of signal: [drive feedback](glossary.md#feedback) per drive, in the physical drive-output frame; raw IMU samples, as the robot's host receives them; and [truth](glossary.md#truth), which covers joint state, base pose and twist, contacts, and in the runtime the sole plates.

Drive feedback and IMU samples are measured signals, the ones the real robot has. Truth exists only in simulation. A policy's inputs are built from measured signals only: `RobotPlant.measured()` in training and the feedback region in the runtime. Truth is for critics, rewards, terminations and tests. The plant has no sensor the robot lacks, and processing on top of the raw signals, such as orientation filtering (Madgwick) and the standing frame, belongs to the consumer, which on the robot is Controls.

## Sensor models and the rig

`src/humanoid_sim/sensors/models.py` has one class per sensor product. A kind's base class (`Imu`, `Camera`, `Lidar`) declares what a product of that kind is described by, with defaults that make an ideal sensor; a product subclasses it and sets its own figures as class attributes, each with its `source`. An instance is that product mounted on a robot: the URDF link it sits at and what the robot sets around it. Instantiating a product gives its whole behaviour.

| Class | Kind | Product | On |
|---|---|---|---|
| `Mti610` | `imu` | Xsens (Movella) MTi-610 | V1 |
| `UnitreeG1Imu` | `imu` | Unitree G1 body IMU (figures unmeasured: an ideal sensor) | G1 |
| `OakDW` | `camera` | Luxonis OAK-D-W | V1 |
| `RealSenseD435i` | `camera` | Intel RealSense D435i | the `g1-cameras-d435` extra rig (the lab G1's camera service finds no D435) |
| `ZedMini` | `camera` | Stereolabs ZED Mini | the `g1-cameras` extra rig |
| `ZedMiniImu` | `imu` | The ZED Mini's built-in IMU (**a modeled guess**: rate from the datasheet, noise and mount not published) | the `g1-cameras` extra rig, as `zed_imu` |
| `LivoxMid360` | `lidar` | Livox Mid-360 | G1 |

A camera class lists its streams as Controls' backend for the product publishes them, and may give each stream's kind (`stream_kinds`: `color`, `mono` or `depth`), its nominal optical centre in the camera frame (`nominal_offsets_m`) and the depth range (`depth_range_m`).

What an instance takes depends on the kind. An `Imu` takes its configured `sample_period_ns` and its delivery to the host (`burst_period_ns`, `in_burst_spacing_ns`; by default every sample as it is taken). A `Camera` takes its resolution, `frame_period_ns` and the unit's calibration per stream: its pinhole `intrinsics` `(fx, fy, cx, cy)` (OpenCV's convention), and where the unit's calibration gives them its lens `distortion` (`(model, coefficients)`, model `plumb_bob` or `equidistant`) and its optical centre (`stream_offsets_m`, in place of the nominal one). A `Lidar` takes only its frame. Figures are checked when a class is defined, an instance's values when it is made.

A `Rig` names a robot's instances. Each robot's rig is in `src/humanoid_sim/sensors/rigs.py`, as on the robot; `rigs.EXTRAS` holds rigs that add sensors a robot's rig does not declare, by name (`g1-cameras`: the ZED Mini and D435i the lab G1 is reported to carry). The runtime names the rig's IMU in its descriptor (`imu`: model, product, frame and delivery; [runtime.md](runtime.md#descriptor)), so a client such as Controls' `xsens_sim` can check which device its samples come from.

```python
def ethrc_v1() -> Rig:
    return Rig(
        imu=Mti610("imu_link", sample_period_ns=2_500_000),  # 400 Hz over CAN, each sample as it is sent
        camera0=OakDW("camera0_camera", width=640, height=400, frame_period_ns=33_333_333, intrinsics={...}),
    )
```

The plant simulates the instance named `imu`, which must sit at the child of the URDF's `imu_joint` (the mount the physics uses); a robot without a rig has no IMU. Cameras and lidars are simulated only by the RTX sensors in the Isaac Sim image ([rtx-sensors.md](rtx-sensors.md)): every lidar, and a camera whose class places and names the kind of every stream and whose instance calibrates every stream (`Camera.rtx_ready`). `Rig.unsimulated()` names what the plant does not simulate, `Rig.unsimulated(rtx=True)` what nothing does. `rigs.load(robot, desc, extra)` builds a robot's rig with an extra rig added and checks every frame against the description; `rigs.imu(desc)` returns its IMU, whose `delivery_cfg()` is what `ImuDelivery` runs:

```python
from humanoid_sim.model.description import Description
from humanoid_sim.sensors import rigs

desc = Description.load("g1_29dof_with_hand_rev_1_0")
rigs.imu(desc)                                  # UnitreeG1Imu(frame='imu_link')
rig = rigs.load(desc.robot, desc, "g1-cameras")
rig.unsimulated()                               # ['mid360', 'zed', 'd435']
rig.unsimulated(rtx=True)                       # []: the RTX sensors simulate them all
```

The V1's OAK-D-W places its streams as Controls' depthai backend does (the left mono at the camera frame's origin, depth aligned to it, the right mono a baseline away); the colour imager's place midway between the monos, and the right mono's and depth's pinholes (the left mono's), are modeled until the unit's calibration is read.

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

### The ZED Mini's IMU (a guess)

Stereolabs publishes only the ZED Mini's rate (800 Hz) for its gyroscope and accelerometer; the chip, noise and bias figures and the mount are "Not published", and `camera_imu_transform` (IMU to left camera) is read per unit from the ZED SDK. `ZedMiniImu` is therefore a placeholder, labelled `MODELED GUESS` in its `source`: 800 Hz, white noise typical of a consumer MEMS part (gyro 0.005 deg/s/√Hz, accelerometer 150 µg/√Hz, no bias terms), at the camera frame `camera0_camera` with that frame's axes, not at the chip. `SimCore(..., imu=True, extra_rig="g1-cameras")` simulates it: `SimCore.extra_imus["zed_imu"]` is an `ImuDelivery` like `SimCore.imu`, stepped with it (not with `cuda_graph`; training's `RobotPlant` has only the body IMU). The 800 Hz samples are linear interpolations between physics-step boundaries with fresh noise per sample (see below), so a 4 ms physics step carries no motion detail above 250 Hz. Replace the figures and the mount once a capture or the unit's transform gives them.

### Body IMU

The plant simulates the rig's `imu`: on the V1 an MTi-610, on the G1 its body IMU. `ImuDelivery` in `src/humanoid_sim/sensors/imu.py` turns the ideal readings of Isaac Lab's IMU sensor into raw samples as the robot's host receives them.

### Mount and ideal input

The sensor sits at the URDF's `imu_joint` (`Description.imu_mount`, `imu_sensor_cfg` in `src/humanoid_sim/physics/isaaclab.py`): parent `base_link`, offset (0, 0.04, 0.07) m, rotation rpy (π/2, π/2, 0) rad. Its frame is the chip frame given by that joint. Isaac Lab's IMU sensor provides, at each step boundary, the angular velocity (rad/s) and the specific force (m/s²) in the sensor frame. Specific force is what an accelerometer measures: at rest it reads gravity's reaction, 9.81 m/s² pointing up.

The plant advances the IMU at the start of every physics step, to that step's start time (`SimCore.step`, `RobotPlant.pre_physics`). What a consumer reads after a step therefore holds the samples measured and delivered up to the start of that step; a sample or burst that falls inside the step shows up one step later.

### Sampling and errors

The sensor takes a sample every `sample_period_ns` (on the V1 2.5 ms, 400 Hz) on its own clock, with a random phase per env drawn at reset. Each sample's value is linearly interpolated between the two step-boundary readings around it.

Each sample and axis gets Gaussian white noise with `σ = density · √(sample rate)`. The densities are the model's; for the MTi-610 its leaflet's (as Controls' `ethrc_ssot/imu_sensors.py` carries them):

| Channel | Density | σ per sample |
|---|---|---|
| Gyro | 1.2217e-4 rad/s/√Hz (0.007 °/s/√Hz) | 2.33e-3 rad/s |
| Accelerometer | 5.884e-4 m/s²/√Hz (60 µg/√Hz) | 0.0112 m/s² |

The model also gives the sensor a bias, per axis and env:

- Turn-on bias (`turn_on`): drawn at every reset from a normal distribution with the model's 1σ (`gyro_turn_on_bias_rad_s`, `accel_turn_on_bias_m_s2`), constant until the next reset. The MTi-610's gyro figure, 1.5e-3 rad/s, is modeled from the size of the bias at one power-on of the V1 (the spread across power-ons is unmeasured); its accelerometer figure is unmeasured (0). The runtime redraws it at every IMU reset, not only at power-on.
- In-run drift (`drift`): a first-order Gauss-Markov process advanced every sample, whose stationary σ is the model's bias instability (`gyro_bias_instability_rad_s`, `accel_bias_instability_m_s2`: for the MTi-610 8 °/h and 15 µg) and whose correlation time is `bias_correlation_time_ns` (100 s, modeled). A reset draws it from its stationary distribution.

On top of the model, each env has a gyro bias `gyro_bias` (rad/s), an accelerometer bias `accel_bias` (m/s²) and a `misalignment` (a rotation vector in rad: the sensor's rotation in its mount). A sample is `R(misalignment) · ideal + bias + turn_on + drift + noise`. In training the three come from the `imu_gyro_bias`, `imu_accel_bias` and `imu_misalignment` [randomization](training.md#randomization) parameters and stay constant through an episode; a task adds them to widen what the model gives. In the runtime they are zero unless an `imu_bias` or `imu_spike` fault is injected (see [drives.md](drives.md#physical-faults)).

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

The plant delivers samples as the rig says. Both robots' rigs deliver every sample as it is taken: the V1's MTi-610 sends each one over CAN (measured on the V1: 400 samples/s, 2.47 ms apart at the median, none missed), the G1's LowState carries each one (modeled). A sensor whose host reads it in [bursts](glossary.md#imu-burst) is described by the burst parameters:

- Measured samples wait in a pending buffer of 16 samples (`ImuDeliveryCfg.pending`); if it fills, the oldest pending sample is dropped.
- A burst arrives every `burst_period_ns`, drawn uniformly from its range for each burst, starting from a random phase per env. It delivers every pending sample measured by the burst time.
- Delivered samples get host-receipt stamps: the oldest sample of a burst is stamped at the burst time and each newer one `in_burst_spacing_ns` later.

A consumer that reads "the latest sample" then sees it up to one burst period late. The V1's earlier serial path did this (bursts every 13.9 to 17.6 ms, 13 µs apart, on the `exc-*` captures), and replays of those captures still read it.

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

`ImuDelivery.set_noise(enabled)` turns the model's errors (white noise, turn-on bias, drift) off (an ideal sensor) or back on, in place, so it also works inside a captured CUDA graph. The sample clock, interpolation, burst delivery and the randomized or injected bias stay as they are. The runtime exposes it as the `SET_FIDELITY` mailbox command; every session starts with `"real"`:

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

In the home pose under an upright base the plates lie flat. See [robot-model.md](robot-model.md#standing-on-the-ground).
