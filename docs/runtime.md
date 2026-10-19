# Real-time runtime

`humanoid-sim serve` runs one robot against the wall clock and publishes the runtime segment, a shared-memory file that one client attaches to. Every bus cycle (one exchange of the robot's host with its drives: 1.5 ms over EtherCAT on the ETHRC V1, 2 ms over `LowCmd` on the G1) the client writes one demand per drive (the V1's 22 EPOS4 drives, ankle cranks included, or the G1's 29 motors) and reads back the physical drive feedback and the raw IMU samples. A mailbox carries commands for reset, pause, supports, disturbances, faults, prescribed poses and the IMU's fidelity, and a truth region gives tests the robot's true state. The usual client is Controls' plant host; see [controls-integration.md](controls-integration.md). The smallest working client is `examples/runtime_client.py` (see [Writing a client](#writing-a-client)), and `humanoid-sim view-runtime` shows a running host in Rerun (see [Watching a host](#watching-a-host-view-runtime)).

| What | Where |
|---|---|
| Command line | `cmd_serve` and `cmd_view_runtime` in `src/humanoid_sim/cli.py` |
| Host loop, sessions, commands | `RuntimeHost` in `src/humanoid_sim/runtime/host.py` |
| Segment layout and protocol, Python | `src/humanoid_sim/runtime/abi/plant_abi.py` (NumPy only) |
| Segment layout, C11 and C++17 | `src/humanoid_sim/runtime/abi/plant_abi.h` |
| Read-only Rerun view | `RuntimeView` in `src/humanoid_sim/runtime/view.py` |
| Example client | `examples/runtime_client.py` |
| Tests | `tests/test_runtime_abi.py` (no physics), `tests/test_runtime_host.py` (Isaac Lab image) |

## Running `serve`

`serve` runs in the Isaac Lab image or on a GPU node, with the robot asset built (see [getting-started.md](getting-started.md)):

```bash
humanoid-sim serve --segment /run/hsim/plant --support suspended
```

| Option | Default | Meaning |
|---|---|---|
| `--segment PATH` | required | The segment file. The host writes `PATH.<pid>.tmp` and renames it over `PATH`, so it needs write access to the directory. Put it on a volume the client also mounts. |
| `--support suspended\|free` | `suspended` | How the robot is held for the whole process (see [Supports](#supports)). |
| `--pose home\|standing` | the profile's `poses.start` | The pose every session starts in (see [The start pose](#the-start-pose)). |
| `--device` | `cuda:0` | `cuda:N`, or `cpu` for Warp's CPU device (below real time; see [Device and performance](#device-and-performance)). |
| `--bus-period-us` | `1500` | The client's bus cycle, written to the header's `bus_period_ns`. Informational: the host does not use it. |
| `--client-timeout-ms` | `500` | End the session when an attached client's latest demands are older than this. |
| `--truth-every N` | `5` | Publish the truth region every N physics steps (and after every accepted command). Use 1 for test oracles that track maxima. |
| `--persist` | off | When a client detaches or is lost, reset the robot and serve the next client on a new session, until stopped. |
| `--max-steps N` | none | Physics periods per session, paused periods included. Reaching it ends the process, with or without `--persist`. |
| `--seed` | `0` | Seed of the IMU's random generator (white noise, sample phase, burst timing). |

Four options of the top-level parser go before `serve`: `--solver` (`mjwarp` or `kamino`, default the profile's), `--robot` (default `ethrc_v1`, also written to the header's `robot_id`), `--assets` (default `outputs/assets`, relative to the working directory) and `--usd` (a robot USD to use instead of the built asset). Without `--usd`, `serve` exits with a message if the asset for the current description and profile is not built.

`serve` prints one JSON line on stdout when a session is ready and one when it ends:

```json
{"segment": "/run/hsim/plant", "session_id": 9134770520263281019, "ready": true}
{"session_id": 9134770520263281019, "steps": 15233, "late_steps": 0, "sim_time_s": 60.944, "attached": true, "exit": 4}
```

### Startup

`RuntimeHost.__init__` builds everything before a client can see the segment:

1. It loads the description and the asset, picks the start pose (`--pose`, else the profile's `poses.start`), and computes the named poses `home` and `standing` (`initial_pose` in `src/humanoid_sim/model/poses.py`) with their base heights: the height of an upright base at which the pose's lowest sole corner is 1 mm above the floor (`STANDING_CLEARANCE_M`).
2. It builds the scene (`SimCore`, with the IMU keeping its 32 latest delivered samples and a contact sensor on every body). On CUDA it captures the plant's work before and after the solver step as two CUDA graphs (a recorded sequence of GPU kernels that replays with one launch) and fuses their kernels with `torch.compile`, and it does the same for the demand intake (`_DemandIntake`). The command line always uses graphs and compilation on CUDA and neither on the CPU.
3. It steps the physics three times and puts the state back (`_warm_up`), so that the first steps' kernel compilation does not happen while a client watches the heartbeat.
4. It opens the first session (`_open_session`): it creates the segment in state creating, writes the start pose, publishes the first feedback and truth records, and then marks the segment ready.

On CUDA, compilation takes about a minute. On the CPU nothing is compiled ahead, but the first steps compile Warp kernels and every step is slow: in the arm64 image on an Apple silicon Mac, with Warp's kernel cache kept on a volume, the segment became ready 25 to 40 s after start. The host tests allow up to three minutes. A client waits with `Segment.open(path, wait_s=...)`, which retries until the segment is ready. Controls' plant host waits up to `--sim-wait` seconds (default 180).

### Supports

The support is chosen at startup and reported in the descriptor's `support` field. Changing it needs a restart.

| Support | Base | What `SET_SUPPORT` can do |
|---|---|---|
| `suspended` | Welded to the world (a fixed-base articulation), 0.3 m above the start pose's height so the feet hang clear | Nothing: `{"support": "suspended"}` is accepted as a no-op, `free` and `gantry` are rejected. Any `base_pose` argument is rejected. |
| `free` | Standing on the floor in the start pose, upright, at the origin | Hang a gantry rope (tension program or slack cable) and release it with `{"support": "free"}` |

The gantry and the floor are described in [world.md](world.md).

### The start pose

Every session starts in the same named pose: the one `--pose` names, else the profile's `poses.start`. The host reports it as `RuntimeHost.start_pose`.

| Robot | `poses.start` | Why |
|---|---|---|
| ETHRC V1 | `standing` | Controls arms a drive only while its position lies inside its window (the joint's `position_limit`, the crank's device-frame limit), and the [home pose](glossary.md#home-pose)'s straight knees lie 0.13 rad past the knees' upper bound, so Controls would refuse to arm a V1 that starts at home. The [standing pose](glossary.md#standing-pose) lies inside every window, and it is where Controls' MuJoCo plant starts the robot. |
| G1 | `home` | Unitree's standing pose, which Controls' G1 lane also resets to |

`test_the_runtime_start_pose_lies_inside_every_drive_window` (in `tests/test_robots.py`) checks every robot's start pose against its windows.

The standing pose leaves the V1's soles tilted under an upright base, which does not matter while the robot hangs (`suspended`, the default). A free robot that holds it falls over within a second or two: in a free session on the CPU device, the base had tilted 99° and dropped to 0.16 m after 2.5 s of plant time. A free V1 that should stand by itself at the baseline gains needs `--pose home`, with both soles flat on the floor: in a 60 s free session on CUDA the base stayed at 0.923 m and within 0.6° of upright. To hold the start pose, take the drive positions the first feedback reports as targets, as `examples/runtime_client.py` does. See [RESET](#reset-1) and [robot-model.md](robot-model.md#standing-on-the-ground).

### Sessions

The header's `session_id` (a random non-zero 64-bit number chosen when the segment file is created) names one client session.

A client attaches by writing a demands record whose `session_id` echoes the header's. Records carrying another session id are ignored, so a client of an earlier session cannot attach to a new one. Until the first demand arrives, the session holds the robot in its [start pose](#the-start-pose): the joints in that pose and, in a free session, the base at its height. The pose is written before every physics step and every drive is treated as off. The client's first demands release the pose, so the robot starts moving when the client does. Physics and plant time run before attachment, and no client timeout applies until then.

A session ends in one of three ways. The segment's `state` becomes ended in every case (`RuntimeHost.run` sets it when it returns):

| End | `"exit"` in the JSON line | Process exit code without `--persist` |
|---|---|---|
| The client sends `DETACH` | 4 (`EXIT_DETACHED`) | 0 |
| An attached client's latest demands are older than `--client-timeout-ms` | 3 (`EXIT_CLIENT_LOST`) | 3 |
| `--max-steps` reached, or SIGINT or SIGTERM | 0 (`EXIT_DONE`) | 0 |

The client timeout compares the host's `CLOCK_MONOTONIC` with the `written_mono_ns` stamp inside the client's latest demands record, so only new demand records keep a session alive; commands do not. A stop signal is handled at the end of the current period once the host is serving. An unexpected error in the host loop also ends the session and exits the process with a traceback.

With `--persist`, a detached or lost session is followed by a new one (`RuntimeHost.new_session`) without recompiling. The host clears the faults, releases any gantry, turns the IMU noise back on and resets the robot to the start pose, which also clears the prescribed pose, the disturbances, the drive bank and the IMU history. It then creates a new segment file with a new session id, unpaused, and renames it over the same path. A client that still maps the old file sees it ended; a client that opens the path gets the new session. The feedback's `step` and `late_steps` restart from 0; plant time and the IMU sample numbers run on. A persistent host ends only on a signal or `--max-steps`, with exit code 0.

A session is tied to the segment file and outlives the client process that drove it. A client that opens the path while the session is still ready, and echoes its session id, continues that session where the previous client left it.

## The clock

The runtime runs in real time only. `RuntimeHost.run` paces each physics period, 4 ms (`physics.period_ns` in `profiles/ethrc_v1.yaml`), to `CLOCK_MONOTONIC`: it sleeps until 100 µs before the step's deadline and spins for the rest (`SPIN_NS`). Each period does, in order (`RuntimeHost.step`):

1. Handle a new mailbox command, if there is one, and write its ack.
2. Take the client's latest demands record, if it changed, into the drive bank. Its arrival time is the step's start, so it is in force for the whole step. The host reads demands once per step: of the records a 1.5 ms bus writes during one 4 ms step only the latest acts, and if no new record arrives the last one stays in force. See [drives.md](drives.md) for the arrival queue and the drive law.
3. Step the physics with the active gantry, disturbances and prescribed pose, unless paused.
4. Publish the feedback, the truth when due, and the heartbeat.

A step that starts more than one period after its deadline counts as late (`late_steps`), and the schedule restarts from that moment. A step that starts late by less than a period is not counted, and the following steps run back to back until they are on schedule again. Plant time advances by exactly one period per step and never jumps, so when the host falls behind, plant time falls behind the wall clock. `keeping_up` is 1 while no step in the last second was late (`KEEPING_UP_WINDOW_NS`). Both are in every feedback record; `late_steps` counts per session.

Plant time (`sim_time_ns`, and the IMU samples' `sample_ns` and `stamp_ns`) starts at 0 when the host starts and runs on across sessions. Each feedback record maps a plant time `t` to the host's clock as `written_mono_ns - (sim_time_ns - t)`. The offset grows whenever steps are late, so take it from the latest record. `*_mono_ns` fields are `CLOCK_MONOTONIC` of the writing process, which is the same clock for every process and container on one machine.

While paused, physics, plant time and the step counter stop. Commands and demands are still taken, and the feedback (with `paused` = 1) and the heartbeat are still written every period. Disturbances and faults are timed in plant time, so they wait as well.

## The segment

The segment is one file of 30,912 bytes: a 272-byte header and six regions. Values are little-endian at natural alignment, and every region starts on a 64-byte boundary. Each region has one writer.

| Region | Offset | Bytes | Writer | Contents |
|---|---|---|---|---|
| header | 0 | 272 | sim | Identity, sizes, session, state, periods, heartbeat |
| descriptor | 320 | 16,408 | sim | JSON: names, units, identity, commands, faults, fidelity gaps |
| demands | 16,768 | 3,632 | client | Bus cycle, client clock, session echo, one SI drive demand per drive |
| feedback | 20,416 | 4,424 | sim | Plant time, lateness, physical drive feedback, the latest raw IMU samples |
| mailbox | 24,896 | 4,128 | client | One command: code and JSON arguments |
| ack | 29,056 | 288 | sim | The last command handled: its number, a result code, a message |
| truth | 29,376 | 1,536 | sim | Base pose and twist, centre of mass, joint state, gantry tension, sole plates |

`MAX_DRIVES` (64) and `MAX_JOINTS` (64) size the arrays. A robot with more drives or joints needs a new major ABI version. Only one client may write the demands and mailbox regions: a second writer would break the protocol below.

### Header

The header is written when the file is created; afterwards only `state` and `heartbeat_mono_ns` change.

| Field | Type | Meaning |
|---|---|---|
| `magic` | char[8] | `HSIMPLNT`, written last when the file is filled |
| `abi_major`, `abi_minor` | u16 | 2, 0 |
| `header_bytes` | u32 | 272 |
| `layout_hash` | u64 | Hash of the layout (see [Validating a segment](#validating-a-segment)) |
| `segment_bytes` | u64 | 30,912 |
| `session_id` | u64 | This session |
| `state` | u32 | 0 creating, 1 ready, 2 ended |
| `num_drives`, `num_joints` | u32 | 22 and 22 for `ethrc_v1` |
| `imu_samples` | u32 | 32, the slots of the feedback's IMU array |
| `physics_period_ns` | u64 | 4,000,000 |
| `bus_period_ns` | u64 | From `--bus-period-us` |
| `heartbeat_mono_ns` | u64 | The host's `CLOCK_MONOTONIC` at its last publish, every period |
| `robot_id` | char[32] | `--robot`, for example `ethrc_v1` |
| `description_digest` | char[64] | Hex SHA-256 of the robot description |
| `sim_revision` | char[48] | Git `HEAD` of the humanoid-sim checkout, or `unknown` when the package is not in a checkout |
| `offsets` | u64[6] | Offsets of the descriptor, demands, feedback, mailbox, ack and truth regions |

Strings are NUL-padded and not NUL-terminated when full: `description_digest` uses all 64 bytes. A client watches `state` and `heartbeat_mono_ns`; Controls' plant host treats a heartbeat older than 500 ms as a lost sim.

### Reading and writing a region

Every region starts with `seq` (u64), `crc` (u32) and 4 bytes of padding. The payload is the rest of the region, from byte 16 to the end of its struct, padding included. The protocol is a sequence lock with a checksum (`Region.write` and `Region.read` in `plant_abi.py`).

A writer takes five steps:

1. It reads `seq`, and adds 1 if it is odd (a writer died mid-write).
2. It stores `seq + 1`. The odd value means a write is in progress.
3. It writes the payload.
4. It stores the CRC-32 of the payload (zlib's) in `crc`.
5. It stores `seq + 2`. The even value means the write is complete.

A reader repeats three steps, up to 1000 times (`attempts`):

1. It reads `seq`. A 0 means the region was never written. An odd value means a write is in progress, so it tries again.
2. It copies the payload and reads `crc`.
3. It accepts the copy if `seq` still has the value from step 1 and the CRC of the copy matches.

`Region.read` returns `(seq, record)`, `None` for a region never written, or raises `StaleRead` when no attempt found a complete write, as when the writer was descheduled in the middle of one. A periodic reader keeps its last record and reads again next cycle. The CRC makes a torn copy detectable without relying on memory ordering, which Python does not give. `seq` grows by 2 per write, so it also tells a reader whether a record is new.

### Validating a segment

`Segment.open(path, robot_id=None, description_digest=None, wait_s=0.0)` opens the file read-write and checks, in order:

| Check | Expected |
|---|---|
| File size | `SEGMENT_BYTES` (30,912) |
| `magic` | `HSIMPLNT` |
| `abi_major` | This side's `ABI_MAJOR` (1) |
| `layout_hash` | This side's `LAYOUT_HASH` |
| `state` | Ready |

While the file is missing or one of these fails it retries every 10 ms until `wait_s` has passed, then raises `FileNotFoundError` or `AbiError`. Once the segment is ready it compares `robot_id` and `description_digest` with the values given and raises `AbiError` on a mismatch. A different `abi_minor` is accepted.

The layout hash (`layout_hash()`) is the first 8 bytes, little-endian, of a SHA-256 over every field's name, offset, type and shape and the region offsets. Any change to the layout changes it. `plant_abi.h` carries the same value as `HSIM_PLANT_LAYOUT_HASH` (`0xfb475e58737ea4a4`), and `tests/test_runtime_abi.py` compiles the header and checks every struct size, field offset, region offset and the hash against the Python layout.

The description digest (`description_digest(urdf, actuators)`) is the SHA-256 over the URDF and `actuators.yaml` of the robot description in `third_party/humanoid-urdf`, each as its file name followed by its bytes. The sim writes `Description.digest`. A client computes the digest from its own checkout of humanoid-urdf and passes it to `Segment.open`: equal digests mean both sides run the same description.

### Descriptor

The descriptor region holds one JSON object (sorted keys, at most 16,384 bytes) written when the file is created. `Segment.descriptor()` returns it as a dict.

| Key | Contents |
|---|---|
| `robot` | `ethrc_v1` |
| `drives` | The drive names, in the order of the demand and feedback arrays |
| `joints` | The joint names, in the order of the truth's joint arrays |
| `support` | `suspended` or `free` |
| `bodies` | The articulation's body names, for `APPLY_DISTURBANCE` and gantry `attachment_bodies` |
| `description` | `robot`, `description_digest` and `description_revision` (the humanoid-urdf checkout's git `HEAD`, or null) |
| `physics` | `engine` (`isaaclab/mjwarp` or `isaaclab/kamino`), `period_ns`, `integrator` |
| `units` | One sentence each for `drive`, `joint`, `imu` and `time` |
| `imu` | `parent`: the link the URDF's `imu_joint` mounts the IMU on |
| `soles` | The two sole bodies, left then right |
| `commands` | Command codes to names, `{"1": "reset", ..., "9": "set_fidelity"}` |
| `faults` | `encoder_freeze`, `imu_spike`, `imu_bias`, `power_sag` |
| `fidelity_gaps` | What the runtime does not model (`FIDELITY_GAPS` in `host.py`) |

The fidelity gaps the runtime publishes:

- drive current, thermal and backlash models absent; applied torque is the envelope-clipped control law
- upper-body passive joint parameters (waist, shoulders, elbows) are description values, not identified
- ankle as pitch and roll hinges driven through the crank transmission (no closed chain)
- IMU white noise as the MTi-610 datasheet gives it (SET_FIDELITY ideal turns it off); no bias drift

### Demands

The client writes the demands region every bus cycle.

| Field | Type | Meaning |
|---|---|---|
| `cycle` | u64 | The client's cycle counter (not used by the host) |
| `written_mono_ns` | u64 | The client's `CLOCK_MONOTONIC` when it wrote the record; the client timeout runs from it |
| `session_id` | u64 | Must equal the header's |
| `num_drives` | u32 | Drives filled in (the host reads as many entries as the header's `num_drives`) |
| `drive` | `HsimDriveDemand[64]` | One demand per drive, in descriptor order |

Each drive's demand is vendor-neutral and in SI units, the `DriveDemand` of Controls' `docs/sim-learning-api-plan.md` (§1.2). The client converts its device's command at its own boundary (an EPOS4's CiA-402 mode and raw JVPT gains, a Unitree motor's `LowCmd` entry; [drives.md](drives.md#the-demand) lists both mappings):

| Field | Type | Meaning |
|---|---|---|
| `law` | u8 | `LAW_DISABLED` (0), `LAW_PD` (1) or `LAW_FREEZE` (2); any other value is disabled |
| `target_position_rad` | f64 | Physical drive-output frame (crank angle for the ankle cranks) |
| `target_velocity_rad_s` | f64 | rad/s |
| `feedforward_torque_nm` | f64 | Nm |
| `stiffness_nm_per_rad` | f64 | P gain |
| `damping_nm_s_per_rad` | f64 | D gain |
| `torque_limit_nm` | f64 | Cap on the applied torque; `inf` for none |

PD tracks the position and velocity targets with the demand's gains and adds the feed-forward torque; freeze holds the position the drive read when it entered the law. A disabled drive applies no torque and its joint, for a 1:1 drive, takes the drive family's back-drive friction. [drives.md](drives.md) gives the drive law, the envelope and the frames. `pd_demand(target, stiffness, damping, *, law=LAW_PD, **fields)` gives every drive's fields as arrays, with no torque limit unless one is passed, and `demands_record(demand, cycle=..., session_id=...)` builds a record from them.

### Feedback

The host writes the feedback region every period, paused or not.

| Field | Type | Meaning |
|---|---|---|
| `step` | u64 | Physics steps taken in this session |
| `sim_time_ns` | i64 | Plant time at the end of the last step |
| `written_mono_ns` | u64 | The host's `CLOCK_MONOTONIC` when it wrote the record |
| `late_steps` | u64 | Late steps in this session (see [The clock](#the-clock)) |
| `keeping_up` | u32 | 1 while no step in the last second was late |
| `paused` | u32 | 1 while paused |
| `num_drives` | u32 | Drives filled in |
| `imu_count` | u32 | Valid entries of `imu`, up to 32 |
| `imu_total` | u64 | IMU samples delivered since the IMU was last reset (session start or `RESET`) |
| `drive` | `HsimDriveFeedback[32]` | Per drive: `position_rad`, `velocity_rad_s`, `applied_torque_nm`, `demanded_torque_nm` |
| `imu` | `HsimImuSample[32]` | The latest delivered IMU samples, newest first |

Drive feedback is SI in the physical drive-output frame, as Controls' `fake_bank_feedback` takes it. `applied` is the torque after the torque-speed envelope and the cap, `demanded` the control law's output before them.

Each IMU sample has `seq` (u64), `sample_ns` (measurement time) and `stamp_ns` (host-receipt time), both plant time, then `gyro_rad_s[3]` and `specific_force_m_s2[3]` in the sensor frame. The samples arrive in bursts, as the robot's MTi-610 delivers them to its computer: about 365 samples per second, in bursts every 13.9 to 17.6 ms (see [sensors.md](sensors.md)). The host keeps the 32 latest delivered samples (`IMU_SAMPLES`), so `imu_count` reaches 32 about 0.1 s of plant time after the IMU was reset (32 samples 2.74 ms apart, plus the wait for the next burst).

`seq` numbers the samples in the order they were measured. It starts at 1 when the host starts and keeps rising for the life of the process: `RESET` and a new session do not restart it, and samples measured but not yet delivered when the IMU is reset are dropped, so the numbers can skip across a reset. A client reading every bus cycle takes the samples whose `seq` is above the last one it saw, oldest first, and keeps its last-seen number across a `RESET`. `imu_count` and `imu_total` do start again from 0 after a reset.

### Mailbox and ack

| Region | Field | Type | Meaning |
|---|---|---|---|
| mailbox | `command` | u32 | Command code (`CMD_*`) |
| | `length` | u32 | Bytes of `args` used |
| | `written_mono_ns` | u64 | The client's clock when it posted the command |
| | `args` | u8[4096] | A JSON object |
| ack | `acked` | u64 | The mailbox `seq` of the command handled |
| | `result` | i32 | `ACK_*` result code |
| | `_pad1` | u32 | |
| | `message` | char[256] | What happened, or why the command was refused |

The protocol is in [Commands](#commands).

### Truth

The truth region is the oracle tests need. The host writes it at the start of a session, every `--truth-every` steps (its `sim_time_ns` says when) and in the period an accepted command was handled. All poses and velocities are in the world frame.

| Field | Type | Meaning |
|---|---|---|
| `sim_time_ns` | i64 | Plant time of this record |
| `base_position_m`, `base_quat_xyzw` | f64[3], f64[4] | Root link pose |
| `base_lin_vel_m_s`, `base_ang_vel_rad_s` | f64[3], f64[3] | Root link twist |
| `com_m` | f64[3] | Whole-robot centre of mass |
| `gantry_tension_n` | f64 | Total rope tension applied on the last step; 0 without a rope |
| `mass_kg` | f64 | Total mass (60.15 kg for `ethrc_v1`) |
| `num_joints` | u32 | Joints filled in |
| `joint_pos_rad`, `joint_vel_rad_s` | f64[64] | Joint state in URDF joint coordinates, in descriptor `joints` order |
| `sole` | `HsimSole[2]` | Left, then right |

Each sole entry describes the sole plate, the contact box fitted to the bottom of each foot (see [robot-model.md](robot-model.md)):

| Field | Type | Meaning |
|---|---|---|
| `position_m`, `quat_xyzw` | f64[3], f64[4] | Pose of the sole body (the body carrying the `foot_roll` joint) |
| `lowest_z_m` | f64 | Height of the plate's lowest corner |
| `normal_force_n` | f64 | Vertical contact force on the sole body |
| `plate_normal_w` | f64[3] | The plate's face normal |
| `plate_face_height_span_m` | f64 | Height difference across the plate's face (0 when it is level) |
| `face_xy_m` | f64[4][2] | x and y of the plate's four bottom-face corners |

## Commands

The client posts one command at a time:

1. `Segment.send(code, args)` writes the mailbox and returns the command's number, the mailbox region's new `seq`.
2. At the start of each period the host reads the mailbox. If its `seq` is above the last one handled, the host runs the command and writes the ack with `acked` set to that number.
3. `Segment.wait_ack(seq, timeout_s=5.0)` polls the ack every millisecond and returns `(result, message)`.

A command posted before the previous one was handled replaces it, and `wait_ack` for the replaced one raises `AbiError` ("superseded"). Wait for each ack before sending the next command. Commands are handled while paused. At real time the ack comes within a period or two; on a slow host keep writing demands while waiting, or the client timeout ends the session.

The ack is written at the start of the period, before that period's physics step and publish. A feedback or truth record read right after the ack can still be the previous period's, from before the command took effect. To see the result of a command, wait for a record whose `sim_time_ns` is later than the one you read when the ack arrived.

| Result | Code | When |
|---|---|---|
| `ACK_OK` | 0 | Applied; the message says what was done |
| `ACK_REJECTED` | 1 | The arguments are not a JSON object, or the handler raised a `KeyError`, `TypeError` or `ValueError`: a key is missing, a value has the wrong type or range, or the command is not allowed in this session. The message is the exception type and text, for example `ValueError: the base is welded to the world in a suspended session`. |
| `ACK_UNKNOWN` | 2 | Unknown command code |
| `ACK_FAILED` | 3 | The handler raised any other exception. The message is its type and text. The host keeps serving the session. |

A rejected command changes nothing: every handler checks its arguments before it touches the state.

Times in arguments are seconds, and durations run in plant time. In the tables below `pose7` is `[x, y, z, qx, qy, qz, qw]` in the world frame with a unit quaternion, and is accepted only in a free session.

### RESET (1)

| Arguments | Effect |
|---|---|
| `{}` or `{"pose": "home"}` | Joints at the description's home pose (legs straight, both soles level and at the same height, see [robot-model.md](robot-model.md#standing-on-the-ground)); in a free session the base upright at the origin, at the height that puts the lowest sole corner 1 mm above the floor |
| `{"pose": "standing"}` | Joints at the description's standing offsets; in a free session the base upright at the origin, at that pose's height |
| `{"pose": "current"}` | Joints where they are; the base untouched |
| `{"joint_pos": {name: rad, ...}, "base_pose": pose7 \| null}` | The named joints at the given positions, the others where they are; the base at `base_pose` when given |

Joint velocities become zero, and the base's too when the base is written. Every position in `joint_pos` must lie within the joint's travel (its recorded mechanical stop, otherwise its position limit; `Description.calibration_travel`). `RESET` also resets the drive bank (queued demands dropped, every drive off until demands are taken again) and the IMU (history and undelivered samples emptied, `imu_count` and `imu_total` from 0, `seq` still rising), releases a prescribed pose, and clears the body and joint disturbances. The client's latest demands are taken again in the same period. Faults, the gantry, pause and the IMU's fidelity are kept. Ack: `reset`.

The home pose is the one to use when a free robot should stand. Both poses are computed at startup, so a reset to either is quick. After the ack, wait for a feedback record newer than the ack (see [Commands](#commands)) and hold the drive positions it reports: demands that still target the old pose pull the joints back to it.

### PAUSE (2) and RESUME (3)

`{}`. Physics and plant time stop and resume; see [The clock](#the-clock). Acks: `paused`, `resumed`.

### SET_SUPPORT (4)

| Arguments | Session | Effect and ack |
|---|---|---|
| `{"support": "free"}` | free | Release the gantry rope: `free` |
| `{"support": "suspended"}` | suspended | No change: `suspended` |
| `{"support": "gantry", "tension_breakpoints_s_n": [[t, N], ...], "attachment_local_m": [x, y, z]}` | free | A vertical tension program on one base-frame point, or on two with `[[x, y, z], [x, y, z]]`: `gantry` |
| `{"support": "gantry", "tension_breakpoints_s_n": [[t, N], ...], "attachment_bodies": [name, ...]}` | free | The same, on one or two points fixed in the base frame where the named bodies are now: `gantry` |
| `{"support": "gantry", "cable": {"overhead_distance_m": m, "length_breakpoints_s_m": [[t, m], ...], "stiffness_n_m": k, "damping_n_s_m": c}, "attachment_local_m": [x, y, z]}` | free | A slack cable: `gantry` |

A tension program has 1 to 8 breakpoints, at non-negative increasing times counted from the command, linear between them and held beyond both ends, with non-negative tensions. The force is vertical and split evenly over the one or two points. Either `attachment_local_m` or `attachment_bodies` is required; when both are given, the bodies are used. A cable runs from one base-frame point to an anchor `overhead_distance_m` above where that point is when the cable is hung. Its unstretched length follows the program (positive lengths), and it pulls along the rope only when stretched: `T = max(0, k · extension + c · extension rate)`, with `k` positive and `c` non-negative. Hanging a rope replaces the previous one; a rejected rope leaves the previous one hanging. [world.md](world.md) describes both.

### APPLY_DISTURBANCE (5)

A wrench on a body:

```json
{"body": "base_link", "force_n": [40.0, 0.0, 0.0], "torque_nm": [0.0, 0.0, 0.0], "duration_s": 0.3}
```

`body` is one of the descriptor's `bodies`. The force and torque are in the world frame and default to zero; `duration_s` is required and positive. Several wrenches can be active at once. Ack: `until <t> ns`, the plant time at which it ends.

A torque on one joint, outside the drive (it is not part of the drive's applied torque):

```json
{"drive": "maxonHEJ50AR4", "torque_nm": 4.0, "frequency_hz": 5.0, "duration_s": null}
```

Name the joint with `"joint"`, or a 1:1 drive with `"drive"` (an ankle crank has no single joint and is rejected). `torque_nm` is constant when `frequency_hz` is 0 (the default), otherwise the amplitude of `torque_nm · sin(2π · frequency_hz · t)` with `t` the plant time. `duration_s` absent or null lasts until cleared. Each joint has at most one such torque: a new command replaces it and `torque_nm` 0 clears it. Acks: `<joint> until <t> ns`, `<joint> until cleared`, `cleared`.

### INJECT_FAULT (6)

```json
{"fault": "power_sag", "drive": null, "value": 0.6, "duration_s": 2.0}
```

| Fault | `drive` | `value` | Effect | No-fault value |
|---|---|---|---|---|
| `encoder_freeze` | required | any non-zero (default 1) | The drive's published position and velocity keep their values from the last feedback before the command. The drive's law and the physics keep using the true state. | 0 |
| `imu_bias` | ignored | rad/s | Added to the gyro bias on all three axes | 0 |
| `imu_spike` | ignored | m/s² | Added to the accelerometer bias on all three axes; lasts one period (4 ms) when `duration_s` is null | 0 |
| `power_sag` | ignored | torque scale in [0, 1] | Scales every drive's torque-speed envelope and torque cap | 1 |

`value` defaults to 1 and must be finite; for `power_sag` it must lie in [0, 1]. `duration_s` null lasts until cleared or the session ends. Injecting a fault again (on the same drive, for `encoder_freeze`) replaces it, and sending its no-fault value clears it. The value is checked before the active fault is touched, so a rejected command, such as `power_sag` 1.5, leaves the fault that was active in place. Acks: `<fault> active`, `cleared`. An unknown fault name is rejected with the list of known ones. Bus and protocol faults, and a drive whose encoder stops reporting, are Controls' fake bank's to inject. [drives.md](drives.md) has more on the faults.

### PRESCRIBE_POSE (7)

```json
{"joint_pos": {"knee_l": -2.0}, "base_pose": [0.0, 0.0, 1.0, 0.0, 0.0871557, 0.0, 0.9961947], "floor_clearance_m": 0.002}
```

The pose is written, with zero velocities, before every physics step until it is released, a `RESET`, or the end of the session. Named joints must lie within their travel, as for `RESET`; the others keep their positions at the time of the command. `base_pose` is optional. `floor_clearance_m` needs a `base_pose` and moves the base vertically so that the lowest sole-plate corner sits that far above the floor. Ack: `prescribed`. `{"release": true}` releases the pose; ack: `released`. The IMU is not reset, so its samples run on.

### DETACH (8)

`{}`. The host acks `detached` and ends the session at the end of that period.

### SET_FIDELITY (9)

`{"imu": "ideal"}` or `{"imu": "real"}`: the IMU without or with its white noise, from the next sample, for this session. Every session starts `real`. Bias and spike faults apply in both. Acks: `imu ideal`, `imu real`.

## Writing a client

A client in Python needs only `plant_abi.py` and NumPy. Controls vendors the file unchanged next to its copy of `plant_abi.h`.

`examples/runtime_client.py` is the smallest working client. It opens the segment with an identity check (robot id and description digest), holds the pose the robot starts in at the description's baseline gains, waves both arms, pushes the left forearm with 40 N for 0.3 s, then detaches. It uses NumPy and the humanoid-sim package for the description, and does not need Isaac Lab:

```bash
humanoid-sim serve --segment /tmp/hsim/plant --device cuda:0 &
python examples/runtime_client.py /tmp/hsim/plant --seconds 20     # --wave-hz 0.4, --push-at 8 (s; negative: no push)
```

It prints the plant time, `late_steps` and the largest applied torque once per second. To stay short it does not watch the heartbeat, does not wait for the push's ack, and does not reset the robot, so in a free session the robot falls (see [Supports](#supports)).

A complete client follows these steps:

1. Open the segment with the robot id and description digest it expects, waiting for it to become ready.
2. Read the descriptor for the drive order.
3. Every bus cycle, write a demands record with the session id and read the feedback.
4. Watch `state` and the heartbeat.
5. Post commands one at a time, and keep writing demands until each is acked.
6. Send `DETACH` and close the segment when leaving.

The client below holds the robot where it starts with a PD law at the description's baseline gains, takes the new IMU samples each cycle, switches the IMU to ideal, reads the truth and detaches. It imports `Description` from humanoid-sim for the digest and the gains; a vendored client computes the digest with `plant_abi.description_digest(urdf_path, actuators_path)` from its own humanoid-urdf checkout.

```python
import time

import numpy as np

from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi

BUS_PERIOD_S = 0.0015  # this client's bus cycle
SIM_STALE_NS = 500_000_000  # give up on a heartbeat older than this


class Client:
    def __init__(self, path, desc):
        self.seg = abi.Segment.open(path, robot_id=desc.robot, description_digest=desc.digest, wait_s=180)
        self.drives = self.seg.descriptor()["drives"]
        n = self.n = self.seg.num_drives
        _, self.fb = self.seg.regions["feedback"].read()
        # PD about the start pose at the baseline gains (Nm/rad, Nm s/rad).
        self.demand = abi.pd_demand(
            self.fb["drive"]["position_rad"][:n],
            [desc.drives[d].baseline_stiffness for d in self.drives],
            [desc.drives[d].baseline_damping for d in self.drives],
        )
        self.cycle = 0
        self.imu_seq = 0

    def bus_cycle(self):
        """Write this cycle's demands, read the latest feedback, return the IMU samples not seen yet."""
        seg = self.seg
        if seg.state != abi.STATE_READY or seg.heartbeat_age_ns() > SIM_STALE_NS:
            raise RuntimeError("the sim ended the session or stopped")
        self.cycle += 1
        seg.regions["demands"].write(abi.demands_record(self.demand, cycle=self.cycle, session_id=seg.session_id))
        try:
            _, self.fb = seg.regions["feedback"].read()
        except abi.StaleRead:
            pass  # the sim is mid-write: keep last cycle's feedback
        imu = self.fb["imu"][: int(self.fb["imu_count"])]
        new = imu[imu["seq"] > self.imu_seq][::-1]  # oldest first
        if len(new):
            self.imu_seq = int(new["seq"][-1])
        time.sleep(BUS_PERIOD_S)
        return new

    def command(self, code, args=None, timeout_s=10.0):
        """Post a command and keep the bus running until the sim acks it."""
        seq = self.seg.send(code, args)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.bus_cycle()
            try:
                return self.seg.wait_ack(seq, timeout_s=0.002)
            except TimeoutError:
                pass
        raise TimeoutError(f"no ack for command {seq}")

    def close(self):
        if self.seg.state == abi.STATE_READY:
            self.seg.send(abi.CMD_DETACH)
        self.seg.close()


client = Client("/run/hsim/plant", Description.load("ethrc_v1"))
try:
    received = []
    for _ in range(2000):  # about 3 s
        received.extend(client.bus_cycle())
    fb = client.fb
    to_mono = int(fb["written_mono_ns"]) - int(fb["sim_time_ns"])  # plant time to this machine's clock
    print("positions", fb["drive"]["position_rad"][: client.n])
    print("late steps", int(fb["late_steps"]), "keeping up", bool(fb["keeping_up"]))
    if received:
        print("last IMU sample received at", int(received[-1]["stamp_ns"]) + to_mono, "ns")
    result, message = client.command(abi.CMD_SET_FIDELITY, {"imu": "ideal"})
    assert result == abi.ACK_OK, message
    _, truth = client.seg.regions["truth"].read()
    print("base height", float(truth["base_position_m"][2]))
finally:
    client.close()
```

`time.sleep` paces the loop only roughly; a real client keeps its own schedule. The tests drive the host the same way: `tests/test_runtime_host.py` runs `serve` on the CPU as a subprocess and acts as its client, and `bench --runtime` (`run_runtime_bench` in `src/humanoid_sim/bench/runner.py`) writes a demand before every unpaced step.

### C and C++ clients

`plant_abi.h` declares the same layout as C11 and C++17 structs (`HsimHeader`, `HsimDemands`, `HsimFeedback`, `HsimMailbox`, `HsimAck`, `HsimTruth` and their elements), the constants (`HSIM_PLANT_LAYOUT_HASH`, `HSIM_PLANT_SEGMENT_BYTES`, `HSIM_OFFSET_*`, `HSIM_CMD_*`, `HSIM_ACK_*`, `HSIM_STATE_*`) and static asserts on the header, feedback and truth sizes. Command arguments are JSON, as documented in `plant_abi.py` and above.

A C or C++ client maps the file with `mmap(MAP_SHARED)` after opening it read-write, checks `magic`, `abi_major`, `layout_hash` and `segment_bytes`, and waits for `state` to be ready. It follows the same region protocol, with the final `seq` store a release store and the reader's `seq` loads acquire loads, as the header's comment says. The CRC is zlib's `crc32(0, payload, sizeof(region) - 16)`. A reader sketch:

```c
#include <stdatomic.h>
#include <string.h>
#include <zlib.h>

#include "plant_abi.h"

/* Copy a region (`size` bytes at `region`) into `out`: 1 on a complete write, 0 if never written or none found. */
static int hsim_read(const void *region, void *out, size_t size) {
  const _Atomic uint64_t *seq = (const _Atomic uint64_t *)region;
  for (int attempt = 0; attempt < 1000; ++attempt) {
    uint64_t s0 = atomic_load_explicit(seq, memory_order_acquire);
    if (s0 == 0) return 0;
    if (s0 & 1u) continue;
    memcpy(out, region, size);
    atomic_thread_fence(memory_order_acquire);
    uint32_t crc;
    memcpy(&crc, (const uint8_t *)out + 8, sizeof crc);
    if (atomic_load_explicit(seq, memory_order_relaxed) == s0 &&
        crc32(0L, (const Bytef *)out + 16, (uInt)(size - 16)) == crc)
      return 1;
  }
  return 0;
}

/* HsimFeedback fb; hsim_read((const uint8_t *)base + HSIM_OFFSET_FEEDBACK, &fb, sizeof fb); */
```

## Device and performance

Run the host on CUDA. With its step captured as CUDA graphs and compiled, one unpaced step took 1.5 ms on average on an L4 (2.7 times faster than real time) with a new demand every step; [benchmarks.md](benchmarks.md) has the full figures. Device-to-host copies go into pinned buffers and the host thread sleeps on a blocking CUDA event while it waits, and it spins only for the last 100 µs before each deadline: the host shares its machine with Controls. Each truth publish waits for one more device-to-host copy, so `--truth-every 1` costs more per step than the default.

`humanoid-sim bench --runtime --device cuda:0` measures the host's whole step, unpaced, with a client writing a demand before every step. It reports `ms_per_step`, `ms_p99`, `ms_max`, `steps_over_period`, `real_time_factor` and whether CUDA graphs were used. [benchmarks.md](benchmarks.md) has the figures.

`--device cpu` runs MJWarp on Warp's CPU device, without graphs or compilation, well below real time. Once a step takes more than two periods, every step counts as late: `late_steps` climbs, `keeping_up` stays 0 and plant time runs slower than the wall clock. The protocol is the same, which makes the CPU device useful for tests and protocol checks. On the CPU, raise `--client-timeout-ms` so that a client which pauses its demands while it waits for an ack is not taken for lost: `tests/test_runtime_host.py` uses 1500 ms and Controls' lane script 10000 ms. [testing-and-gpu.md](testing-and-gpu.md) covers running the tests.

## Watching a host: `view-runtime`

`humanoid-sim view-runtime <segment>` shows a running host in [Rerun](https://rerun.io/), a viewer for 3D scenes and time series. It poses the robot from the truth region and plots, from the feedback and truth regions, each drive's position, velocity, applied torque and demanded torque, the newest IMU sample, the two soles' normal forces, the base height, the gantry tension, and the host's `late_steps` and `keeping_up`. It only reads the descriptor, feedback and truth regions. It never writes demands or commands, so it can watch a session that Controls or another client is driving without changing it, and it does not attach a session or keep one alive.

```bash
humanoid-sim view-runtime /run/hsim/plant                      # opens a Rerun viewer window
humanoid-sim view-runtime /run/hsim/plant --save runtime.rrd   # writes a recording instead
```

| Option | Default | Meaning |
|---|---|---|
| `segment` | required | The file `serve --segment` publishes |
| `--save FILE.rrd` | none | Write a Rerun recording to the file instead of opening a viewer |
| `--connect URL` | none | Stream to a Rerun viewer that is already running, for example `rerun+http://127.0.0.1:9876/proxy` |
| `--fps N` | `30` | Reads of the segment per second |
| `--seconds S` | until the host goes | Stop after this long |
| `--wait S` | `60` | How long to wait for the segment to appear and become ready |

Each read logs the truth and feedback records that are new since the last read, at their `sim_time_ns` on Rerun's `time` timeline (plant time in seconds). The recording therefore samples the host about `--fps` times per second, not every 4 ms step, and the robot's pose moves only as often as the host publishes the truth (every `--truth-every` steps, 20 ms by default).

When a session ends, the viewer follows the next session at the same path if it becomes ready within 10 s (`FOLLOW_S` in `view.py`), as with `serve --persist`, and logs the new session id under `/sim/0/host/session`. Plant time runs on across sessions, so the recording stays on one timeline. If no session follows, or after `--seconds`, or on Ctrl-C, it stops and prints how many robot frames it logged.

The viewer maps the segment file, so it runs on the machine the host runs on. It needs Newton and the Rerun SDK (the `newton` and `viz` extras natively; the Isaac Lab image has both) and the plain robot asset (or `--usd`), whose bodies it places with host kinematics. It refuses a segment whose robot id or description digest differs from the description it loaded (`--robot`). `Segment.open` maps the file read-write, so the viewer's user needs write permission on the file, as a client does, although the viewer writes nothing. With the host in a container, the simplest way is a second container on the same volume writing a recording, opened afterwards on any machine with `rerun runtime.rrd` (see [getting-started.md](getting-started.md#serve-the-runtime)). [visualization.md](visualization.md) shows the viewer's layout with screenshots.
