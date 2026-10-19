# Glossary

Terms used across the docs and the code, in alphabetical order. Each heading is a link target, for example
`glossary.md#drive-frame`.

### ABI

The binary layout and protocol of the [runtime segment](#segment): record types, offsets, command numbers and
their JSON arguments. It is defined in `src/humanoid_sim/runtime/abi/plant_abi.py` and `plant_abi.h`, which a
client copies unchanged. A layout hash in the segment header lets both sides check they agree.

### Ankle crank

Each ankle has two drives (`maxonHEJ50L_ANK1`, `ANK2` on the left) that turn cranks. Rods from the cranks move the
foot, so both cranks together set the ankle's pitch and roll. The sim works with crank angles at the drive
boundary and hinge angles in the physics; `transmissions/ankle.py` converts between them. See
[drives.md](drives.md).

### Arrival

The moment a demand takes effect at its drive. A demand leaves the master at `sent_ns` and arrives
[latency](#latency) later. The arrival queue (`timing/arrival.py`) holds demands in flight per drive.

### Asset

The robot as a USD file that Isaac Lab loads, built by `humanoid-sim build-asset` from the
[description](#description) and the [profile](#profile) into `outputs/assets/`. The `--base-joints` variant adds
yaw, pitch and roll joints above the base for replays.

### Back-drive

Turning a drive's output by an external force while the drive is disabled. The runtime switches a disabled
drive's joint to its back-drive friction; training keeps the enabled friction.

### Baseline gains

The position gains in `actuators.yaml` (`baseline_position_gains`), per drive: for example 320 Nm/rad on the hip
drives and 600 Nm/rad on the knees. The solver's constant PD uses them (see [solver gains](#solver-gains)), and
the examples hold poses with them.

### Bus period

The EtherCAT cycle of Controls' master: 1.5 ms. A runtime client writes one demand record per bus cycle.

### Calibration pose

The pose in which every drive reads zero. URDF joint zero is the calibration pose.

### Capture

A recording from the real robot, such as the excitation captures `datasets/humanoid-v1/exc-*`: drive demands and
feedback at about 200 Hz per drive, IMU samples and the gains in force. Replays read them.

### Decimation

How many physics steps one policy step spans in an Isaac Lab environment. With the 4 ms
[physics period](#physics-period) and a decimation of 5, the policy runs every 20 ms.

### Demand

What the master sends one drive per cycle: mode, enable flag, targets, gains and torque limit. The sim's record
mirrors Controls' `FakeDriveDemand` field by field (`devices/demand.py`, `DriveDemand`).

### Description

The robot facts: the URDF and `actuators.yaml` in the `third_party/humanoid-urdf` submodule, loaded and
hash-checked by `Description.load()`. The same commit is pinned by Controls.

### Digest

A hash of the description files. The runtime segment carries it so a client can refuse to attach to a sim built
from a different description.

### Disturbance

An external force or torque applied for a while: a push on a body (`ForcePush` in training,
`APPLY_DISTURBANCE` in the runtime) or a torque on one joint outside the drive's feedback.

### Domain randomization

Drawing physical and device parameters per environment (friction, masses, latency, gains, sensor biases) so a
policy does not overfit one model. Specs are given in `TrainPlantCfg(randomization=[...])`; see
[training.md](training.md).

### Drive

One EtherCAT servo drive with its motor and gearbox (maxon HEJ 90, HEJ 70 or HEJ 50). The robot has 22. Drives
are named by their device, for example `maxonHEJ90L4` for the left knee.

### Drive frame

Positions and torques as a drive reports them: the joint angle times the joint's sign, relative to the
[calibration pose](#calibration-pose); for an ankle crank, the crank angle.

### Drive law

What a drive does with its demand in each mode: JVPT tracks targets with its own gains, freeze and homing hold the
position they entered with. Implemented in `devices/bank.py`.

### Encoder counts

The resolution of a drive's output encoder: 4096 counts per revolution on the HEJ 90, 16384 on the HEJ 70 and
HEJ 50. Reported positions are quantized to counts and torques to mNm.

### Env

One simulated robot with its own state in a batched simulation. Training runs thousands of envs on one GPU; the
runtime runs one.

### Envelope

The torque-speed envelope of a drive: the peak torque it can apply at a given speed. The applied torque is clipped
to it; the demanded torque is not.

### Fault

A failure injected into the runtime with `INJECT_FAULT`: `encoder_freeze` (a drive's reported position stops
updating), `imu_spike`, `imu_bias` or `power_sag` (every drive's torque scaled down).

### Feedback

What the drives report back: position, velocity, applied torque and demanded torque per drive, plus the IMU
samples. In the runtime it is the `feedback` region of the segment.

### Flat-foot pose

The [standing pose](#standing-pose) with the ankles adjusted so both soles lie level on the ground, and one leg
bent slightly so both soles are at the same height. A robot holding it at the [baseline gains](#baseline-gains)
stands for about 10 s before it leans over; one holding the standing pose tips over within a second. See
[robot-model.md](robot-model.md#standing-on-the-ground).

### Gantry

The overhead support the real robot hangs from. The sim models it as a tension program (a force that follows a
schedule) or as a cable with stiffness and damping. See [world.md](world.md).

### IMU burst

The IMU samples at about 365 Hz, but the host receives them in bursts every 13.9 to 17.6 ms. The sim reproduces
that delivery pattern (`sensors/imu.py`); see [sensors.md](sensors.md).

### Isaac Lab

NVIDIA's robot learning framework. humanoid-sim uses version 3.0 without Omniverse Kit, with Newton as the physics
backend.

### JVPT

The drive mode Controls uses for control (mode -64): the drive tracks a position and velocity target with its own
P and D gains and adds a feed-forward torque. Gains are sent as raw integers, 1000 per SI unit.

### Latency

The time from a demand leaving the master to its [arrival](#arrival) at the drive. Randomized in training; in the
runtime a demand takes effect at the next physics step.

### Mailbox

The command channel of the runtime segment: the client writes one command with JSON arguments, the host answers
with an acknowledgement (OK, rejected, failed, unknown).

### MJWarp

MuJoCo's solver reimplemented in NVIDIA Warp, so it runs batched on a GPU. Newton uses it as its solver here. MuJoCo
itself (the C library) never steps the robot in this repo.

### Newton

NVIDIA's physics engine library that Isaac Lab 3.0 uses. humanoid-sim also uses it on the host, without stepping
physics, for forward kinematics in viewers and model checks.

### Persist

`humanoid-sim serve --persist`: when a client detaches, reset the robot and open a new [session](#session) for the
next client instead of exiting.

### Physics period

The time one physics step covers: 4 ms (250 Hz), set in the profile.

### Plant

Everything between the drive demands and the robot's sensors: drive laws, transmissions, physics, sensors.
`PlantCore` is the batched part shared by every front end; `RobotPlant` wraps it for training and `SimCore` for the
runtime and replays.

### Prescribed pose

A pose the runtime holds the robot in, kinematically, until released (`PRESCRIBE_POSE`), for example to place it
before a test.

### Profile

`profiles/ethrc_v1.yaml`: the modeling choices that are not robot facts, such as the physics period, solver
settings, contact excludes and IMU delivery timing.

### Real-time factor

Simulated time divided by wall time. Above 1 the runtime keeps up with real time.

### Replay

Feeding each simulated drive the demands a real drive received in a [capture](#capture), then comparing what the
sim does with what the robot did.

### Rerun

The visualization tool humanoid-sim logs to. Recordings are `.rrd` files; a layout of views is a blueprint. See
[visualization.md](visualization.md).

### Segment

The shared-memory file `humanoid-sim serve` publishes for one robot: a header, the robot descriptor, the demands,
feedback, mailbox, acknowledgement and truth regions. Each region is written with a seqlock and a checksum so a
reader never sees a half-written record.

### Session

One client's use of a segment, from attach to detach. A new session gets a new session id; demands carrying an
old id are ignored.

### Sole box

The thin box on each foot that touches the ground. The asset build fits it to the foot's lowest collision face
and records it in `asset.json` (`sole_boxes`).

### Solver gains

The constant PD stiffness `k0` and damping `d0` per joint that MJWarp applies implicitly. They are written once at
reset. Every other force law is realized by moving the solver's position target (target shaping); see
[drives.md](drives.md).

### Standing offset

The joint angle of each joint in the robot's standing pose, measured from the [calibration pose](#calibration-pose)
(`standing_offset_rad` in `actuators.yaml`).

### Standing pose

All joints at their standing offsets. It is the reference for drive frames, ankle gains and the left/right mirror
map, and the default spawn pose.

### Support

How the runtime's robot is held: `suspended` (base welded to the world), `free` (on the ground), or `gantry`
(hanging from a modeled gantry). Set at startup and changed with `SET_SUPPORT`.

### Target shaping

Turning a drive's force law into a position target for the solver's constant PD, so that the solver applies the
torque the drive law asks for without changing its gains every step.

### Train-Time API

The `humanoid_sim.train` package: the plant inside an Isaac Lab scene, as an action term, with randomization,
measured and true observations, and signal logging. See [training.md](training.md).

### Truth

The simulated physical state that no sensor on the robot sees: joint state, base pose and twist, contact forces.
`plant.truth()` in training, the `truth` region in the runtime.

### Warp

NVIDIA's Python framework for GPU kernels. MJWarp and Newton are written in it. Warp's CPU device runs the same
kernels on a CPU, which the tests use.
