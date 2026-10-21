# Drives

The plant's input is one [demand](glossary.md#demand) per EtherCAT drive, the same record Controls' fake EtherCAT bank hands to its plant. Its output is what each drive output physically does: position, velocity, the torque the drive applied and the torque its control law asked for. Everything above that line (the standing frame, ankle targets in joint space, slew and blend, trip rules, the policy contract) belongs to Controls. This page covers the path from a demand to the physics solver. What the plant reports back is in [sensors.md](sensors.md).

## The drives

The robot has 22 drives. Eighteen are 1:1 drives: the drive output turns one joint. The other four are the ankle cranks, two per ankle, which move the foot's pitch and roll together through rods.

| Family | Drives | Joints | Peak torque | No-load speed | Encoder |
|---|---|---|---|---|---|
| HEJ90-48-140 | `maxonHEJ90M1`, `M2`, `L1` to `L4`, `R1` to `R4` (10) | hip_central_rotation, waist_upperbody_yaw, leg_flexion, leg_abduction, hip_yaw, knee | 140 Nm | 10.4 rad/s | 4096 counts/rev |
| HEJ70-48-50 | `maxonHEJ70AL1`, `AR1`, `AL2`, `AR2` (4) | shoulder_flexion, shoulder_abduction | 50 Nm | 22 rad/s | 16384 counts/rev |
| HEJ50-48-30 | `maxonHEJ50AL3`, `AR3`, `AL4`, `AR4` (4) | shoulder_rotation, elbow_flexion | 30 Nm | 21 rad/s | 16384 counts/rev |
| HEJ50-48-30 | `maxonHEJ50L_ANK1`, `L_ANK2`, `R_ANK1`, `R_ANK2` (4) | foot_pitch and foot_roll, through the cranks | 30 Nm | 21 rad/s | 16384 counts/rev |

Drive facts (family, sign, gains, torque-speed curves, ankle geometry) come from `third_party/humanoid-urdf/ethrc_v1/actuators.yaml` through `Description` in `src/humanoid_sim/model/description.py`. Encoder resolution is in `profiles/ethrc_v1.yaml` under `devices.encoders`. Every `[E, D]` tensor (envs by drives) uses the description's drive order, the 18 1:1 drives in joint order and then the four cranks; `PlantCore.map.drive_names` lists it.

## One physics step

`PlantCore` in `src/humanoid_sim/core.py` sits between the demands and the solver. For each physics step `[t0, t0 + 4 ms)` it does the following:

1. Converts the joint state to drive-frame state (`DriveJointMap.drive_state`): signs for the 1:1 drives, the crank transmission for the ankles.
2. Runs the drive bank (`DriveBank.compute`) over every demand in force during the step, giving demanded and applied drive torque.
3. Maps drive torque to joint torque (`DriveJointMap.joint_torque`: the sign, or Jᵀ for the ankles) and adds any external joint torque (`PlantCore.external_torque`: the runtime's joint disturbances, and tests).
4. Turns the joint torque into a position target for the solver's constant PD ([target shaping](glossary.md#target-shaping)).
5. After the solver step, `PlantCore.feedback` returns drive-frame feedback.

The physics period is 4 ms (`physics.period_ns` in `profiles/ethrc_v1.yaml`). All times are integer nanoseconds on the plant clock.

## Frames

The plant uses two sets of coordinates.

URDF joint coordinates are what the solver and the URDF use: one angle per joint, with the ankle as two hinges, foot_pitch and foot_roll. URDF zero is the [calibration pose](glossary.md#calibration-pose). A joint's [standing offset](glossary.md#standing-offset) (`standing_offset_rad` in actuators.yaml) is its URDF angle in the robot's reference standing pose. Controls works in standing-relative angles, `q_URDF - standing_offset_rad`; none of the plant's interfaces use them. [robot-model.md](robot-model.md#joint-coordinates) covers joint travel and the base frame.

The physical [drive-output frame](glossary.md#drive-frame) is what a drive measures and what its demand is written in:

| Drive kind | Position | Velocity | Torque to the joints |
|---|---|---|---|
| 1:1 | `q_drive = sign · q_URDF` | `sign · q̇` | `τ_joint = sign · τ_drive` |
| Ankle crank | crank angle relative to the linkage neutral | `J q̇` | `τ_hinges = Jᵀ τ_cranks` |

`sign` is the joint's `sign` (±1) in actuators.yaml. Positions are in rad, velocities in rad/s, torques in Nm at the drive output. The crank's zero (linkage neutral) is the drive's calibration zero; the ankle hinges are then at `ankle_model.semantic_neutral_rad` (pitch -0.329 rad, roll 0), not at URDF zero. Some readings at the [home pose](glossary.md#home-pose), which is URDF zero:

| Drive | Joint(s) | sign | URDF angle at home (rad) | Drive reading (rad) |
|---|---|---|---|---|
| `maxonHEJ90L4` | knee_l | -1 | 0.0 | 0.0 |
| `maxonHEJ90R4` | knee_r | +1 | 0.0 | 0.0 |
| `maxonHEJ50L_ANK1` | foot_pitch_l, foot_roll_l | crank | 0.0, 0.0 | -0.3317 |
| `maxonHEJ50L_ANK2` | foot_pitch_l, foot_roll_l | crank | 0.0, 0.0 | 0.3310 |

To compute them (CPU, no physics backend needed):

```python
import torch
from humanoid_sim.core import PlantCore
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.poses import home_pose

desc = Description.load("ethrc_v1")
joints = list(desc.joints)
plant = PlantCore(desc, joints, num_envs=1)
home = home_pose(desc, load_profile("ethrc_v1"))
q = torch.tensor([[home[n] for n in joints]])  # [1, 22] rad, URDF
q_drive, v_drive, jacobians = plant.map.drive_state(q, torch.zeros_like(q))
print(dict(zip(plant.map.drive_names, q_drive[0].tolist())))
```

### Encoder offsets

In training a drive can read its position with an offset from the truth: `DriveBank.position_offset` (`[E, D]`, rad), drawn by the `encoder_offset` randomization parameter. The drive closes its loop on its own reading, `q + position_offset`, so it settles where the reading equals the target and the true position is off by the offset. `RobotPlant.measured()` includes the offset in the reported position; `RobotPlant.truth()` does not. The runtime host leaves the offset at zero, because Controls' fake bank carries its own encoder bias.

## The demand

`DriveDemand` in `src/humanoid_sim/devices/demand.py` tells each drive which control law to run and with what targets and gains, in SI units and with nothing vendor-specific in it. It is the record of Controls' `docs/sim-learning-api-plan.md` (§1.2). Every field is `[E, D]`.

| Field | dtype | Unit | Meaning |
|---|---|---|---|
| `law` | int8 | | `LAW_DISABLED` (0), `LAW_PD` (1) or `LAW_FREEZE` (2). |
| `target_position_rad` | float | rad | Position target, drive-output frame. PD only. |
| `target_velocity_rad_s` | float | rad/s | Velocity target. PD only. |
| `feedforward_torque_nm` | float | Nm | Feed-forward torque. PD only. |
| `stiffness_nm_per_rad` | float | Nm/rad | P gain. |
| `damping_nm_s_per_rad` | float | Nm·s/rad | D gain. |
| `torque_limit_nm` | float | Nm | Cap on the applied torque; `inf` for none. |

Laws:

| Law | What the drive does |
|---|---|
| Disabled | No torque. |
| PD | Tracks the position and velocity targets with the demand's gains plus the feed-forward torque. |
| Freeze | Holds the position it read when it entered the law. |

A client converts its own device command into this at its boundary. For the ETHRC V1's maxon EPOS4 drives, Controls' plant host maps the CiA-402 mode of operation (object 0x6061): [JVPT](glossary.md#jvpt) (-64) is PD, freeze (-62) and homing (6) are freeze, anything else or a disabled power stage is disabled; the raw JVPT gains are divided by 1000 and a torque cap of 0 becomes `inf`. For a Unitree motor, `LowCmd` mode 1 is PD with the command's `kp`, `kd`, `q`, `dq` and `tau`, and mode 0 is disabled.

Helper constructors:

- `DriveDemand.disabled(num_envs, num_drives, device=..., dtype=...)` disables every drive, with no torque limit. Drives start with this demand, and an env reset puts it in force unless another demand is passed to `reset`.
- `DriveDemand.pd(position, stiffness, damping, *, velocity=None, feedforward=None, torque_limit_nm=None, law=LAW_PD)` builds an enabled demand. `position` is `[E, D]` in rad; `stiffness` (Nm/rad), `damping` (Nm·s/rad) and `torque_limit_nm` may be scalars, `[D]` or `[E, D]`. Pass `law=LAW_FREEZE` for a freeze demand.
- `as_dict()` and `DriveDemand.from_dict(d)` convert to and from a dict of tensors.

```python
import torch
from humanoid_sim.core import PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.model.description import Description

desc = Description.load("ethrc_v1")
joints = list(desc.joints)
plant = PlantCore(desc, joints, num_envs=1)
standing = torch.tensor([[desc.joints[n].standing_offset_rad for n in joints]])
q_drive, _, _ = plant.map.drive_state(standing, torch.zeros_like(standing))

# 0.05 rad past standing on every drive, at the baseline gains (humanoid-urdf baseline_position_gains).
demand = DriveDemand.pd(q_drive + 0.05, plant.drive_stiffness, plant.drive_damping)
print(demand.stiffness_nm_per_rad[0, :3])  # tensor([197.4, 197.4, 197.4]) Nm/rad

plant.submit(demand, torch.tensor([1_000_000]))  # in force 1 ms into the first step
plant.pre_physics(0, standing, torch.zeros_like(standing))
print(plant.last.demanded[0, :3])  # 7.40 Nm each: 197.4 Nm/rad · 0.05 rad over 3/4 of the step
```

## Drive models

Each drive is a product, and `src/humanoid_sim/devices/models.py` has one class per product, as the sensors do: `Hej90`, `Hej70` and `Hej50` (maxon HEJ joints on EPOS4 controllers, `Epos4Drive`) for the ETHRC V1, and `UnitreeN7520_14`, `UnitreeN7520_22`, `UnitreeN5020` and `UnitreeW4010` (`UnitreeMotor`) for the G1. A class names its `actuators.yaml` family; its figures are robot facts, so the description holds them and an instance reads them: `Hej90(desc)` is a HEJ 90 on that robot. `drive_models(desc)` gives every drive its model and refuses a family without a class.

| Attribute | Meaning |
|---|---|
| `family`, `product`, `interface` | The family key, the product, how the robot's controller drives it |
| `drive_delay_ns` | The family's `drive_delay_s` in integer ns: from the controller staging a demand (its APPLIED acknowledgement) to the drive acting on it, which is the bus master's staging, the bus and the drive's interpolation. 0 when the description records none |
| `delay_measured` | Whether the description records the delay; the runtime lists the families that do not in its fidelity gaps |
| `delay_source` | The description's `drive_delay_source` |

On the current V1 description every family has 5.0 ms, one whole-robot figure Controls identified over the 1:1 joints (`ethrc_ssot.command_latency`), until a per-family fit replaces it ([replay-and-sysid.md](replay-and-sysid.md#fitting-paces-actuator-model)). The G1's families record no delay.

## Arrival and latency

A demand leaves the master at `sent_ns` and takes effect at the drive at `arrival_ns`. `ArrivalQueue` in `src/humanoid_sim/timing/arrival.py` holds the demands in flight, per env, with one lane per drive so that each drive can have its own arrival time.

Where the arrival time comes from depends on the front end:

- Training: `RobotPlant.submit(demand, sent_ns)` (`src/humanoid_sim/train/plant.py`) sets `arrival_ns[e, d] = sent_ns[e] + latency_ns[e, d] + drive_delay_ns[d]`. `latency_ns` is the latency before the drive (the controller's path), from the `latency` randomization parameter (seconds, per env and drive, redrawn for envs being reset), 0 when it is not randomized; `drive_delay_ns` is the drive's own delay from its [model](#drive-models), always applied.
- Runtime: the host takes every record the client appended to the demand ring since the last physics step, oldest first, and queues each to arrive at drive `d` at its send time (`sent_mono_ns` mapped to plant time) plus that drive's `drive_delay_ns` ([Drive models](#drive-models); `RuntimeHost._take_demands` in `src/humanoid_sim/runtime/host.py`). Controls' fake bank sends each record when its bus delivers that cycle's outputs, so the lane's timing is the real master's up to the bus, then the drive's delay here.
- Replay: recorded demands with an optional fixed delay; see [replay-and-sysid.md](replay-and-sysid.md).

In training, with `plant = env.action_manager.get_term("drives").plant`:

```python
sent_ns = torch.full((plant.num_envs,), plant.t_ns, dtype=torch.int64, device=plant.device)
plant.submit(demand, sent_ns)  # each drive applies it latency_ns[e, d] later
```

Arrivals rarely fall on step boundaries. For a step `[t0, t1)`, `ArrivalQueue.weights` gives the fraction of the step covered by the record in force at `t0` and by each pending record from its arrival on. The bank evaluates its law once per record, at the step-start state, and sums the results weighted by those fractions. A demand arriving 1 ms into a 4 ms step therefore contributes with weight 0.75, as in the example above. A record whose arrival time is already past counts from the step start.

Other queue rules:

- Per lane, a record never overtakes an earlier one. An arrival earlier than the last queued arrival is moved up to it.
- A record leaves the queue once it has arrived on every lane.
- Capacity is the number of pending records per env: `PlantCfg.queue_capacity` (default 4, used by `SimCore` and the runtime) or `TrainPlantCfg.queue_capacity` (default 8). Size it as `ceil(max latency / demand period) + 1`. For example, 40 ms of latency at a 20 ms policy period needs 3.
- When an env's queue is full, its oldest pending record is dropped and `ArrivalQueue.overflows[e]` counts it. In training that counter is `plant.core.bank.queue.overflows`.

## The drive law

`DriveBank` in `src/humanoid_sim/devices/bank.py` follows Controls' MuJoCo plant (`ethrc_sim/plant.py` `Plant.step`). With `q̃ = q + position_offset` (the drive's reading) and `v` the drive velocity at the step start, each record in force gives:

- PD: `τ = kp (target_position - q̃) + kd (target_velocity - v) + feedforward`, with `kp = stiffness · p_scale` and `kd = damping · d_scale`.
- Freeze: the same law with the target latched to `q̃` when the drive enters the freeze, velocity target 0 and no feed-forward. The latch happens when the freeze demand arrives at the drive: the target is the reading at the start of the step in which the freeze demand first has a share of the step. A freeze demand that is queued but has not arrived latches nothing, and the drive keeps following the demand in force (`test_freeze_latches_when_it_arrives_not_when_queued` in `tests/test_devices.py`). Further freeze demands while the drive is frozen keep the latched target. The gains are those carried by the freeze demand, so a freeze demand with zero gains holds nothing.
- Disabled: zero torque, demanded and applied.

The bank returns a `DriveOutput`, kept as `PlantCore.last`. Each field is `[E, D]` and time-weighted over the step:

| Field | Unit | Meaning |
|---|---|---|
| `demanded` | Nm | Control-law output before any clip |
| `applied` | Nm | After the torque-speed envelope and the torque cap |
| `enabled_fraction` | | Fraction of the step the drive was not disabled |
| `target_velocity` | rad/s | Velocity target (0 when disabled or frozen) |

## Torque-speed envelope and torque cap

`TorqueSpeedEnvelope` in `src/humanoid_sim/devices/envelope.py` limits the applied torque by the drive velocity at the step start, using each family's `torque_speed` curve from actuators.yaml (maxon datasheets, 48 V):

- Motoring (torque and velocity of the same sign): the curve at `|v|`, linearly interpolated between points, and zero at or beyond the last point (the no-load speed).
- Braking (opposite signs) and standstill: the rated peak, the curve's first point.

| Family | Curve points (rad/s, Nm) | Nominal torque (not enforced) |
|---|---|---|
| HEJ90-48-140 | (0, 140), (5, 140), (6, 135), (8, 100), (10, 25), (10.4, 0) | 80 Nm |
| HEJ70-48-50 | (0, 50), (10, 45), (15, 30), (20, 10), (22, 0) | 19 Nm |
| HEJ50-48-30 | (0, 30), (10, 25), (15, 20), (20, 5), (21, 0) | 11 Nm |

The nominal (continuous) torque is `effort_nominal` in actuators.yaml. Nothing clips to it and there is no thermal derating, so a drive can hold its peak indefinitely.

A finite `torque_limit_nm` then clips the applied torque to `±torque_limit_nm`. Both the envelope and the limit are multiplied by the drive's `torque_scale`. The demanded torque is never clipped.

```python
import torch
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.model.description import Description

desc = Description.load("ethrc_v1")
env = TorqueSpeedEnvelope([desc.torque_speed("HEJ90-48-140")])
lower, upper = env.limits(torch.tensor([[-7.0], [0.0], [7.0], [11.0]]))  # rad/s
# lower: -117.5, -140, -140, -140 Nm; upper: 140, 140, 117.5, 0 Nm
```

## Per-drive deviations

The bank also carries how each drive deviates from what it is told. All are `[E, D]`, with no deviation by default, and written by `RobotPlant._write_device` from the randomization parameters (redrawn for envs being reset; see [training.md](training.md)):

| Bank field | Randomization parameter | Unit | Effect |
|---|---|---|---|
| `p_scale` | `drive_stiffness` | ratio | Realized P gain over the demanded one |
| `d_scale` | `drive_damping` | ratio | Realized D gain over the demanded one |
| `torque_scale` | `torque_limit` | ratio | Scales the envelope and the torque cap |
| `position_offset` | `encoder_offset` | rad | Drive's position reading minus the truth |
| `RobotPlant.latency_ns` | `latency` | s, stored as ns | Time from sent to the controller staging it; the drive's own delay follows |

In the runtime these stay at identity, except `torque_scale`, which the `power_sag` fault sets.

## Target shaping with constant solver gains

Isaac Lab's articulation has a joint PD inside the solver. MJWarp's `implicitfast` integrator treats its damping implicitly, which keeps the stiff ankle-crank drives stable at a 4 ms step. Changing the solver's gains at run time requires a model update, so they are fixed when the plant is built. The solver gains `k0`, `d0` of a 1:1 joint are its drive's baseline gains (`baseline_position_gains` in actuators.yaml). For the ankle hinges they are `diag(Jᵀ K J)` and `diag(Jᵀ D J)` with the crank gains K, D and J at the standing pose (`DriveJointMap.joint_space_gains`). They are written to the articulation at spawn (`src/humanoid_sim/physics/joint_properties.py`).

| Joints | `k0` (Nm/rad) | `d0` (Nm·s/rad) |
|---|---|---|
| HEJ90 joints | 197.4 | 12.6 |
| HEJ70 shoulders | 100 | 5 |
| HEJ50 arm joints | 22.1 | 1.41 |
| Ankle hinges (from 22.1 Nm/rad and 1.41 Nm·s/rad per crank) | 47 to 50 | 3.0 to 3.2 |

Every other force law (the drive's own gains, the envelope and cap, arrival blending, external torque) is realized by moving the solver's position target, with its velocity target at 0 (`shaped_position_target` in `src/humanoid_sim/devices/shaping.py`, called from `PlantCore.pre_physics`):

```
target = q + (τ + d0 · q̇) / k0      so that      k0 · (target - q) - d0 · q̇ = τ   at the step start
```

For a 1:1 drive whose demanded gains equal `k0` and `d0` and whose torque is not clipped, the shaped target reduces to the demanded target and the solver runs exactly its implicit PD on it. With other gains the solver applies `τ - k0 Δq - d0 Δq̇` over the step, where Δ is the change during the step. The module documents this as stable while the demanded gains stay within `K ≤ 2 k0` and `D ≤ 2 d0`.

`PlantCore.gain_warnings` (`[E, D]` bool) flags demands outside that region. `PlantCore.submit` sets it for every enabled demand whose P gain exceeds twice the drive's baseline stiffness or whose D gain exceeds twice its baseline damping (`outside_stable_region`). It compares the realized gains, the raw gains times the bank's `p_scale` and `d_scale` at submission, so randomized gain scales count. It only flags the envs the demand reaches (`env_mask`). It latches until the env is reset and does not change the demand. A demand put in force through `reset(env_ids, demand)` does not pass through `submit` and is not checked. In training check `plant.core.gain_warnings.any()`.

The solver's own joint effort limit is the drive peak for 1:1 joints and, for each ankle hinge, `Σ |J| · peak` over its two cranks at standing, so the solver's clamp does not bind before the drive envelope does.

## The ankle transmission

Each ankle has two HEJ50 drives stacked 75 mm apart on the back of the shin: the upper crank (`ANK1`, rod 269.0 mm) and the lower crank (`ANK2`, rod 194.1 mm). Each drive turns a crank; rods from the two cranks to the foot set pitch and roll together. Near standing, a pitch moves the two cranks in opposite directions and a roll moves them in the same direction, each by about 1.03 to 1.07 rad per rad of foot motion.

The simulated foot is two hinges in the URDF tree: foot_pitch, about the lateral axis (close to base x at standing), then foot_roll, about the axis along the foot (close to base y). Cranks and rods are not bodies in the physics: there is no closed chain. `AnkleTransmission` in `src/humanoid_sim/transmissions/ankle.py` maps between hinge state and crank state, batched in torch:

- `cranks(q)` gives crank angles (upper, lower) relative to the linkage neutral from hinge angles `q` (pitch, roll, URDF coordinates). It solves the circle-sphere intersection analytically on the assembly branch connected to the neutral, as humanoid-urdf's `ankle_model` does. With `return_reach=True` it also returns `|K/R|`; values of 1 or more mean the pose is out of the linkage's reach, where the result is clamped.
- `jacobian(q)` gives `J = dθ/dq`, `[..., 2 cranks, 2 (pitch, roll)]`, from the rod-length constraints.
- Crank velocity is `J q̇` and hinge torque is `Jᵀ τ_cranks` (`AnkleTransmission.joint_torque`), so power is conserved through the transmission.
- `joints_from_cranks(theta, q0)` inverts the map with 8 Newton iterations from `q0`. It is used for initial states and replays, never in the step loop.

`DriveJointMap` in `src/humanoid_sim/transmissions/drive_map.py` combines the 1:1 signs and both ankle transmissions into the drive-frame state and the joint torques of the whole robot.

The geometry comes from `ankle_model` in actuators.yaml: measured points and rod length for the right side (`P_shaft`, `n_shaft`, `P_crank_0`, `P_anchor_0`, `L` for the upper and the lower crank), mirrored across the sagittal plane for the left side (`mirror_right`). The linkage model has its own frame, with pitch about its y axis and roll about its x axis (`pitch_axis`, `roll_axis`); `semantic_to_kinematic_sign` maps URDF hinge angles into it. Nothing about the linkage is typed into this repo.

The ankle drives' rotor inertia reaches the hinges as reflected inertia. actuators.yaml provides it as `armature_joint`, the diagonal of `Jᵀ diag(a, a) J` at the neutral with `a` = 0.0056 kg·m² per HEJ50, and the plant uses it as the ankle hinges' armature: 0.0104 kg·m² in pitch and 0.0126 kg·m² in roll. It is constant (taken at the neutral) and diagonal only.

Golden tests in `tests/test_ankle_transmission.py` check the torch transmission on 256 random poses (pitch 0 to 0.7 rad, roll -0.35 to 0.35 rad) against both numpy references:

| Test | Reference | Tolerance |
|---|---|---|
| `test_cranks_match_ankle_kinematics` | `ankle_kinematics.joint_to_motor` (`third_party/ankle_kinematics`) | 1e-4 rad (the reference's angle is absolute, about 1e-5 rad from the neutral) |
| `test_jacobian_matches_reference_and_finite_difference` | `joint_to_motor_jacobian`, and central differences of `cranks` | rtol 1e-6; 1e-7 |
| `test_cranks_match_humanoid_urdf_ankle_model` | humanoid-urdf `ankle_model` | 1e-12 rad |
| `test_float32_batched_and_power_consistent` | power through the transmission | 1e-4 W |
| `test_joints_from_cranks_inverts` | round trip through `cranks` | 1e-10 rad |

They run without Isaac Lab: `uv run pytest tests/test_ankle_transmission.py` (needs the submodules).

## Dynamic friction

The solver's Coulomb friction (MJWarp frictionloss) holds a joint at rest up to the joint's static `friction`, and resists a sliding joint with the same torque. A joint whose `dynamic_friction` is lower gets the difference back while it moves: `PlantCore` adds the joint torque `friction_relief · tanh(q̇ / v_s)` before target shaping, with `friction_relief = friction − dynamic_friction` (`[E, J]`, Nm) and `v_s = SLIDING_VELOCITY_RAD_S` = 0.02 rad/s (`src/humanoid_sim/core.py`). At rest the tanh vanishes and the static friction holds; above a few `v_s` the net friction is the dynamic one. `v_s` is modeled: the drives' breakaway speed has not been measured.

The description states no dynamic friction yet, so the relief is zero by default. Training draws it through `joint_friction_ratio` (dynamic over static) and recomputes it from the solver's static friction, randomized or not, at every reset; `fit-pace` writes it per candidate. A 1:1 joint whose drive is off gets no relief, since its friction is then the back-drive torque below.

## Disabled drives: back-drive friction

A disabled drive is back-driven through its gearbox. Only the runtime host models this (`RuntimeHost._set_drives_off` in `src/humanoid_sim/runtime/host.py`). When a 1:1 drive's law is disabled in the latest demand record, the host sets its joint's Coulomb friction to the family's `backdrive_torque` in place of the enabled-drive friction from actuators.yaml (a family without one keeps the joint's friction). It writes the friction only when the on/off pattern changes. Before a client attaches every drive is disabled.

| Family | `backdrive_torque` | Datasheet figure |
|---|---|---|
| HEJ90-48-140 | 1.5 Nm | upper bound |
| HEJ70-48-50 | 0.5 Nm | approximate |
| HEJ50-48-30 | 0.8 Nm | upper bound |

The ankle hinges keep their friction, since a crank's figure has no joint-space equivalent. `RobotPlant` (training) and `SimCore` (replays, benchmarks) do not switch friction: an off drive there keeps the enabled-drive friction, or the randomized `joint_friction`.

## Physical faults

The runtime host injects physical faults on the `INJECT_FAULT` mailbox command (`RuntimeHost._inject`), with arguments `{"fault": name, "drive": name | null, "value": number, "duration_s": s | null}`. A null duration lasts until cleared; sending the fault's no-fault value clears it. Injecting a fault again replaces the earlier one of the same kind (and drive). The host checks the arguments first: the fault name, the drive for `encoder_freeze`, and a `value` that is finite and, for `power_sag`, within 0 to 1. A rejected command (`ACK_REJECTED`) leaves an active fault of that kind in place. Expiry is checked at the start of each physics step, so a paused host keeps its faults.

| Fault | Drive | `value` | Effect | No-fault value |
|---|---|---|---|---|
| `encoder_freeze` | required | any non-zero (default 1) | The published position and velocity of that drive hold the values of the last feedback before injection. The torques keep updating and the drive's law keeps acting on the true state. | 0 |
| `imu_bias` | ignored | rad/s | Added to the gyro bias on all three axes | 0 |
| `imu_spike` | ignored | m/s² | Added to the accelerometer bias on all three axes, for `duration_s` or else one physics period (4 ms) | 0 |
| `power_sag` | ignored | torque scale, 0 to 1 | Every drive's `torque_scale` is set to `value`, scaling the envelope and the cap | 1 |

Bus and protocol faults, and a drive whose encoder stops reporting, belong to Controls' fake bank (see `FAULTS` in `src/humanoid_sim/runtime/abi/plant_abi.py`). Training has no fault injection; `torque_limit` randomization covers reduced torque there.

```python
from humanoid_sim.runtime.abi import plant_abi as abi

seg = abi.Segment.open("/run/hsim/plant", wait_s=5)
seq = seg.send(abi.CMD_INJECT_FAULT, {"fault": "power_sag", "value": 0.6, "duration_s": 2.0})
print(seg.wait_ack(seq))  # (0, 'power_sag active'); 0 is abi.ACK_OK
```

The mailbox and its other commands are described in [runtime.md](runtime.md).

## What is not modeled

The runtime publishes its known gaps in the segment descriptor (`fidelity_gaps` in `src/humanoid_sim/runtime/host.py`):

- There are no drive current, thermal or backlash models. The applied torque is the envelope-clipped control law.
- The passive joint parameters of the upper body (waist, shoulders, elbows) are the description's values and have not been identified from captures.
- The ankle is two hinges, pitch and roll, driven through the crank transmission. There is no closed chain.
- IMU errors are the rig's model's: white noise, a turn-on bias and a slow drift (`SET_FIDELITY` ideal turns them off). See [sensors.md](sensors.md#sampling-and-errors).
- Each drive's delay is its family's fixed figure, with no per-cycle jitter. A family whose description records no delay (the G1's) has none, and the gap names it.

For the drive model this means the applied torque follows the law within the step, with no current-loop dynamics, and a drive can hold its peak torque indefinitely. Gearbox losses are only the joints' Coulomb friction and viscous damping from actuators.yaml, plus the back-drive friction of an off drive in the runtime. There is no integral term. The crank position windows in actuators.yaml are not enforced by the plant; the ankle hinges have joint stops at their ±30° position limits.
