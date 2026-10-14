# humanoid-sim and humanoid-learning: architecture proposal

Draft, 2026-10-14 (revision 3.2: physics only through Isaac Lab/MJWarp, runtime on a GPU. Earlier revisions are kept as `humanoid-sim-learning.rev3.1.md`, `.rev3.md` and `.rev2.md`).

This proposal is based on:
- the "09 Envisioned stack" figure;
- surveys of ETHRC-Humanoid-Controls (`exp/walk-sim2real` at `447b4acb`) and rc_humanoid_rl_lab (`exp/walk-sim2real` at `dc2c603`);
- the excitation logs in `datasets/humanoid-v1/exc-*`;
- an adversarial review of revision 2, whose key claims were spot-checked against the code.

**What changed in revision 3** (all accepted on 2026-10-14):
1. Controls-in-the-loop runs are real-time only. There is no lockstep or fast mode, and no claim of determinism.
2. The ankle is modeled as hinges plus a transmission first. The closed chain becomes a time-boxed spike, and humanoid-sim takes `ankle_kinematics` as a submodule.
3. Solver gains are constant within an episode. Gain changes and the torque-speed envelope are applied in the device layer by target shaping.
4. Embodiment data comes from the resolved JSON that `ethrc_ssot` already generates, produced in humanoid-urdf's CI. Missing robot facts move into humanoid-urdf.
5. Bundles are identified by a hash of their files, promotion uses tags only, and per-run branches have a retention policy.
6. Isaac Lab is pinned to a `release/3.0.0` commit and re-pinned at GA.
7. Scope is trimmed:
   - two profiles;
   - no 100 µs reference for now;
   - cameras move to M2, Warp first;
   - lighter visualization and eval to start;
   - the Controls model is checked against real robot logs in M1.

**M0 outcomes (revision 3.1, 2026-10-14).** Full report: humanoid-sim `docs/m0-report.md` (branch `m0`).
- **Fidelity.** Drive-level replay of all three `exc-*` captures. Legs 3–16 mrad, inside or near PACE's band, with description values only. 4 ms matches 1 ms (ADR-5 holds). `implicitfast` is required for the ankle cranks (ADR-3). The upper body is the weak spot (waist, shoulder flexion and elbows up to 59 mrad): unmeasured passive parameters and the gantry base motion.
- **Backend parity.** Isaac Lab/MJWarp on the GPU and Newton/MuJoCo C on the CPU agree to ≤0.9 mrad on the same USD.
- **Throughput.** 1,751 sim-s per wall-s at 4096 envs and 6,463 at 16384 (L40S, no self-collision), against walk v2's ~220 for its whole loop. The full convex-hull self-collision set overflows, or runs out of memory, at 4096 envs, so training needs a pruned set (ADR-17).
- **Runtime.** Physics runs only through Isaac Lab on MJWarp (decided 2026-10-14, revision 3.2). MuJoCo C never steps the robot, so the real-time runtime runs on a GPU. Single-robot Isaac Lab/MJWarp on an L40S runs at 0.44× real time today: ~9 ms of per-step overhead, not physics. M2 must reach ≥1.5×. MJWarp on a CPU runs at 0.09× and is used for tests only.
- **Closed-chain ankle.** No-go for M1 (ADR-14): stable only with stiffened loop constraints, 2–2.4× softer than the hinge model under standing load, +19% cost.
- **Asset pipeline.** USD is the canonical sim model, built from humanoid-urdf by humanoid-sim (ADR-16).

Factual corrections from the review are folded in throughout: the drive-boundary shape, feedback fields, timing facts, the trip rule, IMU delivery, and dataset semantics.

## 0. Summary

- **humanoid-sim is the robot below Controls.** It is ROS-free and covers drives, mechanism, body, sensors and the world.
  - **Input:** per-drive demands identical in shape to Controls' `FakeDriveDemand`, for 22 EtherCAT drives including the 4 ankle cranks.
  - **Output:** physical drive-output position, velocity, applied torque and demanded torque, plus raw IMU samples (and camera frames, from M2).
- **Two front ends share one plant:**
  - **Train-Time API:** batched GPU tensors, in-process, used by humanoid-learning. It adds a reporting layer that mimics what Controls' fake drive (`Epos4Slave`) reports: encoder quantization and bias, velocity observer, current reading with bias.
  - **Runtime API:** one shared-memory segment that Controls' fake-coe plant host attaches to, in real time.
    - Per bus cycle: demands in, feedback out.
    - A command mailbox and a truth region carry the test seams.
    - Controls' `Epos4Slave` does the reporting.
- **Ankle.** Simulated as pitch and roll hinges driven through a crank transmission. The transmission is the batched torch port of `ankle_kinematics`. The real closed chain is a 3-day spike in M0.
- **Controller-level behaviour lives in Controls.** humanoid-learning keeps a batched `controls_model` of it.
  - Its stages are taken from the deployed code path.
  - It is checked against real robot logs (`.policyrun`, `exc-*`) from M1, plus golden vectors and declared `training_assumptions`.
- **Robot facts live in humanoid-urdf.** That includes the resolved embodiment JSON, contact primitives, the IMU spec, reading bias and hard stops. Plant modeling lives in humanoid-sim, and task design in humanoid-learning.
- **Physics:**
  - Isaac Lab 3.0 (`release/3.0.0` pin, then GA) with Newton MJWarp, at 250 Hz (4 ms). This is the only physics: `use_mujoco_cpu` is never used.
  - Solver PD gains are constant per episode.
  - MuJoCo stays in Controls for kinematics and logic only.
- **Visualization:**
  - **M0:** Isaac Lab's Rerun visualizer plus a one-call signal logger with naming conventions, and overlays against robot logs.
  - **Later:** a subscription engine, Viser interactive play, and cameras.
- **Evaluation** lives in humanoid-learning as plugins.
  - It starts with Scenario, Metric and Suite, testing checkpoints and real robot logs.
  - Other subjects and domains are added as needed.
- **Artifacts:** W&B tracks runs, and Hugging Face holds checkpoints and bundles.

## 1. What the figure implies, and what to fix

| Figure arrow | Concrete interface |
|---|---|
| humanoid-learning → humanoid-sim, "Train-Time API" | `humanoid_sim.train` plus Isaac Lab terms. In-process, batched, GPU. (§4) |
| Controls → humanoid-sim, "Runtime API" | One shared-memory segment, real-time, attached by Controls' fake-coe plant host. The ROS side stays in Controls. (§5) |
| humanoid-learning → Controls, "Checkpoints" | Bundle `ethrc-policy/v2` on Hugging Face, identified by a hash of its files. Controls pulls by that hash. (§9) |
| Controls ↔ GEAR-SONIC | Unchanged. |
| Controls → SLAM, "Sensor streams" | From M2: sim frame rings, published by a `sim` backend in `ethrc_cameras` on the real `/cameras` contract. (§3.11) |

Corrections to the figure:
1. The caption says "Both run on Isaac Sim". The stack is Isaac Lab 3.0 without the Isaac Sim app.
2. The caption says "humanoid-rl". The name is **humanoid-learning**.
3. Missing nodes:
   - `humanoid-urdf` (the root of the dependency graph);
   - `ankle_kinematics`;
   - Hugging Face and W&B;
   - the data loop robot logs → sysid → humanoid-urdf.
4. The blue group box is titled "t". Isaac Teleop is a client but sits in the plugin region.

## 2. Ownership map

Rule: each value lives in one repo. Others read it at a pinned revision, and CI fails on drift.

| Fact | Owner | Notes |
|---|---|---|
| Kinematics, inertias, meshes, joint names, signs, gear, armature, limits, torque-speed curves and braking peak, friction, ankle geometry, IMU and camera mounts, baseline gains, `standing_offset_rad` | humanoid-urdf (URDF, `actuators.yaml`) | Already the fact source. Pin `e2e47c4`. |
| **Resolved embodiment JSON** | humanoid-urdf CI, using the resolver promoted from `ethrc_ssot.embodiment` | Controls already emits a hashed `embodiment.resolved.json`. The sim and learning read only the JSON, so no ROS or ament dependency. Needs the Controls and urdf owners. |
| **Facts that currently live in Controls code** | Move to humanoid-urdf: sole contact primitives and contact pairs (`build_mjcf.py`), the MTi-610 sensor spec (`ethrc_ssot/imu_sensors.py`), the torque envelope and braking peak (`ethrc_sim/drive_config.py`), JVPT gain conversion (`ethrc_ssot/drive_gains.py`), and hard stops (calibration travel) | The sim must not grow a second copy of any of these. |
| **New facts to measure** | humanoid-urdf | Gear efficiency / current-reading bias (the RL lab's 1.15); crank and rod masses; measured total mass (61–63 kg against the model's 58.7 kg) |
| Drive protocol and drive reporting: CiA-402, OD, homing, encoder counts and bias, velocity observer, current via kt, 100 ms watchdog | Controls fake-coe (`Epos4Slave`) | The sim's training-mode reporting layer mirrors it, with a golden test. |
| Drive electromechanics: PD, envelope, current, friction, backlash, thermal | humanoid-sim device layer | Three copies today: Controls `Plant.step`, RL lab `walk_v2/core.py`, the MuJoCo screen. |
| Ankle math | `ankle_kinematics` v2: Python reference, batched torch port and golden vectors. Submodule in Controls and humanoid-sim; humanoid-learning gets it through humanoid-sim. | Controls is consolidating on one C++ ankle core called from Python. That core is tested against the same golden vectors. **No second C++ implementation.** Needs agreement with Controls before M1. |
| Controller behaviour (see §7.5 for the stage list) | Controls | Stays there. |
| Batched training model of that behaviour | humanoid-learning `controls_model/` | Validated against real logs. |
| Runtime ABI (`plant_abi.h` + `plant_abi.py`) | humanoid-sim; Controls vendors the files | Uses `ethrc_unitree_segment` as the precedent. |
| Plant modeling choices: solver, integrator, contact parameters, randomization defaults | humanoid-sim `profiles/` | |
| Scenes and assets | humanoid-sim, in-tree, Git LFS | |
| Camera calibration store | **New. Owner needed.** | Controls caches only the ZED factory calibration. |
| MDP, rewards, curricula | humanoid-learning | |
| Robot API, bundle loader, golden-vector CI | Controls | The v2 loader and golden-vector checks are new Controls work (§13). |

### 2.1 Repository and submodule graph

```
humanoid-learning
  └─ third_party/humanoid-sim        recursive → humanoid-urdf, ankle_kinematics
  └─ third_party/GEAR-SONIC          fork; same commit Controls pins in sonic_bundles.yaml
  └─ third_party/rsl_rl              only if a fork is needed (prefer subclassing)

humanoid-sim
  └─ third_party/humanoid-urdf
  └─ third_party/ankle_kinematics
  (assets/ in-tree, Git LFS)

ETHRC-Humanoid-Controls
  └─ humanoid-urdf, ankle_kinematics (existing submodules)
  └─ vendors humanoid-sim/runtime/abi/{plant_abi.h, plant_abi.py}
```

**Rules:**
- humanoid-urdf and `ankle_kinematics` must be at the same commits in Controls and humanoid-sim. humanoid-sim's CI reads Controls' pinned SHAs and fails on a mismatch.
- The vendored ABI is checked in CI and at runtime: ABI semver with N−1 support, layout hash, description digest, robot id, session id.
- Nesting stops at two levels. Installs go through uv path sources. CI and Docker use `--recurse-submodules` and skip LFS smudge, except for render jobs.
- Not submodules:
  - Isaac Lab and Newton (the uv lock pins them);
  - datasets, checkpoints and bundles (Hugging Face);
  - humanoid-slam.

## 3. humanoid-sim

### 3.0 Starting point and rules (`AGENTS.md`)

The repo starts fresh. The skeleton (`6d4ac3c`) is discarded with nicosuter's agreement. Rules:
- **Quantities:** requested, applied and measured quantities stay separate. Never synthesize a sensor the robot doesn't have.
- **Time:** integer nanoseconds everywhere, including delays and windows. Never count physics steps.
- **Imports:** Isaac Lab and Newton are imported lazily, so CPU contract tests run without them.
- **Backends:** selected explicitly, with no silent fallback. The runtime is pinned in a lock file.
- **Modeled vs measured:** anything modeled is labeled as modeled until traces verify it. The description is loaded hash-verified.
- **Description-driven:** the model build and device inventory come from the description. Nothing hand- or end-effector-specific is hard-coded.
- **No ROS.**

The skeleton's multi-arch Dockerfile, checksum-pinned `uv` install, `runtime.lock.yaml` and CI workflow are references.

### 3.1 Scope

**In scope:**
- the model build from the resolved embodiment;
- the device layer (electromechanics, plus training-mode reporting);
- physics, including the hinge ankle with transmission;
- hardware-level sensors (drive feedback, raw IMU with its delivery model; cameras from M2);
- the world;
- randomization;
- the Train-Time API and the runtime segment;
- replay and sysid;
- signal logging and viewers;
- benchmarks.

**Out of scope:**
- ROS;
- controller-level behaviour (standing frame, the IMU filter, joint-state estimation, action path, trip rule, policy contract);
- the drive protocol and, in runtime mode, drive reporting (both are `Epos4Slave`'s);
- bus faults;
- tasks, rewards and trainers.

**Coordinates:**
- **Truth:** URDF joint coordinates.
- **Drive interface:** the *physical drive-output frame* (`sign · q`; crank angle for the cranks). This is what Controls' plant already exchanges.
- **Encoder bias and calibration error** are added only by the training-mode reporting layer. Under fake-coe the bank owns them, so applying them in the sim would double-count.

### 3.2 Layers

```
     Train-Time API (batched, in-process)          Runtime segment (shm, real-time, attached by Controls)
                       \                                    /
          SimCore: 4 ms step ordering, reset, snapshot/restore, signal logging
     /             |               |              |              |             \
 devices/      transmissions/   physics/       sensors/        world/       randomization/
 demand →      direct; ankle    Newton MJWarp  IMU raw +       terrain,     typed specs,
 effort;       (ankle_kinematics constant-gain delivery model, scenes,      per-env tensors
 reporting     torch port)      implicit PD    contacts truth  supports
 (train only)
 model/ (resolved embodiment JSON → Newton model; contact primitives; hard stops)
```

### 3.3 Package layout

```
src/humanoid_sim/
  model/          resolved-embodiment loader (hash-verified), name/order tables, URDF → Newton build,
                  contact primitives and pairs, hard stops, mirror map
  timing/         integer-ns clock, demand arrival blending, sensor sampling
  devices/        demand.py (FakeDriveDemand mirror, gain conversion), shaping.py (target shaping:
                  gains, envelope, current limit), reporting.py (Epos4Slave mirror; training only),
                  protection.py, unitree.py (G1, SIL only)
  transmissions/  direct.py, ankle.py (wraps ankle_kinematics' torch port)
  physics/        backend.py (Protocol), newton.py
  sensors/        imu.py (raw + delivery model), contacts.py (truth)
  world/          terrains.py, scenes.py, supports.py, disturbances.py
  randomization/  spec.py, samplers.py, apply.py
  core.py         SimCore
  train/          api.py (RobotPlant), isaaclab/ (scene factory, plant hook, obs/event terms)
  runtime/        host.py, abi/plant_abi.h + plant_abi.py (Python ≥3.10, NumPy only)
  replay/         readers (exc-*, .policyrun, MCAP), replay.py, compare.py
  sysid/          objectives, fitters (CMA-ES, PACE port), reports
  signals/        naming.py, log.py (Rerun + Arrow), layouts/
  cli.py          view | serve | replay | sysid | bench
assets/           scenes (USD), later splats (Git LFS)
profiles/         ethrc_v1.yaml, g1_29dof.yaml, fidelity/{train,runtime}.yaml
third_party/      humanoid-urdf, ankle_kinematics
```

### 3.4 Timing and profiles

**Robot timing facts** (corrected):

| Item | Fact | Where it lives |
|---|---|---|
| Drive PD | 2.5 kHz inside the drive | sim (implicit in the solver) |
| EtherCAT cycle | 1.5 ms; distributed clocks off; drive interpolation period 2 ms | Runtime: real, in Controls. Training: part of bus/drive latency. |
| Master command shaping | `stageSlewedCommand`: velocity slew at the envelope velocity on every drive; a **20 ms** linear endpoint blend on the ankle cranks (`kCommandInterpolationPeriod`); zero velocity and torque feed-forward until the target is reached | Controller-level, so in `controls_model` (§7.5). Not in the sim. |
| Router | Routes per frame. The 200 Hz is the `/rt/joint_states` republish timer, not a routing rate. | `controls_model` |
| Command latency | 18.2 ms mean (2.4 ms std) from the frame stamp, taken *after* inference, to the onset of the drive's position demand. Measured on the 18 one-to-one joints, ankles excluded. Excludes observation age and inference time. PACE separately identified a 1.3 ms command delay, which still needs reconciling with the 5 ms "device" stage. | The latency model is rebuilt from `.policyrun` sample ages, compute time and pacer drops, with observation-side and action-side latency separate. Crank latency gets measured. |
| IMU | MTi-610, stamped on host receipt. Delivered in bursts: across the three `exc-*` datasets the median gap is 13 µs, p99 is 17 ms, the max is 19–22 ms, and the effective rate is about 365 Hz. | Delivery model in sim sensors |

**Physics runs at 250 Hz (4 ms) in both profiles (ADR-5).** The 20 ms policy period is exactly 5 steps.

| Robot behaviour | Representation at 4 ms |
|---|---|
| Drive PD | The solver's PD at **constant per-episode gains** (K0, D0), written only at reset. The integrator is set explicitly; `implicitfast` (implicit in kd only) and `discrete` are compared in M0. |
| Gain changes: leases, rest-start step, 1 s balance→policy fade, lease lapse after 2 s; torque-speed envelope; current limit | **Device-layer target shaping.** Each step computes the force the real drive would produce and writes the position target at which the constant solver PD produces it. This is walk v2's approach, and it avoids Newton's host sync on runtime gain writes. |
| Bus and drive latency | Arrival-time blending of demands within a step |
| Current, thermal, demanded torque | Computed from the shaped force, including the gear-efficiency reading bias from humanoid-urdf |
| Backlash | Randomized target offset plus compliance |
| IMU | Raw samples produced with the measured burst and host-receipt timing |

**Profiles:**
- **`train`:** batched, with the reporting layer. A fixed-base flag covers sysid replay.
- **`runtime`:** a single robot without the reporting layer, real-time, on a GPU (Isaac Lab/MJWarp). MJWarp on Warp's CPU device is for tests only (0.09× real time).

A 100 µs reference simulator is deferred. Fidelity is checked by comparing 4 ms against 1 ms of the same stack, and against `exc-*` real data. A dedicated reference is built only for a term that real data can't settle.

### 3.5 Drive boundary and device layer

**Demand: an exact mirror of `FakeDriveDemand`** (Controls `fake_bank.h`):
- `torque_enabled`;
- `mode`: 6 homing → freeze, −64 JVPT, −62 freeze, 0 none;
- target position, velocity and feed-forward torque;
- raw `p/i/d` gains, converted with the JVPT conversion from humanoid-urdf. A non-zero I-gain is refused, as Controls' plant does today;
- `max_torque_mnm`;
- `homing_hold`.

There is no DAMPING mode: on hardware, damping is JVPT with P=0 and D set by SDO.

**Feedback:**

| Mode | Fields |
|---|---|
| Runtime | Physical frame: position, velocity, applied torque, demanded torque. Exactly what `fake_bank_feedback` accepts today. More fields need a Controls ABI extension and are added only when Controls consumes them. |
| Training | The same physical quantities, then passed through `reporting.py`: encoder counts and bias, the 400 µs velocity observer, ±1 LSB jitter, current via kt with reading bias. This mirrors `Epos4Slave`. A golden test runs `Epos4Slave` through the fake-bank C ABI and compares. |

**Electromechanics**, with per-env `[E, D]` coefficients:

| Term | Notes |
|---|---|
| PD | Constant solver gains plus target shaping (§3.4) |
| Torque-speed envelope and braking peak | From humanoid-urdf |
| Current model | kt, gear efficiency and reading bias, current limits, bus-voltage sag |
| Friction | Coulomb, viscous and Stribeck terms; back-drive friction while disabled |
| Backlash and compliance | Fitted from the motor- vs output-encoder residual |
| Thermal | i²t, winding temperature, derating |
| Armature | Physics inertia |

**Trip.** Controls' trip rule is in Controls: demanded torque above the datasheet peak for 10 ms, with samples at most 5 ms apart. The sim supplies demanded torque *with* the reading bias, so trips match. Controls ask: make sure fake-coe's `demanded_torque` carries the bias, otherwise SIL under-trips.

**Other device kinds.** The device inventory comes from the description, with an adapter per device kind. G1 (Unitree board, LowCmd) runs in SIL only, with stock SONIC. End-effectors on non-EtherCAT buses get their own adapter when they arrive.

### 3.6 Sensors

- **Drive feedback:** §3.5.
- **IMU:**
  - raw gyro and specific force at the description mount, from the MTi-610 spec in humanoid-urdf;
  - bias, random walk, noise, misalignment;
  - **delivery model:** bursts and host-receipt stamps, fitted to `exc-*` `imu.csv`.
  - Madgwick runs in Controls (real nodes) and in learning's `controls_model`. A Python reimplementation already exists in `ethrc_sim/sensors.py`; use it as the seed and oracle for the torch port.
  - Hardware ask: confirm the FTDI latency-timer udev rule is active.
- **Truth:** joint state (URDF), base and link poses and twists, contacts, applied efforts, terrain.
- **Typed split:** in training, actor observations come only from `Measured` (reporting-layer drive feedback and IMU) through `controls_model`.

### 3.7 World

- **Terrain generator** with a difficulty parameter.
- **Supports:** gantry, suspended, free.
- **Hard stops** from the description, needed for the hard-stop and calibration scenarios.
- **Scenes:** USD `hangar`, `empty` (in-tree LFS).
- **Disturbances:** pushes, wrenches, payload.

### 3.8 Randomization

- **Typed `PlantParamSpec`:** nominal value from the embodiment, a distribution, a scope (startup, reset or episode), level scaling.
- **Logging:** samples are logged per env.
- **Coverage:**
  - mass, CoM, payload and inertia;
  - friction;
  - **encoder and calibration offsets.** Widen the ankles beyond today's ±3°: the left ankle zero is suspected to be 7.5° off;
  - gain error, armature, Coulomb friction, torque ceiling, backlash;
  - bus/drive latency;
  - IMU errors and delivery;
  - ankle geometry tolerances.
- **Write costs:** physics writes (mass, inertia, armature, solver gains) happen at startup or reset only.

### 3.9 The ankle

**M1 path: hinges plus transmission.**
- Pitch and roll are hinges. The four crank drives act through `ankle_kinematics`' batched torch port:
  - crank torques → joint torques (Jᵀ);
  - joint state → crank angles and velocities for feedback;
  - reflected crank inertia.
- Three existing implementations already use this structure, so it is the known-good path.

**`ankle_kinematics` v2** (existing ETHRoboticsClub repo, reworked):
- a Python reference, a batched, fixed-iteration, differentiable torch port, and golden vectors keyed by geometry hash;
- the true feasible set (replacing radial projection), the torque polytope, and calibration fitting;
- geometry read from humanoid-urdf;
- v2.0 reproduces today's outputs exactly before any fixes.

Controls' consolidated C++ core is tested against the same golden vectors, and no new C++ implementation is created here. The V1 runtime path today is the Python `ankle_bringup` node, which includes Guard E and nearest-reachable clamps; `controls_model` covers those (§7.5).

**Closed-chain spike, time-boxed to 3 days in M0.**

Known obstacles:
- Isaac Lab 3.0-EA lists closed-loop robots as "not validated on Newton".
- Loops must come from USD `excludeFromArticulation`, and drives, limits and armature on the closing joint are ignored.
- The URDF has no crank or rod links.
- `ankle_model` has no masses.
- The range is a 2-D hull.
- The forward map has spurious assembly modes.

Numeric pass criteria: stable at baseline crank gains (480/8) and walk gains (100/3.5); ankle-angle error under 700 N foot load; reset by settling with weld targets; throughput delta. The go/no-go decision is recorded. If it's a go, humanoid-urdf first needs crank and rod mass properties.

**Open data question:** ankle peak torque 30 vs 62 Nm, continuous 11 vs 9.9 Nm. Needs a measurement or the datasheet.

### 3.10 MuJoCo: kinematics and logic only

- **Stays in Controls:**
  - MuJoCo for `balance_ethrc`, `balance_search.py`, `footprint.py` and the description tools;
  - the MJCF build for those.
- **Replaced by humanoid-sim:**
  - the MuJoCo physics lanes (python lane, fake-coe plant, G1 plant);
  - the MuJoCo evaluator;
  - the RL lab's MuJoCo screen;
  - the WebUI's "MuJoCo viewer".
- **Model consistency.**
  - In M0, check the sim model against Controls' MJCF on 10 poses: FK, CoM, mass, and collision primitives and contact pairs.
  - Later, both builds test against the resolved embodiment JSON in humanoid-urdf's CI.
- **Deletion gate.** The MuJoCo physics lanes are deleted only after two things:
  - the GPU runtime benchmark passes (single robot ≥1.5× real time on Isaac Lab/MJWarp). Mac developers run physics on a remote GPU; locally they get kinematics and viewers;
  - the runtime passes Controls' integration suite (§5.6).

### 3.11 Cameras (M2 and later)

Target hardware: ZED / ZED Mini stereo (HD720 per eye at 30 Hz, fisheye on the Mini, rolling shutter, its own IMU), RealSense, DepthAI, and the Mid-360 LiDAR on the G1.

**Sequence:**
1. **Warp** renderer (kitless, batched) with distortion remap, noise and the camera's own timing.
2. Frame rings in the segment, and a `sim` backend in `ethrc_cameras` publishing on the real `/cameras` contract.
3. **OVRTX**, only after a legal check. It is proprietary-licensed, and the NVIDIA terms restrict redistribution.
4. **splat** (the hangar Gaussian splat) and **kit** (segmentation ground truth) when SLAM eval needs them.

**Prerequisite:** the camera calibration store (§2) needs an owner.

**Rerun:** camera logging is off by default and uses encoded images. Raw stereo 720p at 30 Hz is about 166 MB/s.

## 4. Train-Time API (humanoid-sim → humanoid-learning)

```python
from humanoid_sim.train import RobotPlant, PlantCfg

plant = RobotPlant(PlantCfg(robot="ethrc_v1", fidelity="train", randomization=spec),
                   num_envs=4096, device="cuda:0")

plant.names.devices     # drives from the description (22 for V1, incl. 4 cranks)
plant.submit(demand, at_ns)                # FakeDriveDemand mirror, [E, D]; arrival time per env
plant.pre_physics(); ...Isaac Lab step...; plant.post_physics()
m = plant.measured()    # reporting-layer drive feedback, raw IMU samples with delivery stamps
t = plant.truth()       # URDF joint state, poses, twists, contacts, efforts, terrain
p = plant.params()      # realized randomization
plant.reset(env_ids, state)                # URDF coordinates
plant.log(name, value, envs=...)           # signal logging (§6)
```

**Contract:**
- no host syncs in steady state, checked with a profiler in M0;
- solver gains written only at reset;
- integer-ns time;
- semver, pinned through humanoid-learning's submodule commit.

## 5. Runtime API (Controls → humanoid-sim)

### 5.1 Shape and topology

```
core-body container (Controls, ROS, Python 3.10)                sim container (humanoid-sim, Python 3.12)
 master --transport fake-coe ⇄ Epos4Slave bank (C++)
     │ FakeDriveDemand / fake_bank_feedback
 plant_host  ══ runtime segment (file-backed mmap on a shared volume) ══  sim host process
   wakes on the doorbell; demands → segment;                              4 ms physics, real-time
   feedback → bank; raw IMU → Madgwick path;                              Viser / Rerun, scenario
   mailbox ← Robot API sim actions; /ethrc/plant ← descriptor             control in-process
```

### 5.2 Why a mailbox, not gRPC

gRPC was chosen originally because the stack already uses it. On the merits, the only out-of-process client is Controls' plant host on the same machine, sending a few low-rate commands. One shared-memory segment with a mailbox means one transport, one ABI and no dependencies. It also matches Controls' existing mmap style and the `ethrc_unitree_segment` precedent. Viewers and eval live in the sim host process. A network adapter for HIL comes later.

### 5.3 Segment layout

Fixed layout; each region has one writer and a seqlock.

| Region | Writer | Contents |
|---|---|---|
| Header | sim | ABI semver, layout hash, robot id, session id, description and `ankle_kinematics` digests, bus period |
| Descriptor | sim | JSON: profile, fidelity gaps, revisions, viewer URLs, lateness counters (`keeping_up`, `late_cycles`) |
| Demands | Controls | Bus cycle counter, `FakeDriveDemand[D]` with arrival time |
| Feedback | sim | Sim time, physical feedback `[D]`, raw IMU samples with delivery stamps |
| Mailbox / acks | Controls / sim | Fixed-size commands with `seq`; acks with result codes |
| Truth | sim | The oracle Controls' tests need today: `StandingTruth`, sole-box normals, base pose |
| Doorbell | sim | eventfd or futex. plant_host wakes on it instead of polling. |

**Mailbox commands:**
- `RESET` (named pose or state), `PAUSE`, `RESUME`, `SET_SUPPORT` (gantry, suspended, free), `APPLY_DISTURBANCE`;
- the test seams the python lane offers today: calibration pose, base tilt, disturb;
- `INJECT_FAULT` for physical faults only: encoder dropout or freeze, IMU spike or bias, power sag.

Bus, transport and protocol faults, power cycling and NVRAM seeding stay in fake-coe (`bank.json`).

### 5.4 Clock

**Real-time only.** The sim paces 4 ms steps to the wall clock and answers each 1.5 ms bus cycle with the latest feedback. Controls' bridge, SDK, mailbox, pacer, grants and leases run on wall or steady time by design. Its own docs say a virtual-time lane needs a separate cross-module clock design.

Consequences:
- No lockstep or fast mode, and no determinism claims.
- Deterministic tests live below the bridge (Controls' `VirtualClock`) or in the Train-Time API.
- SIL gates are statistical, with recorded seeds.
- Lockstep, if ever wanted, is a Controls ADR.

### 5.5 Failure semantics

- **Startup:**
  - the sim host writes the header and descriptor, then sets ready;
  - plant_host checks the ABI, robot id and digests, then attaches;
  - the session id binds both sides.
- **Either side dies → the session ends.** The other side detects the stale heartbeat (the bank already goes stale at 500 ms), faults cleanly and exits. Compose restarts both, and a test covers it.
- **Lateness.** Real-time hiccups trip safety: the drive watchdog is 100 ms, and the bridge refuses readings older than 100 ms.
  - Measure the added latency of the extra hop in M2.
  - Publish lateness in the descriptor.
  - Keep GPU stalls (Rerun flushes, recapture) off the step thread.

### 5.6 Compatibility gate and prerequisites

- **Gate.** humanoid-sim is Controls-compatible when Controls' integration suite passes against the runtime segment, including the hard-stop and calibration scenarios.
- **Controls prerequisite.** The fake-coe lane must first pass the suite on MuJoCo.
  - Today the suite defaults to `bus=python`, and the fake-coe plant host lacks `/sim/*` handling and several test seams.
  - These seams become the typed mailbox commands and truth region above.

## 6. Visualization

**Goal:** see the robot with any stream, with almost no setup, in training, eval, SIL and real logs.

**M0:**
- Isaac Lab 3.0's Rerun visualizer renders the scene.
- `log(name, value, envs=…)` writes named signals into the same Rerun recording and into Arrow.
- Naming: `/{source}/{env}/{group}/{name}`. Examples: `/sim/0/drives/<drive>/torque_demand`, `/sim/0/imu/gyro`, `/sim/0/truth/base/twist`, `/real/drives/…`.
- `humanoid-sim view` and `humanoid-sim replay <exc-*> --compare` show sim and real overlaid, with residuals.

**Next:**
- learning logs every manager term automatically;
- Viser interactive play (env picker, gamepad, push, sliders);
- the subscription engine (globs, decimation, pinned side-stream copies), once a second consumer exists;
- layout presets as Rerun blueprints;
- a Controls-side Rerun tap in SIL.

**Recordings:** `.rrd` files go to W&B as artifacts. Real logs come from `exc-*`, `.policyrun` and MCAP (if Controls records rosbag2 as MCAP).

## 7. humanoid-learning

### 7.1 Scope

**In scope:**
- tasks, MDP terms, algorithms, motion data;
- `controls_model` (§7.5);
- training infrastructure and export;
- evaluation.

**Out of scope:** plant parameters, which come from humanoid-urdf and humanoid-sim.

### 7.2 Layout

```
src/humanoid_learning/
  controls_model/  stages from the deployed code path (§7.5), integer-ns timing
  tasks/      locomotion/velocity (walk v2 port first), locomotion/stand, recovery, tracking, teleop
  mdp/        observations (actor: Measured through controls_model; critic: Truth), rewards, terminations,
              commands, curricula, events
  agents/     rsl_rl configs; ScheduledPPO as a subclass; symmetry; distillation
  export/     bundle builder, golden vectors, checks
  eval/       §8
  experiments/  named presets as code
scripts/ train | play | export | eval | promote      sky/
third_party/ humanoid-sim (recursive), GEAR-SONIC, rsl_rl (only if needed)
```

**Which history to inherit:** port walk v2 from `exp/walk-sim2real`, and cherry-pick from `master` the thermal audit, walking gate, distributed Sky launches and PACE (which moves to humanoid-sim). No env-var-driven behaviour, no monkeypatches, no module globals, pinned rsl-rl, CI on every PR.

### 7.3 W&B and Hugging Face

- **W&B:** config, revisions, curves, every reward and termination term, eval reports, videos and `.rrd` artifacts. Uses a team entity.
- **Hugging Face** (private org, currently `ETHRC-humanoid/rc-humanoid-locomotion`):
  - **Checkpoints:** synced during training to `runs/<run-id>` branches (`--hf-sync every:500`, on by default on Sky). Resume pulls from there. **Retention:** prune old `runs/*` branches.
  - **Bundles:** identified by the **sha256 of their files**, which is the same digest Controls computes. A Hugging Face commit is not a content address, because it covers the whole repo.
  - **Promotion:** tags only (`candidate/…`, `sim-gated/…`, …). `MANIFEST.yaml` holds summaries, and editing it creates a new commit, so nothing references bundles by commit.
  - **Controls** pulls by bundle hash.
  - **Credentials:** the HF token and W&B key come from Sky secrets.
- **Note:** an S3 bucket (`s3://ethrc-rc-humanoid-ckpts`) already exists, and uploading from AWS to Hugging Face costs egress. The decision is Hugging Face only. Monitor egress, storage and rate limits (all unverified).

### 7.4 Training capabilities

- asymmetric actor–critic;
- teacher–student distillation;
- symmetry from the description;
- history and RNN encoders;
- velocity estimators;
- AMP;
- SONIC-style tracking (multi-GPU);
- staged curricula;
- finetuning from a bundle;
- sweeps.

### 7.5 `controls_model`: stages from the deployed code

Every stage cites its Controls source, and every stage is tested against real logs.

| Stage | Controls source |
|---|---|
| Observation sampling: latest snapshot per tick, asynchronous sources | deployment `feedback.py`, loop host |
| Joint velocity: the ≥20 ms finite difference; drive velocity is ignored | deployment `feedback.py` |
| IMU: host-receipt bursts → Madgwick → relay rotation | imu bringup, `imu_relay.py`, `ethrc_sim/sensors.py` (Python Madgwick) |
| Obs builder, history, scales | `obs_builder.py` |
| Action path: default + scale, training clip | `controller.py` |
| Ankle radial projection; Guard E envelope and nearest-reachable clamps (ankle node) | `ankle_targets.py`, `ankle_bringup/ankle_core.py` |
| Crank gains in motor space (`ankle_motor_pd`) | bundle and gain lease |
| Gains over time: a step at rest start, a 1 s fade from balance gains at handover, lapse to baseline after 2 s without renewal | Robot API README, `drive_gains.py` |
| Master slew at the envelope velocity, the 20 ms crank endpoint blend, feed-forward suppression | `standalone.cpp` `stageSlewedCommand` |
| Trip: above datasheet peak for 10 ms, max gap 5 ms. **A trip ends the run** (the router rejects the frame). | `TorqueOverload.hpp`, `rt_hardware_bridge.py` |
| Latency: observation age, compute, pacer drops, action path | `.policyrun` records |

**Validation, in order of strength:**
1. **Real-log replay (M1, primary).** Feed `.policyrun` and `exc-*` inputs through `controls_model` and compare observations, targets and demands with the recorded values.
2. **Golden vectors in the bundle,** replayed by Controls CI through PolicyCore.
3. **`training_assumptions` in the bundle,** covering only Controls-side constants (trip dwell and gap, slew, blend, velocity window, Madgwick gain). Controls compares them with its own configuration at load time.

## 8. Evaluation

**Start small:**
- `Scenario`: world, initial state, support, disturbance and command schedules, seed;
- `Metric`: a pure function of a `Trace`, which is Arrow with signal names;
- `Suite`.

**Subjects:**
- `Checkpoint` and `RealLog` first;
- `Bundle` when export lands (M1);
- `ControlsSIL` (real-time, statistical gates) after M2;
- `SlamPipeline` and `Sonic` when those domains start.

**First suite: `locomotion/smoke`.** Velocity sweep, push grid, trip probe, start/stop. Metrics: tracking error, fall rate, trips per robot-minute (deterministic and sampled), saturation, clip activity.

Gates, caching, bootstrap confidence intervals and the domain plugin registry (Python entry points) come when a second domain arrives. The domains planned later are SLAM, tracking, navigation, manipulation and sim-to-real deltas.

## 9. Policy bundle and promotion

**`ethrc-policy/v2`** extends `DeployConfig` / `isaac-manager/v1`. It needs a new Controls loader; the v1 loader stays.

| Part | Contents |
|---|---|
| Network | TorchScript with the embedded contract |
| Contract | `DeployConfig` fields, axis conventions (forward = +y), the signal behind each observation term, `training_assumptions` |
| Identity | Computed hashes for urdf, actuators and calibration; sim, learning and `ankle_kinematics` revisions; image digest; bundle sha256 |
| Golden vectors | Raw inputs → obs → action → demands |
| Eval | Summary and link |

**Storage and promotion:** Hugging Face, identified by bundle hash; promotion by tags (§7.3).

## 10. Sysid and real data

**What the datasets contain:**
- `exc-*` `commands.csv` holds 22 *joint-level standing-frame* commands, including foot pitch and roll and no cranks.
- The drive demand appears only as `pdo_feedback.position_demand_rad` (0x6062, after slew and interpolation), at about 200 Hz.
- `gains.json` gains are static, with no feed-forward or target velocity.

**Replay:**
- Replay 0x6062 into the device layer on a fixed base, with the bus and interpolation latency disabled.
- Compare against `pdo_feedback` at 4 ms and at 1 ms, using the PACE RMS (2–8 mrad) as the baseline.
- Document the convention, and reconcile the 1.3 ms PACE delay with the 5 ms device stage.

**Loop:** fit → report with uncertainty → PR to humanoid-urdf → randomization defaults → retrain. A nightly fidelity score tracks replay error per sim revision.

## 11. Infrastructure

- **Images:**
  - humanoid-sim: wheel plus a multi-arch image;
  - humanoid-learning: built from its checkout;
  - Controls sim lane: a sim container next to `core-body`, sharing the segment through a volume.
- **Pins:**
  - Isaac Lab: a `release/3.0.0` commit (Newton 1.6.1, Warp 1.17), re-pinned at GA, which is targeted for end of October 2026;
  - humanoid-urdf `e2e47c4`.
- **CI:**
  - every PR: CPU contract tests and the ABI golden frame;
  - nightly on GPU: throughput, profiler sync check, `exc-*` replay, 4 ms vs 1 ms parity;
  - release: Controls' integration suite against the runtime.
- **Mac:** Isaac Lab 3.0 has no macOS support, and physics runs only on MJWarp. Macs run the CPU contract tests, kinematics and Rerun viewers natively; physics goes to a remote GPU (Sky). The linux/arm64 image runs MJWarp on the CPU for tests only.

## 12. Hardware asks (M0)

The likely transfer killers are robot facts. Each result lands in humanoid-urdf before M1 retraining:
1. Weigh the robot, and segments if possible. The model is 58.7 kg; the robot weighed 61–63 kg.
2. Re-zero the left ankle. It is suspected to be 7.5° off.
3. Verify the IMU mount. `imu_relay.py` hard-codes `mount_verified=False`.
4. Measure crank and rod masses. Only needed if the closed-chain spike passes.
5. Measure crank command latency. The 18.2 ms excludes the ankles.
6. Confirm the IMU FTDI latency-timer udev rule is active.
7. Settle ankle peak and continuous torque (30 vs 62 Nm, 11 vs 9.9 Nm).

## 13. Controls work list

Each item needs an owner and sign-off before M1:

| Milestone | Item |
|---|---|
| Before M1 | **Alignment note:** realtime-only runtime (no lockstep); the `FakeDriveDemand` and feedback contract; one ankle plan (C++ core plus torch port, shared golden vectors); promoting the embodiment resolver to humanoid-urdf |
| M1 | v2 bundle loader, golden-vector CI, `training_assumptions` check |
| M1–M2 | fake-coe lane passes the integration suite on MuJoCo: `/sim/*` handling, test seams, truth oracle |
| M2 | plant_host on the runtime segment (doorbell wake, failure semantics); fake-coe `demanded_torque` carries the reading bias; raw IMU through the real Madgwick and relay nodes in the sim launch |
| M2+ | `ethrc_cameras` `sim` backend; camera calibration store; a Controls-side Rerun tap; MCAP recording |
| Optional | Bank feedback ABI extension (only for fields Controls consumes); a 2 ms EtherCAT cycle |

## 14. Feature catalog

| Feature | P |
|---|---|
| Resolved embodiment and model build, with contact primitives and hard stops | P0 |
| Device layer: `FakeDriveDemand` mirror, constant gains plus target shaping, envelope, reading bias | P0 |
| Hinge ankle with the `ankle_kinematics` torch transmission | P0 |
| `exc-*` replay and the sim-vs-real viewer | P0 |
| Throughput and no-host-sync benchmark; CPU runtime benchmark | P0 |
| Signal logging with Isaac Lab's Rerun visualizer | P0 |
| `controls_model` with real-log replay parity | P0 |
| Walk v2 port, `locomotion/smoke`, bundle v2 with golden vectors, Hugging Face and W&B | P0 |
| Runtime segment (real-time), failure semantics, compatibility gate | P1 (M2) |
| Training reporting layer with the `Epos4Slave` golden test | P1 |
| IMU delivery model, Madgwick torch port | P1 |
| Current, thermal and backlash models | P1 |
| G1 in SIL; Viser play; manager-term auto-logging | P1 |
| Cameras (Warp) and the `ethrc_cameras` sim backend | P1 (M2) |
| Closed-chain ankle (if the spike passes) | P1 |
| OVRTX (after the legal check), splat, kit, SLAM eval | P2 |
| Subscription engine, gates and caching in eval | P2 |
| HIL network adapter; manipulation and navigation domains | P2 |

## 15. Decisions

| ADR | Decision | Status |
|---|---|---|
| 1 | Runtime: one shared-memory segment with a mailbox and truth region; real-time only | Decided |
| 2 | Drive boundary: an exact `FakeDriveDemand` mirror; physical-frame feedback; per-mode outputs; protocol and runtime reporting stay in fake-coe | Decided |
| 3 | Solver: MJWarp, `implicitfast` (M0: `euler` is 5–9× worse on the ankle cranks), constant per-episode gains plus target shaping; Featherstone and OVPhysX cross-checks later | Decided (M0) |
| 4 | Controller behaviour in Controls; `controls_model` in learning with stages from code, validated by real-log replay, golden vectors and `training_assumptions` | Decided |
| 5 | 250 Hz (4 ms) physics | Decided |
| 6 | Rerun (Isaac Lab visualizer plus signal logging), Viser for interaction | Decided |
| 7 | Hugging Face only; bundle id = sha256 of files; promotion by tags; run-branch retention; W&B tracking | Decided |
| 8 | Lockstep or virtual time in Controls | Out of scope; a Controls ADR if ever |
| 9 | Runtime on a GPU with Isaac Lab/MJWarp. No MuJoCo-C physics anywhere (M0 measured a Newton/MuJoCo-C runtime at 1.16× but rejected it under the physics rule). MuJoCo-lane deletion gated on single-robot ≥1.5× real time (M0: 0.44×, overhead-bound) | Decided (rev 3.2); M2 performance target open |
| 10 | Cameras: Warp first (M2); OVRTX after a legal check; splat and kit later | Decided |
| 11 | Isaac Lab 3.0, `release/3.0.0` pin then GA; MJWarp; MuJoCo for Controls kinematics only | Decided |
| 12 | Ankle: hinges plus `ankle_kinematics` torch transmission; `ankle_kinematics` v2 in Controls and sim; one C++ core in Controls | Decided (pending Controls alignment) |
| 13 | Repo graph as in §2.1 | Decided |
| 14 | Closed-chain ankle: no-go for M1 (needs stiffened loop constraints, 2–2.4× softer than hinges under load, +19% cost); revisit with measured crank and rod masses and Isaac Lab loop joints | Decided (M0) |
| 15 | Embodiment: resolved JSON generated in humanoid-urdf CI; Controls-code facts migrated to humanoid-urdf | Decided (needs the Controls and urdf owners) |
| 16 | Asset pipeline: one USD per description and profile (content-keyed), built through Isaac Lab's kitless URDF converter with welds, calibration-travel ranges, sole boxes and contact excludes applied; every backend loads it | Decided (M0) |
| 17 | Training self-collision: a pruned set (one primitive or hull per body; legs vs legs, arms vs torso). The full hull set is unaffordable at 4096 envs | Decided (M0); build in M1 |

## 16. Roadmap

**M0 (about 2 weeks). Retires the physics, throughput and data-fidelity risks, with no Controls changes.**
1. **Repo and pins:** fresh repo with the §3.0 rules; Isaac Lab `release/3.0.0` commit; humanoid-urdf and `ankle_kinematics` submodules.
2. **Model build:**
   - URDF → Newton with the hinge ankle and the torch transmission;
   - sole boxes and contact excludes ported from `build_mjcf.py`;
   - a check against Controls' MJCF on 10 poses: FK, CoM, mass, collision primitives and contact pairs.
   The resolved-embodiment JSON is used where it already exists.
3. **Device layer:** constant solver gains, target shaping for gain changes and the envelope, integer-ns latency, and an explicit integrator comparison.
4. **Fidelity:** replay 0x6062 from the three `exc-*` datasets on a fixed base. Report per-joint RMS against `pdo_feedback` at 4 ms and 1 ms, against the PACE baseline.
5. **Benchmarks:**
   - 4096-env throughput, with no host syncs (profiler);
   - single-robot real-time factor on Isaac Lab/MJWarp (GPU).
6. **Closed-chain spike:** 3 days, numeric criteria, written go/no-go.
7. **Visualization:** `humanoid-sim view` and `replay`, using Isaac Lab's Rerun visualizer plus logged drive signals for env 0, overlaid on `exc-*`.
8. **Write-ups:** the Controls alignment note (§13) and the hardware asks (§12).

**Exit criteria:**
- `exc-*` replay within 2× the PACE RMS;
- 4096-env throughput no worse than walk v2's current iteration time;
- runtime real-time factor ≥ 1.0, or MuJoCo-lane deletion formally blocked (result: blocked; GPU single robot 0.44×);
- the closed-chain decision recorded.

**M1: walk v2 parity.**
- humanoid-learning with humanoid-sim as a submodule;
- `controls_model` with its stage list and real-log replay parity;
- hardware-ask results landed in humanoid-urdf;
- walk v2 ported and retrained;
- `locomotion/smoke`;
- bundle v2 with golden vectors on Hugging Face, and W&B.

**M2: runtime and cameras.**
- Runtime segment behind fake-coe, after Controls' fake-coe parity prerequisite;
- failure semantics; compatibility gate (V1, then G1 with stock SONIC);
- Warp cameras and the `ethrc_cameras` sim backend;
- the training reporting layer with the `Epos4Slave` golden test;
- MuJoCo physics lanes deleted once the gates pass.

**M3 and later:**
- the `ControlsSIL` subject with statistical gates;
- the subscription engine and Viser play;
- OVRTX, splat and kit, then SLAM eval;
- V1 tracking and SONIC;
- manipulation;
- HIL.
