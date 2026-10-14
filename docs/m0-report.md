# M0 report

2026-10-14. Branch `m0`. Pins:
- Isaac Lab `release/3.0.0` at `32712682` (Newton 1.6.1, Warp 1.17, MuJoCo/MJWarp 3.12);
- humanoid-urdf `e2e47c4`;
- ankle_kinematics `1a3d200`.

These are the same commits Controls pins.

Reproduce:
- **Asset:** `humanoid-sim build-asset`, in the Isaac Lab image.
- **Replays:** `sky/replay.sky.yaml` (Isaac Lab/MJWarp on a GPU), or `humanoid-sim replay <exc-dir> --period-ms 4|1 --integrator implicitfast|euler` on a GPU node. Then `humanoid-sim view-replay <out> --save x.rrd` on any machine.
- **Benchmarks:** `sky/bench.sky.yaml`.

**Physics rule (decided after the first M0 pass).** Physics runs only through Isaac Lab on MJWarp; MuJoCo C never steps the robot. The first pass used Newton with MuJoCo C (`use_mujoco_cpu`) for the Mac replays and a CPU runtime. That path is removed. All fidelity numbers below were re-run on Isaac Lab/MJWarp.
- **Spike:** `spikes/closed_chain/ankle_rig.py`.

## Exit criteria

| Criterion | Result | Status |
|---|---|---|
| `exc-*` replay within 2× the PACE RMS | Legs: 3–16 mrad (PACE 2–8). Ankles: 0.3–14 mrad (PACE 7–13). Waist and arms: 4–59 mrad (PACE 17–26). | **Met for legs and ankles.** Shoulder flexion and elbows on the low-band capture exceed 2× (58/54 and 48/45 mrad); see Fidelity. |
| 4096-env throughput no worse than walk v2 | Physics plus plant: 1,751 sim-s per wall-s at 4096 envs without self-collision (6,463 at 16384). Walk v2's whole training loop runs at ~220 sim-s/s (9.06 s/iter). | **Met, with headroom.** Full convex-hull self-collision does not fit at 4096 envs (see Throughput). |
| Runtime real-time factor ≥ 1.0, or MuJoCo-lane deletion formally blocked | Isaac Lab/MJWarp, single robot on an L40S: 0.44×, bound by ~9 ms of per-step overhead rather than physics. | **Not met. MuJoCo-lane deletion is blocked** until M2 reaches ≥ 1.5×. |
| Closed-chain decision recorded | No-go for M1 (see Spike). | **Done.** |

## What exists

**Model.** The canonical sim model is a USD built from humanoid-urdf by `model/urdf.py` and `model/asset.py`:
- 34 unlisted (ORCA hand) joints welded;
- joint ranges set to the calibration travel (Controls' `joint_travel` rule);
- sole boxes fitted to the plate face (Controls' `sole_plate` algorithm), matching Controls' MJCF within 0.06 mm, normals to 1e-5;
- Isaac Lab's kitless URDF converter (convex hulls, self-collision);
- the 9 contact excludes as `physics:filteredPairs`.

The output directory is keyed by a machine-independent content hash.

**Physics:** Isaac Lab 3.0 articulations on MJWarp (`physics/isaaclab.py`), with per-joint solver properties from one table (`model/joint_properties.py`). On CUDA for real use; on Warp's CPU device for tests (0.09× real time). Newton on the host loads the same USD for kinematics only (`model/kinematics.py`: viewer poses, model checks).

**Plant.** `core.PlantCore`, backend-independent, batched torch:
- an exact `FakeDriveDemand` mirror;
- integer-ns arrival blending of demands within a step;
- Controls' `Plant.step` force law (JVPT, freeze latch, off) and clip (torque–speed envelope, braking at peak, 0x34C6:04 cap);
- I-gain refused as a fault;
- ankle crank transmission: a torch port golden-tested against `ankle_kinematics` and humanoid-urdf `ankle_model` (1e-12);
- constant solver gains with walk v2's target shaping.

Feedback is the 4 physical scalars `fake_bank_feedback` takes.

**Visualization.** `view-replay` writes a Rerun recording:
- the sim robot, with the real robot as a translucent ghost, posed from its measured drives through the same transmission;
- `/sim/0/drives/<drive>/*`, `/real/drives/<drive>/*`, `/demand/...` and `/residual/...` for every drive.

Meshes are logged once (static) and body transforms per frame. Newton's own `ViewerRerun` is not used: with no web viewer it calls `rr.spawn()`, which redirected the stream away from the file into a live viewer, and that viewer dropped data at its memory limit.

**Tests and CI.**
- 28 CPU contract tests: device layer, transmission golden tests, model build, readers;
- 2 physics tests: mass, and standing hold on the built asset;
- CI runs the CPU tests on x86 and arm, and builds the image and asset to run the physics tests.

## Fidelity: drive-level replay of the three `exc-*` captures (Isaac Lab/MJWarp, L40S)

Setup:
- fixed base;
- each drive fed its recorded 0x6062 position demand at the captured gains (JVPT, no feed-forward), with no added latency;
- scored against `native_position_rad`, skipping the first second.

The captures were on a swinging gantry; PACE also fits a fixed root.

Position RMS per drive, in mrad. The real−demand columns show how hard each drive worked. LF is `exc-503f` (0.2–3 Hz), HF is `exc-3a49` (3–10 Hz) and HF2 is `exc-ece4` (3–10 Hz). Ankle "(l)/(r)" is the side.

| Drive | Joint | LF 4 ms | LF 1 ms | HF 4 ms | HF 4 ms euler | HF 1 ms | HF2 4 ms | LF real−demand | HF real−demand |
|---|---|---|---|---|---|---|---|---|---|
| HEJ90M1 | hip_central_rotation | 6.3 | 6.3 | 10.3 | 10.2 | 10.3 | 10.2 | 9.2 | 61.7 |
| HEJ90M2 | waist_upperbody_yaw | 39.3 | 38.7 | 15.9 | 15.9 | 15.8 | 15.6 | 24.1 | 27.4 |
| HEJ90L1 | leg_flexion_l | 10.8 | 10.5 | 7.5 | 7.5 | 7.5 | 7.4 | 25.3 | 63.8 |
| HEJ90R1 | leg_flexion_r | 11.4 | 11.0 | 6.6 | 6.4 | 6.5 | 6.5 | 25.4 | 65.7 |
| HEJ90L2 | leg_abduction_l | 3.9 | 3.8 | 5.5 | 5.5 | 5.5 | 5.3 | 8.3 | 25.0 |
| HEJ90R2 | leg_abduction_r | 3.8 | 3.8 | 4.1 | 3.5 | 3.8 | 4.1 | 7.6 | 21.0 |
| HEJ90L3 | hip_yaw_l | 15.6 | 15.2 | 11.6 | 9.4 | 10.4 | 11.5 | 50.6 | 47.4 |
| HEJ90R3 | hip_yaw_r | 16.5 | 16.2 | 12.2 | 10.7 | 11.4 | 12.0 | 50.5 | 48.5 |
| HEJ90L4 | knee_l | 6.4 | 6.4 | 6.6 | 6.3 | 6.4 | 6.5 | 11.1 | 21.7 |
| HEJ90R4 | knee_r | 3.4 | 3.3 | 3.3 | 2.8 | 3.1 | 3.2 | 11.3 | 22.2 |
| HEJ70AL1 | shoulder_flexion_l | 59.1 | 58.6 | 34.8 | 34.6 | 34.7 | 34.7 | 21.6 | 79.3 |
| HEJ70AR1 | shoulder_flexion_r | 53.7 | 53.2 | 25.3 | 25.0 | 25.2 | 25.2 | 22.4 | 79.4 |
| HEJ70AL2 | shoulder_abduction_l | 18.0 | 17.9 | 7.1 | 7.2 | 7.2 | 7.5 | 15.0 | 54.0 |
| HEJ70AR2 | shoulder_abduction_r | 13.7 | 13.8 | 17.1 | 16.9 | 17.1 | 17.3 | 17.1 | 60.1 |
| HEJ50AL3 | shoulder_rotation_l | 6.4 | 6.1 | 12.8 | 7.9 | 10.3 | 11.7 | 40.7 | 95.3 |
| HEJ50AR3 | shoulder_rotation_r | 3.7 | 3.5 | 10.0 | 4.5 | 7.2 | 9.2 | 40.8 | 97.8 |
| HEJ50AL4 | elbow_flexion_l | 48.9 | 48.0 | 27.7 | 27.7 | 27.7 | 27.2 | 61.8 | 75.4 |
| HEJ50AR4 | elbow_flexion_r | 44.6 | 43.8 | 19.9 | 20.0 | 20.0 | 19.5 | 62.3 | 71.8 |
| HEJ50L_ANK1 | ankle upper (l) | 0.3 | 0.3 | 0.9 | 8.3 | 0.9 | 0.8 | 0.5 | 1.2 |
| HEJ50L_ANK2 | ankle lower (l) | 3.2 | 3.2 | 8.9 | 11.5 | 8.5 | 8.5 | 4.2 | 14.4 |
| HEJ50R_ANK1 | ankle upper (r) | 4.9 | 4.9 | 14.4 | 17.1 | 13.6 | 13.8 | 7.3 | 25.2 |
| HEJ50R_ANK2 | ankle lower (r) | 0.8 | 0.8 | 1.2 | 9.4 | 1.2 | 1.2 | 1.1 | 1.4 |

**Findings.**
1. **4 ms vs 1 ms.** The two agree within 0.5 mrad on every drive except the light shoulder-rotation joints at 3–10 Hz (+2.5 mrad). This supports ADR-5 (250 Hz).
2. **`implicitfast` vs `euler`.** `euler` is better on shoulder rotation but 5–9× worse on the ankle cranks (7.6–17 vs 0.9–14 mrad). Keep `implicitfast` (ADR-3).
3. **Re-run on MJWarp.** Every number in the table comes from Isaac Lab/MJWarp. Across all 198 drive-runs, the superseded first pass on MuJoCo C differed by a median of 0.000 mrad and at most 0.92 mrad, so no conclusion changed.
4. **Legs** are inside or near PACE's band using only description values: no fitted delay, bias or armature. Hip yaw is the weakest at 10–16 mrad.
5. **Upper body.** On the low-band capture the sim is worse than "perfect tracking" for shoulder flexion, the waist and the elbows. Candidates, in order:
   - **Unmeasured passive parameters:** waist damping 10 and friction 0.5, carried from URDF_2; the HEJ70 armature is 0.34 from one sysid run.
   - **Fixed base vs a swinging gantry:** the IMU shows |ω| up to ~1 rad/s, and the shoulders and elbows react to it.
   - **The welded ORCA hand mass on the forearm.**

   Next step: a PACE-style fit of these joints on the sim (the sysid module, M1), and a replay that drives the base from the IMU orientation.

## Throughput (L40S, 4 ms, standing hold with a 20 ms demand stream, torque sync check on)

| Envs | Self-collision | Env-steps/s | Sim-s per wall-s | Envs near standing | Notes |
|---|---|---|---|---|---|
| 1 | on | 109 | 0.44 | 100% | Per-step overhead ~9 ms (Python plus implicit-actuator path, not graph-captured) |
| 1024 | on (default buffers) | 110k | 441 | 96% | MJWarp contact overflow |
| 4096 | on (default buffers) | 415k | 1,661 | 96.5% | Overflow (needs `nconmax` 879, `njmax` 238): invalid |
| 4096 | on (`njmax` 400, `nconmax` 1024) | — | — | — | CUDA out of memory (46 GB) |
| 4096 | off | 438k | 1,751 | 100% | Clean |
| 8192 | off | 854k | 3,415 | 100% | Clean |
| 16384 | off | 1.62M | 6,463 | 100% | Clean |

Per-step wall time is flat from 1 to 4096 envs, so the GPU is not saturated; throughput scales with env count.

No torch-side host sync happens in steady state on the Isaac Lab path. The Newton-direct backend on CUDA does sync, but it is the CPU runtime path.

**Implication for training.** The full convex-hull self-collision set (120 hulls, ~880 broadphase candidates per env) is unaffordable at 4096 envs. M1 needs a pruned self-collision model: one hull or primitive per body, legs vs legs and arms vs torso only. Walk v2 also used a selected set.

## Runtime (one robot, real time)

Under the physics rule the runtime is Isaac Lab/MJWarp on a GPU. Today a single robot runs at **0.44× real time** on an L40S (`bench --num-envs 1`). Per-step wall time is ~9 ms whether there is 1 env or 4096, so the cost is fixed overhead (Python, Isaac Lab's implicit-actuator path outside the CUDA graph, the plant's many small torch ops), not physics.

M2 target: ≥ 1.5× with headroom. Levers, in order:
1. CUDA-graph the plant step and route actuation through Isaac Lab's graphable actuator path, so the whole step is captured;
2. the pruned self-collision set (ADR-17);
3. a dedicated runtime thread.

MuJoCo-lane deletion stays blocked until then.

Superseded first-pass data: a Newton/MuJoCo-C runtime reached 1.16× natively on a Mac and 1.03× in the arm64 container. Isaac Lab on MuJoCo C, via a manager workaround, ran about 20% slower than that. Both were removed under the physics rule. The upstream gap is still worth reporting: Newton's `SolverMuJoCo.get_max_contact_count` and `update_contacts` raise on the MuJoCo-C path, and Isaac Lab 3.0 calls the first.

## Spike: closed-chain ankle (ADR-14). Decision: no-go for M1

The rig:
- right ankle with real geometry from humanoid-urdf;
- cranks on the motor axes, driven by position PD;
- rods attached to the cranks with ball joints, closing onto the foot anchors with ball joints outside the articulation;
- Newton turns each closure into a MuJoCo CONNECT equality.

Crank, rod and foot masses are modeled values (hardware ask #4).

| Criterion | Result |
|---|---|
| Builds and runs on MJWarp | Yes |
| Stable at 480/8 and 100/3.5 | Yes, **only after stiffening the loop constraints** (`solref` 0.008 = 2·dt). With MuJoCo's default (0.02) the 700 N load case diverges (state reaches 1e14). |
| Pose error (cranks commanded to the transmission angles) | 0.8 mrad at 480/8, 2.0 mrad at 100/3.5 |
| 700 N, 5 cm CoP (35 Nm pitch): deflection vs hinge model `(JᵀKJ)⁻¹τ` | −0.077 vs −0.039 rad at 480/8; −0.459 vs −0.189 rad at 100/3.5. The chain is **2–2.4× softer** (constraint compliance at 4 ms). |
| Cost | Not measured at scale on MJWarp: single-env GPU timing is overhead-bound. The first pass measured +19% on MuJoCo C (superseded). |
| Reset by settling | Not needed in this rig: it starts assembled. Full-robot resets would need it. |

Why no-go: the hinges plus transmission already reproduce the linkage exactly (golden tests). The chain adds constraint compliance that the real rigid rods don't have, needs non-default constraint tuning to stay stable, and has no USD loop-joint path tested in Isaac Lab yet.

Revisit if the feasible set or assembly singularities need to be learned against, with:
- measured crank and rod masses;
- Isaac Lab loop joints;
- `solref` authored per constraint.

## Decisions taken in M0 (to fold into the design doc)

- **Asset pipeline (new ADR-16).** USD is the canonical sim model, built by humanoid-sim through Isaac Lab's kitless URDF converter, content-keyed. Every backend loads it.
- **Physics rule and ADR-9.** Isaac Lab/MJWarp is the only physics, and the runtime runs on a GPU. The target is ≥ 1.5× single-robot real time before deleting the MuJoCo lanes.
- **ADR-3, solver.** `implicitfast` confirmed. Training buffers must be sized with a pruned self-collision set.
- **ADR-14.** Closed chain no-go for M1.
- **Joint ranges** come from the calibration travel, as in Controls' MJCF.

## Open items carried into M1

1. Pruned self-collision model; re-run 4096/8192-env throughput with it.
2. Upper-body fidelity: passive-parameter fit (sysid), base motion from the IMU, hand mass.
3. Single-robot GPU runtime from 0.44× to ≥ 1.5× (CUDA-graph the plant and actuation path).
4. Training reporting layer (encoder, velocity observer, current) with the `Epos4Slave` golden test (M2 per plan).
5. The torch ankle transmission moves into ankle_kinematics v2 once Controls agrees.
6. Upstream reports: Newton's MuJoCo-C contact query that Isaac Lab calls; Newton's `ViewerRerun` `spawn` taking over an `rrd` sink.
