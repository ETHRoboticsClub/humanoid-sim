# Running Controls against humanoid-sim

Controls runs humanoid-sim behind two lanes: the ETHRC V1's fake-coe lane, below, and the Unitree G1's lane ([The G1 lane](#the-g1-lane)). Each converts its robot's native device commands into the segment's SI drive demands at its own boundary, so one segment serves both.

Controls' fake-coe lane runs the production EtherCAT master, the `standalone` binary of `ethercat_device_configurator` with `--transport fake-coe`, against a bank of simulated EPOS4 drives (`Epos4Slave` in Controls' `ethrc_fake_ethercat` package). The master speaks CANopen over EtherCAT (CoE) to them as it would to the robot. A separate plant host process owns that bank: each bus cycle it reads every drive's demand from the bank and gives each drive its feedback. By default the plant host steps a MuJoCo plant itself. With `--physics humanoid-sim` it converts each demand to an SI drive demand on humanoid-sim's runtime segment and hands the sim's feedback back to the bank. Everything above the drives (master, safety monitor, router, policies, robot API) is the stack that runs on the robot.

This page covers the Controls side. [runtime.md](runtime.md) describes `humanoid-sim serve`, the segment and the commands.

## Where it lives in Controls

The V1's humanoid-sim plant is on Controls' `master`, in Python. A C++ plant host that keeps the bus period's real-time bound is tracked in [Controls issue #197](https://github.com/ETHRoboticsClub/ETHRC-Humanoid-Controls/issues/197); the IMU already reaches the robot's own chain (below). The files involved, relative to the Controls checkout:

| File | Role |
|---|---|
| `ros2_ws/src/sim/ethrc_fake_ethercat/ethrc_fake_ethercat/plant_host.py` | The plant host; `--physics humanoid-sim` runs `run_segment` |
| `ros2_ws/src/sim/ethrc_fake_ethercat/ethrc_fake_ethercat/sim_segment.py` | `SegmentPlant` (EPOS4 demands to SI, one stamped exchange per bus frame, disturbances), `mti_samples` (the sim's IMU samples as the device reports them), `SegmentTestControl` and `SegmentStandingTruth` (test seams and oracle) |
| `ros2_ws/src/sim/ethrc_fake_ethercat/ethrc_fake_ethercat/mti_can.py` | `MtiCanEmitter`: each IMU sample as the MTi-610's CAN output on a vcan |
| `ros2_ws/src/sim/ethrc_sim/ethrc_sim/humanoid_sim/` | Vendored `plant_abi.py` and `plant_abi.h`, `SegmentClient` (`client.py`) and the IMU mount (`frames.py`), shared with the G1 lane |
| `scripts/sim/launch.py` | `--plant humanoid-sim --sim-segment PATH --sim-wait S` |
| `scripts/sim/humanoid_sim_lane.py` | The lane end to end in two containers |
| `ros2_ws/src/tests/ethrc_integration_tests/test/sim_fixture.py` | `plant_options`: `ETHRC_TEST_PLANT` and `ETHRC_TEST_SIM_SEGMENT` |
| `ros2_ws/src/sim/ethrc_fake_ethercat/test/test_sim_segment.py` | Unit tests of the segment side against a stand-in sim |
| `ros2_ws/src/sim/ethrc_fake_ethercat/test/test_sim_segment_live.py` | The adapter against a live sim (`ETHRC_TEST_HUMANOID_SIM_SEGMENT`, `ETHRC_TEST_HUMANOID_SIM_DESCRIPTION`) |

The two ABI files are copied unchanged from `src/humanoid_sim/runtime/abi/`. Controls pins humanoid-sim as a submodule (`third_party/humanoid-sim`), and `test_vendored_abi_matches_humanoid_sim` (in `ethrc_sim`'s tests) compares both copies with the pinned checkout on every run: ABI major, minor and layout hash, then the files byte for byte (`HUMANOID_SIM_ROOT` overrides the location). When the ABI changes, copy both files again and move the pin; a different layout hash or major version makes `Segment.open` refuse the segment, and `just sim-isaac` runs the pinned humanoid-sim.

## What the plant host does

The plant host runs without ROS. It writes each of the sim's IMU samples as the MTi-610's CAN output (SampleTime, RateOfTurn, Acceleration) on the launcher's `vcan0`, where the robot's own IMU chain reads it: `ethrc_mti_can`, then Madgwick, then `imu_relay`, which publish `/rt/imu`, `/ethrc/imu` and `/ethrc/imu_mount` as on the robot. It writes its projection to Controls' plant link, which `plant_relay` publishes on `/ethrc/plant`. `run_segment` in `plant_host.py` does the following.

At startup:

- It computes the description digest from the description's own `actuators.yaml` and the URDF it names (`description_files`; launch.py passes the file as `--sim-description`, since the session's `actuators.yaml` is a materialized copy). It opens the segment with `Segment.open(path, description_digest=..., wait_s=--sim-wait)` and so refuses a sim running a different description. It does not check the header's `robot_id`.
- It sends `SET_FIDELITY` `real`: the sim's model of the rig's IMU (white noise, turn-on bias, drift) is the sensor, and the chain on the CAN interface does the rest. The descriptor's `imu` names that model ([runtime.md](runtime.md#descriptor), ABI 3.1).
- It writes the plant link's descriptor (the hashes) and checks that every drive in the bus fixture exists in the sim, exiting otherwise.

Every process-data frame the EtherCAT master sends (or, with no frames, every bus period on its own clock):

- `SegmentPlant.exchange_demands` writes one demands record, stamped with the frame's send time, into the sim's demand ring. `demand_fields` converts each drive's bank demand as `references_from_demands` does for MuJoCo: JVPT (-64) becomes PD, freeze (-62) and homing (6) FREEZE, a disabled drive or another mode DISABLED; the raw JVPT gains become SI through `ethrc_ssot.drive_gains.si_gain`, an unset torque cap (0) becomes `inf`, and a non-zero I gain is refused. A sim drive missing from the bus fixture is written disabled.
- It reads the feedback region and passes each drive's position, velocity, applied torque and demanded torque at that instant to the bank (`bank.feedback`). If the region is mid-write it reuses the previous cycle's feedback. Frames served while the host was busy count as late cycles.
- It raises `SimLost` when the segment's state is no longer ready or the sim's heartbeat is older than 500 ms (`SIM_STALE_NS`). The plant host then writes the fault to its status file and exits.
- `mti_samples` takes the IMU samples it has not seen (by `seq`), oldest first, with their sample time on the plant clock and their chip-frame readings, and `MtiCanEmitter` writes each as CAN frames, back to back. The driver's kernel receipt stamp is therefore the plant host's cycle on which the sim delivered the sample. The sim's `seq` keeps rising across a `RESET` and across sessions, so the last number seen stays valid.

Every status period (100 ms):

- It writes its status file (`ethrc-fake-plant/v1`) with `physics: "humanoid-sim"`, `sim_late_steps`, and `keeping_up`, which is true only while the worst lateness of its own cycles so far is within `--max-lateness` (5 ms) and the sim reports `keeping_up`. One cycle later than 5 ms makes it false for the rest of the run.
- It writes the plant projection to the plant link (schema `ethrc-plant/v1`), including the hashes and the fidelity gaps (see [Fidelity gaps](#fidelity-gaps)); `plant_relay` publishes it on `/ethrc/plant`.
- In the device-free test container only (`ETHRC_SIM_TEST_CONTROL` set and `/.dockerenv` present), it serves the test seams. The bank test control's joint disturbances become `APPLY_DISTURBANCE` joint sines on 1:1 drives, sent only when they change (`SegmentPlant.disturb_drives`). The floating-body requests (`calibration-pose`, `gantry`, `base-tilt`, on Controls' `ethrc_sim.test_seams`) become `PRESCRIBE_POSE` and `SET_SUPPORT` commands (`SegmentTestControl.poll`); a request the sim refuses is answered with its reason and changes nothing. With `ETHRC_SIM_TEST_ORACLE_STATE` also set, it writes the plant oracle from the truth region (`SegmentStandingTruth`), whose maxima and normal-force window advance on every new truth record.

On exit, `SegmentPlant.close` sends `DETACH` while the session is still ready, so a `--persist` sim opens the next session at once instead of waiting out its client timeout.

The plant host uses the commands `SET_FIDELITY`, `APPLY_DISTURBANCE` (joint torques), `PRESCRIBE_POSE`, `SET_SUPPORT` and `DETACH`. It does not send `RESET`, `PAUSE`, `RESUME` or `INJECT_FAULT`; bus, protocol and drive faults come from Controls' own fake bank.

`test_sim_segment_live.py` checks the adapter against a running sim. In the humanoid-sim image, with a Controls checkout mounted at `/controls` and its Python packages on `PYTHONPATH`:

```bash
humanoid-sim serve --segment /tmp/plant --device cpu --support suspended --truth-every 1 &
ETHRC_TEST_HUMANOID_SIM_SEGMENT=/tmp/plant \
ETHRC_TEST_HUMANOID_SIM_DESCRIPTION=/workspace/humanoid-sim/third_party/humanoid-urdf/ethrc_v1/actuators.yaml \
  python -m pytest /controls/ros2_ws/src/sim/ethrc_fake_ethercat/test/test_sim_segment_live.py
```

It steps the right elbow 0.2 rad with every drive in JVPT at its baseline gains and checks that the elbow gets there, that every drive, the ankle cranks included, applies `kp (target - q)` at rest, that the sim's IMU samples become the MTi-610's device samples reading gravity up on the still, upright base, each sample once, and that disabled drives apply no torque. It passes on Warp's CPU device.

## Running the lane with launch.py

humanoid-sim serves first, on the same machine, with its segment on a volume the Controls side also mounts:

```bash
humanoid-sim serve --segment /run/hsim/plant --support suspended --truth-every 1
```

Then, in Controls' simulation image (or a checkout with the sim workspace built; see Controls' `docs/sim-bringup.md`), with the same directory mounted:

```bash
python3 scripts/sim/launch.py --headless --hands none --clock wall --bus fake-coe \
  --plant humanoid-sim --sim-segment /run/hsim/plant --profile suspended --environment empty
```

| Option | Default | Notes |
|---|---|---|
| `--plant mujoco\|humanoid-sim` | `mujoco` | `humanoid-sim` needs `--bus fake-coe` (or `unitree`, [The G1 lane](#the-g1-lane)) and `--sim-segment`; launch.py refuses it otherwise |
| `--sim-segment PATH` | none | The file `humanoid-sim serve --segment` publishes |
| `--sim-wait S` | `180` | How long the plant host waits for the segment to become ready. The launcher waits that long plus 30 s for the plant host to report ready. |
| `--profile suspended\|free` | `free` | Must match the sim's `--support`. Nothing compares the two, and `serve` defaults to `suspended`. The two `suspended` differ: Controls' MuJoCo plant welds the pelvis, humanoid-sim hangs the robot from a harness cable on the torso (see below). |
| `--clock plant\|wall` | `plant` with `--hands none`, otherwise `wall` | The fake-coe lane runs on the wall clock: pass `--clock wall` with `--hands none`, since launch.py refuses `plant` with `--bus fake-coe` |

`--fidelity` and `--imu-mount-rpy` shape the MuJoCo plant's IMU chip; launch.py refuses them on humanoid-sim, whose IMU is its own sensor model at the description's mount. The lane needs Linux with the `vcan` kernel module and the right to create the interface (`NET_ADMIN`); Docker Desktop's kernel has no vcan.

## Running the lane check: `humanoid_sim_lane.py`

`scripts/sim/humanoid_sim_lane.py` brings the lane up end to end and checks it. It needs Docker, Controls' simulation image, and either humanoid-sim's image with a checkout whose robot asset is built, or a sim already serving.

```bash
# humanoid-sim in its own container, on a fresh Docker volume
scripts/sim/humanoid_sim_lane.py --sim-image humanoid-sim:dev --sim-repo ../humanoid-sim \
  --controls-image ethrc-controls:simulation --device cuda:0

# humanoid-sim already serving into a host directory (for example on a GPU node, outside a container)
humanoid-sim serve --segment /srv/hsim/plant --device cuda:0 --truth-every 1 --client-timeout-ms 10000 &
scripts/sim/humanoid_sim_lane.py --sim-dir /srv/hsim --controls-image ethrc-controls:simulation
```

With `--sim-image`, the script creates a Docker volume and runs the sim image with `--sim-repo` mounted at `/workspace/humanoid-sim`, adding `--gpus all` for a CUDA device, as `serve --segment /shared/plant --device <device> --support <support> --truth-every 1 --client-timeout-ms 10000`. With `--sim-dir`, the directory must hold the segment as `plant`, and `--support` must match how that sim was started. In both cases it runs the Controls image with the volume or directory at `/shared`:

```bash
python3 /opt/ethrc/repository/scripts/sim/launch.py --prepared --headless --no-api --hands none \
  --robot-id ethrc-sim --domain 141 --profile <support> --environment empty --clock wall \
  --bus fake-coe --plant humanoid-sim --sim-segment /shared/plant
```

| Option | Default | Meaning |
|---|---|---|
| `--sim-image IMAGE` or `--sim-dir DIR` | one is required | Where the sim runs |
| `--sim-repo PATH` | none | humanoid-sim checkout with its asset built; required with `--sim-image` |
| `--controls-image IMAGE` | required | Controls' simulation image |
| `--device` | `cpu` | The sim's device (`--sim-image` only) |
| `--support suspended\|free` | `suspended` | Passed to the sim and as the lane's `--profile` |
| `--timeout S` | `300` | How long to wait for every drive to be enabled |
| `--seconds S` | `60` | How long to run once they are |
| `--keep` | off | Leave the containers and the volume |

The script waits until every drive is enabled or the plant host reports a fault, runs for `--seconds`, and then reads the segment from inside the Controls container. It prints a JSON result: the plant host's status (`physics`, `cycles`, `keeping_up`, `drives`, `drives_operational`, `drives_enabled`, `drives_faulted`, `fault`) and, from the segment, `sim_steps`, `sim_time_s`, `sim_late_steps`, `imu_samples`, `pd_drives` and `max_tracking_error_rad`. `ok` is true, and the exit code 0, when there is no fault, every drive is enabled and none faulted, at least one drive runs the PD law, and the largest PD tracking error is under 0.05 rad.

The sim starts every session in the V1's standing pose (its profile's `poses.start`, see [runtime.md](runtime.md#the-start-pose)), where every drive lies inside the window Controls arms it in; with the V1's standing offsets at 0 it is the home pose.

On the default CPU device the lane comes up but the sim does not keep up with real time.

To watch the robot while Controls drives it, run `humanoid-sim view-runtime` on the same segment, for example `humanoid-sim view-runtime /srv/hsim/plant --save lane.rrd` next to a sim serving into `/srv/hsim`. The viewer only reads the segment, so the lane runs as it would without it, and it follows the sim from session to session under `--persist`. [runtime.md](runtime.md#watching-a-host-view-runtime) lists its options and what it needs.

## Running Controls' integration tests on humanoid-sim

The integration scenarios that run on the fake-coe or Unitree bus can use humanoid-sim as their plant. `plant_options` in `sim_fixture.py` reads two variables:

| Variable | Meaning |
|---|---|
| `ETHRC_TEST_PLANT` | `mujoco` (default) or `humanoid-sim`; any other value fails a fake-coe scenario |
| `ETHRC_TEST_SIM_SEGMENT` | The segment path, required with `humanoid-sim` |

With `ETHRC_TEST_PLANT=humanoid-sim`, every scenario launched on the fake-coe or Unitree bus gets `--plant humanoid-sim --sim-segment $ETHRC_TEST_SIM_SEGMENT`. These are the CTest targets that set `ETHRC_TEST_BUS=fake-coe` or `unitree` (`<scenario>_fake_coe`, `<scenario>_unitree`) and the scenarios that start those lanes themselves (the `g1_*` scenarios); scenarios on the Python bus ignore the variables. For the G1's scenarios, `scripts/sim/humanoid_sim_scenarios.py` starts the sim each one needs ([Running the G1 scenarios](#running-the-g1-scenarios)).

Each scenario starts and stops its own Controls session, so the sim must serve client after client:

```bash
humanoid-sim serve --segment /srv/hsim/plant --support suspended --persist --truth-every 1
```

The sim's support must match the scenario's profile, but the plants hold a suspended robot differently: Controls' MuJoCo plant welds the pelvis to the world, humanoid-sim hangs the robot from a cable on its torso ([world.md](world.md#suspended-the-lab-gantry)), where it can swing and turn slowly. Scenarios that assume a pelvis fixed in space, such as `g1_slam_truth_scenario.py`, hold only on MuJoCo; and gantry requests with `attachment_bodies` pull on the torso here but on the pelvis there. `simulator_command` defaults to `suspended`; a fake-coe scenario that asks for `profile="free"`, such as the opt-in `relative_balance` acceptance test, needs a sim started with `--support free`.

Controls' test runner is `scripts/ci/test-ros.sh [--ros] [--e2e] [--only <ctest regex>]`, run in the `tests` service of `compose.sim.yaml`. That service neither passes the two variables nor mounts the segment's directory, so add both on the command line:

```bash
containers/compose.sh -f compose.sim.yaml -f containers/compose.sim.headless.yaml run --rm \
  -e ETHRC_TEST_PLANT=humanoid-sim -e ETHRC_TEST_SIM_SEGMENT=/shared/plant -v /srv/hsim:/shared \
  tests bash scripts/ci/test-ros.sh --e2e --only '_fake_coe$'
```

Here `/srv/hsim` is the host directory the sim serves into, directly or from its container with the directory mounted; the tests container sees it as `/shared`. The `tests` service drops all capabilities, so the segment file must be readable and writable by the container's root user (see [Shared volume, clock and permissions](#shared-volume-clock-and-permissions)). Run the sim on a GPU for these scenarios: on the CPU device it falls behind real time, and the plant host's status then reports `keeping_up` false.

## The G1 lane

Controls' G1 lane runs the production G1 host (`ethrc_body_host`) on its `LowCmd` bus against a fake motor board (`ethrc_fake_unitree`), through the same shared segment it uses with Unitree's bridge on the robot. The board's plant host steps a MuJoCo plant by default; with `--physics humanoid-sim` humanoid-sim is the plant. The adapter is on Controls' `master`; the gantry, the plant truth, SLAM truth, the compose plant choice and the scenario runner are on Controls' `feat/g1-sim-lane`:

| File | Role |
|---|---|
| `ros2_ws/src/sim/ethrc_sim/ethrc_sim/humanoid_sim/` | Vendored `plant_abi.py` and `plant_abi.h`, and `SegmentClient`, the vendor-neutral side of the segment |
| `ros2_ws/src/sim/ethrc_fake_unitree/ethrc_fake_unitree/sim_plant.py` | `SegmentUnitreePlant`, `SegmentPeer` and `KinematicHands` |
| `ros2_ws/src/sim/ethrc_fake_unitree/ethrc_fake_unitree/plant_host.py` | `--physics humanoid-sim --sim-segment PATH --sim-wait S` |
| `scripts/sim/launch.py` | `--bus unitree --plant humanoid-sim --sim-segment PATH` |
| `ros2_ws/src/sim/ethrc_fake_unitree/test/test_sim_plant.py` | Unit tests against a stand-in sim |
| `ros2_ws/src/sim/ethrc_fake_unitree/test/test_sim_plant_live.py` | The adapter against a live sim (`ETHRC_TEST_HUMANOID_SIM_SEGMENT`) |
| `scripts/sim/humanoid_sim_world.py` | The session's environment as `serve --scene --spawn` |
| `scripts/sim/humanoid_sim_scenarios.py` | Controls' G1 scenarios on humanoid-sim, or on MuJoCo beside it |
| `containers/compose.sim.humanoid-sim.yaml` | `just sim-isaac` (`SIM_ROBOT=g1` for the G1): the sim, its viewer and the stack |

Serve the variant Controls runs, with the support the session asks for and the truth every step:

```bash
humanoid-sim --robot g1_29dof_with_hand_rev_1_0 serve --segment /srv/hsim/plant --support suspended --truth-every 1
```

The G1's motor driver runs each motor's PD itself, and so does humanoid-sim, so the plant host sends laws, not torques. Every period (the description's `control_period_s`, 2 ms) the peer passes the host's command through the relay and the board as on MuJoCo, then hands humanoid-sim each motor's law (`Board.law`): PD with the command's `q`, `dq`, `tau`, `kp` and `kd` for an enabled motor, DISABLED for a limp one, no torque limit (the sim applies the motor's envelope). It keeps doing so while paused, because the sim pauses its own physics on `PAUSE` and ends a session whose demands stop. The motors map to the sim's drives by name: the embodiment's actuator names are the description's joint names, which are the sim's drive names, and the motor frame (`actuator = sign · joint`) is the sim's drive frame.

`LowState` reads what the sim's motors measure. Its IMU gyroscope and accelerometer are the newest sample of the sim's IMU model; its quaternion, on the robot the board's own estimate, is the true orientation: the truth region's base orientation through the URDF's `imu_joint`, as the MuJoCo plant reports its site's. The plant host refuses a truth more than one physics step behind the feedback, hence `--truth-every 1`. The sim welds the Dex3 fingers, so each declared hand is a kinematic `FakeDex3` stepped with the loop, outside the body's physics. Both are in the status's `fidelity_gaps`.

The sim's support is fixed when it starts: the plant host exits if `--profile` differs from it and refuses `/sim/profile` requests. `/sim/reset` sends `RESET` to the home pose and, in a free session, `SET_SUPPORT` free.

### The test gantry and the plant's truth

Controls' G1 scenarios that stand or walk the free robot boot it hanging from the fake board's test-only gantry and lower it under a policy (`g1_walk`, `g1_sonic_navigation`, `g1_sonic_teleop`). On humanoid-sim each gantry request becomes the sim's own commands, posted one at a time after each ack so the plant host's 2 ms loop never waits (`SimGantryRequest` in `sim_plant.py`):

1. `PRESCRIBE_POSE` with the request's joint angles (every other joint at zero), the root where it is, and `floor_clearance_m` as the request's lift: the lowest sole-plate corner that far above the floor.
2. `SET_SUPPORT` with the request's tension program or cable. A request that names the harness body (`attachment_body`, `torso_link` on the G1, as the lab holds it) hangs from the harness hook: the plant host sends the descriptor's `harness.shoulders` as `attachment_bodies` (ABI 3.3), so the rope holds the midpoint of the shoulders on the torso. Any other body is refused.
3. `PRESCRIBE_POSE` `{"release": true}`, so the body moves under the rope.

A request with a field the sim cannot take is refused before any command; a refusal from the sim releases the pose and refuses the request, which changes nothing. A request written before the plant host starts boots the robot hanging: the sim holds its start pose until the first demands, and the plant host completes the request before it writes them. The answer reports the plant time the rope took effect.

The truth file the scenarios use as their oracle (`plant-truth.json`) reads the sim's truth region: the root position, the floor clearance (the lower sole's `lowest_z_m`), the rope's tension, the body's mass and the plant time, stamped with the wall-clock time the sim's feedback maps it to. The scenarios size the boot rope by the sim's own mass (`humanoid_sim_mass_kg` in Controls' `sim_fixture.py`).

### SLAM truth and the world

`--slam-truth` publishes the SLAM contract (`nav_base` pose, status, the map) from the plant's truth: the pose from the sim's truth region, the occupancy grid from Controls' MuJoCo model of the environment. The simulated Mid-360 casts its rays on the same model. Both are only right when the sim serves the world Controls maps, so the plant host compares the descriptor's `scene` with the environment's boxes file and spawn point and exits on a mismatch. `scripts/sim/humanoid_sim_world.py` writes the arguments that serve it:

```bash
python3 scripts/sim/humanoid_sim_world.py --environment hangar --into /srv/hsim   # in Controls' image
humanoid-sim --robot g1_29dof_with_hand_rev_1_0 serve --segment /srv/hsim/plant --support suspended \
  --truth-every 1 --persist $(cat /srv/hsim/world.args)
```

`empty` needs no arguments; `hangar` and `hangar_scan` serve `hangar_proxy.npz` (Git LFS) with the hangar's spawn point at the origin. `just sim-isaac` serves the same world from `SIM_ENVIRONMENT` in its compose file, under the lab gantry's crane. The `room` is built in Controls' code and has no boxes file, so `g1_explore` cannot run on humanoid-sim yet. With SLAM truth served, the launcher starts the G1's navigation (`body-tf`, `sim-lidar`, the map relay and Nav2) on humanoid-sim as on MuJoCo.

`test_sim_plant_live.py` checks the adapter against a running sim. In the humanoid-sim image, with a Controls checkout mounted at `/controls` and its Python packages on `PYTHONPATH`:

```bash
humanoid-sim --robot g1_29dof_with_hand_rev_1_0 serve --segment /tmp/plant --device cpu --support suspended --truth-every 1 &
ETHRC_TEST_HUMANOID_SIM_SEGMENT=/tmp/plant python -m pytest /controls/ros2_ws/src/sim/ethrc_fake_unitree/test/test_sim_plant_live.py
```

It steps the left knee 0.2 rad with every motor at its deploy gains and checks that the knee gets there, that every motor applies `kp (target - q)` at rest, that the IMU reads a still, upright pelvis, and that limp motors apply no torque. It passes on Warp's CPU device.

### Running the G1 scenarios

`scripts/sim/humanoid_sim_scenarios.py` in Controls runs Controls' G1 scenarios on humanoid-sim. Each scenario's session asks for a support and a world; the script starts one `serve --persist --truth-every 1` per support and world, with the world from `humanoid_sim_world.py`, runs each scenario in Controls' simulation-tests image (`scripts/ci/test-ros.sh --e2e --only`) with `ETHRC_TEST_PLANT=humanoid-sim`, samples the plant host's status while it runs, and prints one JSON line per scenario: pass or fail, the plant's real-time factor over the session, the sim's late steps, the board's command-loss timeouts and the host's longest command gap. `--plant mujoco` runs the same scenarios on MuJoCo for a side-by-side, and `sim_plant_live` runs `test_sim_plant_live.py` against the same sim. On a GPU node with humanoid-sim installed:

```bash
scripts/sim/humanoid_sim_scenarios.py --controls-image ethrc-controls:simulation-tests \
  --sim-command humanoid-sim --sim-repo ../humanoid-sim --sim-cpus 0,4 --controls-cpus 1-3,5-7 \
  --env G1_WALK_BUNDLE=/bundle --volume "$PWD/../g1_walk_bundle:/bundle" \
  --volume "$PWD/ros2_ws/policy_checkpoints:/opt/ethrc/repository/ros2_ws/policy_checkpoints" \
  sim_plant_live g1_policy g1_walk g1_sonic_teleop g1_sonic_navigation
```

`g1_walk` needs a walking bundle in `G1_WALK_BUNDLE`, and the SONIC scenarios need `gear-sonic-v1.1` in the policy library (Controls' `scripts/policy/install_sonic.py`). `g1_explore` runs in the room, which has no boxes file, so it cannot run here yet.

### Real time

The G1 lane is wall-clocked: the board's command loss, the body host and the policies run on the wall clock, and the scenarios fail a phase in which the plant falls behind it by more than 5 % (`keeping_up` in Controls' `g1_walk_scenario.py`). Three things decide whether the sim keeps up beside Controls' stack on one machine:

- **The step's own cost.** On an L40S node one step of the free G1 takes 1.6 ms with the truth published every step (1.15 ms every fifth): the physics 0.9 ms, the feedback 0.19 ms, the truth 0.61 ms. A rope adds its own per-step work; the runtime publishes feedback and truth in one transfer and batches the soles, and the rope pushes only the points it uses.
- **The CPU.** The step is a chain of kernel launches and two waits on the GPU, so a CPU thread that the stack preempts starts it late. Controls' stack (SONIC, Nav2, the API) loads 6 to 8 CPUs during a scenario. The scenario script runs the stack 10 nice levels below the sim, and the compose simulation runs the sim 10 levels up (`CAP_SYS_NICE`). On a node with 8 vCPUs, give the sim one physical core of its own (`--sim-cpus 0,4 --controls-cpus 1-3,5-7`; `lscpu -e` shows which CPUs share a core).
- **The GPU.** Another job on the same GPU delays each step's kernels.

Measured on 2026-10-20 (the plant's real-time factor over each scenario's session):

| Host | Scenarios | Real-time factor | Late steps |
|---|---|---|---|
| L40S node, 8 vCPUs, sim on its own core | g1_policy (suspended) | 0.998 | 2 in 33 s |
| | g1_walk (free, gantry, walking) | 0.997 | 24 in 51 s |
| | g1_sonic_navigation (free, hangar) | 0.992 | 86 in 53 s |
| GB10 Spark, 20 cores, sim alone behind a 2 ms client | free, holding still | 1.000 (0.995 with 18 busy processes) | 0 in 30 s (32 with 18 busy) |
| Apple silicon Mac, Warp's CPU device, sim alone | free, holding still | 0.91 | 563 in 30 s |
| L4 node, 8 vCPUs, before the batched truth | free scenarios | 0.90 | about 20 per second |

## Shared volume, clock and permissions

The segment is a file that both processes map. Both must run on the same machine, because the protocol compares `CLOCK_MONOTONIC` values across them: the sim's client timeout, the plant host's heartbeat check and the IMU stamp mapping. Containers on one host share that clock. The file lives on a Docker named volume mounted into both containers (as `humanoid_sim_lane.py --sim-image` does) or in a host directory bind-mounted into the Controls container (`--sim-dir`).

The sim creates `<path>.<pid>.tmp`, sets its mode to `0660` whatever its umask (`Segment.create` calls `fchmod`), and renames it over the path, so it needs write access to the directory. The file belongs to the sim's user and that user's group. The plant host opens the file read-write, because it writes the demands and mailbox regions, so it must run as the same user as the sim or as a member of the file's group. The mode never grants other users access. A `PermissionError` from `Segment.open` is raised at once instead of being retried until `--sim-wait`. The same holds for `humanoid-sim view-runtime`, which also opens the file read-write.

Both images run as root. The lane script's plain `docker run` keeps root's usual capabilities, so the Controls container can open a file created by any user. Controls' compose services, the `tests` service included, drop all capabilities (`cap_drop: [ALL]` in `containers/compose.common.yaml`), so root there gets no exemption from file permissions. Under compose, run the sim as root (its image does), as the user the Controls container runs as, or with a group that user belongs to.

## Fidelity gaps

The plant projection on `/ethrc/plant` lists what the lane does not model, in `fidelity_gaps`. With humanoid-sim the list is the sim's own (the descriptor's `fidelity_gaps`, listed in [runtime.md](runtime.md#descriptor)), followed by the plant host's:

- bank test control 'disturb' reaches the sim at the status period, not on the next bus cycle (`SEGMENT_GAPS`)
- Command path software latency is the host's (the real master on this machine, not the robot's computer); the bus and drive delay is humanoid-sim's drive model (`SEGMENT_COMMAND_PATH_GAP`)

The projection's IMU fields name the sim's sensor model (`imu_model`, `imu_fidelity_gap`). `/rt/imu` comes from Madgwick on the device's raw samples, as on the robot.
