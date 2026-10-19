# Running Controls against humanoid-sim

Controls' fake-coe lane runs the production EtherCAT master, the `standalone` binary of `ethercat_device_configurator` with `--transport fake-coe`, against a bank of simulated EPOS4 drives (`Epos4Slave` in Controls' `ethrc_fake_ethercat` package). The master speaks CANopen over EtherCAT (CoE) to them as it would to the robot. A separate plant host process owns that bank: each bus cycle it reads every drive's demand from the bank and gives each drive its feedback. By default the plant host steps a MuJoCo plant itself. With `--physics humanoid-sim` it forwards each demand to humanoid-sim's runtime segment and hands the sim's feedback back to the bank. Everything above the drives (master, safety monitor, router, policies, robot API) is the stack that runs on the robot.

This page covers the Controls side. [runtime.md](runtime.md) describes `humanoid-sim serve`, the segment and the commands.

## The Controls branch

The humanoid-sim plant is on Controls' branch `exp/humanoid-sim-runtime`, which is not merged into `master`; `master` has the MuJoCo plant only. The files involved, relative to the Controls checkout:

| File | Role |
|---|---|
| `ros2_ws/src/sim/ethrc_fake_ethercat/ethrc_fake_ethercat/plant_host.py` | The plant host; `--physics humanoid-sim` runs `run_segment` |
| `ros2_ws/src/sim/ethrc_fake_ethercat/ethrc_fake_ethercat/sim_segment.py` | `SegmentPlant` (the exchange, IMU samples, commands, detach), `SegmentTestControl` and `SegmentStandingTruth` (test seams and oracle) |
| `ros2_ws/src/sim/ethrc_fake_ethercat/ethrc_fake_ethercat/plant_abi.py` | Vendored copy of humanoid-sim's `plant_abi.py` |
| `ros2_ws/src/sim/ethrc_fake_ethercat/include/ethrc_fake_ethercat/plant_abi.h` | Vendored copy of humanoid-sim's `plant_abi.h` |
| `ros2_ws/src/sim/ethrc_sim/ethrc_sim/sensors.py` | `RawImuChain`: the deployed IMU filter and relay on raw samples |
| `scripts/sim/launch.py` | `--plant humanoid-sim --sim-segment PATH --sim-wait S` |
| `scripts/sim/humanoid_sim_lane.py` | The lane end to end in two containers |
| `ros2_ws/src/tests/ethrc_integration_tests/test/sim_fixture.py` | `plant_options`: `ETHRC_TEST_PLANT` and `ETHRC_TEST_SIM_SEGMENT` |
| `ros2_ws/src/sim/ethrc_fake_ethercat/test/test_sim_segment.py` | Unit tests of the segment side against a stand-in sim |

The two ABI files are copied unchanged from `src/humanoid_sim/runtime/abi/`. With `HUMANOID_SIM_ROOT` set to a humanoid-sim checkout, `test_vendored_abi_matches_humanoid_sim` compares both copies byte for byte. When the ABI changes, copy both files again; a different layout hash or major version makes `Segment.open` refuse the segment.

## What the plant host does

`run_segment` in `plant_host.py` does the following.

At startup:

- It computes the description digest from the description's own `actuators.yaml` and the URDF it names (`description_files`; launch.py passes the file as `--sim-description`, since the session's `actuators.yaml` is a materialized copy). It opens the segment with `Segment.open(path, description_digest=..., wait_s=--sim-wait)` and so refuses a sim running a different description. It does not check the header's `robot_id`.
- It reads the descriptor's drive names and fidelity gaps, and records the hashes it reports: the description digest, the sim session id (`sim_session`, 16 hex digits) and the sim's revision.
- It sends `SET_FIDELITY`: `ideal` for the `minimal` fidelity preset, `real` for the `real` preset (the plant host's default).
- It checks that every drive in the bus inventory exists in the sim, and exits otherwise.

Every bus cycle (the period from Controls' drive configuration):

- `SegmentPlant.exchange` writes one demands record. Each drive's bank demand goes in unchanged (`demand_fields`); a sim drive missing from the bus inventory is written disabled.
- It reads the feedback region and passes each drive's position, velocity, applied torque and demanded torque to the bank (`bank.feedback`). If the region is mid-write it reuses the previous cycle's feedback.
- It raises `SimLost` when the segment's state is no longer ready or the sim's heartbeat is older than 500 ms (`SIM_STALE_NS`). The plant host then writes the fault to its status file and exits.
- `new_imu_samples` takes the IMU samples it has not seen (by `seq`), oldest first, and maps their stamps to its own `CLOCK_MONOTONIC` with the feedback's `written_mono_ns - sim_time_ns`. The sim's `seq` keeps rising across a `RESET` and across sessions, so the last number seen stays valid. `RawImuChain` runs each through the deployed chain (Madgwick with the time step from consecutive receipt stamps, then the relay's rotation into `base_link` by the description's mount), and the plant host publishes the result on `/rt/imu`.

Every status period (100 ms):

- It writes its status file (`ethrc-fake-plant/v1`) with `physics: "humanoid-sim"`, `sim_late_steps`, and `keeping_up`, which is true only while the worst lateness of its own cycles so far is within `--max-lateness` (5 ms) and the sim reports `keeping_up`. One cycle later than 5 ms makes it false for the rest of the run.
- It publishes the latched plant projection on `/ethrc/plant` (schema `ethrc-plant/v1`), including the hashes and the fidelity gaps (see [Fidelity gaps](#fidelity-gaps)).
- In the device-free test container only (`ETHRC_SIM_TEST_CONTROL` set and `/.dockerenv` present), it forwards the test seams. The bank test control's joint disturbances become `APPLY_DISTURBANCE` joint sines on 1:1 drives, sent only when they change (`SegmentPlant.disturb_drives`). The floating-body requests (`calibration-pose`, `gantry`, `base_tilt`) become `PRESCRIBE_POSE` and `SET_SUPPORT` commands (`SegmentTestControl.poll`). With `ETHRC_SIM_TEST_ORACLE_STATE` also set, it writes the plant oracle from the truth region (`SegmentStandingTruth`), whose maxima and normal-force window advance on every new truth record. Run the sim with `--truth-every 1` for per-step windows.

On exit, `SegmentPlant.close` sends `DETACH` while the session is still ready, before ROS shuts down: the launcher kills the plant host 3 s after SIGTERM, and a `--persist` sim then opens the next session at once instead of waiting out its client timeout.

The plant host uses the commands `SET_FIDELITY`, `APPLY_DISTURBANCE` (joint torques), `PRESCRIBE_POSE`, `SET_SUPPORT` and `DETACH`. It does not send `RESET`, `PAUSE`, `RESUME` or `INJECT_FAULT`; bus, protocol and drive faults come from Controls' own fake bank.

## Running the lane with launch.py

humanoid-sim serves first, on the same machine, with its segment on a volume the Controls side also mounts:

```bash
humanoid-sim serve --segment /run/hsim/plant --support suspended
```

Then, in Controls' simulation image (or a checkout with the sim workspace built; see Controls' `docs/sim-bringup.md`), with the same directory mounted:

```bash
python3 scripts/sim/launch.py --headless --hands none --clock wall --bus fake-coe \
  --plant humanoid-sim --sim-segment /run/hsim/plant --profile suspended --environment empty
```

| Option | Default | Notes |
|---|---|---|
| `--plant mujoco\|humanoid-sim` | `mujoco` | `humanoid-sim` needs `--bus fake-coe` and `--sim-segment`; launch.py refuses it otherwise |
| `--sim-segment PATH` | none | The file `humanoid-sim serve --segment` publishes |
| `--sim-wait S` | `180` | How long the plant host waits for the segment to become ready. The launcher waits that long plus 30 s for the plant host to report ready. |
| `--profile suspended\|free` | `free` | Must match the sim's `--support`. Nothing compares the two, and `serve` defaults to `suspended`. |
| `--clock plant\|wall` | `plant` with `--hands none`, otherwise `wall` | The fake-coe lane runs on the wall clock: pass `--clock wall` with `--hands none`, since launch.py refuses `plant` with `--bus fake-coe` |
| `--fidelity minimal\|real` | the plant host's, `real` | Selects the sim's IMU noise through `SET_FIDELITY` |

`--environment` and `--imu-mount-rpy` shape the MuJoCo plant; the plant host ignores them on humanoid-sim.

## Running the lane check: `humanoid_sim_lane.py`

`scripts/sim/humanoid_sim_lane.py` brings the lane up end to end and checks it. It needs Docker, Controls' simulation image, and either humanoid-sim's image with a checkout whose robot asset is built, or a sim already serving.

```bash
# humanoid-sim in its own container, on a fresh Docker volume
scripts/sim/humanoid_sim_lane.py --sim-image humanoid-sim:dev --sim-repo ../humanoid-sim \
  --controls-image ethrc-controls:simulation --device cuda:0

# humanoid-sim already serving into a host directory (for example on a GPU node, outside a container)
humanoid-sim serve --segment /srv/hsim/plant --device cuda:0 --client-timeout-ms 10000 &
scripts/sim/humanoid_sim_lane.py --sim-dir /srv/hsim --controls-image ethrc-controls:simulation
```

With `--sim-image`, the script creates a Docker volume and runs the sim image with `--sim-repo` mounted at `/workspace/humanoid-sim`, adding `--gpus all` for a CUDA device, as `serve --segment /shared/plant --device <device> --support <support> --client-timeout-ms 10000`. With `--sim-dir`, the directory must hold the segment as `plant`, and `--support` must match how that sim was started. In both cases it runs the Controls image with the volume or directory at `/shared`:

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

The script waits until every drive is enabled or the plant host reports a fault, runs for `--seconds`, and then reads the segment from inside the Controls container. It prints a JSON result: the plant host's status (`physics`, `cycles`, `keeping_up`, `drives`, `drives_operational`, `drives_enabled`, `drives_faulted`, `fault`) and, from the segment, `sim_steps`, `sim_time_s`, `sim_late_steps`, `imu_samples`, `jvpt_drives` and `max_tracking_error_rad`. `ok` is true, and the exit code 0, when there is no fault, every drive is enabled and none faulted, at least one drive runs JVPT, and the largest JVPT tracking error is under 0.05 rad.

On the default CPU device the lane comes up but the sim does not keep up with real time.

To watch the robot while Controls drives it, run `humanoid-sim view-runtime` on the same segment, for example `humanoid-sim view-runtime /srv/hsim/plant --save lane.rrd` next to a sim serving into `/srv/hsim`. The viewer only reads the segment, so the lane runs as it would without it, and it follows the sim from session to session under `--persist`. [runtime.md](runtime.md#watching-a-host-view-runtime) lists its options and what it needs.

## Running Controls' integration tests on humanoid-sim

The integration scenarios that run on the fake-coe bus can use humanoid-sim as their plant. `plant_options` in `sim_fixture.py` reads two variables:

| Variable | Meaning |
|---|---|
| `ETHRC_TEST_PLANT` | `mujoco` (default) or `humanoid-sim`; any other value fails a fake-coe scenario |
| `ETHRC_TEST_SIM_SEGMENT` | The segment path, required with `humanoid-sim` |

With `ETHRC_TEST_PLANT=humanoid-sim`, every scenario launched on the fake-coe bus gets `--plant humanoid-sim --sim-segment $ETHRC_TEST_SIM_SEGMENT`. These are the CTest targets that set `ETHRC_TEST_BUS=fake-coe` (`<scenario>_fake_coe`) and the scenarios that start the fake-coe lane themselves; scenarios on the Python or Unitree buses ignore the variables.

Each scenario starts and stops its own Controls session, so the sim must serve client after client:

```bash
humanoid-sim serve --segment /srv/hsim/plant --support suspended --persist --truth-every 1
```

The sim's support must match the scenario's profile. `simulator_command` defaults to `suspended`; a fake-coe scenario that asks for `profile="free"`, such as the opt-in `relative_balance` acceptance test, needs a sim started with `--support free`.

Controls' test runner is `scripts/ci/test-ros.sh [--ros] [--e2e] [--only <ctest regex>]`, run in the `tests` service of `compose.sim.yaml`. That service neither passes the two variables nor mounts the segment's directory, so add both on the command line:

```bash
containers/compose.sh -f compose.sim.yaml -f containers/compose.sim.headless.yaml run --rm \
  -e ETHRC_TEST_PLANT=humanoid-sim -e ETHRC_TEST_SIM_SEGMENT=/shared/plant -v /srv/hsim:/shared \
  tests bash scripts/ci/test-ros.sh --e2e --only '_fake_coe$'
```

Here `/srv/hsim` is the host directory the sim serves into, directly or from its container with the directory mounted; the tests container sees it as `/shared`. The `tests` service drops all capabilities, so the segment file must be readable and writable by the container's root user (see [Shared volume, clock and permissions](#shared-volume-clock-and-permissions)). Run the sim on a GPU for these scenarios: on the CPU device it falls behind real time, and the plant host's status then reports `keeping_up` false.

## Shared volume, clock and permissions

The segment is a file that both processes map. Both must run on the same machine, because the protocol compares `CLOCK_MONOTONIC` values across them: the sim's client timeout, the plant host's heartbeat check and the IMU stamp mapping. Containers on one host share that clock. The file lives on a Docker named volume mounted into both containers (as `humanoid_sim_lane.py --sim-image` does) or in a host directory bind-mounted into the Controls container (`--sim-dir`).

The sim creates `<path>.<pid>.tmp`, sets its mode to `0660` whatever its umask (`Segment.create` calls `fchmod`), and renames it over the path, so it needs write access to the directory. The file belongs to the sim's user and that user's group. The plant host opens the file read-write, because it writes the demands and mailbox regions, so it must run as the same user as the sim or as a member of the file's group. The mode never grants other users access. A `PermissionError` from `Segment.open` is raised at once instead of being retried until `--sim-wait`. The same holds for `humanoid-sim view-runtime`, which also opens the file read-write.

Both images run as root. The lane script's plain `docker run` keeps root's usual capabilities, so the Controls container can open a file created by any user. Controls' compose services, the `tests` service included, drop all capabilities (`cap_drop: [ALL]` in `containers/compose.common.yaml`), so root there gets no exemption from file permissions. Under compose, run the sim as root (its image does), as the user the Controls container runs as, or with a group that user belongs to.

## Fidelity gaps

The plant projection on `/ethrc/plant` lists what the lane does not model, in `fidelity_gaps`. With humanoid-sim the list is the sim's own (the descriptor's `fidelity_gaps`, listed in [runtime.md](runtime.md#descriptor)), followed by the plant host's:

- bank test control 'disturb' reaches the sim at the status period, not on the next bus cycle (`SEGMENT_GAPS`)
- with the `real` preset: "fidelity real: the identified command path latency is not added"; humanoid-sim takes each demand as the bus writes it
- IMU samples from the plant's sensor model (noise, bias, burst delivery); filter and relay as deployed, run in the plant process (`RawImuChain.fidelity_gap`)
- Command path latency is the host's: the real master's software path and no modeled bus or drive delay (ethrc_sim.command_latency runs on the Python bus only) (`COMMAND_PATH_GAP`)

The IMU identity the plant host latches reports the mount revision as `humanoid-sim-` followed by the first 16 hex digits of the description digest, with `mount_verified` false.
