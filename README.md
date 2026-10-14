# humanoid-sim

Shared, batched device models for ETHRC V1, with an Isaac Lab + Newton integration.
Modules live directly in `src/` and install as `humanoid_sim`. RL tasks, rewards,
trainers and deployment controllers belong in their consumer repositories.

## Boundary

```text
device process data → firmware state/demands → actuator effort
                                              ↓
                                    mechanical transmission
                                              ↓
                                         Newton physics
                                              ↓
                                      device measurements
```

`DeviceBank` owns one state per environment and drive. Requested commands,
delivered demands, controller effort, physical effort feedback and timestamps
are distinct. `IsaacEffortAdapter.pre()` stages effort; the caller steps physics
exactly once; `post()` refreshes feedback. Native actuator PD must be disabled.
RL uses these hooks within its existing physics substeps, without a second loop.

`runtime.lock.yaml` pins the Isaac Lab development snapshot resolved on
2026-10-14 and its Newton revision. The container installs that snapshot's frozen
dependency graph: Python 3.12, Torch 2.11, Warp 1.17, and kitless Newton. The
smoke explicitly selects Newton Featherstone and refuses a different backend.
It does not require the full Isaac Sim application. Updating the runtime is an
explicit lock/image change, never a floating install on job startup.

## CPU development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run humanoid-sim smoke --backend cpu --steps 120 --output-dir outputs/cpu
```

CPU smoke uses a single-DOF inertia stub to exercise device sequencing and the
adapter lifecycle. It is not a Newton or whole-robot result. Tests cover delayed
frames, enable sequencing, watchdogs, selective resets, torque mapping, physics
ownership and packaging-compatible imports. CUDA parity runs when CUDA exists.
CI runs the CPU contracts and wheel smoke on native Linux x86-64 and ARM64 runners.

## Newton container

Run on a Linux x86-64 or ARM64 NVIDIA GPU host with Docker and NVIDIA Container
Toolkit. Compose builds for the host architecture. The pinned upstream graph
includes Linux aarch64 (including DGX Spark), with CUDA 13 PyTorch wheels; the
host GPU/driver stack must support that runtime. ARM64 alone does not establish
compatibility with a particular Jetson/JetPack release. Apple Silicon supports
the CPU model tests, not NVIDIA GPU execution through Docker Desktop.

```bash
docker compose build sim
docker compose run --rm sim
docker compose --profile dev run --rm dev
docker compose --profile viewer run --rm --service-ports viewer
```

To publish one image reference for both Linux architectures:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t REGISTRY/humanoid-sim:VERSION --push .
```

Use native build workers for each architecture when available. Both targets use
the same source/dependency lock and their own checksum-pinned `uv` executable.

The default image contains an installed package. Only `dev` uses an explicit
editable source mount. The procedural single-joint scene needs no robot assets
or credentials. Smoke checks actual nonzero effort, joint motion and finite
physical state, and writes a summary/trace to the `runs` volume.

The optional Viser viewer opens at `http://localhost:8080`. It visualizes the
Newton scene in a browser; it is not an Isaac Sim desktop stream. From a remote
host, forward its loopback-bound port with `ssh -L 8080:localhost:8080 HOST`.
Headless runs do not start a viewer. Native Newton GL is also available through
the CLI on a suitably configured display host.

## SkyPilot smoke

Build and publish the image to your chosen registry first. The manifest has no
hardcoded registry, cloud, source checkout or credential. It starts a finite
GPU smoke on a host capable of running Docker; it is not a training job.
Its default L4/A10G resource choices target the x86 cloud lane. For an ARM64
machine pool, select the appropriate SkyPilot resources and use the ARM64 image
or a multi-architecture manifest; the Docker task itself does not force x86.

```bash
sky launch -c humanoid-smoke sky/smoke.yaml \
  --env HUMANOID_SIM_IMAGE=REGISTRY/IMAGE@sha256:DIGEST -i 5
```

The task prints its output directory. Download results before deleting the
cluster; autostop preserves its disk. Training, checkpoints and W&B belong in
`humanoid-rl`. That image should consume this package/runtime at pinned revisions.

## V1 description

`profiles/v1.yaml` pins the authoritative `humanoid-urdf` revision and content
hashes. Provide that checkout explicitly; no local sibling path is embedded.

```python
from humanoid_sim.robot import load_v1

robot = load_v1("/assets/humanoid-urdf", "profiles/v1.yaml")
configs = [drive.config for drive in robot.drives]
```

The inventory is 10 HEJ90, 4 HEJ70 and 8 HEJ50 devices, including two HEJ50
cranks per ankle. Bus addresses are not inferred from list order. This loader
verifies the inventory and supplies baseline gains/peak caps; it does not yet
construct a V1 scene. `CoupledTransmission` requires explicit calibrated
kinematics and uses `J = d(device angle)/d(joint angle)`, so `tau_joint = J.T tau_device`.
No constant or independent-joint ankle approximation is selected implicitly.

## Implemented fidelity

- Batched SI representations of JVPT process-data fields, mode `-64`; raw PDO
  encoding and EtherCAT transport are not implemented.
- Explicit subset of enable/disable/quick-stop/fault-reset states, fixed command
  delay, watchdog fault latch, output PD/feedforward, caps, optional friction
  and first-order torque response. Unsupported controlwords/modes and runtime
  register configuration raise instead of silently succeeding.
- Integer firmware/physics ticks. Physics must resolve each firmware event;
  skipped ticks raise. Default 400 µs is consistent with the description's
  2.5 kHz controller rate. Bus period and policy period are separate concerns.
- Feedback currently exposes sampled physical position/velocity directly.
  Measured effort stays unavailable unless explicitly supplied by a physical
  sensor. Source time and last firmware evaluation time are separate.
- Pending commands and controller state are reset per environment. Command
  validation checks finite inputs; physical finite checks can be disabled in
  the GPU hot path after validation. This foundation is not throughput-qualified.

Initial watchdog policy, transition timing and optional actuator filters are
behavioral assumptions, not verified HEJ firmware. The optional linear speed
envelope is not the measured HEJ map and is not enabled by the V1 loader.
SDO/object dictionary, homing/persistence, encoder quantization/filtering,
thermal/current protection and full calibrated V1/ankle assembly remain to be
implemented against documentation and independent traces. The existing Controls
EPOS4/JVPT emulator is a reference, not firmware ground truth.

## Source references

- [Pinned Isaac Lab source](https://github.com/isaac-sim/IsaacLab/tree/b9a673a29f3a7f09508f339c3ec370fdda2f542f)
- [Pinned Newton source](https://github.com/newton-physics/newton/tree/5ea82eef6f7911388025a8df6ccdb172e2fed286)
- [V1 description](https://github.com/ETHRoboticsClub/humanoid-urdf/tree/a351b4fa25b71ffee466be62127619f8e4fd252e)

The container recipe, GPU Newton smoke and browser viewer require execution on
the target GPU host before they can be called validated. CPU tests cannot
establish those results or hardware fidelity.
