# humanoid-sim documentation

Start with [overview.md](overview.md) for how the parts fit together, then pick the page for what you are doing. [glossary.md](glossary.md) defines the terms the pages use.

## By role

| You are | Start here | Then |
|---|---|---|
| New to the repo | [overview.md](overview.md) | [getting-started.md](getting-started.md), [visualization.md](visualization.md) |
| Hardware engineer checking or changing a robot fact | [robot-model.md](robot-model.md) | [drives.md](drives.md), [sensors.md](sensors.md), [replay-and-sysid.md](replay-and-sysid.md) |
| Writing a training task (humanoid-learning) | [training.md](training.md) | [world.md](world.md), [drives.md](drives.md), [sensors.md](sensors.md) |
| Running Controls against the sim | [controls-integration.md](controls-integration.md) | [runtime.md](runtime.md), [rtx-sensors.md](rtx-sensors.md) |
| Checking the sim against the real robot | [replay-and-sysid.md](replay-and-sysid.md) | [visualization.md](visualization.md) |
| Running tests or GPU jobs | [testing-and-gpu.md](testing-and-gpu.md) | [benchmarks.md](benchmarks.md) |

## All pages

| Page | What it covers |
|---|---|
| [overview.md](overview.md) | The layers, the three front ends, measured versus true quantities |
| [getting-started.md](getting-started.md) | Install, build the robot asset, first replay, first runtime session, first training scene |
| [glossary.md](glossary.md) | Terms, one heading each |
| [robot-model.md](robot-model.md) | Where robot facts come from, joint and drive frames, the asset build, standing on the ground |
| [drives.md](drives.md) | Drive demands and modes, arrival and latency, the drive law, torque envelope, ankle linkage, target shaping, faults |
| [sensors.md](sensors.md) | Encoders, the IMU and its burst delivery, contacts, what is measured and what is true |
| [rtx-sensors.md](rtx-sensors.md) | Cameras and lidar rendered by Isaac Sim's RTX, published as Controls reads them, with their checks and cost |
| [training.md](training.md) | The Train-Time API: the plant in an Isaac Lab task, randomization, observations, logging |
| [world.md](world.md) | Ground and terrain, pushes and joint disturbances, the gantry |
| [runtime.md](runtime.md) | `serve`, the shared-memory segment, sessions, commands, `view-runtime` |
| [controls-integration.md](controls-integration.md) | How Controls' plant host attaches and what it expects |
| [replay-and-sysid.md](replay-and-sysid.md) | Replaying captures, scoring, fitting delays and passive joint parameters |
| [visualization.md](visualization.md) | Rerun recordings of replays, live sessions and training scenes, with screenshots |
| [testing-and-gpu.md](testing-and-gpu.md) | Test markers, CI, the Docker image, SkyPilot GPU tasks |
| [benchmarks.md](benchmarks.md) | Measured speed of the runtime and of training, and how to measure it |
