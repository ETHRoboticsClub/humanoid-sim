# Robot model

humanoid-sim does not define the robot. Geometry, masses, joints, drives and their constants come from the humanoid-urdf description in `third_party/humanoid-urdf`, read in place. This page follows that description from its files to the simulated articulation: how it is loaded and identified, how `humanoid-sim build-asset` turns it into the USD file the physics loads, which per-joint values are written to the solver, the host-side kinematics, the two poses the robot can stand in, the left/right mirror map, and the profile `profiles/ethrc_v1.yaml` that holds this repository's modeling choices. The last section is the workflow for getting a measured value into the simulation.

```
third_party/humanoid-urdf/ethrc_v1/            profiles/ethrc_v1.yaml
  actuators.yaml                                 model:    build choices (asset)
  URDF_5_description/urdf/*.urdf, meshes/        physics:, devices:, sensors:  (spawn and run time)
        |                                              |
        v                                              |
  Description             model/description.py         |
        |                                              |
        +----------------------+-----------------------+
                               v
  PreparedUrdf            model/urdf.py      ->  outputs/assets/ethrc_v1/<key>/robot.urdf, collision/*.stl
                               v
  build_asset             model/asset.py     ->  outputs/assets/ethrc_v1/<key>/usd/robot/robot.usda, asset.json
                               v
  joint_properties        model/joint_properties.py, physics/isaaclab.py   (written at spawn)
```

## The description

### Files

`Description` (`src/humanoid_sim/model/description.py`) maps the robot id `ethrc_v1` to the directory `third_party/humanoid-urdf/ethrc_v1/` and the ament package inside it, `URDF_5_description` (the `ROBOTS` table). It reads these files:

| File (under `ethrc_v1/`) | What humanoid-sim takes from it |
|---|---|
| `actuators.yaml` | Joint and drive inventory, signs, standing offsets, position limits, passive joint parameters, drive families, baseline drive gains, ankle linkage geometry. Its `schema` must be `ethrc-actuators/v1`; anything else raises. |
| `URDF_5_description/urdf/URDF_5.urdf` | The URDF that `actuators.yaml` names in its `urdf:` key. Read for the description digest, the IMU mount (`Description.imu_mount`, the URDF's `imu_joint`) and the mirror map. |
| `URDF_5_description/urdf/URDF_5_orca_hands.urdf` | The variant the profile selects (`model.urdf`) and the asset is built from: the same robot with ORCA hands, 58.68 kg against 55.68 kg for `URDF_5.urdf`. Controls builds its MJCF from this variant too. |
| `URDF_5_description/meshes/` | Visual and collision meshes (STL), referenced as `package://URDF_5_description/meshes/...`. |

The two URDFs are not identical apart from the hands: 19 of the 22 joint origins differ, by up to 1.3 mm in position and up to 0.12 rad (7°) in orientation (`shoulder_abduction_l`; `leg_abduction_l` differs by 0.09 rad). `actuators.yaml` records that the standing sole alignment was matched on the ORCA-hands variant and does not carry over to the plain URDF (`standing_reference.alignment_source`).

### Joint coordinates

URDF joint zero is the [calibration pose](glossary.md#calibration-pose), the pose in which every drive reads zero. A one-to-one drive reads `sign · q`, with `q` in URDF radians and `sign` from the joint's `sign` field (`src/humanoid_sim/transmissions/drive_map.py`).

The [standing offset](glossary.md#standing-offset) `standing_offset_rad` is the URDF angle of each joint in the robot's standing pose: `q_calibration = q_standing + standing_offset_rad`. humanoid-sim uses the vector of standing offsets as the standing pose. The mirror map mirrors about it, the ankle gains and ankle effort limits are evaluated there, and the articulation spawns there unless a scene asks for the flat-foot pose (see [Standing on the ground](#standing-on-the-ground)).

A joint's `position_limit` is in standing-relative radians. `Description.calibration_travel(name)` turns it into the simulated range in URDF radians: both bounds plus the standing offset, then replaced on either side by `mechanical_stop_calibration_rad` where a stop is recorded. Only the knees have one, recorded as the hyperextension stop at 0.0 rad. A stop that lies inside the position limit raises, since the limit could never be reached. This is the rule of Controls' `ethrc_ssot/joint_travel.py`.

The knee stop does not fit the simulated leg. The knee's `position_limit` is [-1.867, +0.28] rad standing-relative, but the stop replaces the upper bound, so the simulated knee travel is [-4.174, 0.0] rad (URDF) on the left and [-4.184, 0.0] rad on the right. In the URDF geometry the leg is straight (the angle between the `leg_flexion`, `knee` and `foot_pitch` joint frames, seen along the knee axis, reaches 180°) at about +0.41 rad standing-relative on the left and +0.43 rad on the right, that is URDF -1.90 and -1.88 rad. From there the simulated knee can bend backwards another 1.9 rad, about 110°, before it meets its joint limit. This is a description fact for humanoid-urdf to correct with a stop that matches the leg geometry (`mechanical_stop_calibration_rad` of `knee_l` and `knee_r`). humanoid-sim applies the travel the description gives and adds no stop of its own.

The ankle hinges get the box from their `position_limit`, ±30° about standing in pitch and in roll. The measured ankle range (`ankle_model.measured_range`) and the crank windows under `drives:` are not used as solver limits.

### Base frame and forward direction

The root body `base_link` carries the base frame. With the base upright, z points up and x is the lateral axis: the left foot sits at -x and the right foot at +x, which makes +y the robot's forward direction. Code that assumes x forward gets this robot wrong, so check the forward direction from the sole boxes. Their long axis runs heel to toe and lies along world y when the robot stands with an upright base (seen from above, within 3° to 5° of y in both poses of [Standing on the ground](#standing-on-the-ground)). That axis is the box x axis, the first column of `rotation` in `asset.json` `sole_boxes`. Its sign comes from a fit (it points to -y on the current asset), so it gives the fore-aft line only; which end is the toe follows from the foot positions above.

### Joints, drives and families

The description has 22 joints and 22 EtherCAT drives. 18 joints have a one-to-one drive, named by the joint's `device:` field. The four ankle joints (`foot_pitch_l/r`, `foot_roll_l/r`) have `device: null`: each foot is moved by two HEJ 50 drives stacked on the back of the shin, the `upper` and `lower` *crank*. Each crank drive turns a short arm that pushes a rod down to the foot, so the two together set both pitch and roll. The cranks are listed under `drives:` with both joints they move. The linkage geometry (`ankle_model`) feeds the transmission described in [drives.md](drives.md).

`Description.joints` holds, per joint: `region`, `device`, `sign`, `standing_offset_rad`, `armature` (for the ankle hinges, `armature_joint`, the crank inertia reflected into joint space), passive viscous `damping` (Nm s/rad), Coulomb `friction` (Nm), peak `effort` and `velocity`. `Description.drives` holds every drive: the 18 one-to-one drives in joint order, then the four cranks. Each drive carries its family, peak and nominal effort, no-load velocity, and its baseline position gains from `baseline_position_gains` (the stiffness and damping the drive's own position loop runs with; see [drives.md](drives.md)).

| Family | Drives | Used for |
|---|---|---|
| `HEJ90-48-140` | 10 | hip central rotation, waist yaw, and per leg: flexion, abduction, hip yaw, knee |
| `HEJ70-48-50` | 4 | shoulder flexion and shoulder abduction |
| `HEJ50-48-30` | 8 | shoulder rotation, elbow flexion, and the four ankle cranks |

From `families:` humanoid-sim reads `torque_speed` (the torque-speed envelope, `src/humanoid_sim/devices/envelope.py`), `controller_rate_hz` (velocity quantization of the training plant's drive feedback, `src/humanoid_sim/devices/reporting.py`) and `backdrive_torque` (the friction of a disabled drive in the runtime host). The family `mass` is a reference humanoid-urdf uses for checking CAD inertials; humanoid-sim does not read it.

To print the inventory and simulated travel from a checkout (core dependencies only):

```bash
uv run python - <<'EOF'
from humanoid_sim.model.description import Description

d = Description.load()
print(d.identity())
for name, j in d.joints.items():
    lo, hi = d.calibration_travel(name)
    print(f"{name:22s} {str(j.device):18s} sign {j.sign:+d}  standing {j.standing_offset_rad:+.4f}  "
          f"travel [{lo:+.3f}, {hi:+.3f}]")
EOF
```

### Identity and hash checks

`Description.digest` is a sha256 over the file name and bytes of `URDF_5.urdf` and `actuators.yaml` (`description_digest` in `src/humanoid_sim/runtime/abi/plant_abi.py`). `Description.revision` is the git commit of the description root, or `None` outside a git checkout. `Description.identity()` returns the robot id, digest and revision. It is recorded in every `asset.json`, in the results of `replay`, `fit-delay` and `fit-passive`, and in the runtime segment's header and metadata.

humanoid-sim has no expected digest built in. The digest is compared where two parties must agree on the description:

- It is part of the asset key (see [Output directory](#output-directory-and-assetjson)). Every command that loads the robot (`replay`, `view-replay`, `view-runtime`, `bench`, `fit-delay`, `fit-passive`, `serve`) looks up the asset built from the current description and profile, and stops with `asset ... is not built for this description and profile` when there is none. `--usd PATH` (a global option, before the subcommand) skips that lookup.
- The runtime host writes it into the segment header. A client that passes its own digest to `Segment.open` (`plant_abi.py`) refuses a segment whose digest differs ("the sim and the client read different descriptions"). See [controls-integration.md](controls-integration.md).

The digest covers `URDF_5.urdf` and `actuators.yaml` only. The simulated variant `URDF_5_orca_hands.urdf` is covered by the asset key (through the prepared URDF), and the mesh files are covered by neither: after a change to an STL alone, rebuild with `humanoid-sim build-asset --force`.

### Using a different description root

`Description.default_root()` is `third_party/humanoid-urdf` in this checkout unless the environment variable `HUMANOID_SIM_DESCRIPTION_ROOT` is set. The root is the humanoid-urdf repository itself, the directory that contains `ethrc_v1/`. There is no command-line flag; the CLI, `SimCore` and the Train-Time API all call `Description.load(robot)` and so follow the variable. From Python, `Description.load("ethrc_v1", root=path)` takes a root directly.

```bash
HUMANOID_SIM_DESCRIPTION_ROOT=$HOME/src/humanoid-urdf uv run python -c \
  "from humanoid_sim.model.description import Description; print(Description.load().identity())"
```

The Isaac Lab image's entrypoint (`docker/entrypoint.sh`) sets the variable to `/workspace/humanoid-sim/third_party/humanoid-urdf` when the repository is mounted there and the variable is not already set. To use another checkout inside the container, mount it and pass `-e HUMANOID_SIM_DESCRIPTION_ROOT=...` with its path in the container.

The profile is not affected by the variable: `load_profile` (`src/humanoid_sim/model/asset.py`) always reads `profiles/<robot>.yaml` from this repository.

## The asset build

Isaac Lab spawns robots only from USD (Universal Scene Description, the scene format Isaac Sim and Isaac Lab use), so the simulated robot is a USD file built from the description. Build it in the Isaac Lab image, from the repository root:

```bash
docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev build-asset
docker run --rm -v "$PWD":/workspace/humanoid-sim -w /workspace/humanoid-sim humanoid-sim:dev build-asset --base-joints
```

Each prints the path of the USD. `--force` rebuilds an existing asset in place. `--assets DIR` (a global option, before the subcommand: `humanoid-sim --assets /tmp/assets build-asset`) changes the output root from `outputs/assets`. The URDF conversion needs Isaac Lab; everything before it runs anywhere with the core dependencies, which is how a host checkout computes the asset key and finds an asset the image built.

### The prepared URDF

`PreparedUrdf.build` (`src/humanoid_sim/model/urdf.py`) applies the profile's modeling choices to the selected URDF, in this order. Every number comes from the meshes, the description or the profile.

1. With `model.weld_unlisted_joints: true`, every movable joint without an entry in `actuators.yaml` becomes a fixed joint at its URDF zero. For the ORCA-hands variant those are the 34 hand joints (17 per hand), the same as Controls' `--fixed-orca-hands`. A listed joint missing from the URDF raises.
2. Each of the 22 joints becomes `revolute` (14 of them are `continuous` in the URDF) with the calibration travel as its range, the drive's peak `effort` and its `velocity`.
3. Each foot gets a sole box. On the body carrying each `foot_roll` joint (profile `model.sole.joints`), the collision mesh with the lowest point, including meshes of links welded to that body, is the foot plate. `fit_sole_plate` fits a thin box to the plate's bottom face: it takes the vertices within 1.5 mm of the lowest point, fits the face normal by SVD, refits with the vertices within `face_depth_m` of the face (at most five rounds, until the set stops changing), and sizes the box from the face's extent along its principal in-plane axis and the axis across it. The box is `box_thickness_m` thick, its bottom face on the plate's bottom face, its x axis along the principal in-plane axis (the sole's long axis, heel to toe) and its z axis pointing up into the plate. The plate mesh's collision is removed and the box (`sole_box_l`, `sole_box_r`) is added. On the current description both boxes are about 250 x 90 x 5 mm. The algorithm is Controls' `build_mjcf.py` `sole_plate`, ported unchanged.
4. The physics collides convex shapes, and the URDF converter turns each collision mesh into its convex hull (the smallest convex shape enclosing it). With `model.collision_hulls: per_body`, `merge_body_collisions` first combines every remaining collision mesh of a body and of the links welded to it into one mesh in the body frame, written as `collision/<body>.stl`, so each body gets one hull: 23 hulls plus the two sole boxes. With `per_mesh` each of the 120 remaining meshes becomes its own hull. Against the ground the hull of the union reaches exactly as far as the separate hulls did; between bodies it fills the gaps between parts. Primitive colliders (the sole boxes) are left alone.
5. With `--base-joints` only, three base joints are added above the root; see [Base joints](#base-joints).
6. `package://` mesh URIs are resolved to absolute paths for the converter. A mesh from any other package raises.

A *body* is a link that no fixed joint merges into a parent: the root and the child of every movable joint. The converter merges welded links into their body (`merge_fixed_joints=True`), so the asset has 23 rigid bodies. Their names come from the CAD export and do not reliably say which side they are on; the joint above each body does:

| Body | Joint above it | Self-collision set |
|---|---|---|
| `base_link` (carries the IMU) | none, it is the root | `trunk` |
| `hej90_hip_abduction_1` | `hip_central_rotation` | `trunk` |
| `Component171_1` (torso) | `waist_upperbody_yaw` | `trunk` |
| `leg_flexion_abduction_connector_l_1`, `..._r_1` | `leg_flexion_l`, `leg_flexion_r` | `leg_l`, `leg_r` |
| `hej90_leg_abduction_l_1`, `..._r_1` | `leg_abduction_l`, `leg_abduction_r` | `leg_l`, `leg_r` |
| `new_lower_leg_Mirror__1`, `new_lower_leg_1` | `hip_yaw_l`, `hip_yaw_r` | `leg_l`, `leg_r` |
| `new_upperl_leg_Mirror__1`, `new_upperl_leg_1` | `knee_l`, `knee_r` | `leg_l`, `leg_r` |
| `universal_joint_cross_l_Mirror__1`, `universal_joint_cross_l_1` | `foot_pitch_l`, `foot_pitch_r` | `leg_l`, `leg_r` |
| `bottom_universal_joint_l_Mirror__1`, `bottom_universal_joint_l_1` (feet, with the sole boxes) | `foot_roll_l`, `foot_roll_r` | `leg_l`, `leg_r` |
| `shoulder_connector_l_1`, `..._r_1` | `shoulder_flexion_l`, `shoulder_flexion_r` | `arm_l`, `arm_r` |
| `shoulder_arm_connector_l_1`, `..._r_1` | `shoulder_abduction_l`, `shoulder_abduction_r` | `arm_l`, `arm_r` |
| `hej50_arm_rotation_l_1`, `..._r_1` | `shoulder_rotation_l`, `shoulder_rotation_r` | `arm_l`, `arm_r` |
| `connector_elbow_forearm_l_1`, `..._r_1` (forearm and welded hand) | `elbow_flexion_l`, `elbow_flexion_r` | `arm_l`, `arm_r` |

`PreparedUrdf.body_tree()` returns this table for any build.

### Conversion, self-collision and contact excludes

`build_asset` (`src/humanoid_sim/model/asset.py`) writes the prepared URDF and merged meshes into the asset directory and runs Isaac Lab's `UrdfConverter` with a floating base, fixed joints merged, convex-hull colliders, self-collision on, and no joint drives (the solver gains are set at spawn, see below). It then authors `physics:filteredPairs` on the USD, which keeps the named body pairs from ever colliding:

- the profile's `model.contact_excludes`, named by body. These are the pairs that overlap at rest once each body is a convex hull: torso, base and hip carrier with each other, the hip carrier with each leg's abduction body, each foot with its shank, and each upper-arm rotation body with its shoulder connector. They are the excludes of Controls' `build_mjcf.py`.
- every body pair whose self-collision sets the profile does not pair up (`PreparedUrdf.self_collision_excludes`).

The self-collision sets are keyed by a joint: a set holds every body at or below that joint, and bodies in no set (base, hip carrier, torso) form the `rest` set, named `trunk` in the profile. `pairs` is either `all` or a list of set pairs that collide. With a list, two bodies collide only if their sets are listed as a pair, so bodies in the same set collide only if the list pairs that set with itself (for example `[leg_l, leg_l]`). The current profile uses `all`, so this step authors no excludes and only the nine contact excludes are filtered.

### Base joints

`--base-joints` builds a separate asset with three revolute joints between a new world-fixed root link `base_world` and the robot's root `base_link`: `base_yaw` about z, `base_pitch` about y and `base_roll` about x, in that order (`R = Rz(yaw) Ry(pitch) Rx(roll)`, `BASE_JOINTS` in `src/humanoid_sim/model/urdf.py`). The added links weigh 1 g each and have no colliders. At spawn the root is fixed and the three joints follow position and velocity targets through a stiff PD (`BASE_JOINT_STIFFNESS`, `BASE_JOINT_DAMPING` in `src/humanoid_sim/physics/isaaclab.py`).

This is the asset `replay`, `fit-delay` and `fit-passive` use with `--base yaw` (the default) or `--base orientation`: the robot hangs on a gantry while the base turns as the capture's IMU recorded. `--base fixed` uses the plain asset with its root fixed. See [replay-and-sysid.md](replay-and-sysid.md).

### Output directory and asset.json

The asset lives in `outputs/assets/ethrc_v1/<key>/`. The key (`asset_key`) is the first 16 hex digits of a sha256 over the asset schema string (`ASSET_SCHEMA`), the description digest, the profile's whole `model` block, and the digest of the prepared URDF text. That text is hashed before mesh paths become absolute, so the key is the same on every machine: a host checkout finds the asset that the image built into the mounted repository.

```
outputs/assets/ethrc_v1/<key>/
  robot.urdf             the prepared URDF (absolute mesh paths of the machine that built it)
  collision/<body>.stl   the merged collision mesh of each body (per_body)
  usd/robot/robot.usda   the USD the physics and host kinematics load
  asset.json             provenance and build record
```

`build_asset` returns the existing USD without rebuilding when `asset.json` is present and `--force` is not given. A change to `actuators.yaml`, to `URDF_5.urdf`, to the selected URDF variant or to anything in the profile's `model` block produces a new key, so the old asset stays on disk and commands ask for a build. Changes to the `physics`, `devices` and `sensors` blocks are applied at spawn or run time and need no rebuild. Mesh file contents are not part of the key, nor are build steps that leave the prepared URDF text unchanged (the converter settings, the authoring of contact excludes); after changing either, rebuild with `--force`.

| `asset.json` field | Content |
|---|---|
| `schema`, `key` | `humanoid-sim-asset/v1` and the key above |
| `usd` | path of the USD relative to the asset directory |
| `robot`, `description_digest`, `description_revision` | `Description.identity()` |
| `base_joints` | the base joint names, or empty |
| `welded_joints` | joints welded in step 1 |
| `sole_boxes` | per side: body, the mesh it replaces, center, rotation and half extents in the body frame |
| `merged_collision_bodies` | bodies that got a merged collision mesh |
| `contact_excludes`, `self_collision_excludes` | the filtered pairs as USD prim paths |
| `self_collision_sets` | body to set name |

The training scene and the runtime host read `sole_boxes` from it (through `asset_meta`) to place the robot on the ground and to compute the flat-foot pose. To find the current asset from Python:

```python
import json
from pathlib import Path

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description

desc = Description.load()
out = asset_dir(desc, load_profile("ethrc_v1"), Path("outputs/assets"))  # base_joints=True: the variant
usd = out / json.loads((out / "asset.json").read_text())["usd"]  # asset.json exists once built
```

Code that is handed a USD path directly (`--usd`, `robot_scene_cfg(num_envs, usd)`) does not check it against the description. Joint ranges and collision geometry come from the USD, while the per-joint solver values below are recomputed from the description at spawn, so a stale USD silently mixes two descriptions.

## Per-joint solver properties

The USD carries geometry, inertia and joint ranges. The remaining per-joint values are written to the articulation when it is spawned, through Isaac Lab's `ImplicitActuatorCfg` in `articulation_cfg` (`src/humanoid_sim/physics/isaaclab.py`). `joint_properties` (`src/humanoid_sim/model/joint_properties.py`) computes them from the plant (`PlantCore`, `src/humanoid_sim/core.py`) and the description:

| Property | Value | Source |
|---|---|---|
| `stiffness`, `damping` | The solver's constant PD gains `k0`, `d0`. One-to-one joints: the drive's `baseline_position_gains`. Ankle hinges: `diag(Jᵀ K J)` with the crank Jacobian `J` at the standing pose (`DriveJointMap.joint_space_gains`). | `actuators.yaml` `baseline_position_gains` |
| `armature` | Rotor inertia reflected to the joint, added to the joint's inertia. Ankle hinges use `armature_joint`. | `actuators.yaml` `armature` / `armature_joint` |
| `friction` | Coulomb friction of an enabled drive (Nm). | `actuators.yaml` `friction` |
| `viscous_friction` | Passive viscous damping (Nm s/rad), separate from the PD damping. | `actuators.yaml` `damping` |
| `joint_effort_limit` | One-to-one joints: the drive's peak `effort`. Ankle hinges: the torque both cranks reach at their peak at the standing pose (per hinge, the sum over the two cranks of the Jacobian entry's magnitude times the crank's peak), so the solver's clamp never binds before the drive model's own torque envelope does. | `actuators.yaml` `effort` |
| initial position | The standing pose, or the flat-foot pose when a training scene asks for it (`robot_scene_cfg(..., pose="flat_feet")`). | `standing_offset_rad` |
| range | Calibration travel, built into the USD. | `position_limit`, `mechanical_stop_calibration_rad` |

The solver gains are written at reset only; during a step the plant shapes the position target so that the solver applies the drive model's torque ([drives.md](drives.md)). How stiff the joint stops are is set by the profile's `physics.joint_limits`. The runtime host changes one value while running: when a one-to-one drive is disabled, its joint's Coulomb friction becomes the family's `backdrive_torque` until the drive is enabled again (`RuntimeHost._set_drives_off`, `src/humanoid_sim/runtime/host.py`); the ankle hinges keep theirs. Training can randomize several of these values per environment, see [training.md](training.md).

To print the values as the solver receives them (core dependencies only):

```bash
uv run python - <<'EOF'
from humanoid_sim.train.scene import plant_joint_properties

p = plant_joint_properties()
for row in zip(p.names, p.stiffness, p.damping, p.armature, p.friction, p.passive_damping, p.effort_limit):
    print("{:22s} k0 {:8.1f}  d0 {:5.1f}  armature {:.4f}  friction {:.3f}  damping {:.3f}  effort {:6.1f}".format(*row))
EOF
```

## Host kinematics

`RobotKinematics` (`src/humanoid_sim/model/kinematics.py`) loads the asset USD into a Newton model on the CPU and evaluates forward kinematics: body poses from joint positions, with the root fixed at `base_height_m`. It has no solver and steps nothing. The replay viewer, the flat-foot pose and the model checks use it. It needs `uv sync --extra newton` and a built asset.

```python
import torch

from humanoid_sim.model.kinematics import RobotKinematics

kin = RobotKinematics(usd, list(desc.joints))  # usd and desc as in the asset.json snippet above
q = torch.tensor([[desc.joints[n].standing_offset_rad for n in desc.joints]])
poses = kin.body_poses(q)  # [bodies, 7]: position (m), quaternion xyzw; names in kin.model.body_label
```

## Standing on the ground

The robot can be put on flat ground in two poses. Both have the base upright and the lowest sole-box corner on the ground.

| Pose | Joints | Soles under an upright base | Base height |
|---|---|---|---|
| `standing` | The description's standing offsets | Pitched by about 10° (left) and 17° (right) | 0.945 m |
| `flat_feet` | The flat-foot pose, derived below | Level, both at one height | 0.900 m |

Under an upright base the description's [standing pose](glossary.md#standing-pose) does not put the soles flat on the ground. A robot that holds it rigidly (every drive in JVPT at the [baseline gains](glossary.md#baseline-gains), targets at the standing pose) lands on the sole edges, saturates the ankle cranks at their 30 Nm peak, hops and tips over within about a second. Held the same way, the [flat-foot pose](glossary.md#flat-foot-pose) stands. The physics test `test_a_robot_holding_the_flat_foot_pose_stands` (`tests/test_world.py`) checks that after 1.5 s the base is still upright and above 0.85 m, with the weight on the ground. Holding a pose is not balancing, though: with no randomization on flat ground the robot stays up for about 10 s, then a slow sideways lean builds and it tips over at about 13 s. Anything longer needs a controller.

### The flat-foot pose

`flat_foot_pose(usd, desc, soles)` (`src/humanoid_sim/model/kinematics.py`) derives the pose from the standing pose by forward kinematics on the built asset, with the base upright:

1. For each foot it solves `foot_pitch` and `foot_roll` so that the sole box lies level. On the current description that pitches the left ankle by +0.17 rad and the right by +0.30 rad; the roll changes by under 0.01 rad.
2. The sole that now hangs lower (the left, by 15 mm) is raised to the other's height by bending that leg at `leg_flexion` and `knee`, with its ankle solved again so the sole stays level and keeps its fore-aft position along its long axis.
3. Every other joint keeps its standing offset. A pose outside a joint's travel raises `ValueError`.

The joints that move, on the current description (rad, URDF):

| Joint | Standing | Flat-foot | Change |
|---|---|---|---|
| `leg_flexion_l` | +0.183 | +0.268 | +0.085 |
| `knee_l` | -2.307 | -2.474 | -0.167 (more flexion) |
| `foot_pitch_l` | +0.391 | +0.649 | +0.258 (0.17 to level the sole, the rest for the knee bend) |
| `foot_roll_l` | +0.004 | -0.006 | -0.010 |
| `foot_pitch_r` | +0.260 | +0.559 | +0.299 |
| `foot_roll_r` | +0.008 | +0.010 | +0.002 |

`tests/test_physics.py` `test_flat_foot_pose_levels_both_soles_at_one_height` checks that both soles are level, their bottoms within 0.5 mm of each other, and that only these leg joints move.

The flat-foot pose is a starting pose. The description's standing pose stays the reference for everything else: the standing-relative coordinates Controls works in (`q - standing_offset_rad`), the ankle hinges' solver gains and effort limits (taken through the crank Jacobian at standing), the mirror map's center, and the runtime's start pose for each session.

`standing_base_height(usd, desc, soles, q=None)` returns the height of an upright base at which joint positions `q` (`[J]`, description joint order; the standing pose when omitted) touch the ground with their lowest sole-box corner. Both functions need Newton (`uv sync --extra newton` on the host) and a built asset. `soles` is the `sole_boxes` list of the asset's `asset.json`:

```python
from humanoid_sim.model.asset import asset_meta
from humanoid_sim.model.kinematics import flat_foot_pose, standing_base_height

soles = asset_meta(usd)["sole_boxes"]                 # usd and desc as in the asset.json snippet above
q = flat_foot_pose(usd, desc, soles)                  # [22] rad, description joint order
height = standing_base_height(usd, desc, soles, q)    # 0.900 m; without q, the standing pose: 0.945 m
```

### Selecting a pose

Training: `robot_scene_cfg(num_envs, usd, pose="flat_feet")` (`src/humanoid_sim/train/scene.py`) spawns every env in the flat-foot pose. The pose becomes the articulation's default joint positions (`robot.data.default_joint_pos`), so `reset_scene_to_default` and a task's action offsets use it. `pose="standing"` is the default. The pose is computed when the config is built, which takes several seconds. `humanoid_sim.train.scene.initial_pose(usd, robot, pose)` returns the joint positions and base height of a pose, for a task that writes its own resets. `POSES` lists the names, `("standing", "flat_feet")`; any other name raises `ValueError`.

Runtime: the mailbox command `RESET` with `{"pose": "flat_feet"}` puts the joints in the flat-foot pose, and in a free session the base upright at that pose's height (see [runtime.md](runtime.md#reset-1)). The host computes the pose on the first such reset and keeps it for later ones. That computation takes several seconds (about 10 s on a laptop CPU), during which the host neither steps nor publishes a heartbeat; the period then counts as late. A session still starts in the standing pose.

The training scene places the base 1 mm (`STANDING_CLEARANCE_M`) above the pose's height, and so does the runtime host in a free session. A suspended session welds the base 0.3 m above the standing pose's height, with the feet in the air.

## Mirror map

`mirror_map(desc)` (`src/humanoid_sim/model/mirror.py`) returns a `MirrorMap`: which joint and drive is whose left/right mirror image, and with which sign. Joint positions mirror about the standing pose `c`: `q'[perm[i]] = c[perm[i]] + sign[i] · (q[i] - c[i])`. Velocities, torques and deviations map with the signs alone. Everything is derived from the description:

- Pairs are joints named `*_l` / `*_r`. Drives follow their joints; the ankle cranks pair upper with upper and lower with lower.
- The sagittal plane is the plane between the first pair's joint frames, snapped to the nearest base axis. In the base frame of this URDF that is the plane normal to x (`lateral_axis = 0`; see [Base frame and forward direction](#base-frame-and-forward-direction)).
- A joint's sign is +1 where its partner's axis equals the mirror image of its own axis and -1 where it is the opposite. A rotation axis is a pseudovector, so its mirror image is `-M a` for the reflection `M`. Joints on the plane map to themselves: -1 for a yaw or roll joint, +1 for a pitch joint.
- A one-to-one drive's sign combines the joint sign with both drives' `sign` fields. For the cranks it is the sign that maps the left cranks of an ankle pose onto the right cranks of its mirror image, checked over ±15° about standing.

| Joint (pair) | Joint sign |
|---|---|
| `hip_central_rotation`, `waist_upperbody_yaw` (map to themselves) | -1 |
| `leg_flexion`, `hip_yaw` | -1 |
| `leg_abduction`, `knee`, shoulders, elbows, `foot_pitch`, `foot_roll` | +1 |

With the current description every drive sign is -1.

The robot is not mirror-symmetric. Left and right standing offsets differ (`foot_pitch_l` stands at 0.391 rad, `foot_pitch_r` at 0.260 rad), left and right leg axes differ by a few degrees, and at the standing pose the feet sit several centimetres from each other's mirror image. `MirrorMap.standing_asymmetry_m` is the largest mismatch of a joint frame at the standing pose, 0.058 m on the current description. Because each side mirrors about its own standing pose, the mirror of the standing pose is the standing pose, but a mirrored state is the other side doing the mirrored motion with its own geometry, which is close to the geometric mirror image without being equal to it. The map reads `URDF_5.urdf`, the URDF `actuators.yaml` names, and not the ORCA-hands variant the asset is built from; the signs are the same for both.

```python
import numpy as np

from humanoid_sim.model.description import Description
from humanoid_sim.model.mirror import mirror_map

m = mirror_map(Description.load())
q = np.array(m.center)                       # standing pose, description joint order
q[m.joint_names.index("knee_l")] += 0.1
q_mirror = m.mirror_joints(q)                # knee_r 0.1 rad off standing, everything else at standing
```

`mirror_joints`, `mirror_joint_rates` and `mirror_drives` take numpy arrays or torch tensors with the joint or drive axis last. Nothing inside humanoid-sim calls the map; `tests/test_mirror.py` checks it against forward kinematics.

## The profile

`profiles/ethrc_v1.yaml` holds this repository's modeling choices for the robot. `load_profile(robot)` reads `profiles/<robot>.yaml` and checks that its `robot` key matches. Each entry that mirrors Controls names its source in a comment.

| Key | Read by | Effect |
|---|---|---|
| `schema` | nothing | Format tag, `humanoid-sim-profile/v1`. Not checked. |
| `robot` | `load_profile` | Must equal the robot id (`--robot`, default `ethrc_v1`). |
| `model.urdf` | `PreparedUrdf` | URDF variant to build, relative to `URDF_5_description/`. |
| `model.weld_unlisted_joints` | `PreparedUrdf.weld_unlisted` | Weld movable joints that `actuators.yaml` does not list. Default `false`. |
| `model.sole.joints` | `PreparedUrdf.add_sole_boxes` | Side to joint; the joint's child body gets the sole box. |
| `model.sole.box_thickness_m` | `fit_sole_plate` | Sole box thickness (0.005). |
| `model.sole.face_depth_m` | `fit_sole_plate` | How far along the face normal a vertex may lie and still belong to the plate face (0.001). |
| `model.contact_excludes` | `add_contact_excludes` | Body pairs that never collide. |
| `model.collision_hulls` | `PreparedUrdf.build` | `per_body` or `per_mesh`. Default `per_mesh`; anything else raises. |
| `model.self_collision.sets`, `.rest`, `.pairs` | `PreparedUrdf.body_sets`, `self_collision_excludes` | Self-collision sets and which pairs of them collide. Unknown joints or set names raise. |
| `model.ground_friction` | `SimCore`, `robot_scene_cfg` | Friction of the flat ground (1.0) unless the caller overrides it or passes a terrain. See [world.md](world.md). |
| `devices.encoders.<family>.counts_per_revolution` | `DriveReporting` | Output encoder resolution in counts per revolution: the integer units the drive reports position in. The training plant quantizes measured feedback with it; in the runtime, Controls' fake drive does that. |
| `physics.period_ns` | `SimCore`, `RobotPlant`, `robot_sim_cfg` | Physics step, 4000000 ns (4 ms, 250 Hz). `replay --period-ms` sets it for replays. |
| `physics.solver` | nothing | Records MJWarp, the only solver. |
| `physics.integrator` | `simulation_cfg` | MJWarp integrator, `implicitfast`. `replay --integrator` overrides it. |
| `physics.cone` | `simulation_cfg` | Friction cone model, `pyramidal`. |
| `physics.iterations`, `physics.ls_iterations` | `simulation_cfg` | Solver and line-search iteration limits. |
| `physics.njmax`, `physics.nconmax` | `simulation_cfg` | MJWarp buffers: constraint rows per world (a maximum) and contacts per world (an average; the pool is shared by all worlds). |
| `physics.self_collision` | `SimCore`, `robot_scene_cfg` | Self-collision on the spawned articulation. |
| `physics.joint_limits` | `apply_joint_limits` | `mujoco`: MuJoCo's own joint-limit constraint, as Controls' MJCF has it. `newton`: Newton's force-space default, softer for the legs. |
| `sensors.imu.*` | `ImuDeliveryCfg.from_profile` | IMU sample period, burst delivery timing and noise densities. See [sensors.md](sensors.md). |
| `solver_gains` | nothing | Records that the solver PD is the description's baseline gains. |

Only the `model` block is part of the asset key. The `physics` block is described in more detail in [world.md](world.md) and [testing-and-gpu.md](testing-and-gpu.md).

### Robot facts kept in the profile

The repository rule is that robot facts live in humanoid-urdf. A few currently live in the profile because humanoid-urdf does not carry them yet; each cites its source in Controls:

| Entry | Fact | Source |
|---|---|---|
| `devices.encoders` | Output encoder counts per revolution per drive family (4096 for HEJ 90, 16384 for HEJ 70 and HEJ 50) | Controls `ethercat_device_configurator` device configurations, `Hardware.position_encoder_resolution` |
| `sensors.imu.sample_period_ns`, `burst_period_ns`, `in_burst_spacing_ns` | How IMU samples reach the host | measured on the `exc-*` captures |
| `sensors.imu.gyro_noise_density_rad_s_sqrt_hz`, `accel_noise_density_m_s2_sqrt_hz` | MTi-610 white noise | Controls `ethrc_ssot/imu_sensors.py` |

The rest of `model` (the URDF variant, sole box parameters, contact excludes, ground friction) mirrors how Controls builds its MuJoCo model; those are modeling choices. One robot constant lives in code: `JVPT_PER_SI_GAIN` in `src/humanoid_sim/devices/demand.py`, the factor (1000) between SI gains and the raw gains a drive demand carries in JVPT, the drive mode (-64) that tracks position and velocity targets with the drive's own P and D gains (see [drives.md](drives.md)). humanoid-urdf records it only in a comment.

## Changing a robot fact

A measured value reaches the simulation through humanoid-urdf. Find where the fact lives:

| You measured | Where it lives | What it changes in humanoid-sim |
|---|---|---|
| Joint Coulomb friction or viscous damping | `actuators.yaml` `joints.<joint>.friction` / `damping`, with `damping_friction_source` | Solver friction and passive damping at spawn |
| Back-driving torque of a disabled drive | `actuators.yaml` `families.<family>.backdrive_torque` | Friction of a disabled drive's joint in the runtime host |
| Rotor inertia | `joints.<joint>.armature` (ankles: `armature_joint`) | Solver armature |
| A joint's zero or the standing pose | `joints.<joint>.standing_offset_rad` and `standing_reference`; follow `ethrc_v1/AGENTS.md` in humanoid-urdf (never move URDF joint origins for this) | Standing pose, flat-foot pose, simulated travel, ankle gains and effort limits, mirror center, spawn height |
| Joint travel or a mechanical stop | `joints.<joint>.position_limit` (standing-relative) or `mechanical_stop_calibration_rad` (URDF radians) | Joint range in the USD |
| Drive gains, peak torque, torque-speed curve | `baseline_position_gains`, `effort`, `families.<family>.torque_speed` | Solver PD, effort limits, torque envelope |
| A link's mass, inertia or geometry | The link's `<inertial>`, `<origin>` or meshes in the URDF. humanoid-urdf edits the calibration-coordinate sources in `ethrc_v1/tools/calibration_model/` and publishes them with `tools/sync_calibration_model.py` (see `ethrc_v1/tools/README.md` there). humanoid-sim simulates `URDF_5_orca_hands.urdf`, so the change must reach that file. | Mass, inertia and collision shapes in the USD |
| Encoder resolution, IMU timing or noise | `profiles/ethrc_v1.yaml` in this repository | Training plant feedback, IMU delivery |

Then:

1. Make the change in humanoid-urdf with its `*_source` field and get it merged to humanoid-urdf's master. To try it here first, point `HUMANOID_SIM_DESCRIPTION_ROOT` at your humanoid-urdf checkout, or check out your branch inside `third_party/humanoid-urdf`.
2. Move the submodule pin. Controls pins the same repository, and CI's `controls-pins` job fails unless `third_party/humanoid-urdf` is at the commit Controls pins on `CONTROLS_REF` (default `master`), so Controls and humanoid-sim move together:

   ```bash
   git -C third_party/humanoid-urdf fetch origin
   git -C third_party/humanoid-urdf checkout <commit>
   git add third_party/humanoid-urdf
   ```

3. Rebuild both assets in the Isaac Lab image (`humanoid-sim build-asset` and `humanoid-sim build-asset --base-joints`). Any change to `actuators.yaml` changes the description digest and with it the asset key, even for values such as friction that are applied at spawn, so commands refuse to run until the new asset exists. A change to a mesh file alone does not change the key; use `--force`.
4. Re-run the checks: `uv run pytest -m "not physics"` on the host (model build, mirror map, ankle transmission, device layer) and the physics tests (`pytest -m physics`) in the image, as [testing-and-gpu.md](testing-and-gpu.md) describes. The physics tests skip when the asset for the current description is not built.
5. Check the effect against the robot: replay an excitation capture (`humanoid-sim replay <exc-dir>`) and compare the position error before and after, or refit the passive parameters with `humanoid-sim fit-passive`. A fit reports values against the description's and writes nothing back; fitted values go into `actuators.yaml` by hand, with their source. See [replay-and-sysid.md](replay-and-sysid.md).
6. Restart `humanoid-sim serve`. Controls must read the same description, or attaching fails on the digest check.
