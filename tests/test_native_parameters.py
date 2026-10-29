from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.description import Description
from humanoid_sim.physics.native import NativeDrivePlant
from humanoid_sim.sysid.pace import PaceSpace, blocks, start_values, structural_fixed, write
from humanoid_sim.train.plant import RobotPlant


def native_plant():
    desc = Description.load("ethrc_v1")
    core = PlantCore(desc, list(desc.joints), 2, PlantCfg())
    return core, NativeDrivePlant(core)


def test_native_sysid_writes_physical_inventory_and_preserves_zero_inertia():
    core, native = native_plant()
    names = list(reversed(native.names))
    written = {}

    def record(**kwargs):
        ids = kwargs.pop("joint_ids")
        for key, value in kwargs.items():
            written[key] = dict(zip([names[index] for index in ids.tolist()], value[0].tolist(), strict=True))

    robot = SimpleNamespace(device="cpu", joint_names=names,
                            write_joint_armature_to_sim_index=record,
                            write_joint_viscous_friction_coefficient_to_sim_index=record,
                            write_joint_friction_coefficient_to_sim_index=record)
    start = start_values(core.desc, native.names)
    space = PaceSpace(native.names, core.map.drive_names, start, fixed=structural_fixed(start))
    values = space.decode(np.ones((2, space.dim)))
    write(SimpleNamespace(plant=core, native=native, backend=SimpleNamespace(robot=robot)), space, values)
    for linkage in core.desc.linkages:
        for name in linkage.joints:
            assert written["armature"][name] == 0
        spec = core.desc.ankle_model.payload["sides"][linkage.side]
        for key in ("upper", "lower"):
            name = spec["usd_joints"][key]
            assert written["armature"][name] == pytest.approx(core.desc.physics_joints[name].armature * 10)
    assert torch.equal(core.friction_relief, native.core.friction_relief[:, native.semantic_ids])
    partitions = blocks(space, core, list(range(len(space.drives))))
    covered = np.concatenate([indices for indices, _ in partitions])
    free = np.ones(space.dim, dtype=bool)
    for name, mask in space.fixed.items():
        free[space.slices()[name]] &= ~mask
    assert sorted(covered.tolist()) == np.flatnonzero(free).tolist()


def test_training_randomization_writes_native_ids_and_projects_relief():
    core, native = native_plant()
    plant = RobotPlant.__new__(RobotPlant)
    plant.core, plant.native = core, native
    plant.device, plant.num_envs = torch.device("cpu"), 2
    plant.physics_names, plant.drives = native.names, core.map.drive_names
    plant.desc = core.desc
    plant.physics_ids = torch.arange(len(native.names)) + 3
    plant._solver, plant.imu = "kamino", None
    armature = torch.tensor([core.desc.physics_joints[name].armature for name in native.names]).expand(2, -1)
    friction = torch.ones_like(armature)
    plant._nominal_physics = {"joint_armature": armature, "joint_friction": friction}
    plant._ratio = plant._device_nominal()["joint_friction_ratio"]
    values = {"joint_armature": armature * 2, "joint_friction": friction * 2,
              "joint_friction_ratio": torch.full_like(friction, 0.25)}
    plant.randomizer = SimpleNamespace(values=values)
    written = {}

    def record(**kwargs):
        assert torch.equal(kwargs.pop("joint_ids"), plant.physics_ids)
        written.update(kwargs)

    plant.robot = SimpleNamespace(write_joint_armature_to_sim_index=record,
                                  write_joint_friction_coefficient_to_sim_index=record)
    plant._write_physics()
    plant._write_device(torch.arange(2))
    assert torch.equal(written["armature"], armature * 2)
    assert torch.all(native.core.friction_relief == 1.5)
    assert torch.all(core.friction_relief == 1.5)
    values["joint_armature"] = torch.ones_like(armature)
    with pytest.raises(ValueError, match="zero/nonzero"):
        plant._write_physics()
