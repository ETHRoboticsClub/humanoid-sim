from types import SimpleNamespace

import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.description import Description
from humanoid_sim.physics.joint_properties import joint_dynamics
from humanoid_sim.train.plant import RobotPlant
from humanoid_sim.transmissions.drive_map import native_joint_names


def native_plant():
    desc = Description.load("ethrc_v1")
    core = PlantCore(desc, list(desc.joints), 2, PlantCfg(), solver_joint_names=native_joint_names(desc))
    return core




def test_training_randomization_writes_native_ids_and_projects_relief():
    core = native_plant()
    plant = RobotPlant.__new__(RobotPlant)
    plant.core = core
    plant.device, plant.num_envs = torch.device("cpu"), 2
    plant.drives = core.map.drive_names
    plant.command_latency_ns = torch.zeros(len(plant.drives), dtype=torch.int64)
    plant.imu_latency_ns = 0
    plant.desc = core.desc
    plant.solver_joint_ids = torch.arange(len(core.solver_map.joint_names)) + 3
    plant._solver, plant.imu = "kamino", None
    armature = torch.tensor([joint_dynamics(core.desc)["armature"][core.solver_map.joint_names.index(name)] for name in core.solver_map.joint_names]).expand(2, -1)
    friction = torch.ones_like(armature)
    plant._nominal_physics = {"joint_armature": armature, "joint_friction": friction}
    plant._ratio = plant._device_nominal()["joint_friction_ratio"]
    values = {"joint_armature": armature * 2, "joint_friction": friction * 2,
              "joint_friction_ratio": torch.full_like(friction, 0.25)}
    plant.randomizer = SimpleNamespace(values=values)
    written = {}

    def record(**kwargs):
        assert torch.equal(kwargs.pop("joint_ids"), plant.solver_joint_ids)
        written.update(kwargs)

    plant.robot = SimpleNamespace(write_joint_armature_to_sim_index=record,
                                  write_joint_friction_coefficient_to_sim_index=record)
    plant._write_physics()
    plant._write_device(torch.arange(2))
    assert torch.equal(written["armature"], armature * 2)
    assert torch.all(core.friction_relief == 1.5)
    values["joint_armature"] = torch.ones_like(armature)
    with pytest.raises(ValueError, match="zero/nonzero"):
        plant._write_physics()
