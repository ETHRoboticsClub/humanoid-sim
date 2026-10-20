"""Replay's start pose from measured drive positions, at the plant's default precision."""

import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.description import Description
from humanoid_sim.replay.replay import initial_joint_state

# A right-ankle crank pair (upper, lower), relative to the linkage neutral: inside both drives' travel
# (actuators.yaml R_ANK1 [-0.72, 0.73], R_ANK2 [-0.78, 0.72]) with the foot tilted 25.7 deg, inside the
# 30 deg cone. Found by the ankle comparison harness: float32 Newton settles 1.2e-6 rad from these cranks.
CRANKS_R = (-0.6184375, 0.0825)
# One count of the drives' 17-bit encoders is 2 pi / 2**17 = 4.8e-5 rad; a start pose that reproduces
# the readings to a fifth of that is exact for replay.
START_TOLERANCE_RAD = 2 * torch.pi / 2**17 / 5


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def test_start_pose_inside_the_cone_at_default_precision(desc):
    joints = list(desc.joints)
    plant = PlantCore(desc, joints, 1, PlantCfg())  # float32, as SimCore builds it
    standing = torch.tensor([[desc.joints[n].standing_offset_rad for n in joints]], dtype=plant.cfg.dtype)
    native, _, _ = plant.map.drive_state(standing, torch.zeros_like(standing))
    _, cranks, joint_ids, _ = next(a for a in plant.map.ankles if a[0] == "r")
    native[0, cranks] = torch.tensor(CRANKS_R, dtype=plant.cfg.dtype)

    q = initial_joint_state(plant, native[0].numpy())

    exact = desc.ankle_transmission("r", dtype=torch.float64)
    back = exact.cranks(q[0, joint_ids].to(torch.float64))
    assert torch.allclose(back, torch.tensor(CRANKS_R, dtype=torch.float64), rtol=0, atol=START_TOLERANCE_RAD)
