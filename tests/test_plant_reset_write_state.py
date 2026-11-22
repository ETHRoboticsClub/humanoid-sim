"""RobotPlant.reset(write_state=False): everything a reset does but the joint state write, which the caller makes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from humanoid_sim.train.plant import RobotPlant


def _plant():
    plant = SimpleNamespace(robot=Mock(), core=Mock(), randomizer=Mock(), sampler=Mock(), imu=Mock(),
                            native_state=object(), t_ns=7, _hist_ns=torch.zeros(3, 2, dtype=torch.int64),
                            _targets=torch.zeros(3, 4), joint_ids=torch.tensor([0, 2]), _write_device=Mock(),
                            write_joint_state=Mock())
    plant.joint_state = lambda: (torch.ones(3, 2), torch.zeros(3, 2))
    return plant


@pytest.mark.parametrize("write_state", (True, False))
def test_a_reset_writes_the_native_state_unless_the_caller_does(write_state):
    plant, ids = _plant(), torch.tensor([0, 2])
    RobotPlant.reset(plant, ids, write_state=write_state)
    assert plant.write_joint_state.call_count == int(write_state)
    # the rest of the reset is the same either way
    plant.core.reset.assert_called_once_with(ids, None)
    plant.randomizer.draw_reset.assert_called_once_with(ids)
    plant._write_device.assert_called_once_with(ids)
    plant.sampler.reset.assert_called_once_with(ids)
    plant.imu.reset.assert_called_once_with(ids, 7)
    assert (plant._hist_ns[ids] == -1).all() and (plant._hist_ns[1] == 0).all()
    assert (plant._targets[ids][:, [0, 2]] == 1).all() and (plant._targets[1] == 0).all()  # held at the pose


def test_a_reset_that_leaves_the_state_takes_none():
    with pytest.raises(ValueError, match="write_state=False"):
        RobotPlant.reset(_plant(), torch.tensor([1]), joint_pos=torch.zeros(1, 2), write_state=False)
