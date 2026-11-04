"""RobotPlant.feedback_history: the history ``measured`` exposes, read without a new sample."""

from types import SimpleNamespace

import torch

from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.train.plant import RobotPlant


def _plant(history_ns):
    z = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
    plant = SimpleNamespace(_hist=DriveFeedback(z, z + 10, z + 20, z + 30), _hist_ns=history_ns, imu=None,
                            samples=0)

    def sample():
        plant.samples += 1
        return DriveFeedback(*(torch.zeros(2, 1) for _ in range(4)))

    plant.solver_state = lambda: (None, None)
    plant.core = SimpleNamespace(feedback=lambda q, qd: None)
    plant._sample = lambda fb: sample()
    plant.feedback_history = lambda: RobotPlant.feedback_history(plant)
    return plant


def test_the_history_and_its_times_come_back_without_a_sample():
    ns = torch.tensor([[3, 2, 1], [3, -1, -1]])
    plant = _plant(ns)
    history, history_ns = RobotPlant.feedback_history(plant)
    assert plant.samples == 0
    assert torch.equal(history.position_rad, plant._hist.position_rad)
    assert history_ns is ns


def test_measured_exposes_the_same_history_with_one_sample():
    plant = _plant(torch.tensor([[3, 2, 1], [3, -1, -1]]))
    measured = RobotPlant.measured(plant)
    history, history_ns = RobotPlant.feedback_history(plant)
    assert plant.samples == 1
    assert torch.equal(measured.history.position_rad, history.position_rad)
    assert measured.history_ns is history_ns
    assert measured.history is not plant._hist  # a copy of the container: the plant's own is not handed out
