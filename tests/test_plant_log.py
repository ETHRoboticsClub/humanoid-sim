"""RobotPlant.log: signals at plant time, per env, through the attached signal log."""

from types import SimpleNamespace

import pytest
import torch

from humanoid_sim.train.plant import RobotPlant


class _Log:
    def __init__(self):
        self.calls = []

    def at(self, t_ns):
        self.t = t_ns

    def log(self, name, value, *, group, envs):
        self.calls.append((self.t, name, value.tolist(), group, tuple(envs)))


def test_log_writes_at_plant_time_through_the_attached_log():
    plant = SimpleNamespace(t_ns=8_000_000)
    with pytest.raises(RuntimeError, match="no signal log"):
        RobotPlant.log(plant, "x", torch.zeros(2))
    log = _Log()
    RobotPlant.attach_log(plant, log)
    RobotPlant.log(plant, "reward", torch.tensor([1.0, 2.0]), envs=(1,), group="mdp")
    assert log.calls == [(8_000_000, "reward", [1.0, 2.0], "mdp", (1,))]
