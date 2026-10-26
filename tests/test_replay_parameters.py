"""Replay parameter writes on an explicitly recording backend double; no physics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.replay.parameters import apply_parameters


class RecordingRobot:
    device = "cpu"
    joint_names = ["b", "a"]

    def __init__(self):
        self.written = {}

    def write_joint_armature_to_sim_index(self, armature, joint_ids):
        self.written["armature"] = (armature.clone(), joint_ids.clone())

    def write_joint_viscous_friction_coefficient_to_sim_index(self, joint_viscous_friction_coeff, joint_ids):
        self.written["damping"] = (joint_viscous_friction_coeff.clone(), joint_ids.clone())

    def write_joint_friction_coefficient_to_sim_index(self, joint_friction_coeff, joint_ids):
        self.written["friction"] = (joint_friction_coeff.clone(), joint_ids.clone())


@pytest.fixture
def sim():
    plant = SimpleNamespace(
        num_envs=4,
        map=SimpleNamespace(joint_names=["a", "b"], drive_names=["da", "db"]),
        cfg=SimpleNamespace(dtype=torch.float64, device="cpu"),
        friction_relief=torch.zeros(4, 2, dtype=torch.float64),
        bank=SimpleNamespace(position_offset=torch.zeros(4, 2, dtype=torch.float64)),
    )
    return SimpleNamespace(plant=plant, backend=SimpleNamespace(robot=RecordingRobot()))


def test_named_subset_and_per_environment_values(sim):
    values = np.arange(4, dtype=float)[:, None] + 1
    apply_parameters(
        sim,
        ["a"],
        friction=values,
        friction_ratio=np.full((4, 1), 0.25),
        drive_names=["db"],
        position_offset=-values,
    )
    written, ids = sim.backend.robot.written["friction"]
    assert ids.tolist() == [1]
    np.testing.assert_array_equal(written.numpy(), values)
    np.testing.assert_array_equal(sim.plant.friction_relief[:, 0], values[:, 0] * 0.75)
    assert not sim.plant.friction_relief[:, 1].any()
    np.testing.assert_array_equal(sim.plant.bank.position_offset[:, 1], -values[:, 0])
    assert not sim.plant.bank.position_offset[:, 0].any()
    assert "damping" not in sim.backend.robot.written


@pytest.mark.parametrize(
    "bad",
    [
        dict(damping=np.ones((3, 2))),
        dict(damping=np.full((4, 2), np.nan)),
        dict(armature=-np.ones((4, 2))),
        dict(friction_ratio=np.ones((4, 2))),
        dict(friction=np.ones((4, 2)), friction_ratio=np.full((4, 2), 1.1)),
        dict(position_offset=np.full((4, 2), np.inf)),
    ],
)
def test_invalid_candidate_is_rejected_before_any_write(sim, bad):
    with pytest.raises(ValueError):
        apply_parameters(sim, ["a", "b"], **(dict(damping=np.ones((4, 2))) | bad))
    assert sim.backend.robot.written == {}
    assert not sim.plant.bank.position_offset.any()


def test_replacing_full_candidate_does_not_leave_previous_offsets_or_relief(sim):
    for ratio, offset in [(0.0, 0.1), (1.0, 0.0)]:
        apply_parameters(
            sim,
            ["a", "b"],
            friction=np.ones((4, 2)),
            friction_ratio=np.full((4, 2), ratio),
            position_offset=np.full((4, 2), offset),
        )
    assert not sim.plant.friction_relief.any()
    assert not sim.plant.bank.position_offset.any()


def test_unknown_or_duplicate_names_do_not_write(sim):
    for names in [["a", "a"], ["missing"]]:
        with pytest.raises(ValueError):
            apply_parameters(sim, names, damping=np.ones((4, len(names))))
    assert sim.backend.robot.written == {}
