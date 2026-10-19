"""Solver selection from the profile's physics block (no Isaac Lab needed)."""

import pytest

from humanoid_sim.model.asset import load_profile
from humanoid_sim.physics.isaaclab import SOLVERS, solver_settings


def test_profile_selects_its_solver_and_has_a_block_per_solver():
    physics = load_profile("ethrc_v1")["physics"]
    assert solver_settings(physics)[0] == physics["solver"]
    for name in SOLVERS:
        solver, settings = solver_settings(physics, name)
        assert solver == name and "integrator" in settings and "iterations" in settings


def test_unknown_or_unconfigured_solver_raises():
    physics = load_profile("ethrc_v1")["physics"]
    with pytest.raises(ValueError, match="physics solver"):
        solver_settings(physics, "physx")
    with pytest.raises(ValueError, match="no 'kamino' block"):
        solver_settings({k: v for k, v in physics.items() if k != "kamino"}, "kamino")
