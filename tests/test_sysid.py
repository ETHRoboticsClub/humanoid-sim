"""Passive-parameter fit: search space and CMA-ES."""

import math

import numpy as np
import pytest

from humanoid_sim.sysid.passive import Cmaes, FitSpace, Group


def test_space_names_and_bounds():
    s = FitSpace([Group("waist", ("waist_upperbody_yaw",)), Group("elbow", ("elbow_flexion_l", "elbow_flexion_r"))])
    assert s.dim == 6
    assert s.names[:3] == ["waist.damping", "waist.friction", "waist.armature"]
    m = s.multipliers(np.array([0.0, 5.0, -5.0, math.log(2.0), 0.0, 0.0]))
    assert np.allclose(m[:4], [1.0, 10.0, 0.1, 2.0])


def test_cmaes_minimizes_a_shifted_ellipsoid():
    target = np.array([0.5, -1.0, 0.2, 1.5])
    scale = np.array([1.0, 3.0, 10.0, 0.5])
    es = Cmaes(np.zeros(4), 0.5, 32, seed=1)
    for _ in range(80):
        x = es.ask()
        es.tell(x, (((x - target) * scale) ** 2).sum(axis=1))
    assert np.allclose(es.mean, target, atol=1e-3)


def test_bound_penalty_keeps_cmaes_near_a_boundary_optimum():
    from humanoid_sim.sysid.passive import BOUND_PENALTY

    s = FitSpace([Group("g", ("j",))], params=("damping",), log_bound=1.0)
    es = Cmaes(np.zeros(1), 0.5, 16, seed=0)
    for _ in range(60):
        x = es.ask()
        f = (np.clip(x[:, 0], -1, 1) - 3.0) ** 2  # optimum beyond the bound: flat outside without the penalty
        es.tell(x, f + BOUND_PENALTY * 1e3 * s.out_of_bounds(x))
    assert abs(es.mean[0] - 1.0) < 0.3 and es.sigma < 1.0


class _Robot:
    """Records what the fit writes to the solver."""

    def __init__(self, joints):
        self.joint_names, self.device, self.written = list(joints), "cpu", {}

    def write_joint_viscous_friction_coefficient_to_sim_index(self, joint_viscous_friction_coeff, joint_ids):
        self.written["damping"] = joint_viscous_friction_coeff

    def write_joint_friction_coefficient_to_sim_index(self, joint_friction_coeff, joint_ids):
        self.written["friction"] = joint_friction_coeff

    def write_joint_armature_to_sim_index(self, armature, joint_ids):
        self.written["armature"] = armature


def test_fit_writes_start_values_times_multipliers_and_leaves_the_rest():
    from humanoid_sim.model.description import Description
    from humanoid_sim.sysid.passive import _write

    desc = Description.load()
    elbow = ("elbow_flexion_l", "elbow_flexion_r")
    space = FitSpace([Group("elbow", elbow)], params=("damping",), start={"elbow.armature": 0.0055})
    robot = _Robot(desc.joints)
    _write(robot, desc, space, np.array([[0.0], [math.log(2.0)]]))
    d0 = [desc.joints[j].damping for j in elbow]
    assert np.allclose(robot.written["damping"].numpy(), [d0, [2 * d for d in d0]])
    assert np.allclose(robot.written["armature"].numpy(), 0.0055)  # start value, not fitted
    assert "friction" not in robot.written  # neither fitted nor moved: the solver keeps the description's


def test_fit_space_refuses_unknown_names():
    import pytest

    with pytest.raises(ValueError, match="unknown parameters"):
        FitSpace([Group("g", ("j",))], params=("stiffness",))
    with pytest.raises(ValueError, match="unknown group"):
        FitSpace([Group("g", ("j",))], start={"h.armature": 1.0})


def test_cmaes_scale_sets_the_first_steps_per_coordinate():
    es = Cmaes(np.zeros(2), 0.5, 4000, seed=0, scale=np.array([1.0, 0.1]))
    xs = es.ask()
    assert xs[:, 0].std() == pytest.approx(0.5, rel=0.05) and xs[:, 1].std() == pytest.approx(0.05, rel=0.05)


def test_cmaes_never_selects_a_diverged_candidate():
    """With most of a population non-finite, the mean moves only toward the finite ones."""
    es = Cmaes(np.zeros(2), 0.5, 16, seed=0)
    xs = es.ask()
    f = np.full(16, np.nan)
    f[:3] = [1.0, 2.0, 3.0]
    es.tell(xs, f)
    lo, hi = xs[:3].min(axis=0), xs[:3].max(axis=0)
    assert ((lo - 1e-12 <= es.mean) & (es.mean <= hi + 1e-12)).all()
    sigma = es.sigma
    es.ask()
    es.tell(es.ask(), np.full(16, np.nan))
    assert es.sigma == sigma / 2
