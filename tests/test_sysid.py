"""Passive-parameter fit: search space and CMA-ES."""

import math

import numpy as np

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
