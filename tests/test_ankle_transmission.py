"""Golden tests: the torch ankle transmission against humanoid-urdf ``ankle_model``, the description geometry
and frame signs this package reads."""

import numpy as np
import pytest
import torch

from humanoid_sim.model.description import Description
from humanoid_sim.transmissions.ankle import AnkleTransmission

RNG = np.random.default_rng(0)
# Offsets from the linkage neutral, |pitch|, |roll| <= 0.35 around the standing pose (0.33 rad of pitch from it).
OFFSETS = RNG.uniform(-0.35, 0.35, size=(256, 2)) + np.array([0.35, 0.0])


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


@pytest.mark.parametrize("side", ["l", "r"])
def test_jacobian_matches_finite_difference(desc, side):
    tr = desc.ankle_transmission(side, dtype=torch.float64)
    q = torch.as_tensor(OFFSETS[:32] + desc.ankle_neutral(side))
    eps = 1e-6
    fd = torch.stack([(tr.cranks(q + eps * e) - tr.cranks(q - eps * e)) / (2 * eps)
                      for e in torch.eye(2, dtype=torch.float64)], dim=-1)
    assert torch.allclose(tr.jacobian(q), fd, atol=1e-7)


@pytest.mark.parametrize("side", ["l", "r"])
def test_cranks_match_humanoid_urdf_ankle_model(desc, side):
    module = desc.ankle_model_module()
    model = module.load(desc.actuators_path)
    tr = desc.ankle_transmission(side, dtype=torch.float64)
    sign = np.asarray(desc.ankle_kinematic_sign(side))
    ref, _ = module._motor_angles_numpy(OFFSETS * sign, model.geometries[side])
    assert np.allclose(tr.cranks(torch.as_tensor(OFFSETS + model.neutral(side))).numpy(), ref, atol=1e-12)
    # Both cranks read zero at the description's linkage neutral.
    assert np.allclose(tr.cranks(torch.as_tensor(model.neutral(side))).numpy(), 0.0, atol=1e-12)


def test_float32_batched_and_power_consistent(desc):
    tr = desc.ankle_transmission("l", dtype=torch.float32)
    q = torch.as_tensor(OFFSETS + desc.ankle_neutral("l"), dtype=torch.float32).reshape(16, 16, 2)
    qd = torch.randn_like(q)
    theta, omega, jac = tr.crank_state(q, qd)
    assert theta.shape == (16, 16, 2) and jac.shape == (16, 16, 2, 2)
    tau_m = torch.randn_like(q)
    # Power through the transmission is conserved: tau_m . omega == (J^T tau_m) . qd
    tau_q = AnkleTransmission.joint_torque(jac, tau_m)
    assert torch.allclose((tau_m * omega).sum(-1), (tau_q * qd).sum(-1), atol=1e-4)


@pytest.mark.parametrize("side", ["l", "r"])
def test_joints_from_cranks_inverts(desc, side):
    tr = desc.ankle_transmission(side, dtype=torch.float64)
    q = torch.as_tensor(OFFSETS + desc.ankle_neutral(side))
    back = tr.joints_from_cranks(tr.cranks(q), torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(back, q, atol=1e-10)
