"""Golden tests: the torch ankle transmission against both numpy references.

* ``ankle_kinematics`` @ Controls' pin: ``joint_to_motor`` and ``joint_to_motor_jacobian`` (kinematic frame,
  geometry in ``geometry.py``).
* humanoid-urdf ``ankle_model``: the description geometry and frame signs this package reads.
"""

import sys

import numpy as np
import pytest
import torch

from humanoid_sim.model.description import Description
from humanoid_sim.transmissions.ankle import AnkleTransmission

sys.path.insert(0, str(Description.default_root().parent / "ankle_kinematics"))
ak = pytest.importorskip("ankle_kinematics")

RNG = np.random.default_rng(0)
# Inside the measured range and the cone: |pitch|, |roll| <= 0.35 around the standing pose.
SAMPLES = RNG.uniform(-0.35, 0.35, size=(256, 2)) + np.array([0.35, 0.0])


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def _ak_leg(side):
    return ak.LEFT_LEG if side == "l" else ak.RIGHT_LEG


@pytest.mark.parametrize("side", ["l", "r"])
def test_cranks_match_ankle_kinematics(desc, side):
    tr = desc.ankle_transmission(side, dtype=torch.float64)
    q = torch.as_tensor(SAMPLES)
    got = tr.cranks(q).numpy()
    sign = np.asarray(desc.ankle_kinematic_sign(side))
    for (p, r), g in zip(SAMPLES, got, strict=True):
        ref = ak.joint_to_motor(_ak_leg(side), p * sign[0], r * sign[1])
        # ankle_kinematics' theta is absolute; ours is relative to the neutral solution (|theta0| ~ 1e-5).
        assert np.allclose(g, ref, atol=1e-4), (side, p, r, g, ref)


@pytest.mark.parametrize("side", ["l", "r"])
def test_jacobian_matches_reference_and_finite_difference(desc, side):
    tr = desc.ankle_transmission(side, dtype=torch.float64)
    q = torch.as_tensor(SAMPLES[:32])
    jac = tr.jacobian(q).numpy()
    sign = np.asarray(desc.ankle_kinematic_sign(side))
    for qi, ji in zip(SAMPLES[:32], jac, strict=True):
        qk = qi * sign
        motors = ak.joint_to_motor(_ak_leg(side), *qk)
        ref = ak.mapping.joint_to_motor_jacobian(_ak_leg(side), *qk, motors) * sign
        assert np.allclose(ji, ref, rtol=1e-6, atol=1e-8)
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
    ref, _ = module._motor_angles_numpy(SAMPLES * sign, model.geometries[side])
    assert np.allclose(tr.cranks(torch.as_tensor(SAMPLES)).numpy(), ref, atol=1e-12)


def test_float32_batched_and_power_consistent(desc):
    tr = desc.ankle_transmission("l", dtype=torch.float32)
    q = torch.as_tensor(SAMPLES, dtype=torch.float32).reshape(16, 16, 2)
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
    q = torch.as_tensor(SAMPLES)
    back = tr.joints_from_cranks(tr.cranks(q), torch.tensor([0.35, 0.0], dtype=torch.float64))
    assert torch.allclose(back, q, atol=1e-10)
