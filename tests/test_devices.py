import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.bank import DriveBank
from humanoid_sim.devices.demand import LAW_FREEZE, DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.devices.shaping import shaped_position_target, solver_force
from humanoid_sim.model.description import Description
from humanoid_sim.timing import NS_PER_MS

D64 = torch.float64
STEP = 4 * NS_PER_MS
HEJ90 = [(0.0, 140.0), (5.0, 140.0), (6.0, 135.0), (8.0, 100.0), (10.0, 25.0), (10.4, 0.0)]


def _bank():
    return DriveBank(TorqueSpeedEnvelope([HEJ90], dtype=D64), 1, 1, dtype=D64)


def _pd(pos, kp=320.0, kd=12.0, **kw):
    return DriveDemand.pd(torch.tensor([[pos]], dtype=D64), kp, kd, **kw)


def test_pd_law_matches_controls_plant():
    bank = _bank()
    bank.submit(_pd(0.1, feedforward=torch.tensor([[2.0]], dtype=D64)), torch.tensor([0]))
    out = bank.compute(0, STEP, torch.tensor([[0.05]], dtype=D64), torch.tensor([[0.5]], dtype=D64))
    expect = 320.0 * 0.05 + 12.0 * (0 - 0.5) + 2.0
    assert out.demanded.item() == pytest.approx(expect)
    assert out.applied.item() == pytest.approx(expect)


def test_envelope_and_cap_clip_applied_not_demanded():
    bank = _bank()
    bank.submit(_pd(2.0, torque_limit_nm=torch.tensor([[100.0]], dtype=D64)), torch.tensor([0]))
    out = bank.compute(0, STEP, torch.zeros(1, 1, dtype=D64), torch.tensor([[7.0]], dtype=D64))
    assert out.demanded.item() == pytest.approx(320 * 2.0 - 12 * 7.0)
    assert out.applied.item() == pytest.approx(100.0)  # cap 100 < motoring limit 117.5 at 7 rad/s


def test_freeze_latches_on_entry_and_off_is_zero():
    bank = _bank()
    d = _pd(0.0)
    d.law[:] = LAW_FREEZE
    bank.submit(d, torch.tensor([0]))
    bank.compute(0, STEP, torch.tensor([[0.3]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    out = bank.compute(STEP, 2 * STEP, torch.tensor([[0.2]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    assert out.demanded.item() == pytest.approx(320 * (0.3 - 0.2))
    bank.submit(DriveDemand.disabled(1, 1, dtype=D64), torch.tensor([2 * STEP]))
    out = bank.compute(2 * STEP, 3 * STEP, torch.zeros(1, 1, dtype=D64), torch.ones(1, 1, dtype=D64))
    assert out.applied.item() == 0 and out.enabled_fraction.item() == 0


def test_freeze_latches_when_it_arrives_not_when_queued():
    bank = _bank()
    bank.submit(_pd(0.0), torch.tensor([0]))
    bank.compute(0, STEP, torch.tensor([[0.5]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    d = _pd(0.0)
    d.law[:] = LAW_FREEZE
    bank.submit(d, torch.tensor([3 * STEP]))  # queued now, in force from the start of the fourth step
    bank.compute(STEP, 2 * STEP, torch.tensor([[0.4]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    bank.compute(2 * STEP, 3 * STEP, torch.tensor([[0.3]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    bank.compute(3 * STEP, 4 * STEP, torch.tensor([[0.2]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    out = bank.compute(4 * STEP, 5 * STEP, torch.tensor([[0.1]], dtype=D64), torch.zeros(1, 1, dtype=D64))
    assert out.demanded.item() == pytest.approx(320 * (0.2 - 0.1))  # held where it was when freeze began


def test_a_disabled_drive_applies_no_torque():
    bank = _bank()
    bank.submit(DriveDemand.disabled(1, 1, dtype=D64), torch.tensor([0]))
    out = bank.compute(0, STEP, torch.tensor([[0.3]], dtype=D64), torch.tensor([[1.0]], dtype=D64))
    assert out.applied.item() == 0 and out.demanded.item() == 0 and out.enabled_fraction.item() == 0


def test_arrival_blends_forces():
    bank = _bank()
    bank.submit(_pd(0.0), torch.tensor([0]))
    bank.compute(0, STEP, torch.zeros(1, 1, dtype=D64), torch.zeros(1, 1, dtype=D64))
    bank.submit(_pd(0.1), torch.tensor([STEP + 1 * NS_PER_MS]))  # lands 1 ms into the next step
    out = bank.compute(STEP, 2 * STEP, torch.zeros(1, 1, dtype=D64), torch.zeros(1, 1, dtype=D64))
    assert out.demanded.item() == pytest.approx(0.75 * 320 * 0.1)


def test_shaping_is_identity_at_solver_gains():
    q, qd, k0, d0 = torch.tensor(0.2), torch.tensor(-1.0), torch.tensor(320.0), torch.tensor(12.0)
    force = k0 * (0.5 - q) + d0 * (0.3 - qd)
    target = shaped_position_target(q, qd, force, k0, d0)
    assert target.item() == pytest.approx(0.5 + 12.0 * 0.3 / 320.0)
    assert solver_force(target, q, qd, k0, d0).item() == pytest.approx(force.item())


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def test_plant_core_on_description(desc):
    joints = list(desc.joints)
    plant = PlantCore(desc, joints, 2, PlantCfg(dtype=D64))
    assert plant.map.num_drives == 22 and len(plant.map.ankles) == 2
    standing = torch.tensor([desc.joints[n].standing_offset_rad for n in joints], dtype=D64).expand(2, -1)
    qdrv, _, _ = plant.map.drive_state(standing, torch.zeros_like(standing))
    demand = DriveDemand.pd(qdrv.clone(), plant.drive_stiffness.expand(2, -1), plant.drive_damping.expand(2, -1))
    plant.submit(demand, torch.zeros(2, dtype=torch.long))
    target = plant.pre_physics(0, standing, torch.zeros_like(standing))
    # Demanding the current pose at the baseline gains: zero force, target = current pose.
    assert torch.allclose(target, standing, atol=1e-9)
    fb = plant.feedback(standing, torch.zeros_like(standing))
    assert torch.allclose(fb.applied_torque_nm, torch.zeros(2, 22, dtype=D64), atol=1e-9)
    # Ankle solver gains are the crank gains through J at standing (pitch ~ 480/0.96^2 per crank pair).
    i = joints.index("foot_pitch_l")
    assert 700 < plant.k0[0, i].item() < 1100


def test_drive_deviations_and_per_drive_arrival():
    env = TorqueSpeedEnvelope([HEJ90, HEJ90], dtype=D64)
    bank = DriveBank(env, 1, 2, dtype=D64)
    bank.p_scale[:] = torch.tensor([[1.1, 1.0]], dtype=D64)
    bank.position_offset[:] = torch.tensor([[0.0, 0.02]], dtype=D64)  # drive 1 reads 0.02 rad high
    bank.torque_scale[:] = torch.tensor([[1.0, 0.5]], dtype=D64)
    demand = DriveDemand.pd(torch.tensor([[0.1, 2.0]], dtype=D64), 320.0, 12.0)
    bank.submit(demand, torch.tensor([[0, STEP // 2]]))  # drive 1 arrives half way through the step
    out = bank.compute(0, STEP, torch.zeros(1, 2, dtype=D64), torch.zeros(1, 2, dtype=D64))
    assert out.demanded[0, 0].item() == pytest.approx(1.1 * 320.0 * 0.1)
    # Drive 1: off for the first half, then 320 * (2.0 - 0.02) demanded, capped at half the 140 Nm peak.
    assert out.demanded[0, 1].item() == pytest.approx(0.5 * 320.0 * 1.98)
    assert out.applied[0, 1].item() == pytest.approx(0.5 * 70.0)


def test_zero_torque_scale_applies_no_torque():
    """A full power sag (torque scale 0) leaves the drive law's demand and applies nothing, without NaNs."""
    bank = _bank()
    bank.torque_scale.zero_()
    bank.submit(_pd(0.0), torch.tensor([0]))
    out = bank.compute(0, STEP, torch.zeros(1, 1, dtype=D64), torch.zeros(1, 1, dtype=D64))
    assert out.applied.item() == 0.0 and out.demanded.item() == 0.0
    bank.submit(_pd(0.1), torch.tensor([STEP]))
    out = bank.compute(STEP, 2 * STEP, torch.zeros(1, 1, dtype=D64), torch.zeros(1, 1, dtype=D64))
    assert out.applied.item() == 0.0 and out.demanded.item() == pytest.approx(32.0)
