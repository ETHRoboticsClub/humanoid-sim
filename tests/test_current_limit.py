"""The output current limit in the drive bank (from #103), on fixture drives, and its torque from the description."""

import math

import pytest
import torch

from humanoid_sim.devices.bank import DriveBank
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.devices.winding import WindingI2t
from humanoid_sim.model.description import Description
from humanoid_sim.timing import NS_PER_MS

D64 = torch.float64
STEP = 4 * NS_PER_MS
CURVE = [(0.0, 140.0), (10.0, 140.0), (12.0, 0.0)]  # fixture envelope: 140 Nm peak
ZERO = torch.zeros(1, 1, dtype=D64)


def _bank(current=None, winding=None, drives=1):
    limit = None if current is None else torch.full((drives,), current, dtype=D64)
    return DriveBank(TorqueSpeedEnvelope([CURVE] * drives, dtype=D64), 1, drives, current_limit_nm=limit,
                     winding=winding, dtype=D64)


def _hold(bank, torque, t0=0):
    ff = torch.full((1, bank.num_drives), torque, dtype=D64)
    bank.submit(DriveDemand.pd(torch.zeros(1, bank.num_drives, dtype=D64), 0.0, 0.0, feedforward=ff),
                torch.tensor([t0]))


@pytest.mark.parametrize("torque", [200.0, -200.0])
def test_the_current_limit_clips_the_applied_torque_not_the_demanded(torque):
    bank = _bank(current=130.0)
    _hold(bank, torque)
    out = bank.compute(0, STEP, ZERO, ZERO)
    assert out.demanded.item() == torque and out.applied.item() == math.copysign(130.0, torque)


def test_a_current_limit_above_the_envelope_changes_nothing():
    limited, plain = _bank(current=184.0), _bank()
    for bank in (limited, plain):
        _hold(bank, 200.0)
    for k in range(10):
        a = limited.compute(k * STEP, (k + 1) * STEP, ZERO, ZERO)
        b = plain.compute(k * STEP, (k + 1) * STEP, ZERO, ZERO)
        assert torch.equal(a.applied, b.applied) and torch.equal(a.demanded, b.demanded)


def test_the_current_limit_and_the_winding_combine_the_lower_bound_wins():
    tau_ns = 1_000_000_000
    winding = WindingI2t(torch.tensor([80.0], dtype=D64), torch.tensor([tau_ns]))
    bank = _bank(current=120.0, winding=winding)
    _hold(bank, 200.0)
    first = math.ceil(tau_ns * math.log(120.0**2 / (120.0**2 - 80.0**2)) / STEP)  # heats at the clipped 120 Nm
    applied = [bank.compute(k * STEP, (k + 1) * STEP, ZERO, ZERO).applied.item() for k in range(first + 1)]
    assert applied[0] == 120.0 and applied[first - 1] == 120.0 and applied[first] == 80.0


@pytest.mark.parametrize("limit", [[0.0], [-5.0], [10.0, 20.0]])
def test_invalid_current_limits_are_refused(limit):
    with pytest.raises(ValueError):
        DriveBank(TorqueSpeedEnvelope([CURVE], dtype=D64), 1, 1, current_limit_nm=torch.tensor(limit, dtype=D64))


def test_the_description_gives_the_current_limit_torque():
    desc = Description.load("ethrc_v1")
    for name, drive in desc.drives.items():
        fam = desc.table["families"][drive.family]
        want = fam["max_current_a"] * fam["effort_nominal"] / fam["nominal_current_a"]
        assert desc.current_limit_torque_nm(name) == pytest.approx(want)


def test_a_robot_without_current_figures_has_no_current_limit():
    desc = Description.load("g1_29dof_rev_1_0")
    assert all(desc.current_limit_torque_nm(name) is None for name in desc.drives)
