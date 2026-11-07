"""The winding I²t limit, on fixture drives: no value here comes from the description."""

import math

import pytest
import torch

from humanoid_sim.devices.bank import DriveBank
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.devices.winding import WindingI2t
from humanoid_sim.timing import NS_PER_MS

D64 = torch.float64
STEP = 4 * NS_PER_MS
TAU_NS = 1_000_000_000  # fixture winding time constant: 1 s
NOMINAL = 80.0  # fixture nominal torque
CURVE = [(0.0, 140.0), (10.0, 140.0), (12.0, 0.0)]  # fixture envelope: 140 Nm peak


def _winding(drives=1, nominal=NOMINAL, tau_ns=TAU_NS):
    return WindingI2t(torch.full((drives,), nominal, dtype=D64), torch.full((drives,), tau_ns, dtype=torch.int64))


def _bank(envs=1, drives=1, winding=True):
    env = TorqueSpeedEnvelope([CURVE] * drives, dtype=D64)
    return DriveBank(env, envs, drives, winding=_winding(drives) if winding else None, dtype=D64)


def _hold(bank, torque, t0=0):
    """Demand ``torque`` [E, D] (or a number) as pure feedforward from ``t0``: the law's output is exactly it."""
    ff = torch.as_tensor(torque, dtype=D64).expand(bank.num_envs, bank.num_drives)
    bank.submit(DriveDemand.pd(torch.zeros(bank.num_envs, bank.num_drives, dtype=D64), 0.0, 0.0, feedforward=ff),
                torch.tensor([t0]))


def _step(bank, k):
    z = torch.zeros(bank.num_envs, bank.num_drives, dtype=D64)
    return bank.compute(k * STEP, (k + 1) * STEP, z, z)


def _steps_to_limit(torque, nominal=NOMINAL, tau_ns=TAU_NS):
    """The first step whose applied torque is limited, from cold: heat after k steps is T²(1 − e^(−k dt/τ))."""
    t = tau_ns * math.log(torque**2 / (torque**2 - nominal**2))
    return math.ceil(t / STEP)


def test_below_nominal_is_never_limited():
    bank = _bank()
    _hold(bank, 60.0)
    for k in range(10 * TAU_NS // STEP):
        assert _step(bank, k).applied.item() == pytest.approx(60.0)
    assert not bank.winding_heat.limited.any()


@pytest.mark.parametrize("torque", [120.0, -120.0, 140.0])
def test_above_nominal_falls_back_to_nominal_after_the_i2t_time(torque):
    bank = _bank()
    _hold(bank, torque)
    first = _steps_to_limit(abs(torque))
    for k in range(first):
        assert _step(bank, k).applied.item() == pytest.approx(torque), k
    for k in range(first, first + 50):
        out = _step(bank, k)
        assert out.applied.item() == pytest.approx(math.copysign(NOMINAL, torque)), k
        assert out.demanded.item() == pytest.approx(torque)  # the law's output is not limited
    assert bank.winding_heat.limited.all()


def test_a_cooled_winding_gives_the_full_torque_again():
    bank = _bank()
    _hold(bank, 120.0)
    first = _steps_to_limit(120.0)
    k = 0
    while k < first + 10:
        _step(bank, k)
        k += 1
    bank.submit(DriveDemand.disabled(1, 1, dtype=D64), torch.tensor([k * STEP]))
    for _ in range(20 * TAU_NS // STEP):  # off: no torque, the winding cools
        assert _step(bank, k).applied.item() == 0
        k += 1
    assert not bank.winding_heat.limited.any()
    _hold(bank, 120.0, t0=k * STEP)
    applied = []
    for _ in range(first + 1):
        applied.append(_step(bank, k).applied.item())
        k += 1
    assert applied[first - 1] == pytest.approx(120.0) and applied[first] == pytest.approx(NOMINAL)


def test_a_limited_drive_under_nominal_demand_recovers_its_peak_gradually():
    bank = _bank()
    _hold(bank, 120.0)
    first = _steps_to_limit(120.0)
    for k in range(first + 1):
        _step(bank, k)
    k = first + 1
    _hold(bank, 40.0, t0=k * STEP)  # under nominal: the winding cools while the drive works
    assert _step(bank, k).applied.item() == pytest.approx(40.0)
    heat = bank.winding_heat.heat.item()
    assert heat < NOMINAL**2


def test_drives_and_envs_are_limited_independently_and_reset_clears_one_env():
    bank = _bank(envs=2, drives=2)
    torque = torch.tensor([[120.0, 60.0], [60.0, 60.0]], dtype=D64)
    _hold(bank, torque)
    first = _steps_to_limit(120.0)
    for k in range(first + 1):
        out = _step(bank, k)
    torch.testing.assert_close(out.applied, torch.tensor([[NOMINAL, 60.0], [60.0, 60.0]], dtype=D64))
    assert bank.winding_heat.limited.tolist() == [[True, False], [False, False]]
    bank.reset(torch.tensor([0]))
    assert bank.winding_heat.heat[0].tolist() == [0.0, 0.0]
    warm = 60.0**2 * (1 - math.exp(-(first + 1) * STEP / TAU_NS))
    assert bank.winding_heat.heat[1].tolist() == pytest.approx([warm, warm])


def test_without_a_winding_the_bank_is_unchanged():
    torque = 130.0
    plain, inert = _bank(winding=False), DriveBank(TorqueSpeedEnvelope([CURVE], dtype=D64), 1, 1,
                                                    winding=_winding(nominal=math.inf), dtype=D64)
    _hold(plain, torque)
    _hold(inert, torque)
    for k in range(2 * _steps_to_limit(torque)):
        a, b = _step(plain, k), _step(inert, k)
        assert torch.equal(a.applied, b.applied) and torch.equal(a.demanded, b.demanded)
    assert not plain.winding_heat.heat.any()


def test_a_step_split_between_two_demands_heats_by_their_time_weighted_square():
    bank = _bank()
    _hold(bank, 100.0)
    _hold(bank, 0.0, t0=STEP // 4)  # the step's first quarter at 100 Nm, the rest at 0
    _step(bank, 0)
    assert bank.winding_heat.heat.item() == pytest.approx(0.25 * 100.0**2 * (1 - math.exp(-STEP / TAU_NS)))


@pytest.mark.parametrize("nominal, tau_ns", [(0.0, TAU_NS), (-1.0, TAU_NS), (NOMINAL, 0)])
def test_invalid_windings_are_refused(nominal, tau_ns):
    with pytest.raises(ValueError):
        _winding(nominal=nominal, tau_ns=tau_ns)


def test_time_constants_must_be_integer_nanoseconds():
    with pytest.raises(ValueError, match="int64"):
        WindingI2t(torch.tensor([NOMINAL]), torch.tensor([1.0]))
