"""Drive protection: the output current limit and the torque overload trip (devices/protection.py)."""

import types

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.bank import DriveBank
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.devices.envelope import TorqueSpeedEnvelope
from humanoid_sim.devices.protection import STATUS_OVERLOAD, DriveProtection, profile_response
from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.timing import NS_PER_MS

D64 = torch.float64
STEP = NS_PER_MS
DWELL = 10 * NS_PER_MS
HEJ90 = [(0.0, 140.0), (5.0, 140.0), (6.0, 135.0), (8.0, 100.0), (10.0, 25.0), (10.4, 0.0)]
LIMIT = 100.0  # fixture torque limit, below the fixture envelope's 140 Nm peak
ZERO = torch.zeros(1, 1, dtype=D64)


def _protection(response="cut", *, drives=1, limit=LIMIT, current=float("inf"), dwell=DWELL):
    return DriveProtection(response, dwell, torch.full((drives,), limit, dtype=D64),
                           torch.full((drives,), current, dtype=D64))


def _bank(protection, envs=1):
    return DriveBank(TorqueSpeedEnvelope([HEJ90], dtype=D64), envs, 1, protection=protection, dtype=D64)


def _hold(bank, torque, envs=1):
    """A PD demand whose output is ``torque`` Nm at zero state: all feed-forward."""
    d = DriveDemand.pd(torch.zeros(envs, 1, dtype=D64), 0.0, 0.0, feedforward=torch.full((envs, 1), torque, dtype=D64))
    bank.submit(d, torch.zeros(envs, dtype=torch.int64))


def _run(bank, steps, start=0, envs=1):
    """(applied, tripped) per step; the drive sits at rest."""
    applied, tripped = [], []
    for k in range(start, start + steps):
        out = bank.compute(k * STEP, (k + 1) * STEP, torch.zeros(envs, 1, dtype=D64), torch.zeros(envs, 1, dtype=D64))
        applied.append(out.applied[:, 0].clone())
        tripped.append(bank.overload.tripped[:, 0].clone())
    return torch.stack(applied), torch.stack(tripped)


def test_holding_above_the_limit_trips_at_the_dwell_and_cuts():
    bank = _bank(_protection("cut"))
    _hold(bank, 120.0)
    applied, tripped = _run(bank, 30)
    first = int(tripped[:, 0].nonzero()[0])
    assert first * STEP == DWELL  # the first overloaded step starts at 0; one dwell later the drive has tripped
    assert torch.all(applied[:first, 0] == 120.0)  # held for exactly the dwell
    assert torch.all(applied[first:, 0] == 0.0) and tripped[first:].all()


def test_flag_reports_the_trip_and_keeps_the_torque():
    bank = _bank(_protection("flag"))
    _hold(bank, 120.0)
    applied, tripped = _run(bank, 30)
    assert tripped[:, 0].nonzero()[0].item() * STEP == DWELL and tripped[-1, 0]
    assert torch.all(applied == 120.0)


@pytest.mark.parametrize("response", ["flag", "cut"])
def test_at_or_below_the_limit_never_trips(response):
    bank = _bank(_protection(response))
    _hold(bank, LIMIT)  # at the limit is not above it
    applied, tripped = _run(bank, 500)
    assert not tripped.any() and torch.all(applied == LIMIT)


def test_a_step_under_the_limit_restarts_the_count():
    bank = _bank(_protection("cut"))
    _hold(bank, 120.0)
    _, tripped = _run(bank, 9)  # 9 ms over: one step short of the dwell
    assert not tripped.any()
    bank.submit(DriveDemand.pd(ZERO, 0.0, 0.0, feedforward=torch.full((1, 1), 50.0, dtype=D64)),
                torch.tensor([9 * STEP]))
    _, tripped = _run(bank, 1, start=9)
    assert not tripped.any()
    bank.submit(DriveDemand.pd(ZERO, 0.0, 0.0, feedforward=torch.full((1, 1), 120.0, dtype=D64)),
                torch.tensor([10 * STEP]))
    _, tripped = _run(bank, 9, start=10)
    assert not tripped.any()  # 9 ms since the restart again
    _, tripped = _run(bank, 1, start=19)
    assert not tripped.any()
    _, tripped = _run(bank, 1, start=20)  # a dwell after the restart at step 10
    assert tripped.all()


def test_the_sign_of_the_torque_does_not_matter():
    bank = _bank(_protection("cut"))
    _hold(bank, -120.0)
    applied, tripped = _run(bank, 12)
    assert tripped[:, 0].nonzero()[0].item() * STEP == DWELL and applied[-1, 0] == 0.0 and applied[0, 0] == -120.0


def test_a_trip_latches_until_the_drive_is_turned_off():
    bank = _bank(_protection("cut"))
    _hold(bank, 120.0)
    _run(bank, 12)
    _hold(bank, 10.0)  # demand back under the limit: still tripped
    bank.submit(DriveDemand.pd(ZERO, 0.0, 0.0, feedforward=torch.full((1, 1), 10.0, dtype=D64)),
                torch.tensor([12 * STEP]))
    applied, tripped = _run(bank, 5, start=12)
    assert tripped.all() and torch.all(applied == 0.0)
    bank.submit(DriveDemand.disabled(1, 1, dtype=D64), torch.tensor([17 * STEP]))  # turn it off
    _, tripped = _run(bank, 1, start=17)
    assert not tripped.any() and bank.overload.status().item() == 0
    bank.submit(DriveDemand.pd(ZERO, 0.0, 0.0, feedforward=torch.full((1, 1), 10.0, dtype=D64)),
                torch.tensor([18 * STEP]))  # turn it back on
    applied, tripped = _run(bank, 3, start=18)
    assert not tripped.any() and torch.all(applied == 10.0)


def test_the_status_word_and_env_reset():
    bank = _bank(_protection("cut"), envs=2)
    d = DriveDemand.pd(torch.zeros(2, 1, dtype=D64), 0.0, 0.0, feedforward=torch.tensor([[120.0], [50.0]], dtype=D64))
    bank.submit(d, torch.zeros(2, dtype=torch.int64))
    _, tripped = _run(bank, 12, envs=2)
    assert tripped[-1].tolist() == [True, False]  # envs are judged independently
    assert bank.overload.status().flatten().tolist() == [STATUS_OVERLOAD, 0]
    bank.reset(torch.tensor([0]))
    assert bank.overload.status().flatten().tolist() == [0, 0] and bank.overload.over_since_ns[0, 0] == -1


def test_the_current_limit_clips_the_applied_torque_and_bounds_the_judged_demand():
    bank = _bank(_protection("cut", current=130.0))
    _hold(bank, 200.0)
    out = bank.compute(0, STEP, ZERO, ZERO)
    assert out.demanded.item() == 200.0 and out.applied.item() == 130.0  # the demanded torque is never clipped
    # A current limit under the torque limit leaves nothing to overload: the judged demand saturates below it.
    bank = _bank(_protection("cut", current=90.0))
    _hold(bank, 200.0)
    applied, tripped = _run(bank, 50)
    assert not tripped.any() and torch.all(applied == 90.0)


def test_the_overload_is_judged_on_the_blended_demand_of_a_partial_step():
    bank = _bank(_protection("flag"))
    _hold(bank, 120.0)
    _run(bank, 3)
    # A demand under the limit arriving half way through step 3 weighs the step's demand down to 85 Nm: not over.
    bank.submit(DriveDemand.pd(ZERO, 0.0, 0.0, feedforward=torch.full((1, 1), 50.0, dtype=D64)),
                torch.tensor([3 * STEP + STEP // 2]))
    _, tripped = _run(bank, 20, start=3)
    assert not tripped.any()


def test_without_protection_the_bank_is_unchanged():
    torch.manual_seed(0)
    plain, inert = _bank(None), _bank(_protection("flag", limit=1e9, current=float("inf")))
    assert plain.protection is None
    for k in range(40):
        d = DriveDemand.pd(torch.randn(1, 1, dtype=D64), 300.0, 10.0, feedforward=torch.randn(1, 1, dtype=D64) * 150)
        for bank in (plain, inert):
            bank.submit(d, torch.tensor([k * STEP]))
        q, v = torch.randn(1, 1, dtype=D64), torch.randn(1, 1, dtype=D64) * 5
        a, b = plain.compute(k * STEP, (k + 1) * STEP, q, v), inert.compute(k * STEP, (k + 1) * STEP, q, v)
        for name in ("demanded", "applied", "enabled_fraction", "target_velocity"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not plain.overload.tripped.any()


def test_the_time_may_be_a_device_tensor_and_per_env():
    bank = _bank(_protection("cut"), envs=2)
    d = DriveDemand.pd(torch.zeros(2, 1, dtype=D64), 0.0, 0.0, feedforward=torch.full((2, 1), 120.0, dtype=D64))
    bank.submit(d, torch.zeros(2, dtype=torch.int64))
    z = torch.zeros(2, 1, dtype=D64)
    for k in range(12):
        t0 = torch.tensor(k * STEP)
        bank.compute(t0, t0 + STEP, z, z)
    assert bank.overload.tripped.all()


# -- description ---------------------------------------------------------------------------------


def test_the_description_gives_the_dwell_and_the_current_limit_torque():
    desc = Description.load("ethrc_v1")
    assert desc.overload_dwell_ns() == round(desc.table["backend"]["overload_dwell_s"] * 1e9)
    for name, drive in desc.drives.items():
        fam = desc.table["families"][drive.family]
        want = fam["max_current_a"] * fam["effort_nominal"] / fam["nominal_current_a"]
        assert desc.current_limit_torque_nm(name) == pytest.approx(want)
        assert want > drive.effort_peak  # the current limit lies above the torque limit, as on the robot


def test_a_robot_without_current_figures_has_no_current_limit():
    desc = Description.load("g1_29dof_rev_1_0")
    protection = DriveProtection.for_drives(desc, list(desc.drives), "cut", dtype=D64)
    assert torch.isinf(protection.current_limit_nm).all() and protection.dwell_ns == desc.overload_dwell_ns()
    assert torch.equal(protection.limit_nm, torch.tensor([d.effort_peak for d in desc.drives.values()], dtype=D64))


def test_protection_needs_the_dwell_and_a_known_response():
    desc = types.SimpleNamespace(robot="x", overload_dwell_ns=lambda: None)
    with pytest.raises(ValueError, match="overload_dwell_s"):
        DriveProtection.for_drives(desc, [], "cut")
    assert DriveProtection.for_drives(desc, [], None) is None
    with pytest.raises(ValueError, match="unknown drive protection response"):
        _protection("trip")
    assert profile_response({"devices": {}}) is None and profile_response({}) is None
    assert profile_response({"devices": {"protection": {"response": "cut"}}}) == "cut"


def test_the_plant_trips_a_held_peak_after_the_description_dwell():
    desc = Description.load("ethrc_v1")
    joints = list(desc.joints)
    period = 4 * NS_PER_MS
    plant = PlantCore(desc, joints, 1, PlantCfg(protection="cut", physics_period_ns=period, dtype=D64))
    peak = torch.tensor([[d.effort_peak for d in plant.desc.drives.values()]], dtype=D64)
    zero = torch.zeros(1, len(joints), dtype=D64)
    n = plant.map.num_drives
    plant.reset(torch.tensor([0]), DriveDemand.pd(torch.zeros(1, n, dtype=D64), 0.0, 0.0,
                                                  feedforward=1.2 * peak))
    first = None
    for k in range(10):
        plant.pre_physics(k * period, zero, zero)
        if first is None and plant.bank.overload.tripped.all():
            first = k
    dwell = desc.overload_dwell_ns()
    assert first == -(-dwell // period)  # the first step to start a dwell after the first overloaded one
    assert bool(plant.bank.overload.tripped.all()) and plant.last.applied.abs().max() == 0


def test_the_default_plant_has_no_protection():
    desc = Description.load("ethrc_v1")
    assert PlantCfg().protection is None
    plant = PlantCore(desc, list(desc.joints), 1, PlantCfg(dtype=D64))
    assert plant.bank.protection is None


# -- the runtime reports the status --------------------------------------------------------------


def test_the_status_is_in_the_feedback_record_and_the_layout():
    assert abi.DRIVE_FEEDBACK.names[-2:] == ("status", "_pad0") and abi.DRIVE_STATUS_OVERLOAD == STATUS_OVERLOAD
    assert abi.ABI_MINOR >= 6


def test_the_host_writes_the_status_word(tmp_path):
    from humanoid_sim.runtime.host import RuntimeHost

    n, h = 3, 2
    seg = abi.Segment.create(tmp_path / "plant", robot_id="r", description_digest="d" * 64, sim_revision="t",
                             num_drives=n, num_joints=n, physics_period_ns=4_000_000, bus_period_ns=1_500_000,
                             descriptor={})
    overload = types.SimpleNamespace(status=lambda: torch.tensor([[0, STATUS_OVERLOAD, 0]]))
    gyro = torch.zeros(1, h, 3)
    host = object.__new__(RuntimeHost)
    host.__dict__.update(
        sim=types.SimpleNamespace(imu=types.SimpleNamespace(samples=types.SimpleNamespace(gyro=gyro)), t_ns=5,
                                  plant=types.SimpleNamespace(bank=types.SimpleNamespace(overload=overload))),
        drives=["a", "b", "c"], seg=seg, step_count=1, late_steps=0, paused=0, _last_late_ns=0, _demands_taken=0,
        _demand_overruns=0, _faults={}, imu_log=None, extra_imu_logs={})
    drive_rows = np.concatenate([np.arange(4 * n, dtype=np.float64), overload.status().numpy()[0]])
    host._write_feedback(np.concatenate([drive_rows, np.zeros(6 * h), np.zeros(4 * h + 1)]))  # IMU, integers
    got = seg.regions["feedback"].read()[1]["drive"]
    assert got["status"][:n].tolist() == [0, STATUS_OVERLOAD, 0]
    assert got["demanded_torque_nm"][:n].tolist() == [9.0, 10.0, 11.0] and got["position_rad"][:n].tolist() == [0, 1, 2]


@pytest.mark.physics
@pytest.mark.parametrize("compile", [False, True])
def test_the_captured_drive_step_trips_as_the_eager_one_does(compile):
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    from humanoid_sim.train.drive_step import DriveStepGraph

    desc = Description.load()
    names = list(desc.joints)
    period_ns = NS_PER_MS
    cfg = PlantCfg(device="cuda:0", physics_period_ns=period_ns, queue_capacity=8, protection="cut")
    eager, captured = [PlantCore(desc, names, 4, cfg) for _ in range(2)]
    q = torch.zeros(4, len(names), device="cuda:0")
    qd = torch.zeros_like(q)
    graph = DriveStepGraph(captured, q, qd, compile=compile)
    peak = torch.tensor([[d.effort_peak for d in desc.drives.values()]], device="cuda:0")
    for step in range(60):
        t_ns = step * period_ns
        if step in (0, 30):  # env 0 holds 1.2 x the peak, env 1 half of it; env 2 restarts at step 30
            scale = torch.tensor([[1.2], [0.5], [1.2 if step == 0 else 0.0], [1.2]], device="cuda:0")
            demand = DriveDemand.pd(torch.zeros(4, peak.shape[1], device="cuda:0"), 0.0, 0.0, feedforward=scale * peak)
            for core in (eager, captured):
                core.submit(demand, torch.full((4,), t_ns, device="cuda:0", dtype=torch.int64))
        if step == 45:
            for core in (eager, captured):
                core.reset(torch.tensor([3], device="cuda:0"))
        eager.feedback(q, qd)
        expected = eager.pre_physics(t_ns, q, qd, eager.post_state)
        actual, _ = graph.step(t_ns, q, qd)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(captured.last.applied, eager.last.applied, atol=1e-5, rtol=1e-5)
        assert torch.equal(captured.bank.overload.tripped, eager.bank.overload.tripped)
        assert torch.equal(captured.bank.overload.over_since_ns, eager.bank.overload.over_since_ns)
    assert eager.bank.overload.tripped[0].all() and not eager.bank.overload.tripped[1].any()
