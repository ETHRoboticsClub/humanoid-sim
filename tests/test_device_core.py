import pytest
import torch

from humanoid_sim.clock import TickScheduler
from humanoid_sim.contracts import CommandBatch, DriveConfig, DriveState, MODE_JVPT
from humanoid_sim.firmware import DeviceBank


PERIOD = 400_000


def command(bank, cw=15, position=1.0, velocity=0.0, effort=0.0, mode=MODE_JVPT):
    def floats(value):
        return torch.full(bank.shape, value, device=bank.device, dtype=bank.dtype)

    return CommandBatch(
        torch.full(bank.shape, cw, device=bank.device, dtype=torch.int64),
        torch.full(bank.shape, mode, device=bank.device, dtype=torch.int64),
        floats(position),
        floats(velocity),
        floats(effort),
    )


def state(bank, now):
    zeros = torch.zeros(bank.shape, device=bank.device, dtype=bank.dtype)
    return bank.advance(now, zeros, zeros)


def enable(bank):
    for index, cw in enumerate((6, 7, 15)):
        bank.submit(command(bank, cw), index * PERIOD)
        state(bank, index * PERIOD)


def test_enable_sequence_pd_feedforward_and_sensor_truth():
    bank = DeviceBank([DriveConfig("one", stiffness=10, damping=2, torque_limit=30)], 3)
    bank.submit(command(bank, 15), 0)
    assert not state(bank, 0).any()  # Direct enable does not jump state machine.
    for now, cw in ((PERIOD, 6), (2 * PERIOD, 7), (3 * PERIOD, 15)):
        bank.submit(command(bank, cw, position=1, velocity=2, effort=3), now)
        torque = state(bank, now)
    assert torch.equal(torque, torch.full(bank.shape, 17.0))
    assert (bank.feedback.statusword == DriveState.OPERATION_ENABLED).all()
    assert torch.isnan(bank.feedback.measured_effort).all()
    bank.refresh_feedback(torch.ones(bank.shape), torch.zeros(bank.shape), torch.full(bank.shape, 4.0))
    assert (bank.feedback.measured_effort == 4).all()
    assert (bank.feedback.applied_effort == 17).all()


def test_delay_delivery_boundary_and_copied_commands():
    bank = DeviceBank(
        [DriveConfig("fast", command_delay_ns=0), DriveConfig("slow", command_delay_ns=PERIOD + 1)], 2
    )
    frame = command(bank, 6, position=2)
    bank.submit(frame, 0)
    frame.position.fill_(99)
    state(bank, 0)
    assert (bank.feedback.demand_position[:, 0] == 2).all()
    assert (bank.feedback.demand_position[:, 1] == 0).all()
    state(bank, PERIOD)
    assert (bank.feedback.statusword[:, 1] == DriveState.SWITCH_ON_DISABLED).all()
    state(bank, 2 * PERIOD)
    assert (bank.feedback.statusword[:, 1] == DriveState.READY_TO_SWITCH_ON).all()
    assert (bank.feedback.demand_position[:, 1] == 2).all()


def test_watchdog_boundary_frame_prevents_expiry_then_fault_latches():
    bank = DeviceBank([DriveConfig("one", stiffness=10, torque_limit=20, watchdog_ns=2 * PERIOD)], 2)
    enable(bank)
    state(bank, 3 * PERIOD)
    bank.submit(command(bank), 4 * PERIOD)  # Frame at exact timeout precedes watchdog.
    assert (state(bank, 4 * PERIOD) == 10).all()
    state(bank, 5 * PERIOD)
    assert not state(bank, 6 * PERIOD).any()
    assert bank.feedback.watchdog_expired.all()
    bank.submit(command(bank), 7 * PERIOD)
    assert not state(bank, 7 * PERIOD).any()
    assert (bank.feedback.statusword == DriveState.FAULT).all()
    bank.submit(command(bank, 128), 8 * PERIOD)
    state(bank, 8 * PERIOD)
    assert (bank.feedback.statusword == DriveState.SWITCH_ON_DISABLED).all()
    assert not bank.feedback.watchdog_expired.any()


def test_disabled_watchdog_and_per_environment_control_are_independent():
    bank = DeviceBank(
        [
            DriveConfig("no_watchdog", stiffness=10, torque_limit=20, watchdog_ns=0),
            DriveConfig("short_watchdog", stiffness=10, torque_limit=20, watchdog_ns=PERIOD),
        ],
        2,
    )
    enable(bank)
    frame = command(bank)
    frame.controlword[1, 0] = 0
    bank.submit(frame, 3 * PERIOD)
    state(bank, 3 * PERIOD)
    state(bank, 4 * PERIOD)
    assert bank.feedback.applied_effort[0, 0].item() == 10
    assert bank.feedback.applied_effort[1, 0].item() == 0
    assert not bank.feedback.watchdog_expired[:, 0].any()
    assert bank.feedback.watchdog_expired[:, 1].all()
    assert (bank.feedback.statusword[:, 1] == DriveState.FAULT).all()


def test_selective_reset_drops_pending_frames_and_filter_for_only_selected_env():
    bank = DeviceBank(
        [DriveConfig("one", stiffness=10, torque_limit=20, torque_filter_tau_s=0.001, watchdog_ns=0)], 2
    )
    enable(bank)
    before = bank.feedback.actuator_effort.clone()
    assert (before > 0).all()
    bank.submit(command(bank, position=3), 5 * PERIOD)
    bank.reset([0])
    assert bank.feedback.actuator_effort[0].item() == 0
    assert bank.feedback.actuator_effort[1].item() == before[1].item()
    for tick in (3, 4, 5):
        state(bank, tick * PERIOD)
    assert bank.feedback.demand_position[0].item() == 0
    assert bank.feedback.demand_position[1].item() == 3
    assert bank.feedback.statusword[0].item() == DriveState.SWITCH_ON_DISABLED
    assert bank.feedback.statusword[1].item() == DriveState.OPERATION_ENABLED
    bank.reset(now_ns=20 * PERIOD)
    assert not state(bank, 20 * PERIOD).any()


def test_invalid_frame_is_atomic_and_queue_is_bounded():
    bank = DeviceBank([DriveConfig("one")], 2, queue_capacity=1)
    invalid = command(bank, 6)
    invalid.position[1, 0] = float("nan")
    with pytest.raises(ValueError):
        bank.submit(invalid, 0)
    assert not bank.feedback.commanded_position.any()
    with pytest.raises(NotImplementedError):
        bank.submit(command(bank, 6, mode=10), 0)
    with pytest.raises(NotImplementedError):
        bank.submit(command(bank, 3), 0)
    bank.submit(command(bank, 6), 0)
    with pytest.raises(OverflowError):
        bank.submit(command(bank, 7, position=2), 0)
    assert (bank.feedback.commanded_position == 1).all()
    state(bank, 0)
    assert (bank.feedback.statusword == DriveState.READY_TO_SWITCH_ON).all()
    with pytest.raises(NotImplementedError):
        bank.configure("gain", 12)


def test_selective_reset_releases_queue_capacity_when_no_recipients_remain():
    bank = DeviceBank([DriveConfig("one", command_delay_ns=10 * PERIOD)], 1, queue_capacity=1)
    bank.submit(command(bank, 6, position=9), 0)
    bank.reset([0])
    bank.submit(command(bank, 6, position=2), 0)
    for tick in range(11):
        state(bank, tick * PERIOD)
    assert bank.feedback.demand_position.item() == 2
    assert bank.feedback.statusword.item() == DriveState.READY_TO_SWITCH_ON

    # A frame still targeting another environment must retain its queue slot.
    bank = DeviceBank([DriveConfig("one", command_delay_ns=10 * PERIOD)], 2, queue_capacity=1)
    bank.submit(command(bank, 6), 0)
    bank.reset([0])
    with pytest.raises(OverflowError):
        bank.submit(command(bank, 7), 0)


def test_integer_clock_rejects_unresolved_physics_and_skipped_firmware():
    with pytest.raises(ValueError):
        TickScheduler(PERIOD, PERIOD * 2)
    with pytest.raises(ValueError):
        TickScheduler(PERIOD, 300_000)
    scheduler = TickScheduler(PERIOD, PERIOD // 2)
    assert scheduler.current().firmware
    assert not scheduler.step().firmware
    assert scheduler.step().firmware
    bank = DeviceBank([DriveConfig("one")], 1)
    state(bank, 0)
    assert not state(bank, PERIOD // 2).any()
    with pytest.raises(ValueError):
        state(bank, PERIOD + 1)
    state(bank, PERIOD)


def test_physical_timestamp_is_independent_of_firmware_tick():
    bank = DeviceBank([DriveConfig("one")], 1, check_finite=False)
    state(bank, 0)
    position, velocity = torch.ones(bank.shape), torch.zeros(bank.shape)
    bank.refresh_feedback(position, velocity, timestamp_ns=PERIOD // 2)
    assert bank.feedback.timestamp_ns == PERIOD // 2
    assert bank.feedback.firmware_timestamp_ns == 0
    assert bank.feedback.position.item() == 1
    with pytest.raises(ValueError):
        bank.refresh_feedback(position, velocity, timestamp_ns=0)
    with pytest.raises(ValueError):
        state(bank, 0)
    state(bank, PERIOD)
    assert bank.feedback.timestamp_ns == PERIOD
    assert bank.feedback.firmware_timestamp_ns == PERIOD
    bank.reset(now_ns=10 * PERIOD)
    assert bank.feedback.timestamp_ns == 10 * PERIOD
    assert bank.feedback.firmware_timestamp_ns == -1


def test_symmetric_speed_cap_filter_and_passive_friction():
    bank = DeviceBank(
        [
            DriveConfig(
                "one", torque_limit=10, speed_limit=2, viscous_friction=0.5, coulomb_friction=1, watchdog_ns=0
            )
        ],
        2,
    )
    enable(bank)
    bank.submit(command(bank, effort=100), 3 * PERIOD)
    velocity = torch.tensor([[1.0], [-1.0]])
    torque = bank.advance(3 * PERIOD, torch.zeros(bank.shape), velocity)
    assert torch.equal(bank.feedback.actuator_effort, torch.full(bank.shape, 5.0))
    assert torch.equal(torque, torch.tensor([[3.5], [6.5]]))
    bank.submit(command(bank, 0), 4 * PERIOD)
    torque = bank.advance(4 * PERIOD, torch.zeros(bank.shape), velocity)
    assert torch.equal(torque, torch.tensor([[-1.5], [1.5]]))
    bank.submit(command(bank, 6), 5 * PERIOD)
    bank.advance(5 * PERIOD, torch.zeros(bank.shape), velocity * 10)
    assert not bank.feedback.actuator_effort.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_batched_trajectory():
    config = [DriveConfig("one", stiffness=20, damping=1, torque_limit=10)]
    cpu, gpu = DeviceBank(config, 128), DeviceBank(config, 128, device="cuda")
    for tick, cw in enumerate((6, 7, 15, 15)):
        for bank in (cpu, gpu):
            bank.submit(command(bank, cw, effort=2), tick * PERIOD)
            state(bank, tick * PERIOD)
        assert torch.equal(cpu.feedback.applied_effort, gpu.feedback.applied_effort.cpu())
