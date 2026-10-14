"""CPU contract coverage; these tests do not establish an Isaac/Newton runtime."""

from types import SimpleNamespace
import json
import subprocess
import sys

import pytest
import torch

from humanoid_sim.cli import main
from humanoid_sim.contracts import CommandBatch, DriveConfig, MODE_JVPT
from humanoid_sim.firmware import DeviceBank
from humanoid_sim.isaac import IsaacEffortAdapter


class ArticulationStub:
    def __init__(self, num_envs=2, num_joints=1):
        q = torch.zeros((num_envs, num_joints))
        self.data = SimpleNamespace(
            joint_pos=q, joint_vel=q.clone(), joint_stiffness=q.clone(), joint_damping=q.clone()
        )
        self.cfg = SimpleNamespace(
            actuators={"effort": SimpleNamespace(class_type=SimpleNamespace(is_implicit_model=True))}
        )
        self.actuators = SimpleNamespace(target_command=self)
        self.effort = q.clone()
        self.writes = self.updates = self.steps = 0

    def set_effort_index(self, *, value, joint_ids=None, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        columns = slice(None) if joint_ids is None else joint_ids
        if isinstance(rows, list) and isinstance(columns, list):
            self.effort[torch.tensor(rows)[:, None], columns] = value
        else:
            self.effort[rows, columns] = value

    def write_data_to_sim(self):
        self.writes += 1

    def update(self, dt):
        self.updates += 1

    def step(self, dt):
        self.steps += 1
        self.data.joint_vel.add_(self.effort * dt)
        self.data.joint_pos.add_(self.data.joint_vel * dt)


def bank(num_envs=2):
    return DeviceBank([DriveConfig("hinge", stiffness=20, damping=1, torque_limit=5)], num_envs)


def submit(devices, now_ns, word=15):
    def real(value):
        return torch.full(devices.shape, value)

    def integer(value):
        return torch.full(devices.shape, value, dtype=torch.int64)

    devices.submit(CommandBatch(integer(word), integer(MODE_JVPT), real(0.2), real(0.0), real(0.0)), now_ns)


def enable(adapter, art):
    for word in (6, 7, 15):
        submit(adapter.devices, adapter.clock.now_ns, word)
        adapter.pre()
        art.step(adapter.dt)
        adapter.post()


def test_adapter_never_steps_and_rejects_phase_errors():
    art, devices = ArticulationStub(), bank()
    adapter = IsaacEffortAdapter(art, devices)
    submit(devices, 0, 6)
    adapter.pre()
    assert art.steps == 0
    with pytest.raises(RuntimeError, match="pre called twice"):
        adapter.pre()
    art.step(adapter.dt)
    feedback = adapter.post()
    assert art.steps == 1 and art.writes == 1 and art.updates == 1
    assert feedback.timestamp_ns == 400_000
    assert feedback.firmware_timestamp_ns == 0
    with pytest.raises(RuntimeError, match="post requires"):
        adapter.post()


def test_torque_is_held_between_firmware_ticks():
    art, devices = ArticulationStub(), bank()
    adapter = IsaacEffortAdapter(art, devices, physics_period_ns=200_000)
    calls = []
    original = devices.advance

    def counted(now_ns, q, dq):
        calls.append(now_ns)
        return original(now_ns, q, dq)

    devices.advance = counted
    for index in range(8):
        if adapter.clock.current().firmware:
            submit(devices, adapter.clock.now_ns, (6, 7, 15, 15)[index // 2])
        previous = devices.applied_effort.clone()
        applied = adapter.pre()
        if index % 2:
            torch.testing.assert_close(applied, previous)
        art.step(adapter.dt)
        feedback = adapter.post()
        assert feedback.timestamp_ns == (index + 1) * 200_000
    assert calls == [0, 400_000, 800_000, 1_200_000]
    assert bool(art.data.joint_pos.gt(0).all())


def test_hooks_can_delegate_write_and_update_to_isaac_scene():
    art = ArticulationStub()
    adapter = IsaacEffortAdapter(art, bank())
    adapter.pre(write_to_sim=False)
    assert art.writes == 0
    # Existing env.scene performs these operations once for all assets.
    art.write_data_to_sim()
    art.step(adapter.dt)
    art.update(adapter.dt)
    adapter.post(update_articulation=False)
    assert (art.writes, art.steps, art.updates) == (1, 1, 1)


@pytest.mark.parametrize("gain", ["joint_stiffness", "joint_damping"])
def test_rejects_second_pd_controller(gain):
    art = ArticulationStub()
    getattr(art.data, gain).fill_(1.0)
    with pytest.raises(ValueError, match="must be zero"):
        IsaacEffortAdapter(art, bank())


def test_rejects_explicit_isaac_actuator():
    art = ArticulationStub()
    art.cfg.actuators["effort"].class_type.is_implicit_model = False
    with pytest.raises(ValueError, match="explicit Isaac actuators"):
        IsaacEffortAdapter(art, bank())


def test_joint_order_and_transmission_are_used_for_state_and_effort():
    class SignTransmission:
        def device_state(self, q, dq):
            return -q, -dq

        def joint_effort(self, q, device_effort):
            return -device_effort

    art, devices = ArticulationStub(num_joints=2), bank()
    art.data.joint_pos[:, 0] = 0.8
    adapter = IsaacEffortAdapter(art, devices, joint_ids=[1], transmission=SignTransmission())
    enable(adapter, art)
    assert bool(art.effort[:, 0].eq(0).all())
    assert bool(art.effort[:, 1].lt(0).all())
    torch.testing.assert_close(devices.feedback.position, -art.data.joint_pos[:, [1]])
    assert bool(art.data.joint_pos[:, 0].eq(0.8).all())


def test_selective_reset_clears_only_selected_environment_effort():
    art = ArticulationStub()
    adapter = IsaacEffortAdapter(art, bank())
    enable(adapter, art)
    before = art.effort[1].clone()
    adapter.reset([0])
    assert bool(art.effort[0].eq(0).all())
    torch.testing.assert_close(art.effort[1], before)
    assert adapter.clock.now_ns == 1_200_000


def test_cpu_smoke_emits_movement_and_trace(tmp_path, capsys):
    assert main(["smoke", "--steps", "120", "--output-dir", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["backend"] == "cpu-inertia-stub"
    assert summary["max_displacement_rad"] > 0.0
    assert summary["peak_applied_effort_nm"] > 0.0
    trace = [json.loads(row) for row in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(trace) == 120
    assert trace[-1]["physics_time_ns"] == 48_000_000
    assert trace[-1]["firmware_time_ns"] == 47_600_000


def test_cli_import_does_not_load_physics_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import humanoid_sim.cli; "
            "assert 'isaaclab' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
    )
