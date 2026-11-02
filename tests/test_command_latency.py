import pytest
import torch

from humanoid_sim.train.plant import command_latency_ns

DELAYS = torch.tensor([5_000_000, 2_000_000, 0])


def test_without_a_command_path_the_controller_adds_nothing():
    assert torch.equal(command_latency_ns({"devices": {}}, DELAYS), torch.zeros(3, dtype=torch.int64))


def test_each_drive_waits_the_measured_path_less_its_own_delay():
    out = command_latency_ns({"devices": {"command_path": {"frame_to_drive_ns": 6_600_000}}}, DELAYS)
    assert out.tolist() == [1_600_000, 4_600_000, 6_600_000]
    assert torch.equal(out + DELAYS, torch.full((3,), 6_600_000))


@pytest.mark.parametrize("total", [4_999_999, -1, 6.6e6, True])
def test_a_path_shorter_than_a_drive_or_not_whole_ns_is_refused(total):
    with pytest.raises(ValueError, match="frame_to_drive_ns"):
        command_latency_ns({"devices": {"command_path": {"frame_to_drive_ns": total}}}, DELAYS)
