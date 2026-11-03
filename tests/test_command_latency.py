import math

import pytest
import torch

from humanoid_sim.timing import ArrivalQueue
from humanoid_sim.train.plant import command_arrival_ns, command_latency_ns

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


# -- jitter on a demand's arrival (off unless asked for) ------------------------------------------------------------

LATENCY = torch.tensor([[1_600_000, 4_600_000, 6_600_000]])


def test_without_jitter_a_demand_arrives_after_the_latency_and_the_drive_delay():
    sent = torch.tensor([10_000_000, 20_000_000])
    out = command_arrival_ns(sent, LATENCY.expand(2, -1), DELAYS)
    assert torch.equal(out, sent.unsqueeze(1) + LATENCY + DELAYS)


def test_jitter_is_uniform_in_whole_ns_within_its_bound_and_never_before_the_send():
    gen = torch.Generator().manual_seed(0)
    sent = torch.zeros(40_000, dtype=torch.int64)
    nominal = command_arrival_ns(sent, LATENCY.expand(len(sent), -1), DELAYS)
    out = command_arrival_ns(sent, LATENCY.expand(len(sent), -1), DELAYS, 500_000, gen)
    delta = out - nominal
    assert out.dtype == torch.int64 and delta.abs().max() == 500_000
    assert abs(delta.double().mean().item()) < 5_000 and abs(delta.double().std().item() - 500_000 / math.sqrt(3)) < 5_000
    # a drive with nothing in front of it is not reached before the demand was sent
    short = command_arrival_ns(sent, torch.zeros(len(sent), 3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64),
                               500_000, gen)
    assert short.min() == 0 and (short > 0).any()


def test_a_negative_jitter_is_refused():
    with pytest.raises(ValueError, match="negative"):
        command_arrival_ns(torch.zeros(1, dtype=torch.int64), LATENCY, DELAYS, -1)


def test_jittered_demands_still_arrive_in_the_order_they_were_sent():
    gen = torch.Generator().manual_seed(1)
    e, d = 64, 3
    queue = ArrivalQueue({"x": torch.zeros(e, d)}, capacity=8, lanes=d)
    for i in range(6):  # 2 ms apart with 3 ms of jitter either way: arrivals alone would cross
        sent = torch.full((e,), 2_000_000 * i, dtype=torch.int64)
        queue.submit({"x": torch.full((e, d), float(i))}, command_arrival_ns(sent, LATENCY.expand(e, -1), DELAYS,
                                                                           3_000_000, gen))
    arrival = queue.arrival[:, :6]
    assert (arrival[:, 1:] >= arrival[:, :-1]).all()
    assert (arrival[:, 1:] == arrival[:, :-1]).any()  # some were held back behind an earlier one
