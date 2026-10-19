import torch

from humanoid_sim.timing import NS_PER_MS, ArrivalQueue

STEP = 4 * NS_PER_MS


def _queue(envs=2, capacity=3):
    return ArrivalQueue({"x": torch.zeros(envs, 2)}, capacity)


def _step_blend(q, t0):
    w = q.weights(t0, t0 + STEP)
    out = q.blend(q.records()["x"], w)
    q.advance(t0 + STEP)
    return w, out


def test_record_landing_inside_step_is_time_weighted():
    q = _queue()
    q.submit({"x": torch.ones(2, 2)}, torch.tensor([1 * NS_PER_MS, 4 * NS_PER_MS]))
    w, out = _step_blend(q, 0)
    assert torch.allclose(w.sum(1), torch.ones(2, 1, dtype=torch.float64))
    assert torch.allclose(out[0], torch.full((2,), 0.75))  # in force for 3 of 4 ms
    assert torch.allclose(out[1], torch.zeros(2))  # arrives exactly at t1: next step
    _, out = _step_blend(q, STEP)
    assert torch.allclose(out, torch.ones(2, 2))
    assert int(q.count.sum()) == 0


def test_two_arrivals_in_one_step_and_fifo_order():
    q = _queue(envs=1)
    q.submit({"x": torch.full((1, 2), 1.0)}, torch.tensor([1 * NS_PER_MS]))
    q.submit({"x": torch.full((1, 2), 3.0)}, torch.tensor([3 * NS_PER_MS]))
    _, out = _step_blend(q, 0)
    assert torch.allclose(out, torch.full((1, 2), (0 * 1 + 1 * 2 + 3 * 1) / 4))
    assert torch.allclose(q.active["x"], torch.full((1, 2), 3.0))


def test_arrival_never_overtakes_earlier_record():
    q = _queue(envs=1)
    q.submit({"x": torch.full((1, 2), 1.0)}, torch.tensor([10 * NS_PER_MS]))
    q.submit({"x": torch.full((1, 2), 2.0)}, torch.tensor([2 * NS_PER_MS]))  # clamped to 10 ms
    assert q.arrival[0, :2, 0].tolist() == [10 * NS_PER_MS, 10 * NS_PER_MS]


def test_overflow_drops_oldest_and_counts():
    q = _queue(envs=1, capacity=2)
    for i in range(3):
        q.submit({"x": torch.full((1, 2), float(i))}, torch.tensor([(10 + i) * NS_PER_MS]))
    assert int(q.overflows[0]) == 1
    assert q.pending["x"][0, :, 0].tolist() == [1.0, 2.0]


def test_masked_submit_and_reset():
    q = _queue()
    q.submit({"x": torch.ones(2, 2)}, torch.tensor([0, 0]), env_mask=torch.tensor([True, False]))
    _step_blend(q, 0)
    assert q.active["x"][:, 0].tolist() == [1.0, 0.0]
    q.reset(torch.tensor([0]), {"x": torch.full((1, 2), 5.0)})
    assert q.active["x"][0, 0].item() == 5.0 and int(q.count[0]) == 0


def test_lanes_arrive_at_their_own_times():
    q = ArrivalQueue({"x": torch.zeros(1, 2)}, 3, lanes=2)
    q.submit({"x": torch.ones(1, 2)}, torch.tensor([[1 * NS_PER_MS, 6 * NS_PER_MS]]))
    w = q.weights(0, STEP)
    out = q.blend(q.records()["x"], w)
    assert out[0].tolist() == [0.75, 0.0]
    q.advance(STEP)
    assert q.active["x"][0].tolist() == [1.0, 0.0] and int(q.count[0]) == 1  # lane 1 still pending
    q.submit({"x": torch.full((1, 2), 2.0)}, torch.tensor([[5 * NS_PER_MS, 10 * NS_PER_MS]]))
    w = q.weights(STEP, 2 * STEP)
    out = q.blend(q.records()["x"], w)
    assert torch.allclose(out[0], torch.tensor([1.0 * 0.25 + 2.0 * 0.75, 0.5 * 0 + 0.5 * 1.0], dtype=out.dtype))
    q.advance(2 * STEP)
    assert q.active["x"][0].tolist() == [2.0, 1.0] and int(q.count[0]) == 1


def test_a_reset_queue_does_not_depend_on_what_was_queued_before():
    """Reset empties the slots to the record put in force, as on a fresh queue: a queue that held demands before
    (the runtime's graph warm-up queues some) and one that never did compare equal after the same reset."""
    used, fresh = _queue(), _queue()
    for i in range(3):
        used.submit({"x": torch.full((2, 2), float(i + 1))}, torch.tensor([(10 + i) * NS_PER_MS] * 2))
    for q in (used, fresh):
        q.reset(torch.tensor([0, 1]), {"x": torch.full((2, 2), 7.0)})
    for k in fresh.pending:
        assert torch.equal(used.pending[k], fresh.pending[k]) and (used.pending[k] == 7.0).all()
    assert torch.equal(used.arrival, fresh.arrival) and torch.equal(used.count, fresh.count)
