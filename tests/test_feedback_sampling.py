"""Feedback age: the drive feedback the master holds between the drive's samples (devices/sampling.py)."""

import math

import pytest
import torch

from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.devices.sampling import FeedbackSampler
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.timing.clock import NS_PER_MS

CYCLE = 1_500_000
# Spacings of the PDO samples of the V1 excitation captures exc-53f09771 and exc-5fd2bcca (2026-10-28), in bus cycles
# of 1.5 ms: 35,814 of them (the table of the profile's devices.feedback_sampling, rounded).
MEASURED_PMF = {1: 0.0224, 2: 0.1317, 3: 0.4206, 4: 0.3310, 5: 0.0693, 6: 0.0140, 7: 0.0072, 8: 0.0025, 9: 0.0008,
                10: 0.0006}
MEASURED_MEAN_MS, MEASURED_P5_P95_MS = 5.08, (3.0, 7.5)


def _sampler(e=1, d=3, *, profile=True, **kw):
    if profile:
        block = load_profile("ethrc_v1")["devices"]["feedback_sampling"]
        kw.setdefault("cycles", block["cycles"])
        kw.setdefault("probability", block["probability"])
    return FeedbackSampler(e, d, cycle_ns=CYCLE, dtype=torch.float64, **kw)


def _ramp(t_ns, e, d):
    """A reading that is its own time: position t in ms, velocity -t, torques 2t and 3t."""
    x = torch.full((e, d), t_ns / NS_PER_MS, dtype=torch.float64)
    return DriveFeedback(x, -x, 2 * x, 3 * x)


def _run(s, e, d, ticks, tick_ns):
    """The plant time of every new sample per env: sample_ns after each tick, and what was held."""
    taken, held = [], []
    for i in range(ticks):
        t = i * tick_ns
        out = s.step(t, _ramp(t, e, d))
        taken.append(s.sample_ns[:, 0].clone())
        held.append(out.position_rad[:, 0].clone())
    return torch.stack(taken), torch.stack(held)  # [T, E]


def _spacings_ns(taken):
    out = []
    for e in range(taken.shape[1]):
        t = torch.unique_consecutive(taken[:, e])
        out.append(t[1:] - t[:-1])
    return torch.cat(out)


def test_spacings_are_whole_bus_cycles_with_the_measured_distribution():
    e, tick = 400, 100_000  # 0.1 ms ticks resolve a 1.5 ms cycle
    s = _sampler(e, 1, seed=1)
    taken, _ = _run(s, e, 1, 4000, tick)
    gaps = _spacings_ns(taken)
    assert len(gaps) > 30_000
    assert (gaps % CYCLE == 0).all()
    k = gaps // CYCLE
    for cycles, p in MEASURED_PMF.items():
        got = (k == cycles).double().mean().item() if cycles < 10 else (k >= 10).double().mean().item()
        assert abs(got - p) < 5 * math.sqrt(p * (1 - p) / len(k)) + 0.002, (cycles, got, p)
    ms = gaps.double() / NS_PER_MS
    assert abs(ms.mean().item() - MEASURED_MEAN_MS) < 0.05
    p5, p95 = torch.quantile(ms, torch.tensor([0.05, 0.95], dtype=torch.float64)).tolist()
    assert (p5, p95) == pytest.approx(MEASURED_P5_P95_MS, abs=0.01)


def test_a_physics_step_longer_than_a_cycle_keeps_the_sampling_rate():
    e = 400
    s = _sampler(e, 1, seed=2)
    taken, _ = _run(s, e, 1, 6000, NS_PER_MS)  # 1 ms steps: a sample waits for the next step
    gaps = _spacings_ns(taken).double() / NS_PER_MS
    assert abs(gaps.mean().item() - MEASURED_MEAN_MS) < 0.05
    assert gaps.min().item() >= 1.0


def test_a_sample_repeats_exactly_until_the_next_one():
    s = _sampler(3, 4, seed=3)
    seen = {}
    for i in range(500):
        t = i * 400_000
        out = s.step(t, _ramp(t, 3, 4))
        for e in range(3):
            stamp = s.sample_ns[e, 0].item()
            assert out.position_rad[e, 0].item() == stamp / NS_PER_MS  # the reading of the sample's own time
            assert out.velocity_rad_s[e, 0].item() == -stamp / NS_PER_MS
            assert out.applied_torque_nm[e, 0].item() == 2 * stamp / NS_PER_MS
            assert out.demanded_torque_nm[e, 0].item() == pytest.approx(3 * stamp / NS_PER_MS)
            seen.setdefault(e, set()).add(stamp)
        assert (t - s.sample_ns).max().item() <= 15 * CYCLE
    assert all(len(v) > 30 for v in seen.values())  # it renews (500 steps of 0.4 ms is 200 ms)


def test_a_second_call_at_the_same_time_changes_nothing():
    s = _sampler(50, 3, seed=4)
    for i in range(40):
        t = i * NS_PER_MS
        first = s.step(t, _ramp(t, 50, 3))
        nxt = s.next_ns.clone()
        again = s.step(t, DriveFeedback(*(torch.full((50, 3), 99.0, dtype=torch.float64) for _ in range(4))))
        assert torch.equal(first.position_rad, again.position_rad)
        assert torch.equal(nxt, s.next_ns)


def test_the_time_may_be_a_device_tensor():
    a, b = _sampler(20, 3, seed=5), _sampler(20, 3, seed=5)
    for i in range(60):
        t = i * NS_PER_MS
        assert torch.equal(a.step(t, _ramp(t, 20, 3)).position_rad,
                           b.step(torch.tensor(t), _ramp(t, 20, 3)).position_rad)


def test_drives_of_an_env_sample_together_unless_asked_otherwise():
    together, apart = _sampler(8, 6, seed=6), _sampler(8, 6, seed=6, per_drive=True)
    for i in range(300):
        t = i * 500_000
        together.step(t, _ramp(t, 8, 6)), apart.step(t, _ramp(t, 8, 6))
    assert (together.sample_ns == together.sample_ns[:, :1]).all()
    assert not (apart.sample_ns == apart.sample_ns[:, :1]).all()


def test_reset_takes_a_fresh_sample_for_those_envs_only():
    s = _sampler(4, 2, seed=7)
    for i in range(30):
        s.step(i * NS_PER_MS, _ramp(i * NS_PER_MS, 4, 2))
    s.reset(torch.tensor([1, 3]))
    t = 30 * NS_PER_MS
    out = s.step(t, _ramp(t, 4, 2))
    assert (s.sample_ns[[1, 3]] == t).all()
    assert (out.position_rad[[1, 3]] == t / NS_PER_MS).all()
    assert (s.sample_ns[[0, 2]] < t).sum() > 0 or (s.sample_ns[[0, 2]] == t).all()  # theirs keep their own clock
    assert (out.position_rad[[0, 2]] == s.sample_ns[[0, 2]] / NS_PER_MS).all()


def test_the_profile_block_gives_the_measured_distribution():
    desc, profile = Description.load(), load_profile("ethrc_v1")
    s = FeedbackSampler.for_robot(desc, profile, 2, 5)
    assert s.cycle_ns == CYCLE and s.cycles.tolist() == list(range(1, 11))
    pmf = torch.diff(torch.cat([torch.zeros(1, dtype=torch.float64), s._cdf, torch.ones(1, dtype=torch.float64)]))
    for k, p in MEASURED_PMF.items():
        assert pmf[k - 1].item() == pytest.approx(p, abs=1e-4)
    assert (pmf * s.cycles).sum().item() * CYCLE / NS_PER_MS == pytest.approx(MEASURED_MEAN_MS, abs=0.01)


def test_a_robot_without_measured_sampling_has_none():
    assert FeedbackSampler.for_robot(Description.load(), {"devices": {}}, 2, 5) is None


@pytest.mark.parametrize("kw", [dict(cycles=[], probability=[]), dict(cycles=[0], probability=[1.0]),
                                dict(cycles=[1, 2], probability=[1.0]), dict(cycles=[1], probability=[0.0]),
                                dict(cycles=[1, 2], probability=[1.0, -0.5])])
def test_malformed_distributions_are_refused(kw):
    with pytest.raises(ValueError):
        FeedbackSampler(1, 1, cycle_ns=CYCLE, **kw)
