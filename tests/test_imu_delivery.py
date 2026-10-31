"""IMU sampling, errors and burst delivery."""

import math

import pytest
import torch

from humanoid_sim.sensors.imu import ImuDelivery, ImuDeliveryCfg, pending_slots, rotvec_matrix

STEP = 4_000_000
CFG = ImuDeliveryCfg(sample_period_ns=2_740_000, burst_period_ns=(13_900_000, 17_600_000), in_burst_spacing_ns=13_000,
                     gyro_noise_density=0.0, accel_noise_density=0.0)


def _run(imu, steps, fn, t0=0):
    for k in range(1, steps + 1):
        t = t0 + k * STEP
        g, f = fn(t)
        imu.step(t, g, f)


def test_ramp_is_sampled_on_the_sensor_clock_and_delivered_in_bursts():
    e = 8
    imu = ImuDelivery(CFG, e, dtype=torch.float64)
    imu.reset(torch.arange(e), 0, torch.zeros(e, 6, dtype=torch.float64))

    def ramp(t):  # gyro x = t in seconds, exactly linear
        g = torch.zeros(e, 3, dtype=torch.float64)
        g[:, 0] = t / 1e9
        return g, torch.zeros(e, 3, dtype=torch.float64)

    _run(imu, 250, ramp)  # 1 s
    s = imu.samples
    valid = s.stamp_ns >= 0
    assert valid[:, :5].all()
    # Interpolated value equals the ramp at each sample's measurement time.
    assert torch.allclose(s.gyro[..., 0][valid], s.sample_ns[valid].double() / 1e9, atol=1e-9)
    # Newest first: measurement times and sequence numbers descend; a sample is received after it is measured.
    assert (s.sample_ns[:, :-1] > s.sample_ns[:, 1:]).all()
    assert (s.seq[:, :-1] == s.seq[:, 1:] + 1).all()
    assert (s.stamp_ns >= s.sample_ns).all()
    # ~365 samples per second delivered (bursts every 14-18 ms carry 5-7 samples each).
    rate = s.count.double()
    assert ((rate > 330) & (rate < 370)).all()
    age = (s.stamp_ns[:, 0] - s.sample_ns[:, 0]).double()
    assert (age >= 0).all() and (age < CFG.burst_period_ns[1]).all()


def test_burst_stamps_and_sizes():
    imu = ImuDelivery(CFG, 1, dtype=torch.float64, seed=3)
    imu.reset(torch.arange(1), 0, torch.zeros(1, 6, dtype=torch.float64))
    stamps = []
    for k in range(1, 500):
        imu.step(k * STEP, torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64))
        stamps.append(imu.samples.stamp_ns[0].clone())
    s = imu.samples.stamp_ns[0]
    gaps = (s[:-1] - s[1:]).tolist()
    in_burst = [g for g in gaps if g < 1_000_000]
    between = [g for g in gaps if g >= 1_000_000]
    assert set(in_burst) == {CFG.in_burst_spacing_ns}
    assert all(CFG.burst_period_ns[0] - 2 * CFG.in_burst_spacing_ns * 8 <= g <= CFG.burst_period_ns[1] for g in between)


def test_errors_bias_misalignment_and_noise():
    e = 4
    cfg = ImuDeliveryCfg(**{**CFG.__dict__, "gyro_noise_density": 1e-3, "accel_noise_density": 0.0})
    imu = ImuDelivery(cfg, e, dtype=torch.float64)
    imu.reset(torch.arange(e), 0)
    imu.gyro_bias[:] = torch.tensor([0.01, 0.0, 0.0])
    imu.misalignment[:, 2] = math.pi / 2  # 90° about z: x → y
    f = torch.tensor([[1.0, 0.0, 9.81]], dtype=torch.float64).expand(e, 3)
    g = torch.zeros(e, 3, dtype=torch.float64)
    imu.step(0, g, f)  # sets the interpolation start
    for k in range(1, 100):
        imu.step(k * STEP, g, f)
    s = imu.samples
    assert torch.allclose(s.specific_force[:, 0], torch.tensor([0.0, 1.0, 9.81], dtype=torch.float64), atol=1e-9)
    gx = s.gyro[..., 0][s.stamp_ns >= 0]
    sigma = 1e-3 * math.sqrt(1e9 / cfg.sample_period_ns)
    assert gx.mean().item() == pytest.approx(0.01, abs=4 * sigma / math.sqrt(gx.numel()))
    assert gx.std().item() == pytest.approx(sigma, rel=0.3)
    r = rotvec_matrix(torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64))[0]
    assert torch.allclose(r.T @ r, torch.eye(3, dtype=torch.float64), atol=1e-12)


def test_noise_off_gives_an_ideal_sensor_and_back():
    cfg = ImuDeliveryCfg(**{**CFG.__dict__, "gyro_noise_density": 1e-3, "accel_noise_density": 1e-3})
    imu = ImuDelivery(cfg, 1, dtype=torch.float64)
    imu.reset(torch.arange(1), 0)
    f = torch.tensor([[0.0, 0.0, 9.81]], dtype=torch.float64)
    g = torch.zeros(1, 3, dtype=torch.float64)

    def run(start):
        for k in range(start, start + 60):
            imu.step(k * STEP, g, f)
        valid = imu.samples.stamp_ns[0] >= 0
        return imu.samples.gyro[0][valid], imu.samples.specific_force[0][valid]

    imu.step(0, g, f)
    imu.set_noise(False)
    gyro, acc = run(1)
    assert gyro.abs().max().item() == 0.0 and torch.allclose(acc, f.expand_as(acc), atol=1e-12)
    imu.set_noise(True)
    gyro, _ = run(200)
    assert gyro.std().item() > 0


def test_reset_clears_an_env_only():
    imu = ImuDelivery(CFG, 2, dtype=torch.float64)
    imu.reset(torch.arange(2), 0, torch.zeros(2, 6, dtype=torch.float64))
    z = torch.zeros(2, 3, dtype=torch.float64)
    _run(imu, 50, lambda t: (z, z))
    before = imu.samples.count.clone()
    imu.reset(torch.tensor([1]), 50 * STEP)
    assert imu.samples.count.tolist() == [before[0].item(), 0]
    assert (imu.samples.stamp_ns[1] == -1).all() and (imu.samples.stamp_ns[0, :5] >= 0).all()
    _run(imu, 50, lambda t: (z, z), t0=50 * STEP)
    assert imu.samples.seq[1, 0].item() == imu.samples.count[1].item()


def test_turn_on_bias_is_drawn_per_env_at_reset():
    e = 4000
    cfg = ImuDeliveryCfg(**{**CFG.__dict__, "gyro_turn_on_bias": 1.5e-3, "accel_turn_on_bias": 0.02})
    imu = ImuDelivery(cfg, e, dtype=torch.float64)
    imu.reset(torch.arange(e), 0)
    std = imu.turn_on.std(dim=0)
    assert torch.allclose(std[:3], torch.full((3,), 1.5e-3, dtype=torch.float64), rtol=0.05)
    assert torch.allclose(std[3:], torch.full((3,), 0.02, dtype=torch.float64), rtol=0.05)
    before = imu.turn_on.clone()
    imu.reset(torch.tensor([0]), 0)
    assert not torch.equal(imu.turn_on[0], before[0]) and torch.equal(imu.turn_on[1:], before[1:])
    # A sample at rest reads the bias (no white noise configured).
    z = torch.zeros(e, 3, dtype=torch.float64)
    _run(imu, 10, lambda t: (z, z))
    assert torch.allclose(imu.samples.gyro[:, 0], imu.turn_on[:, :3], atol=1e-12)


def test_bias_drift_is_stationary_and_decorrelates():
    e, tau_ns = 2000, 200_000_000
    cfg = ImuDeliveryCfg(**{**CFG.__dict__, "gyro_bias_instability": 1e-3, "bias_correlation_time_ns": tau_ns})
    imu = ImuDelivery(cfg, e, dtype=torch.float64)
    imu.reset(torch.arange(e), 0)
    z = torch.zeros(e, 3, dtype=torch.float64)
    start = imu.drift[:, 0].clone()
    _run(imu, 100, lambda t: (z, z))  # 0.4 s, two correlation times
    end = imu.drift[:, 0]
    assert end.std().item() == pytest.approx(1e-3, rel=0.1)
    assert torch.corrcoef(torch.stack([start, end]))[0, 1].item() == pytest.approx(math.exp(-2), abs=0.06)
    assert imu.samples.gyro[:, 0, 0].std().item() == pytest.approx(1e-3, rel=0.1)  # the samples carry it


def test_noise_off_also_drops_the_models_bias():
    cfg = ImuDeliveryCfg(**{**CFG.__dict__, "gyro_turn_on_bias": 1e-2, "gyro_bias_instability": 1e-3})
    imu = ImuDelivery(cfg, 3, dtype=torch.float64)
    imu.reset(torch.arange(3), 0)
    imu.gyro_bias[:] = 0.5  # the randomization's bias stays
    imu.set_noise(False)
    z = torch.zeros(3, 3, dtype=torch.float64)
    _run(imu, 20, lambda t: (z, z))
    assert torch.allclose(imu.samples.gyro[:, 0], torch.full((3, 3), 0.5, dtype=torch.float64), atol=1e-12)


def test_a_latency_holds_each_sample_at_least_that_long_per_env():
    cfg = ImuDeliveryCfg(**{**CFG.__dict__, "sample_period_ns": 2_500_000, "burst_period_ns": (2_500_000, 2_500_000),
                            "in_burst_spacing_ns": 0})
    latency = torch.tensor([0, 1_000_000, 2_100_000, 5_000_000, 12_000_000, 60_000_000])
    e = len(latency)
    cfg = ImuDeliveryCfg(**{**cfg.__dict__, "pending": pending_slots(cfg, int(latency.max()), STEP)})
    imu = ImuDelivery(cfg, e, dtype=torch.float64)
    imu.reset(torch.arange(e), 0, torch.zeros(e, 6, dtype=torch.float64))
    imu.latency_ns.copy_(latency)
    _run(imu, 250, lambda t: (torch.zeros(e, 3, dtype=torch.float64), torch.zeros(e, 3, dtype=torch.float64)))
    s = imu.samples
    age = s.stamp_ns - s.sample_ns
    assert (age >= latency.unsqueeze(1)).all()
    # Bursts every sample period: at most one period beyond the latency, and none missed.
    assert (age < latency.unsqueeze(1) + cfg.burst_period_ns[1]).all()
    assert (s.seq[:, :-1] == s.seq[:, 1:] + 1).all()
