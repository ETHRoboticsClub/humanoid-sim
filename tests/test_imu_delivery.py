"""IMU sampling, errors and burst delivery."""

import math

import pytest
import torch

from humanoid_sim.sensors.imu import ImuDelivery, ImuDeliveryCfg, rotvec_matrix

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
