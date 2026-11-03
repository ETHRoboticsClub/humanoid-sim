"""Drive reporting against Controls' fake drive: the values its own tests pin (Epos4Slave @ 447b4acb,
slave_bank_test.cpp Encoder.* and test_bank_binding.py)."""

import math
import statistics

import pytest
import torch

from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.devices.reporting import DriveReporting
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description


def _fb(position, velocity, torque):
    t = lambda v: torch.tensor([[v]], dtype=torch.float64)  # noqa: E731
    return DriveFeedback(t(position), t(velocity), t(torque), t(torque))


def test_counts_velocity_and_torque_as_epos4slave_reports_them():
    r = DriveReporting([4096], [2500.0], jitter=False, dtype=torch.float64)
    out = r.report(_fb(1.0, 5.0, 10.0))
    assert out.position_rad.item() == 652 * 2 * math.pi / 4096  # round(4096 / 2pi) counts
    milli_rpm = round(out.velocity_rad_s.item() * 60 / (2 * math.pi) * 1000)
    assert milli_rpm == 36621  # one count per 400 us sample
    assert out.applied_torque_nm.item() == 10.0
    hi = DriveReporting([16384], [2500.0], jitter=False, dtype=torch.float64)
    assert round(hi.position(torch.tensor([[-0.5]], dtype=torch.float64)).item() * 16384 / (2 * math.pi)) == -1304


def test_velocity_is_whole_counts_per_controller_sample():
    r = DriveReporting([4096], [2500.0], jitter=False, dtype=torch.float64)
    step = 2 * math.pi / 4096 / 400e-6
    for true, counts in ((0.5 * step, 0), (0.6 * step, 1), (2.4 * step, 2), (-1.7 * step, -2)):
        v = r.velocity(torch.tensor([[true]], dtype=torch.float64)).item()
        assert abs(v - counts * step) < 1e-9, (true, counts)


def test_jitter_is_one_count_at_most_and_takes_every_value():
    jittered = DriveReporting([4096] * 3, [2500.0] * 3, seed=3, dtype=torch.float64)
    exact = DriveReporting([4096] * 3, [2500.0] * 3, jitter=False, dtype=torch.float64)
    x = torch.full((500, 3), 0.25, dtype=torch.float64)
    delta = torch.round((jittered.position(x) - exact.position(x)) * 4096 / (2 * math.pi))
    assert set(delta.unique().tolist()) == {-1.0, 0.0, 1.0}


def test_profile_covers_every_drive_family():
    desc = Description.load()
    r = DriveReporting.for_drives(desc, list(desc.drives), load_profile("ethrc_v1"))
    hej90 = [i for i, d in enumerate(desc.drives) if desc.drives[d].family.startswith("HEJ90")]
    assert torch.allclose(r.counts_per_rad[hej90], torch.tensor(4096 / (2 * math.pi)))


# -- the measured errors a drive's reading carries on top of the quantization --------------------------------------

NORMAL = statistics.NormalDist()
# Fraction of nonzero velocity readings of a drive at rest and standard deviation of its torque reading, per family,
# in the V1 excitation captures exc-53f09771 and exc-5fd2bcca (2026-10-28), pdo_feedback.csv, windows of 100 samples
# in which the position stayed within 2 counts: 65 HEJ90, 16 HEJ70 and 35 HEJ50 windows.
MEASURED_AT_REST = {"HEJ90-48-140": (0.334, 0.463), "HEJ70-48-50": (0.314, 0.107), "HEJ50-48-30": (0.318, 0.052)}


def _rest(n, d=1):
    z = torch.zeros(n, d, dtype=torch.float64)
    return DriveFeedback(z, z.clone(), z.clone(), z.clone())


def test_a_drive_at_rest_reads_zero_without_the_dither():
    r = DriveReporting([4096], [2500.0], jitter=False, dtype=torch.float64)
    assert r.velocity(torch.full((1000, 1), 0.4 * 3.8, dtype=torch.float64)).abs().max() == 0  # under half a count


@pytest.mark.parametrize("sigma", [0.3, 0.5, 0.52])
def test_dither_nonzero_rate_matches_the_gaussian_before_the_rounding(sigma):
    n = 400_000
    r = DriveReporting([4096], [2500.0], jitter=False, dtype=torch.float64, seed=1, velocity_dither_counts=[sigma])
    counts = torch.round(r.velocity(torch.zeros(n, 1, dtype=torch.float64)) / (2 * math.pi / 4096 * 2500))
    p = 2 * (1 - NORMAL.cdf(0.5 / sigma))
    assert abs((counts != 0).double().mean().item() - p) < 5 * math.sqrt(p * (1 - p) / n)
    assert abs(counts.sum().item()) < 5 * math.sqrt(p * n)  # symmetric
    assert set(counts.unique().tolist()) <= {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}


def test_dither_adds_to_the_motion_it_rounds():
    r = DriveReporting([4096], [2500.0], jitter=False, dtype=torch.float64, seed=2, velocity_dither_counts=[0.5])
    step = 2 * math.pi / 4096 * 2500
    v = r.velocity(torch.full((100_000, 1), 7.0 * step, dtype=torch.float64)) / step  # 7 counts per sample
    assert abs(v.mean().item() - 7.0) < 0.01
    assert abs(v.std().item() - math.sqrt(0.25 + 1 / 12)) < 0.02  # the rounding adds 1/12 to the variance


@pytest.mark.parametrize("family", MEASURED_AT_REST)
def test_profile_dither_reproduces_the_nonzero_rate_measured_at_rest(family):
    desc = Description.load()
    drives = [d for d in desc.drives if desc.drives[d].family == family][:1]
    r = DriveReporting.for_drives(desc, drives, load_profile("ethrc_v1"), jitter=False, seed=4)
    n = 200_000
    z = torch.zeros(n, 1, device=r.device, dtype=torch.float32)
    counts = torch.round(r.velocity(z) / (2 * math.pi / r.cpr / r.sample_s).float())
    assert abs((counts != 0).float().mean().item() - MEASURED_AT_REST[family][0]) < 0.015


@pytest.mark.parametrize("family", MEASURED_AT_REST)
def test_profile_torque_noise_has_the_standard_deviation_measured_at_rest(family):
    desc = Description.load()
    drives = [d for d in desc.drives if desc.drives[d].family == family][:1]
    r = DriveReporting.for_drives(desc, drives, load_profile("ethrc_v1"), jitter=False, seed=5)
    out = r.report(_rest(200_000)).applied_torque_nm.double()
    sigma = MEASURED_AT_REST[family][1]
    assert abs(out.std().item() / sigma - 1) < 0.04  # the reading is 0.462 against 0.463 Nm; the profile rounds it
    assert abs(out.mean().item()) < 5 * sigma / math.sqrt(len(out))
    assert torch.equal(out, torch.round(out * 1000) / 1000)  # still whole mNm
    assert torch.equal(r.report(_rest(10)).demanded_torque_nm, torch.zeros(10, 1, dtype=torch.float64))


def test_noise_off_reports_exactly_and_noise_is_reproducible_per_seed():
    desc, profile = Description.load(), load_profile("ethrc_v1")
    quiet = DriveReporting.for_drives(desc, list(desc.drives), profile, noise=False, jitter=False)
    out = quiet.report(DriveFeedback(*(torch.zeros(500, len(desc.drives)) for _ in range(4))))
    assert not out.velocity_rad_s.any() and not out.applied_torque_nm.any()
    a, b = (DriveReporting.for_drives(desc, list(desc.drives), profile, seed=9) for _ in range(2))
    fb = DriveFeedback(*(torch.zeros(50, len(desc.drives)) for _ in range(4)))
    assert torch.equal(a.report(fb).velocity_rad_s, b.report(fb).velocity_rad_s)
    assert a.report(fb).velocity_rad_s.any()


def test_dither_and_noise_need_one_non_negative_sigma_per_drive():
    with pytest.raises(ValueError, match="sigmas"):
        DriveReporting([4096, 4096], [2500.0] * 2, velocity_dither_counts=[0.5])
    with pytest.raises(ValueError, match="sigmas"):
        DriveReporting([4096], [2500.0], torque_noise_nm=[-0.1])
