"""Drive reporting against Controls' fake drive: the values its own tests pin (Epos4Slave @ 447b4acb,
slave_bank_test.cpp Encoder.* and test_bank_binding.py)."""

import math

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
