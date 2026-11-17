"""The runtime's drive feedback as the master reads it: the training plant's reporting and sampling models."""

import math

import pytest
import torch

from humanoid_sim.devices.feedback import DriveFeedback
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.runtime.host import _MeasuredFeedback
from humanoid_sim.timing.clock import NS_PER_MS

STEP = NS_PER_MS  # the native V1's physics period


def _measured(robot="ethrc_v1", **kw):
    desc, profile = Description.load(robot), load_profile(robot)
    drives = list(desc.drives)
    return _MeasuredFeedback(desc, drives, profile, torch.device("cpu"), torch.float32, seed=kw.pop("seed", 0),
                             cuda_graph=False), desc, profile, drives


def _rest(d, position=0.1):
    x = torch.full((1, d), position)
    z = torch.zeros(1, d)
    return DriveFeedback(x, z.clone(), z.clone(), z.clone())


def test_a_robot_without_measurements_reads_the_exact_state():
    m, _, _, drives = _measured("g1_29dof_rev_1_0")
    fb = _rest(len(drives))
    assert not m.active and m(0, fb) is fb


def test_the_v1_is_reported_and_held_by_the_training_plants_models():
    m, desc, profile, drives = _measured()
    assert m.active and m.reporting is not None and m.sampler is not None
    assert m.reporting.jitter is False  # the position's count jitter is Controls' fake drive's
    assert m.sampler.publish_delay_ns == profile["devices"]["feedback_sampling"]["publish_delay_ns"]
    assert m.sampler.cycle_ns == desc.control_period_ns()


def test_positions_are_whole_encoder_counts_within_half_a_count():
    m, desc, profile, drives = _measured()
    cpr = torch.tensor([profile["devices"]["encoders"][desc.drives[d].family]["counts_per_revolution"]
                        for d in drives], dtype=torch.float64)
    exact = 0.123456
    worst = 0.0
    for i in range(200):
        out = m(i * STEP, _rest(len(drives), exact))
        counts = out.position_rad.double() * cpr / (2 * math.pi)
        assert torch.allclose(counts, counts.round(), atol=2e-3)
        worst = max(worst, ((counts - exact * cpr / (2 * math.pi)).abs().max()).item())
    assert worst <= 0.5 + 2e-3  # a one-count jitter on top would reach 1.5


def test_a_drive_at_rest_reads_the_measured_torque_noise_and_velocity_dither():
    m, desc, profile, drives = _measured(seed=1)
    seen, velocity, torque, last = 0, [], [], None
    for i in range(12_000):
        t = i * STEP
        out = m(t, _rest(len(drives)))
        stamp = m.sampler.sample_ns.clone()
        if last is None or not torch.equal(stamp, last):  # a new sample: draws of the noise are independent
            velocity.append(out.velocity_rad_s.clone())
            torque.append(out.applied_torque_nm.clone())
            seen += 1
        last = stamp
    assert seen > 1500
    velocity, torque = torch.cat(velocity), torch.cat(torque)
    for k, drive in enumerate(drives):
        family = desc.drives[drive].family
        sigma = profile["devices"]["feedback_noise"][family]["torque_nm"]
        assert torque[:, k].std().item() == pytest.approx(sigma, rel=0.1), drive
        if family == "HEJ90-48-140":
            assert (velocity[:, k] != 0).double().mean().item() == pytest.approx(0.334, abs=0.03), drive


def test_the_reading_is_as_old_as_the_masters_sample_plus_the_publish_delay():
    m, desc, profile, drives = _measured(seed=2)
    block = profile["devices"]["feedback_sampling"]
    cycle = desc.control_period_ns()
    p = torch.tensor(block["probability"], dtype=torch.float64)
    p = p / p.sum()
    spacing = torch.tensor(block["cycles"], dtype=torch.float64) * cycle
    held_age = (p * spacing**2).sum() / (2 * (p * spacing).sum())  # the age of a held sample at a random time
    expected_ms = (held_age.item() + block["publish_delay_ns"]) / NS_PER_MS
    ages = []
    for i in range(20_000):
        t = i * STEP
        m(t, _rest(len(drives)))
        if t > 50 * NS_PER_MS:
            ages.append((t - m.sampler.sample_ns[0, 0]).item() / NS_PER_MS)
    mean = sum(ages) / len(ages)
    # The sample is taken at the first step past its time and shown at the first step past its delay: up to
    # a step each way.
    assert expected_ms - 1.0 <= mean <= expected_ms + 1.0
    assert 4.0 < mean < 6.5  # about 5 ms of the robot's 8.4: the rest is Controls' own cycle


def test_a_reset_reads_the_new_state_at_once():
    m, _, _, drives = _measured(seed=3)
    for i in range(50):
        m(i * STEP, _rest(len(drives), 0.0))
    m.reset()
    out = m(50 * STEP, _rest(len(drives), 0.5))
    assert (out.position_rad > 0.4).all()
