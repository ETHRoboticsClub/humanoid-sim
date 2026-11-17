"""The runtime's measured timing and IMU state, on by default: arrival jitter, a deadline-based clock mapping, the
IMU's bias per power-on and its mount rotation, the feedback as the master reads it."""

import math

import numpy as np
import pytest
from conftest import built_usd

from humanoid_sim.model.asset import load_profile
from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.sensors.imu import mount_misalignment_rad
from humanoid_sim.timing import NS_PER_MS, NS_PER_S, command_jitter_ns

pytestmark = pytest.mark.physics


def _host(tmp_path, **kw):
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.runtime.host import RuntimeHost

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    host = RuntimeHost(tmp_path / "plant", built_usd("ethrc_v1"), device=device, support="free", compile=False, **kw)
    return host, abi.Segment.open(tmp_path / "plant", wait_s=5)


def _feedback(seg):
    got = seg.regions["feedback"].read()
    assert got is not None
    return got[1]


@pytest.mark.parametrize("jitter_ms", [0.0, 3.0])
def test_host_jitter_does_not_move_demand_arrivals_or_the_clock_mapping(tmp_path, monkeypatch, jitter_ms):
    """Steps run on a schedule; the host's clock reads late by a random amount when it writes (its compute time, up to
    ``jitter_ms``, with a spike). A demand sent on the schedule still reaches each drive its delay after the plant
    time that moment has, exactly, and the feedback's own mapping of plant time to the host's clock does not move."""
    host, client = _host(tmp_path)
    try:
        host._jitter_ns = 0  # arrival jitter is the next test's
        n, period = len(host.drives), host.period_ns
        demand = abi.pd_demand(np.zeros(n), 10.0, 1.0)
        rng = np.random.default_rng(0)
        epoch = abi.mono_ns() + 10 * NS_PER_S
        reading = {"at": 0}
        monkeypatch.setattr(abi, "mono_ns", lambda: reading["at"])
        offsets, stamps, queued = [], [], []
        t0 = host.sim.t_ns
        for k in range(14):
            deadline = epoch + k * period
            if k == 9:
                host.paused = True  # a paused host keeps publishing: the stamps still advance
            late = int(rng.uniform(0, jitter_ms) * NS_PER_MS) + (int(2.3 * NS_PER_MS) if jitter_ms and k % 5 == 4 else 0)
            reading["at"] = deadline + late
            sent = deadline - 200_000  # on the schedule, just before this step
            if k:
                client.write_demands(demand, cycle=k, sent_mono_ns=sent)
            host._deadline_ns = deadline
            start = host.sim.t_ns
            host.step()
            fb = _feedback(client)
            offsets.append(int(fb["written_mono_ns"]) - int(fb["sim_time_ns"]))
            stamps.append(int(fb["written_mono_ns"]))
            if k and not host.paused:
                q = host.sim.plant.bank.queue
                arrival = q.arrival[0, int(q.count[0]) - 1].cpu().numpy()
                queued.append((arrival, start + (sent - deadline)))
        assert queued
        for arrival, plant_sent in queued:
            assert (arrival == plant_sent + host._drive_delays_ns).all()
        assert all(b > a for a, b in zip(stamps, stamps[1:], strict=False))
        # While it runs, plant time and the schedule advance together, so the mapping is one constant.
        assert set(offsets[:9]) == {epoch - t0}
    finally:
        host.close()


def test_a_demand_arrives_its_delay_after_its_send_within_the_profiles_jitter(tmp_path):
    host, client = _host(tmp_path)
    try:
        jitter = command_jitter_ns(load_profile("ethrc_v1"))
        assert host._jitter_ns == jitter == 1_200_000
        n = len(host.drives)
        arrivals = []
        host._intake = lambda drive, arrival_ns: arrivals.append(np.array(arrival_ns))  # what the host queues
        demand = abi.pd_demand(np.zeros(n), 10.0, 1.0)
        sent = abi.mono_ns()
        for k in range(1, 401):
            client.write_demands(demand, cycle=k, sent_mono_ns=sent)
            host._take_demands()
        assert len(arrivals) == 400
        nominal = sent - host._mono_minus_plant_ns + host._drive_delays_ns
        delta = np.array(arrivals) - nominal
        assert np.abs(delta).max() <= jitter and np.abs(delta).max() > 0.9 * jitter
        assert abs(delta.mean()) < 50_000 and delta.std() == pytest.approx(0.68e6, abs=0.03e6)
        assert len({tuple(row) for row in delta}) > 390  # each record and drive its own draw
    finally:
        host.close()


def test_the_demand_queue_holds_every_record_in_flight(tmp_path):
    host, _ = _host(tmp_path)
    try:
        bus = host._segment_args["bus_period_ns"]
        assert bus == host.desc.control_period_ns()
        longest = int(host._drive_delays_ns.max()) + host._jitter_ns
        in_flight = math.ceil(longest / bus) + 1
        assert host.sim.plant.bank.queue.capacity >= in_flight + math.ceil(2 * host.period_ns / bus)
    finally:
        host.close()


def test_the_imu_bias_is_drawn_at_a_session_start_and_kept_through_reset(tmp_path):
    host, client = _host(tmp_path)
    try:
        imu = host.sim.imu
        first = imu.turn_on.clone()
        assert first.abs().max() > 0
        for _ in range(3):
            host.step()
        assert host._reset({"pose": host.start_pose}) == "reset"
        assert (imu.turn_on == first).all() and imu.samples.count[0] == 0  # kept, the buffers restart
        assert host._reset({"joint_pos": {}}) == "reset"
        assert (imu.turn_on == first).all()
        host.new_session()  # the sensor's power is cycled
        assert not (imu.turn_on == first).any()
    finally:
        host.close()


def test_the_runtime_imu_is_rotated_in_its_mount_as_the_profile_says(tmp_path):
    host, _ = _host(tmp_path)
    try:
        mount = mount_misalignment_rad(load_profile("ethrc_v1"))
        assert mount[1] != 0 and mount[2] != 0
        want = np.array(mount)
        assert np.allclose(host.sim.imu.misalignment[0].cpu().numpy(), want)
        host._reset({"pose": host.start_pose})
        host.new_session()
        assert np.allclose(host.sim.imu.misalignment[0].cpu().numpy(), want)
    finally:
        host.close()


def test_the_published_drive_feedback_is_reported_held_and_delayed(tmp_path):
    host, client = _host(tmp_path)
    try:
        n = len(host.drives)
        m = host._measured
        assert m.reporting is not None and m.sampler is not None
        stamps = set()
        for _ in range(40):
            host.step()
            stamps.add(int(m.sampler.sample_ns[0, 0]))
        assert 3 <= len(stamps) <= 25  # renewed every 3 to 7.5 ms, not every step
        encoders = load_profile("ethrc_v1")["devices"]["encoders"]
        cpr = np.array([encoders[host.desc.drives[d].family]["counts_per_revolution"] for d in host.drives],
                       dtype=np.float64)
        counts = _feedback(client)["drive"]["position_rad"][:n] * cpr / (2 * np.pi)
        assert np.allclose(counts, np.round(counts), atol=2e-3)  # whole encoder counts
    finally:
        host.close()
