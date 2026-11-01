"""A lidar sweep is one RTX render's output: one emitter state, traced at one pose, stamped at that render."""
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.sensors.lidar import VALID, RtxLidar

N = 20


def lidar():
    lid = object.__new__(RtxLidar)
    lid.fire_ns = np.arange(N, dtype=np.int64) * 5_000_000
    lid.period_ns = 100_000_000
    lid._rtx = SimpleNamespace(parse_generic_model_output_data=lambda g: g)
    lid._init_reads()
    return lid


def output(emitters, state, pos=(1.0, 2.0, 3.0), r=2.0):
    """An RTX output: ``emitters`` in report order, each at the azimuth ``state * 100 + emitter`` (deg/100)."""
    e = np.asarray(emitters)
    return SimpleNamespace(size=1, numElements=len(e), emitterId=e, x=(state * 100 + e) / 100.0, y=np.zeros(len(e)),
                           z=np.full(len(e), r), flags=np.full(len(e), VALID, np.uint8),
                           frameEnd=SimpleNamespace(posM=pos, orientation=(0.0, 0.0, 0.0, 1.0)))


def feed(lid, out):
    lid.runtime = SimpleNamespace(get_data=lambda _: (out, None))


def test_an_output_that_wraps_mid_pattern_is_one_sweep_of_its_own_state_pose_and_render_time():
    lid = lidar()
    split = 8  # the sensor clock's sweep boundary falls inside the render: emitters 8.. then 0..7
    feed(lid, output(list(range(split, N)) + list(range(split)), state=3, pos=(1.0, 2.0, 3.0)))
    (sweep,) = lid.read(1_000_000_000)
    assert sweep.start_ns == 1_000_000_000
    np.testing.assert_array_equal(sweep.time_s, 0.0)
    assert sweep.hit.all() and len(sweep.points_m) == N
    az = np.degrees(np.arctan2(sweep.points_m[:, 1], sweep.points_m[:, 0]))
    np.testing.assert_allclose(az * 100, 300 + np.arange(N), atol=1e-3)  # by emitter, every ray of state 3
    np.testing.assert_allclose(sweep.origin_w, (1.0, 2.0, 3.0))
    feed(lid, output(list(range(split, N)) + list(range(split)), state=4, pos=(5.0, 0.0, 0.0)))
    (nxt,) = lid.read(1_100_000_000)
    az = np.degrees(np.arctan2(nxt.points_m[:, 1], nxt.points_m[:, 0]))
    np.testing.assert_allclose(az * 100, 400 + np.arange(N), atol=1e-3)  # no ray of the previous render
    np.testing.assert_allclose(nxt.origin_w, (5.0, 0.0, 0.0))


def test_a_ray_reported_twice_on_the_boundary_is_kept_once():
    lid = lidar()
    feed(lid, output(list(range(N)) + [0], state=1))
    (sweep,) = lid.read(0)
    assert len(sweep.points_m) == N


def test_partial_outputs_while_the_sensor_starts_are_dropped_and_counted():
    lid = lidar()
    feed(lid, output(range(5, N), state=0))
    assert lid.read(0) == [] and lid.dropped == 1
    feed(lid, output(range(N), state=1))
    assert len(lid.read(100_000_000)) == 1


def test_a_partial_output_after_a_whole_one_is_refused():
    lid = lidar()
    feed(lid, output(range(N), state=0))
    lid.read(0)
    feed(lid, output(range(N // 3), state=1))  # renders faster than the sweep period
    with pytest.raises(RuntimeError, match="once per sweep period"):
        lid.read(33_000_000)


def test_an_unknown_emitter_is_refused():
    lid = lidar()
    feed(lid, output(list(range(N)) + [N], state=0))
    with pytest.raises(ValueError, match="emitters"):
        lid.read(0)


def test_a_missed_ray_is_zero_and_not_a_hit():
    lid = lidar()
    out = output(range(N), state=0)
    out.flags[3] = 0
    feed(lid, out)
    (sweep,) = lid.read(0)
    assert not sweep.hit[3] and not sweep.points_m[3].any() and sweep.hit.sum() == N - 1
