"""Camera frame delivery (sensors/delivery.py) on CPU: each frame's latency is drawn in its range and frames keep their
order, losses come at their rate with a stereo pair lost together, the draws repeat per seed and leave each other and
the imaging alone, and a session hands a frame to its feed only once it has arrived, stamped with its exposure."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.sensors import feeds, rigs
from humanoid_sim.sensors.delivery import (
    FrameDelivery,
    check_overrides,
    delivery_settings,
    latency_from_ms,
)
from humanoid_sim.sensors.geometry import camera_spec
from humanoid_sim.sensors.imaging import CAMERA_EFFECTS, ImagingChain, check_effects, parse_effects
from humanoid_sim.sensors.models import OakDW
from humanoid_sim.sensors.session import SensorSession

PERIOD_NS = 33_333_333
MS = 1_000_000


def oak(**kw) -> OakDW:
    """The V1's OAK-D-W (its calibration only) with ``kw`` replacing its instance arguments."""
    c = rigs.ethrc_v1().sensors["camera0"]
    args = dict(width=c.width, height=c.height, frame_period_ns=c.frame_period_ns, intrinsics=c.intrinsics,
                distortion=c.distortion, stream_offsets_m=c.stream_offsets_m,
                stream_rotations_xyzw=c.stream_rotations_xyzw, depth_focal_px=c.depth_focal_px) | kw
    camera = OakDW(c.frame, **args)
    camera.name = "camera0"
    return camera


SPEC = camera_spec(oak())
STREAMS = [s.name for s in SPEC.streams]  # rgb, mono_left, mono_right, depth


def run(delivery: FrameDelivery, frames: int, start_ns: int = 1_000_000_000) -> dict[str, list]:
    """Each stream's (stamp, arrival) over ``frames`` exposures one period apart."""
    out = {s: [] for s in STREAMS}
    for i in range(frames):
        stamp = start_ns + i * PERIOD_NS
        for s in STREAMS:
            out[s].append((stamp, delivery.decide(s, stamp)))
    return out


def test_an_instance_checks_its_latency_and_drop_probability():
    camera = oak()
    assert camera.latency_ns == (0, 0) and camera.drop_probability == 0.0
    assert oak(latency_ns=(5 * MS, 9 * MS), drop_probability=0.25).latency_ns == (5 * MS, 9 * MS)
    for bad in [(9 * MS, 5 * MS), (-1, 5), (1.5, 2), (1,), (True, 2)]:
        with pytest.raises(ValueError, match="latency_ns"):
            oak(latency_ns=bad)
    for bad in [-0.1, 1.5, float("nan"), True]:
        with pytest.raises(ValueError, match="drop_probability"):
            oak(drop_probability=bad)


def test_the_delivery_effects_are_camera_effects():
    assert {"latency", "drops"} <= set(CAMERA_EFFECTS) == set(parse_effects("all"))
    assert check_effects(["drops", "noise", "latency"]) == ("noise", "latency", "drops")


def test_an_override_needs_its_effect_and_a_valid_value():
    check_overrides(("latency", "drops"), (MS, 2 * MS), 0.5)
    check_overrides((), None, None)
    with pytest.raises(ValueError, match="'latency' camera effect is off"):
        check_overrides(("drops",), (MS, 2 * MS), None)
    with pytest.raises(ValueError, match="'drops' camera effect is off"):
        check_overrides(("latency",), None, 0.1)
    with pytest.raises(ValueError, match="probability"):
        check_overrides(("drops",), None, 2.0)
    assert latency_from_ms(None) is None
    assert latency_from_ms([30]) == (30 * MS, 30 * MS)
    assert latency_from_ms([7.5, 25]) == (7_500_000, 25 * MS)
    for bad in ([20, 10], [1, 2, 3], [-1]):
        with pytest.raises(ValueError):
            latency_from_ms(bad)


def test_the_settings_are_the_instances_its_overrides_or_none_when_off():
    camera = oak(latency_ns=(5 * MS, 9 * MS), drop_probability=0.25)
    assert delivery_settings(camera, ()) == ((0, 0), 0.0)
    assert delivery_settings(camera, ("latency",)) == ((5 * MS, 9 * MS), 0.0)
    assert delivery_settings(camera, ("drops",)) == ((0, 0), 0.25)
    assert delivery_settings(camera, ("latency", "drops"), (MS, MS), 0.5) == ((MS, MS), 0.5)


def test_a_fixed_latency_delivers_every_frame_that_much_later():
    for s, frames in run(FrameDelivery(SPEC, (20 * MS, 20 * MS), 0.0, 0), 50).items():
        assert all(arrival == stamp + 20 * MS for stamp, arrival in frames), s


def test_the_latency_is_drawn_in_its_range_and_frames_keep_their_order():
    low, high = 10 * MS, 80 * MS  # longer than a period apart: draws would overtake
    frames = run(FrameDelivery(SPEC, (low, high), 0.0, 3), 3000)
    for s, pairs in frames.items():
        stamps, arrivals = (np.array(v, dtype=np.int64) for v in zip(*pairs, strict=True))
        assert (np.diff(arrivals) >= 0).all(), s
        assert (arrivals >= stamps + low).all() and (arrivals - stamps).min() < low + MS
        assert (arrivals - stamps).max() <= high + PERIOD_NS  # held behind an earlier frame at most
    # Below a period of spread nothing is held back: the latencies are the uniform draws themselves.
    latency = np.array([a - t for t, a in run(FrameDelivery(SPEC, (low, low + 30 * MS), 0.0, 4), 4000)["rgb"]])
    assert latency.min() >= low and latency.max() <= low + 30 * MS
    assert abs(latency.mean() - (low + 15 * MS)) < 0.5 * MS  # sigma of the mean: 0.14 ms
    assert abs(latency.std() - 30 * MS / math.sqrt(12)) < 0.5 * MS


def test_frames_are_lost_at_their_rate_a_stereo_pair_together():
    p, n = 0.2, 5000
    d = FrameDelivery(SPEC, (0, 0), p, 1)
    frames = run(d, n)
    lost = {s: np.array([a is None for _, a in frames[s]]) for s in STREAMS}
    assert (lost["mono_left"] == lost["mono_right"]).all()
    assert not (lost["rgb"] == lost["mono_left"]).all() and not (lost["rgb"] == lost["depth"]).all()
    sigma = math.sqrt(n * p * (1 - p))
    for s in ("rgb", "mono_left", "depth"):
        assert abs(lost[s].sum() - n * p) < 4 * sigma, s
    assert d.frames == 4 * n and d.dropped == sum(int(v.sum()) for v in lost.values())
    assert all(a is not None for f in run(FrameDelivery(SPEC, (0, 0), 0.0, 1), 200).values() for _, a in f)
    assert all(a is None for f in run(FrameDelivery(SPEC, (0, 0), 1.0, 1), 200).values() for _, a in f)


def test_the_draws_repeat_per_seed_and_losses_leave_the_latencies_alone():
    def make(p, seed=7, camera_id=None):
        spec = SPEC if camera_id is None else SimpleNamespace(camera_id=camera_id, streams=SPEC.streams)
        return run(FrameDelivery(spec, (5 * MS, 25 * MS), p, seed), 300)

    assert make(0.3) == make(0.3)
    assert make(0.3) != make(0.3, seed=8) and make(0.3) != make(0.3, camera_id="other")
    kept, lossy = make(0.0), make(0.3)
    for s in STREAMS:
        for (_, a), (_, b) in zip(kept[s], lossy[s], strict=True):
            assert b is None or a == b  # 20 ms of spread within a 33 ms period: no frame is held back


def test_delivery_alone_leaves_the_imaging_off():
    import torch

    camera = oak()
    plain = ImagingChain(camera, SPEC, (), 0)
    delivered = ImagingChain(camera, SPEC, ("latency", "drops"), 0)
    assert delivered.effects == ()
    hdr = torch.full((8, 8, 3), 0.18)
    rgb = next(s for s in SPEC.streams if s.name == "rgb")
    assert np.array_equal(plain.image(rgb, hdr), delivered.image(rgb, hdr))


@pytest.mark.parametrize("make", [rigs.ethrc_v1, rigs.g1_cameras_d435])
def test_every_rig_camera_can_be_delivered(make):
    for camera in make().sensors.values():
        if camera.kind == "camera":
            latency, drops = delivery_settings(camera, ("latency", "drops"))
            d = FrameDelivery(camera_spec(camera), latency, drops, 0)
            assert set(d.group) == set(camera.streams)


# -- the session's delivery to its feeds ----------------------------------------------------------------------------

def session(monkeypatch, latency_ns, drop_probability=0.0):
    """A session of the stereo pair of :data:`SPEC` without Kit: the renders are blank, the feed records what it is
    handed and when."""
    monkeypatch.setattr(feeds.time, "time", lambda: 90.0)
    published = []
    s = SensorSession.__new__(SensorSession)
    stereo = SimpleNamespace(camera_id=SPEC.camera_id,
                             streams=[st for st in SPEC.streams if st.name in ("mono_left", "mono_right")])
    s.cfg = SimpleNamespace(images=None, depth_noise=False)
    s.sim = SimpleNamespace(t_ns=1_000_000_000)
    s.clock = None
    s.recorder = None
    s.camera_specs = [stereo]
    s.cameras = SimpleNamespace(image=lambda *_: np.zeros((2, 2), dtype=np.uint8))
    s.feeds = {SPEC.camera_id: SimpleNamespace(publish=lambda name, stamp, _image:
                                               published.append((name, stamp, s.sim.t_ns)))}
    s.imaging = {}
    s.delivery = {SPEC.camera_id: FrameDelivery(stereo, latency_ns, drop_probability, 0)}
    s.last_delivery, s._pending, s.last_images, s._saved = {}, [], {}, {}
    import itertools

    s._order = itertools.count()
    s.counts = {"camera_frames": 0, "camera_frames_dropped": 0}
    s.ms = {"camera_read": [], "publish": []}
    return s, published


def test_a_frame_goes_to_its_feed_when_it_arrives_stamped_with_its_exposure(monkeypatch):
    s, published = session(monkeypatch, (20 * MS, 20 * MS))
    exposure = s.sim.t_ns
    s._cameras(exposure)
    assert published == [] and s.last_delivery[(SPEC.camera_id, "mono_left")] == (exposure, exposure + 20 * MS)
    s.sim.t_ns += 19 * MS
    s.deliver(s.sim.t_ns)
    assert published == []
    s.sim.t_ns += 6 * MS  # 5 ms after the frames arrived: the renderer was late
    s.deliver(s.sim.t_ns)
    assert [name for name, _, _ in published] == ["mono_left", "mono_right"]
    assert all(stamp == pytest.approx(90.0 - 0.025) for _, stamp, _ in published)  # the exposure, 25 ms ago
    late = s.delivery[SPEC.camera_id].describe()
    assert late["delivered"] == 2 and late["late_ms"]["max"] == pytest.approx(5.0)
    s.deliver(s.sim.t_ns)
    assert len(published) == 2


def test_a_lost_frame_never_reaches_its_feed(monkeypatch):
    s, published = session(monkeypatch, (0, 0), drop_probability=1.0)
    s._cameras(s.sim.t_ns)
    s.deliver(s.sim.t_ns + 100 * MS)
    assert published == [] and s.counts["camera_frames_dropped"] == 2 and s.counts["camera_frames"] == 2
    assert s.last_delivery[(SPEC.camera_id, "mono_right")] == (s.sim.t_ns, None)


def test_frames_on_their_way_are_forgotten_with_their_session(monkeypatch):
    s, published = session(monkeypatch, (20 * MS, 20 * MS))
    s._cameras(s.sim.t_ns)
    s.drop_pending()
    s.deliver(s.sim.t_ns + 100 * MS)
    assert published == []


def test_render_runtime_takes_the_latency_and_drop_options(monkeypatch):
    from humanoid_sim import cli
    from humanoid_sim.runtime import render_workers

    seen = []
    monkeypatch.setattr(render_workers, "render_runtime_workers", lambda _segment, cfg, **_kw: seen.append(cfg))
    assert cli.main(["render-runtime", "seg", "--camera-effects", "latency,drops", "--camera-latency-ms", "7.5", "25",
                     "--camera-drop-probability", "0.1"]) == 0
    assert seen[-1].camera_effects == ("latency", "drops")
    assert seen[-1].camera_latency_ns == (7_500_000, 25 * MS) and seen[-1].camera_drop_probability == 0.1
    assert cli.main(["render-runtime", "seg", "--camera-effects", "all"]) == 0
    assert seen[-1].camera_latency_ns is None and seen[-1].camera_drop_probability is None
    with pytest.raises(ValueError, match="'drops' camera effect is off"):
        cli.main(["render-runtime", "seg", "--camera-effects", "latency", "--camera-drop-probability", "0.1"])
