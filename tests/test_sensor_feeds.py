"""The simulated camera feed (``sensors/feeds.py``) read back by Controls' own reference reader
(``ethrc_cameras.sim_feed``, from the checkout ``HUMANOID_SIM_CONTROLS`` names): the descriptor validates, the rings
open with the headers the descriptor implies, and every published frame comes back with its stamp."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from humanoid_sim.model.description import Description
from humanoid_sim.sensors import feeds, rigs
from humanoid_sim.sensors.geometry import camera_spec

G1 = "g1_29dof_with_hand_rev_1_0"
CAMERAS = "g1-cameras-d435"  # sensors/rigs.py EXTRAS
SPECS = [camera_spec(s) for s in rigs.load(G1, Description.load(G1), CAMERAS).sensors.values()
         if s.kind == "camera"]


@pytest.fixture
def sim_feed(controls_cameras):
    return importlib.import_module("ethrc_cameras.sim_feed")


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.camera_id)
def test_controls_reads_the_device(tmp_path, sim_feed, spec):
    feed = feeds.CameraFeed(tmp_path, spec, capacity=4)
    rng = np.random.default_rng(1)
    sent = {}
    for n in range(6):  # more than the capacity: the reader sees the newest entries
        for s in spec.streams:
            h, w = s.lens.height_px, s.lens.width_px
            image = (rng.integers(0, 65535, (h, w), dtype=np.uint16) if s.kind == "depth"
                     else rng.integers(0, 255, (h, w) if s.kind == "mono" else (h, w, 3), dtype=np.uint8))
            feed.publish(s.name, 1000.0 + n * 0.033, image)
            sent[(s.name, n)] = image
    d = sim_feed.read_descriptor(tmp_path / spec.camera_id)
    assert (d.identity.camera_id, d.identity.product, d.session) == (spec.camera_id, spec.product, feed.session)
    assert {fs.spec.name: fs.spec.shutter for fs in d.streams} == {s.name: s.shutter for s in spec.streams}
    for s in spec.streams:
        cal = d.calibration.intrinsics[s.name]
        assert (cal.width_px, cal.height_px, cal.fx_px, cal.cy_px) == (s.lens.width_px, s.lens.height_px,
                                                                      s.lens.fx_px, s.lens.cy_px)
        assert d.calibration.sensor_poses[s.name].translation_m == pytest.approx(s.translation_m)
        stream = next(fs for fs in d.streams if fs.spec.name == s.name)
        reader = sim_feed.open_ring(tmp_path / spec.camera_id / f"{s.name}.ring",
                                    sim_feed.ring_header(stream, d.session, 4))
        assert reader.published() == 6
        entry = reader.entry(5)
        assert entry.stamp_s == pytest.approx(1000.0 + 5 * 0.033)
        image = sent[(s.name, 5)]
        expect = image[..., ::-1] if s.encoding == "bgr8" else image
        assert entry.payload == np.ascontiguousarray(expect).tobytes()
        assert reader.entry(1) is None  # overwritten
        reader.close()
    feed.close()
    assert not (tmp_path / spec.camera_id / "descriptor.json").exists()


def test_depth_is_millimetres_within_range():
    d = np.array([[0.1, 0.5, 2.3456], [np.inf, 11.0, np.nan]])
    np.testing.assert_array_equal(feeds.depth_mm(d, (0.28, 10.0)), [[0, 500, 2346], [0, 0, 0]])
