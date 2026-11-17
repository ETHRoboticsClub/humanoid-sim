"""BagCapture windows and pins: the span cut to seconds after the bag's start, the sha256, gains judged on the drives."""

import numpy as np
import pytest

pytest.importorskip("rosbags")

from test_bag_reader import DRIVES, PERIOD, T0, _write  # noqa: E402

from humanoid_sim.replay.bag import BagCapture  # noqa: E402


def _bag(path, **kw):
    """A bag whose metadata says it started at T0 (the writer stamps it with the first message's receipt time)."""
    root = _write(path, **kw)
    return root


def test_window_cuts_the_span_and_start_ns_is_the_metadata_start(tmp_path):
    root = _bag(tmp_path / "bag")
    cap = BagCapture(root, drives=DRIVES)
    assert cap.time_span_ns()[0] == T0
    assert abs(cap.start_ns - T0) < 1_000_000_000  # the bag's own start, from metadata.yaml
    lo = (T0 - cap.start_ns) / 1e9 + 0.1
    win = BagCapture(root, drives=DRIVES, window_s=(lo, lo + 0.2))
    assert win.time_span_ns() == (cap.start_ns + round(lo * 1e9), cap.start_ns + round((lo + 0.2) * 1e9))


def test_window_outside_the_recorded_span_is_refused(tmp_path):
    root = _bag(tmp_path / "bag")
    with pytest.raises(ValueError, match="not inside"):
        BagCapture(root, drives=DRIVES, window_s=(-50.0, 0.1)).time_span_ns()
    with pytest.raises(ValueError, match="start must precede"):
        BagCapture(root, window_s=(2.0, 1.0))


def test_gains_are_judged_over_the_window_and_on_the_asked_drives(tmp_path):
    root = _bag(tmp_path / "bag", change_at=50)
    first = (T0 - BagCapture(root).start_ns) / 1e9
    with pytest.raises(ValueError, match="gains changed"):
        BagCapture(root, drives=DRIVES).gains()
    early = BagCapture(root, drives=DRIVES, window_s=(first, first + 49 * PERIOD / 1e9)).gains()
    assert early["maxonHEJ90L1"].stiffness == 320.0
    late = BagCapture(root, drives=DRIVES, window_s=(first + 51 * PERIOD / 1e9, first + 98 * PERIOD / 1e9)).gains()
    assert late["maxonHEJ90L1"].stiffness == 640.0
    # an unpowered hand drive in the gains message does not stop the body drives' gains
    assert BagCapture(_bag(tmp_path / "u", unpowered=True), drives=DRIVES).gains()["maxonHEJ50L_ANK1"].damping == 9.0


def test_sha256_pins_the_contents(tmp_path):
    a = BagCapture(_bag(tmp_path / "a"))
    assert len(a.sha256) == 64 and a.dataset_id == f"bag-{a.sha256[:16]}"
    assert a.sha256 == BagCapture(a.root).sha256
    assert np.array_equal([a.sha256 != BagCapture(_bag(tmp_path / "b", n=90)).sha256], [True])
