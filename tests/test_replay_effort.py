"""Independent current/effort fixtures, not robot calibration evidence."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.replay.effort import CurrentConversion, EffortCfg, prepare_effort, sample_current
from humanoid_sim.replay.exc import PDO_FIELDS, DriveTrace, ExcitationCapture


def fixture(tmp_path, *, gains=(2.0, -3.0), zeros=(0.5, -0.25)):
    desc = SimpleNamespace(digest="fixture-description", drives={"direct": None, "ankle-crank": None})
    t = 1_800_000_000_000_000_000 + np.arange(9, dtype=np.int64) * 500_000_000
    traces = {d: DriveTrace(d, t, {"current_a": np.full(9, 1.5), "current_demand_a": np.full(9, 12.0)})
              for d in desc.drives}
    cap = SimpleNamespace(dataset_id="synthetic", pdo=lambda: traces)
    doc = {"schema": "ethrc-effort-calibration/v1", "description_digest": desc.digest,
           "current_signal": "current_a", "torque_reference": "drive-output-equivalent",
           "status": "synthetic", "source": "declared fixture values", "assumptions": "ideal algebra only",
           "drives": {d: {"gain_nm_per_a": g, "zero_a": z}
                      for d, g, z in zip(desc.drives, gains, zeros, strict=True)}}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(doc))
    cfg = EffortCfg("current", path, max_gap_ns=500_000_000)
    return desc, cap, traces, cfg


def test_conversion_order_sign_offset_and_inverse(tmp_path):
    desc, _, _, cfg = fixture(tmp_path)
    conv = CurrentConversion.load(cfg.calibration, desc, ["ankle-crank", "direct"])
    np.testing.assert_array_equal(conv.torque(np.array([[1.5, 1.5], [-1.25, -0.5]])),
                                  [[-5.25, 2], [3, -2]])
    np.testing.assert_array_equal(conv.equivalent_current(np.array([[-5.25, 2]])), [[1.5, 1.5]])
    assert len(conv.metadata["sha256"]) == 64 and conv.metadata["status"] == "synthetic"


@pytest.mark.parametrize("bad", ["schema", "description_digest", "current_signal", "torque_reference", "source", "assumptions", "status", "gain", "missing"])
def test_no_guessed_calibration_fallback(tmp_path, bad):
    desc, _, _, cfg = fixture(tmp_path)
    doc = json.loads(cfg.calibration.read_text())
    if bad == "gain":
        doc["drives"]["direct"]["gain_nm_per_a"] = 0
    elif bad == "missing":
        del doc["drives"]["direct"]
    else:
        doc[bad] = ""
    cfg.calibration.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        CurrentConversion.load(cfg.calibration, desc, list(desc.drives))


def test_epoch_timing_exact_samples_gaps_and_invalid_neighbors():
    epoch = 1_800_000_000_000_000_000
    t = epoch + np.array([0, 10, 20, 100, 110], dtype=np.int64)
    values = np.array([0., 2., np.nan, 8., 10.])
    q = epoch + np.array([-1, 0, 5, 10, 15, 20, 30, 100, 105, 110, 111], dtype=np.int64)
    actual, mask = sample_current(t, values, q, 20)
    np.testing.assert_array_equal(mask, [False, True, True, True, False, False, False, True, True, True, False])
    np.testing.assert_array_equal(actual[mask], [0, 1, 2, 8, 9, 10])
    # Even finite endpoints cannot bridge the long gap.
    values[2] = 3
    assert not sample_current(t, values, np.array([epoch + 30], dtype=np.int64), 20)[1][0]


def test_compare_uses_actual_applied_pair_and_independent_metrics(tmp_path):
    desc, cap, traces, cfg = fixture(tmp_path)
    prepared = prepare_effort(cap, desc, list(desc.drives), cfg)
    t = traces["direct"].t_ns
    # Independent expected reference: [2*(1.5-.5), -3*(1.5+.25)].
    applied = np.tile([3.0, -7.25], (len(t), 1))
    arrays, report = prepared.compare(t, applied)
    assert report["drives"]["direct"]["rms_torque_nm"] == 1
    assert report["drives"]["ankle-crank"]["mean_torque_error_nm"] == -2
    assert report["drives"]["direct"]["actual"]["valid_samples"] == 7
    assert report["drives"]["direct"]["actual"]["covered_duration_ns"] == 3_000_000_000
    np.testing.assert_array_equal(arrays["recorded_current_demand_a"], 12)
    np.testing.assert_array_equal(arrays["current_derived_torque_nm"][0], [2, -5.25])
    traces["direct"].fields["current_a"][:] = 100
    # Prepared input has its own snapshot and cannot follow later source mutation.
    assert prepared.actual[0][0] == 1.5


def test_partial_current_never_changes_interval_or_zero_fills(tmp_path):
    desc, cap, traces, cfg = fixture(tmp_path)
    traces["direct"].fields["current_a"][4] = np.nan
    del traces["direct"].fields["current_demand_a"]
    prepared = prepare_effort(cap, desc, ["direct"], cfg)
    t = traces["direct"].t_ns
    arrays, report = prepared.compare(t, np.full((9, 1), 2.0))
    assert len(arrays["t_ns"]) == 9
    assert np.isnan(arrays["current_derived_torque_nm"][4, 0])
    assert report["drives"]["direct"]["demand"]["valid_samples"] == 0
    assert report["drives"]["direct"]["actual"]["valid_samples"] == 6
    strict = prepare_effort(cap, desc, ["direct"], EffortCfg("current", cfg.calibration, 500_000_000, 1.0))
    with pytest.raises(ValueError, match="coverage"):
        strict.compare(t, np.full((9, 1), 2.0))


def test_raw_mode_has_no_fabricated_current_or_torque_model(tmp_path):
    desc, cap, traces, _ = fixture(tmp_path)
    prepared = prepare_effort(cap, desc, ["direct"], EffortCfg(max_gap_ns=500_000_000))
    arrays, report = prepared.compare(traces["direct"].t_ns, np.zeros((9, 1)))
    assert "current_derived_torque_nm" not in arrays
    assert "rms_torque_nm" not in report["drives"]["direct"]
    assert report["metadata"]["calibration"] is None


def test_missing_actual_and_empty_score_are_refused(tmp_path):
    desc, cap, traces, cfg = fixture(tmp_path)
    prepared = prepare_effort(cap, desc, ["direct"], cfg)
    with pytest.raises(ValueError, match="warmup"):
        prepared.compare(traces["direct"].t_ns[:2], np.zeros((2, 1)))
    traces["direct"].fields["current_a"][:] = np.nan
    with pytest.raises(ValueError, match="no finite actual"):
        prepare_effort(cap, desc, ["direct"], cfg)


def test_csv_optional_currents_keep_position_reader_usable(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "ethrc-excitation/v8", "complete": True}))
    columns = [f for f in PDO_FIELDS if f not in ("current_a", "current_demand_a")]
    with (tmp_path / "pdo_feedback.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["device", "t_source_ros_ns", *columns])
        writer.writerows([["direct", t, *([0] * len(columns))] for t in (20, 10, 10)])
    trace = ExcitationCapture(tmp_path).pdo()["direct"]
    np.testing.assert_array_equal(trace.t_ns, [10, 20])
    assert np.isnan(trace["current_a"]).all() and np.isnan(trace["current_demand_a"]).all()


@pytest.mark.parametrize("present", [True, False])
def test_bag_uses_source_time_and_preserves_optional_actual_demand(present):
    from humanoid_sim.replay.bag import FEEDBACK_FIELDS, FEEDBACK_TOPIC, BagCapture

    messages = []
    for stamp in (20, 10, 10):
        values = {field: [0.0] for field in FEEDBACK_FIELDS.values()}
        for field, val in (("current_a", 2.0), ("current_demand_a", -5.0)):
            if present:
                values[field] = [val]
            else:
                del values[field]
        messages.append(SimpleNamespace(device=["direct"], source_stamp_ns=[stamp], **values))
    bag = BagCapture.__new__(BagCapture)
    bag.root, bag._drives = "synthetic-bag", None
    bag.__dict__["_messages"] = {FEEDBACK_TOPIC: messages}
    trace = bag.pdo()["direct"]
    np.testing.assert_array_equal(trace.t_ns, [10, 20])
    if present:
        np.testing.assert_array_equal(trace["current_a"], 2)
        np.testing.assert_array_equal(trace["current_demand_a"], -5)
    else:
        assert np.isnan(trace["current_a"]).all()
