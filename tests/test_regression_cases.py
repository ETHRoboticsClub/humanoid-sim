"""Regression cases: the file's schema, the verdicts of the bands, the command conversion, and (data) the pinned bags."""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from humanoid_sim.model.description import Description
from humanoid_sim.replay import regression as R
from humanoid_sim.replay.exc import CommandTrace
from humanoid_sim.replay.metrics import MetricResult, Stat

CASES = Path(__file__).resolve().parents[1] / "regression" / "cases.yaml"
RECORDINGS = Path(os.environ.get("HUMANOID_SIM_RECORDINGS",
                                 Path(__file__).resolve().parents[3] / "data" / "recordings"))


def _stat(mean, n=5):
    return Stat(mean, 0.0, mean, mean, n)


def _case(bands):
    return R.Case("c", "d", {"s3": "s3://x/y", "sha256": "0" * 64, "path": "y"}, (10.0, 20.0), 2.0, {}, bands,
                  support="gantry hang", scoring="open_loop")


def _run(case, results):
    return R.CaseRun(case, {m: MetricResult(m, r, s, 5) for m, (r, s) in results.items()})


def test_the_cases_file_loads_with_its_pins_and_consistent_bands():
    cases = R.load_cases(CASES)
    assert cases.version == 1 and {c.id for c in cases.cases} >= {
        "v11-11400-quiet-stand", "v15-12900-growing-shake", "c10-1.4hz-limit-cycle"}
    for c in cases.cases:
        assert c.bag["s3"].startswith("s3://") and c.bag["s3"].endswith(c.bag["path"])
        assert len(c.bag["sha256"]) == 64 and c.bands and c.sources and c.robot_windows >= 3
        assert c.support == "floor, rope slack" and c.scoring == "closed_loop"  # all three stood on the floor
        assert not any(m.startswith("sim_real_rms.") for m in c.bands)
        for metric, b in c.bands.items():
            assert b["lo"] <= b["ref"] <= b["hi"] or metric.startswith("sim_real_rms"), (c.id, metric, b)
            if metric.startswith("sim_real_rms"):
                assert b["lo"] == 0.0 < b["hi"]
        assert any(m.startswith("tracking_rms.") for m in c.bands)
        assert any(m.startswith("hip_tau_over_target.") for m in c.bands)


def test_cases_file_round_trips_through_dump_and_load(tmp_path):
    cases = R.load_cases(CASES)
    path = tmp_path / "cases.yaml"
    path.write_text(R.dump_cases(cases))
    again = R.load_cases(path)
    assert [c.bands for c in again.cases] == [c.bands for c in cases.cases]
    assert [c.bag for c in again.cases] == [c.bag for c in cases.cases]
    assert again.tolerances == cases.tolerances


@pytest.mark.parametrize("edit,message", [
    (lambda d: d["cases"].append(dict(d["cases"][0])), "duplicate case ids"),
    (lambda d: d["cases"][0].update(window_s=[5.0, 1.0]), "window"),
    (lambda d: d["cases"][0]["bag"].pop("sha256"), "bag needs"),
    (lambda d: d.update(format="other/v1"), "format"),
    (lambda d: d["cases"][0].pop("description"), "lacks"),
])
def test_malformed_cases_files_are_refused(tmp_path, edit, message):
    doc = yaml.safe_load(CASES.read_text())
    edit(doc)
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match=message):
        R.load_cases(path)


def test_judge_passes_inside_the_band_and_fails_outside_with_circular_phase():
    bands = {"tracking_rms.d": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125},
             "hip_q_over_target.3-6Hz.phase_deg": {"ref": 175.0, "lo": 155.0, "hi": 195.0},
             "hip_q_over_target.3-6Hz.gain": {"ref": 1.0, "lo": 0.7, "hi": 1.3}}
    run = _run(_case(bands), {
        "tracking_rms.d": (_stat(0.010), _stat(0.020)),  # sim tracks worse than the band allows
        "hip_q_over_target.3-6Hz.phase_deg": (_stat(175.0), _stat(-175.0)),  # 10 deg away across the wrap: inside
        "hip_q_over_target.3-6Hz.gain": (_stat(1.0), _stat(1.0)),
        "tracking_offset.d": (_stat(0.01), _stat(0.0))})
    v = {x.metric: x.status for x in R.judge(run)}
    assert v == {"tracking_rms.d": "fail", "hip_q_over_target.3-6Hz.phase_deg": "pass",
                 "hip_q_over_target.3-6Hz.gain": "pass", "tracking_offset.d": "info"}
    assert not R.case_passed(R.judge(run)) and R.counts(R.judge(run))["pass"] == 2


def test_a_floor_case_is_not_scored_on_a_prescribed_base_and_a_free_base_would_score_it():
    case = _case({"tracking_rms.a": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125}})
    case.support, case.scoring = "floor, rope slack", "closed_loop"
    run = _run(case, {"tracking_rms.a": (_stat(0.010), _stat(0.050))})
    verdicts = R.judge(run)
    assert [v.status for v in verdicts] == ["skip"] and "closed loop" in verdicts[0].reason
    assert R.case_state(case, verdicts) == "NOT SCORED" and R.case_passed(verdicts)
    assert [v.status for v in R.judge(run, base_prescribed=False)] == ["fail"]
    assert R.case_state(_case({}), []) == "NOT SCORED"
    ok = _run(_case({"tracking_rms.a": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125}}), {"tracking_rms.a": (_stat(0.010), _stat(0.010))})
    assert R.case_state(ok.case, R.judge(ok)) == "PASS"


def test_a_case_needs_its_support_state_and_a_known_scoring(tmp_path):
    doc = yaml.safe_load(CASES.read_text())
    doc["cases"][0].pop("support")
    path = tmp_path / "cases.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="support"):
        R.load_cases(path)
    doc["cases"][0]["support"], doc["cases"][0]["scoring"] = "floor", "free"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="scoring"):
        R.load_cases(path)


def test_judge_skips_what_needs_the_sims_torso_stale_bands_and_missing_values():
    bands = {"imu.pitch_sway_deg": {"ref": 1.0, "lo": 0.5, "hi": 1.5},
             "tracking_rms.a": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125},
             "tracking_rms.b": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125},
             "tracking_rms.c": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125}}
    run = _run(_case(bands), {"imu.pitch_sway_deg": (_stat(1.0), None),
                              "tracking_rms.a": (_stat(0.011), _stat(0.010)),  # the robot moved off its pin
                              "tracking_rms.b": (_stat(0.010), None)})  # the sim has no value; c is not computed
    v = {x.metric: x.status for x in R.judge(run)}
    assert v == {"imu.pitch_sway_deg": "skip", "tracking_rms.a": "stale", "tracking_rms.b": "missing",
                 "tracking_rms.c": "missing"}
    assert "skip" not in {x.status for x in R.judge(run, base_prescribed=False)}
    assert not R.case_passed(R.judge(run))
    assert R.case_passed(R.judge(_run(_case({"imu.pitch_sway_deg": bands["imu.pitch_sway_deg"]}),
                                      {"imu.pitch_sway_deg": (_stat(1.0), None)})))


def test_proposed_bands_follow_the_robot_and_the_stated_tolerances():
    t = R.DEFAULT_TOLERANCES
    run = _run(_case({}), {
        "tracking_rms.d": (_stat(0.010), None),
        "hip_tau_over_target.3-6Hz.gain": (_stat(200.0), None),
        "hip_tau_over_target.3-6Hz.phase_deg": (_stat(-10.0), None),
        "imu.pitch_sway_deg": (_stat(1.0), None),
        "imu.pitch_rate_rms.3-6Hz": (_stat(0.04), None),
        "tracking_offset.d": (_stat(0.05), None)})
    run.results["hip_tau_over_target.6-12Hz.gain"] = MetricResult("x", _stat(5.0), None, 2)  # too few windows
    bands, sources = R.propose_bands(run, t)
    tol = t["tracking_rms"]["rel"] * 0.010 + t["tracking_rms"]["abs"]
    assert bands["tracking_rms.d"] == pytest.approx({"ref": 0.010, "lo": 0.010 - tol, "hi": 0.010 + tol})
    assert bands["sim_real_rms.d"]["hi"] == pytest.approx(0.5 * 0.010 + t["sim_real_rms"]["abs"])
    assert bands["hip_tau_over_target.3-6Hz.gain"]["hi"] == pytest.approx(200.0 * 1.3)
    assert bands["hip_tau_over_target.3-6Hz.phase_deg"] == {"ref": -10.0, "lo": -30.0, "hi": 10.0}
    assert bands["imu.pitch_sway_deg"]["hi"] == pytest.approx(1.0 + 0.3 + 0.05)
    assert bands["imu.pitch_rate_rms.3-6Hz"]["hi"] == pytest.approx(0.04 * 1.3 + 0.005)
    assert "tracking_offset.d" not in bands and "hip_tau_over_target.6-12Hz.gain" not in bands
    assert {"tracking_rms", "sim_real_rms", "response", "imu"} <= set(sources)


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def test_hip_pair_finds_the_flexion_drives_and_their_signs(desc):
    drives = list(desc.drives)
    pair = R.hip_pair(desc, drives)
    left, right = desc.drives[drives[pair.left]], desc.drives[drives[pair.right]]
    assert (left.joints[0], right.joints[0]) == ("hip_flexion_l", "hip_flexion_r")
    assert (pair.sign_left, pair.sign_right) == (float(left.sign), float(right.sign))


class _Bag:
    """Just the COMMAND frames of a bag."""

    def __init__(self, trace):
        self._trace = trace

    def commands(self):
        return self._trace


def test_drive_commands_are_the_frames_in_drive_frame_from_the_first_complete_one(desc):
    import torch

    from humanoid_sim.transmissions.drive_map import DriveJointMap

    drives, joints = list(desc.drives), list(desc.joints)
    rng = np.random.default_rng(1)
    q = 0.05 * rng.standard_normal((6, len(joints)))
    q[:2, joints.index("knee_flexion_l")] = np.nan  # the first frames do not name every joint yet
    trace = CommandTrace(np.arange(6, dtype=np.int64) * 20_000_000 + 1_000, joints, q)
    out = R.drive_commands(_Bag(trace), desc, drives)
    assert out.drives == tuple(drives) and out.reference == "frame"
    assert out.t_ns.tolist() == trace.t_ns[2:].tolist() and out.position_rad.shape == (4, len(drives))
    standing = np.array([desc.joints[j].standing_offset_rad for j in joints])
    expect, _, _ = DriveJointMap(desc, joints, drives, dtype=torch.float64).drive_state(
        torch.as_tensor(q[2:] + standing), torch.zeros(4, len(joints), dtype=torch.float64))
    assert np.allclose(out.position_rad, expect.numpy())
    # a 1:1 drive is sign * (target + standing offset)
    i = drives.index("maxonHEJ90L1")
    d = desc.drives["maxonHEJ90L1"]
    assert out.position_rad[:, i] == pytest.approx(
        d.sign * (q[2:, joints.index(d.joints[0])] + desc.joints[d.joints[0]].standing_offset_rad))
    with pytest.raises(ValueError, match="never name"):
        R.drive_commands(_Bag(CommandTrace(trace.t_ns, joints[:-1], q[:, :-1])), desc, drives)


@pytest.mark.data
@pytest.mark.skipif(not RECORDINGS.exists(), reason="robot recordings not available (HUMANOID_SIM_RECORDINGS)")
def test_pinned_bags_match_their_sha256_and_the_robots_pinned_values():
    """Each case's bag is the pinned one, and the robot's values on its window are the ones the bands were set from
    (a changed bag, window or metric makes the bands stale)."""
    cases = R.load_cases(CASES)
    desc = Description.load("ethrc_v1")
    drives = list(desc.drives)
    seen = 0
    for case in cases.cases:
        if not (RECORDINGS / case.bag["path"] / "metadata.yaml").is_file():
            continue
        seen += 1
        run = R.robot_only(case, R.resolve_bag(case, RECORDINGS), desc, drives)
        verdicts = R.judge(run)
        stale = [v.metric for v in verdicts if v.status == "stale"]
        assert not stale, (case.id, stale[:5])
        bands, _ = R.propose_bands(run, cases.tolerances)
        assert set(bands) == set(case.bands)
    if not seen:
        pytest.skip("none of the cases' bags is under the recordings directory")


def test_stale_is_judged_on_all_the_robots_windows_and_sim_minus_robot_needs_no_robot_value():
    bands = {"tracking_rms.a": {"ref": 0.010, "lo": 0.0075, "hi": 0.0125},
             "sim_real_rms.a": {"ref": 0.0, "lo": 0.0, "hi": 0.006}}
    run = _run(_case(bands), {"tracking_rms.a": (_stat(0.0105), _stat(0.010)),  # fewer common windows: another mean
                              "sim_real_rms.a": (None, _stat(0.004))})
    run.results["tracking_rms.a"].robot_all = _stat(0.010)
    assert {v.metric: v.status for v in R.judge(run)} == {"tracking_rms.a": "pass", "sim_real_rms.a": "pass"}
