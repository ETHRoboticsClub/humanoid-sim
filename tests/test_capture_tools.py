"""convert-capture, segment-capture and pace-report on small synthetic captures and fits (no physics)."""

import csv
import json

import numpy as np
import pytest

from humanoid_sim.model.description import Description
from humanoid_sim.replay.convert import convert_capture, is_legacy, joint_map, legacy_table
from humanoid_sim.replay.exc import FORMAT, ExcitationCapture
from humanoid_sim.replay.segments import Window, cut_window, plan_windows

T0 = 1_793_210_373_000_000_000
PERIOD = 20_000_000  # 50 Hz command frames


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def write_capture(root, joints, q_of, n=200, *, sweep=None, gap_after=None):
    """A minimal exc-* capture: command frames naming ``joints`` with ``q_of(k, joint)``, one PDO row per frame."""
    root.mkdir(parents=True)
    action = {"sweep": sweep} if sweep else {}
    (root / "manifest.json").write_text(json.dumps({"format": FORMAT, "complete": True, "dataset_id": root.name,
                                                   "effective_action": action}))
    stamps = [T0 + k * PERIOD + (int(5e9) if gap_after is not None and k > gap_after else 0) for k in range(n)]
    signal = [k * PERIOD / 1e9 if gap_after is None or k <= gap_after else (k - gap_after - 1) * PERIOD / 1e9
              for k in range(n)]
    with (root / "commands.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "t_frame_ros_ns", "t_signal_s", "joint_id", "q_nominal_rad", "q_mirror_rad"])
        for k in range(n):
            for j in joints:
                w.writerow([k, stamps[k], signal[k], j, repr(float(q_of(k, j))), repr(float(q_of(k, j)))])
    with (root / "pdo_feedback.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "t_source_ros_ns", "native_position_rad", "position_demand_rad"])
        for k in range(n):
            w.writerow(["maxonHEJ90M1", stamps[k] + 1000, 0.0, 0.0])
    (root / "gains.json").write_text("{}")
    (root / "notes.zip").write_text("not copied")
    return root


def legacy_q(k, j):
    """Old command values: small sines, the ankles inside their reach."""
    amp = 0.05 if j.startswith("foot_") else 0.2
    return amp * np.sin(0.05 * k + sum(map(ord, j)) % 7)


def test_legacy_table_is_controls_urdf6():
    old, path = legacy_table("urdf6-b14158e")
    assert old["urdf"] == "urdf/URDF_6.urdf" and path.name == "urdf6-b14158e-actuators.yaml"
    with pytest.raises(ValueError, match="no legacy description"):
        legacy_table("urdf9")


def test_joint_map_matches_every_drive_and_flips_the_waist(desc):
    jm = joint_map(desc, legacy_table("urdf6-b14158e")[0])
    assert {nj for nj, _, _ in jm.linear.values()} == {j for j, v in desc.joints.items() if v.device}
    assert jm.linear["waist_upperbody_yaw"][:2] == ("torso_axial_rotation", -1.0)
    assert jm.old_ankles == {"l": ("foot_pitch_l", "foot_roll_l"), "r": ("foot_pitch_r", "foot_roll_r")}


def test_conversion_keeps_every_drive_target(desc, tmp_path):
    old = legacy_table("urdf6-b14158e")[0]
    src = write_capture(tmp_path / "exc-a", list(old["joints"]), legacy_q)
    assert is_legacy(src, desc)
    out = convert_capture(src, tmp_path / "v1", desc)
    assert not is_legacy(out, desc) and not (out / "notes.zip").exists()
    report = json.loads((out / "conversion.json").read_text())
    assert report["drive_targets"]["frames"] == 200
    assert max(report["drive_targets"]["max_abs_rad"].values()) < 1e-9
    cmd = ExcitationCapture(out).commands()
    assert set(cmd.joints) == set(desc.joints)
    waist = cmd.positions_rad[:, cmd.joints.index("torso_axial_rotation")]
    assert np.allclose(waist, [-legacy_q(k, "waist_upperbody_yaw") for k in range(200)])
    with pytest.raises(FileExistsError):
        convert_capture(src, tmp_path / "v1", desc)


def test_hold_keeps_the_ankles_at_the_first_frame(desc, tmp_path):
    old = legacy_table("urdf6-b14158e")[0]
    src = write_capture(tmp_path / "exc-b", list(old["joints"]), legacy_q)
    out = convert_capture(src, tmp_path / "v1", desc, ankles="hold")
    cmd = ExcitationCapture(out).commands()
    for linkage in desc.linkages:
        q = cmd.positions_rad[:, [cmd.joints.index(j) for j in linkage.joints]]
        assert np.ptp(q, axis=0).max() < 1e-12
    report = json.loads((out / "conversion.json").read_text())
    assert report["ankles"] == "hold" and report["drive_targets"]["max_abs_rad"]["maxonHEJ90M1"] < 1e-9


def test_conversion_refuses_unknown_joints(desc, tmp_path):
    old = legacy_table("urdf6-b14158e")[0]
    src = write_capture(tmp_path / "exc-c", [*old["joints"], "tail"], legacy_q, n=5)
    with pytest.raises(ValueError, match="tail"):
        convert_capture(src, tmp_path / "v1", desc)


def test_a_log_chirp_is_cut_into_equal_octave_bands(desc, tmp_path):
    src = write_capture(tmp_path / "exc-d", list(desc.joints), lambda k, j: 0.0, n=1500,
                        sweep={"f0_hz": 0.2, "f1_hz": 10.0, "duration": 30_000_000_000})
    windows = plan_windows(src, bands=3)
    assert [w.name for w in windows] == ["b0", "b1", "b2"]
    assert [round(w.start_s, 3) for w in windows] == [0.0, 10.0, 20.0] and windows[-1].end_s == pytest.approx(30.0)
    assert windows[0].note == "log chirp 0.2-0.74 Hz"
    out = cut_window(src, tmp_path / "seg", windows[1])
    cap = ExcitationCapture(out)
    assert cap.dataset_id == "exc-d-b1" and len(cap.commands().t_ns) == 500
    assert json.loads((out / "manifest.json").read_text())["segment"]["of"] == "exc-d"


def test_pieces_split_at_a_pause_and_short_windows_are_dropped(desc, tmp_path):
    src = write_capture(tmp_path / "exc-e", list(desc.joints), lambda k, j: 0.0, n=1000, gap_after=599)
    windows = plan_windows(src, min_s=5.0)
    assert [w.name for w in windows] == ["p0", "p1"] and windows[1].start_s == pytest.approx(17.0)
    assert plan_windows(src, min_s=10.0) == [windows[0]]
    explicit = [Window("x", 1.0, 2.0, "explicit")]
    assert plan_windows(src, explicit=explicit) == explicit


def test_report_scores_holdouts_by_group_and_flags_bounds(desc, tmp_path):
    from humanoid_sim.sysid.pace_report import render, summarize

    drives = ["maxonHEJ70AL1", "maxonHEJ70AR1", "maxonHEJ50L_ANK1"]
    joints = ["shoulder_flexion_l", "shoulder_flexion_r"]
    vals = {j: {"armature": 0.3, "damping": 1.0, "friction": 0.2, "friction_ratio": 0.5} for j in joints}
    for name, xa in (("fit-b0", 0.99), ("fit-b1", 0.1)):
        d = tmp_path / name
        d.mkdir()
        (d / "fit-pace.json").write_text(json.dumps({
            "drive_order": drives, "joint_order": joints, "captures": [name[-2:]],
            "objective_start": 0.02, "objective": 0.01, "x": [xa, 0.0, 0.0],
            "names": ["armature.shoulder_flexion_l", "damping.shoulder_flexion_l", "bias.maxonHEJ70AL1"],
            "free": ["armature.shoulder_flexion_l", "damping.shoulder_flexion_l"],
            "values": {"joints": vals, "drives": {dr: {"bias_rad": 0.001} for dr in drives},
                       "delay_ns": {"HEJ70-48-50": 5_000_000}},
            "start": {j: {"armature": 0.34, "damping": 1.0, "friction": 0.2, "friction_ratio": 1.0} for j in joints},
            "scored": drives[:2],
            "holdout": {"captures": ["other"], "rms_start": [[0.02, 0.04, 0.5]], "rms_fit": [[0.01, 0.02, 0.5]]}}))
    s = summarize([tmp_path / "fit-b0", tmp_path / "fit-b1"], desc)
    assert s["fits"]["fit-b0"]["at_bound"] == ["armature.shoulder_flexion_l (upper)"]
    assert s["fits"]["fit-b1"]["at_bound"] == []
    assert s["holdout_by_group"]["shoulder_flexion"] == {"before_mrad": pytest.approx(30.0),
                                                         "after_mrad": pytest.approx(15.0), "improved": 2, "of": 2}
    assert s["fits"]["fit-b0"]["holdout"]["other"]["scored_mrad"] == (pytest.approx(30.0), pytest.approx(15.0))
    assert s["table"]["shoulder_flexion_l"]["datasheet_armature"] == 0.0055
    text = render(s)
    assert "| shoulder_flexion | 30.00 | 15.00 | -50% | 2/2 |" in text and "maxonHEJ50L_ANK1" in text
