"""Fitting the sweeps an excitation capture finished: the opt-in, which sweeps count and the frequency cut."""

import json
import math

import numpy as np
import pytest

from humanoid_sim.replay.exc import FORMAT, ExcitationCapture

# A small schedule of the manifest's shape: three sweeps of 2 s with 0.5 s rests, 0.1 -> 7 Hz.
F0, F1, SWEEP_S, GAP_S, SCALES = 0.1, 7.0, 2.0, 0.5, [1.0, 0.6, 0.3]
T0 = 1_000_000_000_000  # the ROS stamp of signal time 0
FRAME_S = 0.02


def write_capture(root, *, last_signal_s: float, complete: bool) -> None:
    """Frames every 20 ms of signal time up to ``last_signal_s`` (after an anchor frame at signal 0 a second
    earlier), one drive's feedback every 5 ms over the same span."""
    signals = [{"joint_id": j, "f0_hz": F0, "f1_hz": F1, "duration_s": SWEEP_S, "sequence_gap_s": GAP_S,
                "start_offset_ns": 0, "sequence_scales": SCALES} for j in ("knee_flexion_l", "knee_flexion_r")]
    (root / "manifest.json").write_text(json.dumps({"format": FORMAT, "complete": complete, "dataset_id": "exc-x",
                                                    "signals": signals}))
    rows = ["seq,t_frame_ros_ns,t_signal_s,joint_id,q_nominal_rad,q_mirror_rad"]
    frames = [(T0 - 1_000_000_000, 0.0)]
    frames += [(T0 + round(s * 1e9), s) for s in np.arange(0.0, last_signal_s + 1e-9, FRAME_S)]
    for seq, (t, s) in enumerate(frames, start=1):
        for j in ("knee_flexion_l", "knee_flexion_r"):
            rows.append(f"{seq},{t},{s:.6f},{j},{s:.6f},{s:.6f}")
    (root / "commands.csv").write_text("\n".join(rows) + "\n")
    pdo = ["device,t_source_ros_ns,native_position_rad,velocity_rad_s,torque_nm,current_a,position_demand_rad,"
           "current_demand_a,motor_position_rad"]
    for t in range(frames[0][0], frames[-1][0] + 1, 5_000_000):
        pdo.append(f"m1,{t},0,0,0,0,0,0,0")
    (root / "pdo_feedback.csv").write_text("\n".join(pdo) + "\n")


def signal_of(t_ros_ns) -> np.ndarray:
    return (np.asarray(t_ros_ns) - T0) / 1e9


def test_an_incomplete_capture_is_refused_unless_opened_for_its_finished_sweeps(tmp_path):
    write_capture(tmp_path, last_signal_s=2.68, complete=False)  # into the rest after sweep 0
    with pytest.raises(ValueError, match="not complete"):
        ExcitationCapture(tmp_path)
    (only,) = ExcitationCapture(tmp_path, allow_incomplete=True).sequences()
    assert only.index == 0 and only.dataset_id == "exc-x#sequence0"
    assert only.provenance["capture_complete"] is False and only.provenance["sequence_scale"] == 1.0
    # From the capture's start (the anchor frame included) to the end of the sweep, not into the rest.
    assert only.commands().t_ns[0] == T0 - 1_000_000_000
    assert signal_of(only.commands().t_ns[-1]) == pytest.approx(SWEEP_S)
    assert signal_of(only.pdo()["m1"].t_ns[-1]) <= SWEEP_S


def test_a_later_sweep_starts_at_its_own_start_with_the_frame_in_force(tmp_path):
    write_capture(tmp_path, last_signal_s=3 * SWEEP_S + 2 * GAP_S, complete=True)
    windows = ExcitationCapture(tmp_path).sequences()
    assert [w.index for w in windows] == [0, 1, 2] and [w.scale for w in windows] == SCALES
    second = windows[1]
    start = SWEEP_S + GAP_S
    frames = signal_of(second.commands().t_ns)
    assert frames[0] == pytest.approx(start - FRAME_S)  # the frame in force when the window opens
    assert frames[1] == pytest.approx(start) and frames[-1] == pytest.approx(start + SWEEP_S)
    feedback = signal_of(second.pdo()["m1"].t_ns)
    assert feedback[0] >= start - 1e-9 and feedback[-1] <= start + SWEEP_S + 1e-9


def test_a_capture_that_finished_no_sweep_has_nothing_to_fit(tmp_path):
    write_capture(tmp_path, last_signal_s=1.5, complete=False)
    with pytest.raises(ValueError, match="finished none"):
        ExcitationCapture(tmp_path, allow_incomplete=True).sequences()


def test_the_frequency_cut_ends_each_sweep_where_its_chirp_reaches_it(tmp_path):
    write_capture(tmp_path, last_signal_s=3 * SWEEP_S + 2 * GAP_S, complete=True)
    top = 1.0
    for w in ExcitationCapture(tmp_path).sequences(max_frequency_hz=top):
        start = w.index * (SWEEP_S + GAP_S)
        local = signal_of(w.commands().t_ns)[-1] - start
        # Controls' chirp: f(t) = f0 (f1/f0)^(t/T). The last frame kept is at or below the cut, the next above it.
        assert F0 * (F1 / F0) ** (local / SWEEP_S) <= top * (1 + 1e-9)
        assert F0 * (F1 / F0) ** ((local + FRAME_S) / SWEEP_S) > top
        assert w.provenance["max_frequency_hz"] == top
        assert w.provenance["top_frequency_hz"] == pytest.approx(top)
    with pytest.raises(ValueError, match="not above"):
        ExcitationCapture(tmp_path).sequences(max_frequency_hz=F0 / 2)


def test_joints_on_different_schedules_cannot_be_cut_into_sweeps(tmp_path):
    write_capture(tmp_path, last_signal_s=2.68, complete=False)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["signals"][1]["start_offset_ns"] = math.floor(0.25e9)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different schedules"):
        ExcitationCapture(tmp_path, allow_incomplete=True).sequences()
