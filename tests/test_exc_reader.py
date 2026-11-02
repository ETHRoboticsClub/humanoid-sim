import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.replay.exc import ExcitationCapture, sample_hold, sample_linear

ROOT = Path(os.environ.get("HUMANOID_SIM_DATASETS", Path(__file__).resolve().parents[2] / "datasets/humanoid-v1"))
CAPTURES = sorted(ROOT.glob("exc-*")) if ROOT.exists() else []


def test_gains_leave_out_actuators_without_si_gains(tmp_path):
    """The robot API lists the hands' actuators with null SI gains; the body's drives are read as before."""
    import json

    from humanoid_sim.replay.exc import FORMAT

    (tmp_path / "manifest.json").write_text(json.dumps({"format": FORMAT, "complete": True, "dataset_id": "exc-x"}))
    body = {"stiffness_nm_per_rad": 100.0, "damping_nm_s_per_rad": 2.0, "raw_p_gain": 100000, "raw_d_gain": 2000}
    hand = {"stiffness_nm_per_rad": None, "damping_nm_s_per_rad": None, "raw_p_gain": 0, "raw_d_gain": 0}
    actuators = {"maxonHEJ70AL1": body, "forearm_pronation_l": hand}
    (tmp_path / "gains.json").write_text(json.dumps({"start": {"actuators": actuators}, "end": {"actuators": actuators}}))
    gains = ExcitationCapture(tmp_path).gains()
    assert list(gains) == ["maxonHEJ70AL1"] and gains["maxonHEJ70AL1"].stiffness == 100.0


def test_sampling_helpers():
    t = np.array([0, 10, 20], dtype=np.int64)
    v = np.array([0.0, 1.0, 3.0])
    assert sample_hold(t, v, np.array([-5, 9, 10, 25])).tolist() == [0.0, 0.0, 1.0, 3.0]
    assert sample_linear(t, v, np.array([5, 15])).tolist() == [0.5, 2.0]


@pytest.mark.data
@pytest.mark.skipif(not CAPTURES, reason="exc-* captures not available")
def test_captures_load_and_match_description():
    from humanoid_sim.model.description import Description

    desc = Description.load()
    for root in CAPTURES:
        cap = ExcitationCapture(root)
        pdo = cap.pdo()
        assert set(pdo) == set(desc.drives)
        for tr in pdo.values():
            assert np.all(np.diff(tr.t_ns) > 0)
            assert np.isfinite(tr["position_demand_rad"]).all()
        gains = cap.gains()
        # The captures ran the description's baseline gains.
        for name, d in desc.drives.items():
            assert gains[name].stiffness == pytest.approx(d.baseline_stiffness)
            assert gains[name].damping == pytest.approx(d.baseline_damping)


@pytest.mark.data
@pytest.mark.skipif(not CAPTURES, reason="exc-* captures not available")
def test_imu_stream_is_base_frame_and_bursty():
    from humanoid_sim.replay.base import BaseMotion, sample_times_ns

    for root in CAPTURES:
        cap = ExcitationCapture(root)
        imu = cap.imu()
        f = imu.specific_force.mean(axis=0)
        assert f[2] == pytest.approx(9.8, abs=0.1) and np.hypot(f[0], f[1]) < 0.2  # base z up, at rest-ish
        ts = sample_times_ns(imu)
        assert np.all(np.diff(ts) > 0) and np.all(ts <= imu.t_ns)
        assert (imu.t_ns - ts).max() < 25_000_000  # never older than a burst period plus jitter
        a, b = cap.time_span_ns()
        bm = BaseMotion.from_imu(imu, np.arange(a, b, 4_000_000))
        assert np.ptp(bm.angles[:, 0]) > np.radians(5)  # the gantry turns the base
        assert not bm.angles[:, 1:].any()
