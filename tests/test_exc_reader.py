import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.replay.exc import ExcitationCapture, sample_hold, sample_linear

ROOT = Path(os.environ.get("HUMANOID_SIM_DATASETS", Path(__file__).resolve().parents[2] / "datasets/humanoid-v1"))
CAPTURES = sorted(ROOT.glob("exc-*")) if ROOT.exists() else []


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
