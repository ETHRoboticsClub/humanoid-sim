"""Demand-delay scan: each env replays the capture with its demand series shifted by its own delay."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.sysid.delay import drive_regions

ROOT = Path(os.environ.get("HUMANOID_SIM_DATASETS", Path(__file__).resolve().parents[2] / "datasets/humanoid-v1"))
CAPTURES = sorted(ROOT.glob("exc-*")) if ROOT.exists() else []
ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"


def test_regions_cover_every_drive_once():
    desc = Description.load()
    drives = list(desc.drives)
    regions = drive_regions(desc, drives)
    assert sorted(i for idx in regions.values() for i in idx) == list(range(len(drives)))
    assert len(regions["ankle"]) == 4


@pytest.mark.physics
@pytest.mark.data
@pytest.mark.skipif(not CAPTURES, reason="exc-* captures not available")
def test_delayed_demand_series_and_scan(request):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.replay.exc import ExcitationCapture, sample_linear
    from humanoid_sim.replay.replay import _prepare
    from humanoid_sim.sim import SimCore
    from humanoid_sim.sysid.delay import scan_delay

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    delays = np.array([-4_000_000, 0, 8_000_000])
    sim = SimCore(usd, device="cpu", num_envs=len(delays), fixed_base=True)
    request.addfinalizer(sim.close)
    cap = ExcitationCapture(CAPTURES[0])
    run = _prepare(cap, sim, 1.0, None, delays)
    drive = sim.plant.map.drive_names[0]
    pdo = cap.pdo()[drive]
    for e, d in enumerate(delays):
        expect = sample_linear(pdo.t_ns, pdo["position_demand_rad"], run.t_grid - d)
        np.testing.assert_allclose(run.demand[:, e, 0], expect)
    result = scan_delay(sim, [cap], delays, base_motion=None, duration_s=1.5, skip_s=0.5)
    rms = np.array(result["rms_rad"])
    assert rms.shape == (3, len(sim.plant.map.drive_names)) and np.isfinite(rms).all()
    assert set(result["regions"]) >= {"ankle"}
