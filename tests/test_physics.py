"""Physics on the built robot asset through Isaac Lab / MJWarp (Warp CPU device). Runs in the Isaac Lab image."""

import json
from pathlib import Path

import pytest
import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"


@pytest.fixture(scope="module")
def usd():
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def test_kinematic_model_mass_and_bodies(usd):
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    from humanoid_sim.model.kinematics import RobotKinematics

    kin = RobotKinematics(usd, list(Description.load().joints))
    assert float(kin.model.body_mass.numpy().sum()) == pytest.approx(58.684, abs=0.01)
    assert kin.model.body_count == 23


@pytest.mark.physics
def test_holds_standing_on_fixed_base(usd):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.sim import SimCore

    sim = SimCore(usd, device="cpu", fixed_base=True)
    q, qd = sim.joint_state()
    p = sim.plant
    qdrv, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.jvpt(qdrv, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    for _ in range(100):
        fb = sim.step()
    q1, _ = sim.joint_state()
    assert (q1 - q).abs().max().item() < 0.006  # gravity sag at baseline gains
    assert torch.isfinite(fb.applied_torque_nm).all()
