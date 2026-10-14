"""Physics on the built robot asset (Newton + MuJoCo C, CPU). Needs the physics extra and a built asset."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("newton")
pytest.importorskip("pxr")
pytestmark = pytest.mark.physics

from humanoid_sim.devices.demand import DriveDemand  # noqa: E402
from humanoid_sim.model.asset import asset_dir, load_profile  # noqa: E402
from humanoid_sim.model.description import Description  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"


@pytest.fixture(scope="module")
def usd():
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    import json

    return out / json.loads((out / "asset.json").read_text())["usd"]


def test_holds_standing_on_fixed_base(usd):
    from humanoid_sim.sim import SimCore

    sim = SimCore("newton", usd, fixed_base=True)
    q, qd = sim.joint_state()
    p = sim.plant
    qdrv, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.jvpt(qdrv, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    for _ in range(250):
        fb = sim.step()
    q1, _ = sim.joint_state()
    assert (q1 - q).abs().max().item() < 0.006  # gravity sag at baseline gains
    assert torch.isfinite(fb.applied_torque_nm).all()


def test_model_mass_matches_description_urdf(usd):
    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.joint_properties import joint_properties
    from humanoid_sim.physics.newton_direct import NewtonCfg, NewtonRobot

    desc = Description.load()
    plant = PlantCore(desc, list(desc.joints), 1, PlantCfg())
    robot = NewtonRobot(usd, joint_properties(plant, list(desc.joints)), NewtonCfg(kinematics_only=True))
    assert float(robot.model.body_mass.numpy().sum()) == pytest.approx(58.684, abs=0.01)
    assert robot.model.body_count == 23
