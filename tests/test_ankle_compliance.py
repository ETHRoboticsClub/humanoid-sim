"""Ankle linkage compliance: the crank springs' statics, the asset variant that carries them, and their knobs."""
import json

import numpy as np
import pytest
import torch

from humanoid_sim.model.asset import asset_dir, build_asset, compliance_springs, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.randomization import ParamSpec, Randomizer
from humanoid_sim.transmissions import compliance


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def _poses(desc, n=64, seed=0):
    """Jacobians of every ankle at poses spread around its linkage neutral, float64."""
    generator = torch.Generator().manual_seed(seed)
    out = []
    for linkage in desc.linkages:
        tr = desc.ankle_transmission(linkage.side, dtype=torch.float64)
        q = tr.neutral + (torch.rand(n, 2, generator=generator, dtype=torch.float64) - 0.5) * 0.4
        out.append(tr.jacobian(q))
    return torch.cat(out)


def test_spring_statics_agree_from_the_load_and_from_the_drive_torques(desc):
    jac = _poses(desc)
    generator = torch.Generator().manual_seed(1)
    k = 100.0 + 2000.0 * torch.rand(len(jac), 2, generator=generator, dtype=torch.float64)
    load = 20.0 * (torch.rand(len(jac), 2, generator=generator, dtype=torch.float64) - 0.5)
    # The cranks hold the hinge load: J^T tau_d + tau = 0.
    drive = -torch.linalg.solve(jac.transpose(-1, -2), load.unsqueeze(-1)).squeeze(-1)
    from_load = compliance.deflection_under_load(jac, k, load)
    torch.testing.assert_close(compliance.deflection_from_drive_torque(jac, k, drive), from_load)
    # The deflection does work against the load (C is positive definite) and undoes the foot stiffness.
    assert ((from_load * load).sum(-1) > 0).all()
    torch.testing.assert_close(compliance.foot_stiffness(jac, k) @ from_load.unsqueeze(-1), load.unsqueeze(-1))
    c = compliance.foot_compliance(jac, k)
    torch.testing.assert_close(c, c.transpose(-1, -2))


def test_no_load_no_deflection_and_stiff_springs_are_rigid(desc):
    jac = _poses(desc)
    k = torch.full((len(jac), 2), 500.0, dtype=torch.float64)
    assert compliance.deflection_under_load(jac, k, torch.zeros(len(jac), 2, dtype=torch.float64)).abs().max() == 0
    load = torch.full((len(jac), 2), 10.0, dtype=torch.float64)
    soft = compliance.deflection_under_load(jac, k, load).abs().max()
    stiff = compliance.deflection_under_load(jac, k * 1e6, load).abs().max()
    assert soft > 1e-3 and stiff == pytest.approx(float(soft) * 1e-6, rel=1e-9)


def test_equal_crank_stiffness_gives_the_pitch_stiffness_and_the_split_only_couples(desc):
    jac = _poses(desc, n=8)
    k = compliance.equal_crank_stiffness(jac, 800.0)
    assert (k[..., 0] == k[..., 1]).all()
    torch.testing.assert_close(compliance.foot_stiffness(jac, k)[..., 0, 0], torch.full((len(jac),), 800.0,
                                                                                         dtype=torch.float64))
    # Moving stiffness from one crank to the other keeps the diagonal near where it was and couples pitch and roll.
    split = k * torch.tensor([1.5, 0.5], dtype=torch.float64)
    equal, uneven = compliance.foot_stiffness(jac, k), compliance.foot_stiffness(jac, split)
    torch.testing.assert_close(torch.diagonal(uneven, dim1=-2, dim2=-1), torch.diagonal(equal, dim1=-2, dim2=-1),
                               rtol=0.1, atol=0)
    assert (uneven[..., 0, 1].abs() > 5 * equal[..., 0, 1].abs().clamp(min=1.0)).all()


def test_every_crank_drive_has_one_spring(desc):
    springs = compliance.spring_joints(desc)
    cranks = [d for d, drive in desc.drives.items() if drive.crank is not None]
    assert list(springs) == cranks
    from humanoid_sim.model.usd import crank_joints

    assert springs == {d: crank_joints(desc)[d] + compliance.SPRING_SUFFIX for d in cranks}


def test_rigid_asset_is_untouched_and_the_variant_adds_one_spring_per_crank(desc, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from humanoid_sim.model.usd import crank_joints

    profile = load_profile(desc.robot)
    rigid = build_asset(desc, profile, tmp_path)
    assert rigid.parent == asset_dir(desc, profile, tmp_path)
    assert asset_dir(desc, profile, tmp_path, ankle_compliance=False) == rigid.parent
    variant = build_asset(desc, profile, tmp_path, ankle_compliance=True)
    assert variant.parent == asset_dir(desc, profile, tmp_path, ankle_compliance=True) != rigid.parent

    rigid_stage = Usd.Stage.Open(str(rigid))
    assert not [p for p in rigid_stage.Traverse()
                if p.GetName().endswith((compliance.SPRING_SUFFIX, compliance.OUTPUT_SUFFIX))]
    assert "ankle_compliance" not in json.loads((rigid.parent / "asset.json").read_text())
    assert compliance_springs(rigid) == {}

    stage = Usd.Stage.Open(str(variant))
    springs = compliance_springs(variant)
    assert springs == compliance.spring_joints(desc)
    prims = {p.GetName(): p for p in stage.Traverse()}
    xforms = UsdGeom.XformCache()

    def pose(name):
        return np.asarray(xforms.GetLocalToWorldTransform(prims[name])).T

    def frame(joint, side):
        pos = getattr(joint, f"GetLocalPos{side}Attr")().Get()
        rot = getattr(joint, f"GetLocalRot{side}Attr")().Get()
        m = Gf.Matrix4d(1.0)
        m.SetRotate(Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary())))
        m.SetTranslateOnly(Gf.Vec3d(*pos))
        return np.asarray(m).T

    for drive, spring in springs.items():
        crank = UsdPhysics.RevoluteJoint(prims[crank_joints(desc)[drive]])
        source = desc.usd_model.joints[crank_joints(desc)[drive]]
        shank, arm = desc.usd_model.joint_bodies(crank.GetPrim().GetName())
        output = compliance.output_body(crank.GetPrim().GetName())
        # The drive's joint keeps its shank, frames, axis and stops, and now turns the output body.
        assert [t.name for t in crank.GetBody0Rel().GetTargets()] == [shank]
        assert [t.name for t in crank.GetBody1Rel().GetTargets()] == [output]
        assert crank.GetLocalPos1Attr().Get() == source.GetLocalPos1Attr().Get()
        assert prims[output].HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(prims[output]).GetMassAttr().Get() > 0
        # The output sits where the crank arm sits, so both joints' frames coincide at zero spring angle.
        np.testing.assert_allclose(pose(output), pose(arm), atol=1e-9)
        joint = UsdPhysics.RevoluteJoint(prims[spring])
        assert [t.name for t in joint.GetBody0Rel().GetTargets()] == [output]
        assert [t.name for t in joint.GetBody1Rel().GetTargets()] == [arm]
        assert joint.GetAxisAttr().Get() == crank.GetAxisAttr().Get()
        assert not joint.GetExcludeFromArticulationAttr().Get()
        world = [pose(output) @ frame(joint, 0), pose(arm) @ frame(joint, 1), pose(output) @ frame(crank, 1)]
        for other in world[1:]:
            np.testing.assert_allclose(other, world[0], atol=1e-6)
    # Every delivered closure is still there.
    closures = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                and UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
    assert closures == {n for n, j in desc.usd_model.joints.items() if j.GetExcludeFromArticulationAttr().Get()}


def test_spring_gains_come_from_the_profile_and_are_required(tmp_path):
    from humanoid_sim.physics.isaaclab import spring_gains

    usd = tmp_path / "robot.usda"
    usd.write_text("")
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda", "ankle_compliance": {
        "springs": {"A": "a_spring", "B": "b_spring"}}}))
    physics = {"ankle_compliance": {"stiffness_nm_per_rad": 800.0, "damping_nm_s_per_rad": {"A": 0.5, "B": 0.25}}}
    assert spring_gains(usd, physics) == {"a_spring": (800.0, 0.5), "b_spring": (800.0, 0.25)}
    for bad in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="stiffness_nm_per_rad"):
            spring_gains(usd, {"ankle_compliance": {"stiffness_nm_per_rad": bad, "damping_nm_s_per_rad": 0.5}})
    with pytest.raises(ValueError, match="damping"):
        spring_gains(usd, {"ankle_compliance": {"stiffness_nm_per_rad": 800.0}})
    # A rigid asset needs nothing, and the defaults ask for nothing.
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda"}))
    assert spring_gains(usd, load_profile("ethrc_v1")["physics"]) == {}


def test_the_defaults_keep_the_linkage_rigid():
    knobs = load_profile("ethrc_v1")["physics"]["ankle_compliance"]
    assert knobs == {"stiffness_nm_per_rad": None, "damping_nm_s_per_rad": None}


def test_spring_knobs_select_cranks_and_draw_per_env():
    spec = ParamSpec.from_dict({"param": "ankle_spring_stiffness", "cranks": ".*L_.*", "dist": "log_uniform",
                                "range": [200.0, 2000.0], "op": "set"})
    assert (spec.kind, spec.startup, spec.select) == ("physics", True, ".*L_.*")
    with pytest.raises(ValueError, match="does not apply to a crank"):
        ParamSpec.from_dict({"param": "ankle_spring_damping", "drives": ".*", "dist": "uniform",
                             "range": [0.1, 1.0], "op": "set"})
    cranks = ["HEJ50L_ANK1", "HEJ50L_ANK2", "HEJ50R_ANK1", "HEJ50R_ANK2"]
    nominal = {"ankle_spring_stiffness": torch.full((32, 4), 1000.0)}
    r = Randomizer([spec], nominal, {"crank": cranks}, num_envs=32)
    v = r.values["ankle_spring_stiffness"]
    assert ((v[:, :2] >= 200.0) & (v[:, :2] <= 2000.0)).all() and (v[:, 2:] == 1000.0).all()
    assert v[:, 0].unique().numel() > 1
