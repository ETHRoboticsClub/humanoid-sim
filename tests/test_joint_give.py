"""Joint give: a spring between a drive joint's encoder side and its link: the asset variant that carries it, where its
gains come from, its knobs and its statics."""
import json

import numpy as np
import pytest
import torch

from humanoid_sim.model.asset import (
    asset_dir,
    build_asset,
    compliance_springs,
    default_give,
    give_springs,
    load_profile,
)
from humanoid_sim.model.description import Description
from humanoid_sim.randomization import ParamSpec, Randomizer
from humanoid_sim.transmissions import compliance

# Test inputs: the joints a check gives give to, by pattern (a leg's hip flexion and abduction, knee flexion).
GIVE = ("hip_flexion_.*", "hip_abduction_.*", "knee_flexion_.*")


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


@pytest.fixture(scope="module")
def names(desc):
    return compliance.resolve_give(desc, GIVE)


def test_patterns_select_joints_in_the_descriptions_order_and_only_drive_joints(desc, names):
    assert len(names) >= 3 and list(names) == [n for n in desc.joints if n in names]
    assert compliance.resolve_give(desc, [names[0]]) == (names[0],)
    assert compliance.resolve_give(desc, [names[0], names[0], names[-1]]) == (names[0], names[-1])
    with pytest.raises(ValueError, match="selects no joint"):
        compliance.resolve_give(desc, ["no_such_joint"])
    # An ankle hinge is moved by cranks, not by a drive of its own: it has no give.
    hinge = desc.linkages[0].joints[0]
    assert hinge in desc.joints and hinge not in compliance.give_candidates(desc)
    with pytest.raises(ValueError, match="selects no joint"):
        compliance.resolve_give(desc, [hinge])
    pytest.importorskip("pxr")
    from humanoid_sim.model.usd import _checked_give

    with pytest.raises(ValueError, match="cannot carry give"):
        _checked_give(desc, [hinge])


def test_the_static_deflection_is_the_load_over_the_stiffness():
    k = torch.tensor([300.0, 1500.0, 40.0])
    load = torch.tensor([6.0, -15.0, 0.0])
    torch.testing.assert_close(compliance.give_deflection(load, k), torch.tensor([0.02, -0.01, 0.0]))


def test_the_description_can_carry_a_joints_give_on_the_joint_or_its_family():
    desc = Description.load("ethrc_v1")
    joint = next(iter(desc.table["joints"]))
    family = desc.table["joints"][joint]["family"]
    other = next(n for n, j in desc.table["joints"].items() if j["family"] == family and n != joint)
    desc.table["joints"][joint]["give_stiffness_nm_per_rad"] = 700.0
    desc.table["families"][family]["give_damping_nm_s_per_rad"] = 2.5
    desc.table["families"][family]["give_stiffness_nm_per_rad"] = 900.0
    assert (desc.joints[joint].give_stiffness, desc.joints[joint].give_damping) == (700.0, 2.5)
    assert (desc.joints[other].give_stiffness, desc.joints[other].give_damping) == (900.0, 2.5)
    for bad in (0.0, -1.0, float("nan")):
        broken = Description.load("ethrc_v1")
        broken.table["joints"][joint]["give_damping_nm_s_per_rad"] = bad
        with pytest.raises(ValueError, match="give_damping_nm_s_per_rad"):
            broken.joints  # noqa: B018


def test_the_description_as_delivered_asks_for_nothing_that_is_not_there():
    desc = Description.load("ethrc_v1")
    assert all(j.give_stiffness is None or j.give_stiffness > 0 for j in desc.joints.values())
    assert all(j.give_damping is None or j.give_damping > 0 for j in desc.joints.values())


def test_the_rigid_asset_is_untouched_and_the_variant_adds_one_give_per_joint(desc, names, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    profile = load_profile(desc.robot)
    rigid = build_asset(desc, profile, tmp_path, joint_give=())
    assert rigid.parent == asset_dir(desc, profile, tmp_path, joint_give=())
    # Without joints named, an asset carries the profile's joint give (the V1's hips and knees).
    assert asset_dir(desc, profile, tmp_path) == asset_dir(desc, profile, tmp_path,
                                                           joint_give=default_give(desc, profile)) != rigid.parent
    variant = build_asset(desc, profile, tmp_path, joint_give=names)
    assert variant.parent == asset_dir(desc, profile, tmp_path, joint_give=names) != rigid.parent
    # The order of the names does not matter; a different set is a different asset.
    assert asset_dir(desc, profile, tmp_path, joint_give=names[::-1]) == variant.parent
    assert asset_dir(desc, profile, tmp_path, joint_give=names[:1]) not in (variant.parent, rigid.parent)
    # It is its own asset: neither the crank springs' nor the rigid one's.
    assert asset_dir(desc, profile, tmp_path, ankle_compliance=True, joint_give=names) not in (variant.parent,
                                                                                                rigid.parent)

    rigid_stage = Usd.Stage.Open(str(rigid))
    assert not [p for p in rigid_stage.Traverse() if p.GetName().endswith((compliance.GIVE_SUFFIX,
                                                                          compliance.OUTPUT_SUFFIX))]
    assert "joint_give" not in json.loads((rigid.parent / "asset.json").read_text())
    assert give_springs(rigid) == {} and compliance_springs(variant) == {}

    stage = Usd.Stage.Open(str(variant))
    springs = give_springs(variant)
    assert springs == {n: compliance.give_joint(n) for n in names}
    prims = {p.GetName(): p for p in stage.Traverse()}
    rigid_prims = {p.GetName(): p for p in rigid_stage.Traverse()}
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

    def ends(joint):
        return [[t.name for t in rel.GetTargets()] for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]

    def linked(on, body):
        """The joints of a stage that have ``body`` at one end."""
        return {p.GetName() for p in on.Traverse() if p.IsA(UsdPhysics.Joint)
                and body in [t.name for rel in (UsdPhysics.Joint(p).GetBody0Rel(), UsdPhysics.Joint(p).GetBody1Rel())
                             for t in rel.GetTargets()]}

    for name, give in springs.items():
        source = desc.usd_model.joints[name]
        parent, link = desc.usd_model.joint_bodies(name)
        output = compliance.output_body(name)
        drive = UsdPhysics.RevoluteJoint(prims[name])
        # The drive's joint keeps its name, parent, frames, axis and stops, and now turns the output body.
        assert ends(drive) == [[parent], [output]]
        assert drive.GetLocalPos1Attr().Get() == source.GetLocalPos1Attr().Get()
        assert drive.GetAxisAttr().Get() == UsdPhysics.RevoluteJoint(source.GetPrim()).GetAxisAttr().Get()
        rigid_drive = UsdPhysics.RevoluteJoint(rigid_prims[name])
        assert [drive.GetLowerLimitAttr().Get(), drive.GetUpperLimitAttr().Get()] == [
            rigid_drive.GetLowerLimitAttr().Get(), rigid_drive.GetUpperLimitAttr().Get()]
        assert prims[output].HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(prims[output]).GetMassAttr().Get() > 0
        # The output sits where the link sits, so every joint's frames coincide at zero give angle.
        np.testing.assert_allclose(pose(output), pose(link), atol=1e-9)
        joint = UsdPhysics.RevoluteJoint(prims[give])
        assert ends(joint) == [[output], [link]]
        assert joint.GetAxisAttr().Get() == drive.GetAxisAttr().Get()
        assert not joint.GetExcludeFromArticulationAttr().Get()
        world = [pose(output) @ frame(joint, 0), pose(link) @ frame(joint, 1), pose(output) @ frame(drive, 1)]
        for other in world[1:]:
            np.testing.assert_allclose(other, world[0], atol=1e-6)
        # The link keeps its other joints.
        assert linked(stage, link) - {give, name} == linked(rigid_stage, link) - {name}
    meta = json.loads((variant.parent / "asset.json").read_text())
    # The pair each joint kept apart before the output body came between them stays out of contact with each other.
    excluded = {frozenset(path.split("/")[-1] for path in pair) for pair in meta["self_collision_excludes"]}
    for name in names:
        assert frozenset(desc.usd_model.joint_bodies(name)) in excluded, name
    # Every delivered closure and the ankle's rods are still there.
    closures = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                and UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
    assert closures == {n for n, j in desc.usd_model.joints.items() if j.GetExcludeFromArticulationAttr().Get()}
    # Every joint the give does not touch is as the rigid asset has it.
    rigid_joints = {p.GetName(): ends(UsdPhysics.Joint(p)) for p in rigid_stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Joint) and p.GetName() in rigid_joints and p.GetName() not in names:
            assert ends(UsdPhysics.Joint(p)) == rigid_joints[p.GetName()], p.GetName()


def test_give_gains_come_from_the_profile_then_the_description_and_are_required(tmp_path):
    from humanoid_sim.physics.isaaclab import compliance_gains, give_gains

    desc = Description.load("ethrc_v1")
    joints = [n for n in compliance.give_candidates(desc)][:3]
    usd = tmp_path / "robot.usda"
    usd.write_text("")
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda", "joint_give": {
        "springs": {n: compliance.give_joint(n) for n in joints}}}))
    a, b, c = joints
    physics = {"joint_give": {"stiffness_nm_per_rad": 800.0, "damping_nm_s_per_rad": {a: 0.5, b: 0.25, c: 4.0}}}
    assert give_gains(usd, physics, desc) == {compliance.give_joint(a): (800.0, 0.5),
                                              compliance.give_joint(b): (800.0, 0.25),
                                              compliance.give_joint(c): (800.0, 4.0)}
    # A joint the profile leaves out takes the description's, and the profile wins where it gives one.
    described = Description.load("ethrc_v1")
    described.table["joints"][b]["give_stiffness_nm_per_rad"] = 1200.0
    described.table["joints"][b]["give_damping_nm_s_per_rad"] = 3.0
    described.table["joints"][c]["give_stiffness_nm_per_rad"] = 1300.0
    partial = {"joint_give": {"stiffness_nm_per_rad": {a: 800.0, c: None}, "damping_nm_s_per_rad": {a: 0.5, c: 1.0}}}
    assert give_gains(usd, partial, described) == {compliance.give_joint(a): (800.0, 0.5),
                                                   compliance.give_joint(b): (1200.0, 3.0),
                                                   compliance.give_joint(c): (1300.0, 1.0)}
    # Neither source: refused, naming the joint.
    for bad in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="stiffness_nm_per_rad"):
            give_gains(usd, {"joint_give": {"stiffness_nm_per_rad": bad, "damping_nm_s_per_rad": 0.5}}, desc)
    with pytest.raises(ValueError, match="damping"):
        give_gains(usd, {"joint_give": {"stiffness_nm_per_rad": 800.0}}, desc)
    assert compliance_gains(usd, physics, desc) == give_gains(usd, physics, desc)
    # An asset without give needs nothing.
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda"}))
    assert give_gains(usd, load_profile("ethrc_v1")["physics"], desc) == {}


def test_the_v1_gives_its_hips_and_knees_by_default_at_the_measured_values(tmp_path):
    desc = Description.load("ethrc_v1")
    profile = load_profile("ethrc_v1")
    given = default_give(desc, profile)
    assert set(given) == {f"{j}_{s}" for j in ("hip_flexion", "hip_abduction", "knee_flexion") for s in "lr"}
    assert list(given) == [n for n in desc.joints if n in given]
    # Every given joint has a positive stiffness and damping from the profile.
    from humanoid_sim.physics.isaaclab import give_gains

    usd = tmp_path / "robot.usda"
    usd.write_text("")
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda", "joint_give": {
        "springs": {n: compliance.give_joint(n) for n in given}}}))
    gains = give_gains(usd, profile["physics"], desc)
    assert {k for k, _ in gains.values()} == {600.0, 950.0} and all(c == 1.2 for _, c in gains.values())
    # A robot whose profile names none gives none.
    assert load_profile("g1_29dof_rev_1_0")["model"]["joint_give"] == []


def test_give_knobs_select_joints_and_draw_per_env():
    spec = ParamSpec.from_dict({"param": "joint_give_stiffness", "give_joints": "knee_.*", "dist": "log_uniform",
                                "range": [200.0, 2000.0], "op": "set"})
    assert (spec.kind, spec.startup, spec.select) == ("physics", True, "knee_.*")
    with pytest.raises(ValueError, match="does not apply to a give parameter"):
        ParamSpec.from_dict({"param": "joint_give_damping", "joints": ".*", "dist": "uniform", "range": [0.1, 1.0],
                             "op": "set"})
    joints = ["hip_a", "knee_l", "knee_r"]
    nominal = {"joint_give_stiffness": torch.full((32, 3), 1000.0)}
    r = Randomizer([spec], nominal, {"give": joints}, num_envs=32)
    v = r.values["joint_give_stiffness"]
    assert (v[:, 0] == 1000.0).all() and ((v[:, 1:] >= 200.0) & (v[:, 1:] <= 2000.0)).all()
    assert v[:, 1].unique().numel() > 1
    with pytest.raises(ValueError, match="selects no give"):
        Randomizer([ParamSpec.from_dict({"param": "joint_give_damping", "give_joints": "elbow.*", "dist": "uniform",
                                         "range": [0.1, 1.0], "op": "set"})],
                   {"joint_give_damping": torch.ones(4, 3)}, {"give": joints}, num_envs=4)
