"""Load tilt: a leg joint's link tilts past what the joint's encoder implies, by an angle set by the sole's vertical
load: the asset variant that carries it, where its gains and nominal values come from, its law and its knobs."""
import json

import numpy as np
import pytest
import torch

from humanoid_sim.model.asset import asset_dir, build_asset, load_profile, tilt_springs
from humanoid_sim.model.description import Description
from humanoid_sim.randomization import ParamSpec, Randomizer
from humanoid_sim.transmissions import compliance

# Test inputs: the joints a check tilts, by pattern (the knee, whose link is the shank), and one it also gives.
TILT = ("knee_flexion_.*",)


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


@pytest.fixture(scope="module")
def profile():
    return load_profile("ethrc_v1")


@pytest.fixture(scope="module")
def names(desc, profile):
    return compliance.resolve_tilt(desc, profile, TILT)


def test_the_tilt_is_the_load_over_the_saturation_load_up_to_the_angle():
    angle = torch.tensor([0.05, 0.05, -0.04, 0.05, 0.05, 0.0])
    sat = torch.tensor([400.0, 400.0, 250.0, 400.0, 400.0, 100.0])
    load = torch.tensor([100.0, 400.0, 125.0, 4000.0, -30.0, 50.0])
    got = compliance.load_tilt(load, angle, sat)
    torch.testing.assert_close(got, torch.tensor([0.0125, 0.05, -0.02, 0.05, 0.0, 0.0]))
    # Off is off, whatever the load; no load is no tilt; it never exceeds its angle.
    assert (compliance.load_tilt(load, torch.zeros(6), sat) == 0).all()
    assert (compliance.load_tilt(torch.zeros(6), angle, sat) == 0).all()
    ramp = compliance.load_tilt(torch.linspace(0, 3000, 61), torch.tensor(0.06), torch.tensor(500.0))
    assert (ramp.diff() >= 0).all() and ramp.max() == pytest.approx(0.06)


def test_each_leg_joint_has_one_sole_and_a_joint_above_both_legs_has_none(desc, profile):
    legs = compliance.tilt_legs(desc, profile)
    soles = set(legs.values())
    assert len(soles) == len(profile["model"]["sole"]["joints"])
    candidates = compliance.give_candidates(desc)
    assert set(legs) <= set(candidates) and len(legs) < len(candidates)
    assert list(legs) == [n for n in candidates if n in legs]
    # The two knees are on different legs; the joints above the legs (the waist) are on both.
    knees = compliance.resolve_tilt(desc, profile, TILT)
    assert len(knees) == 2 and len({legs[n] for n in knees}) == 2
    above = [n for n in candidates if n not in legs]  # on neither leg's chain (the waist, the arms) or on both
    assert above
    # Each leg joint's sole is the one at the end of the chain that has it.
    for name in knees:
        sole = legs[name]
        chain, body = [], sole
        parents = {child: (up, joint) for joint in desc.joints for up, child in [desc.usd_model.joint_bodies(joint)]}
        while body in parents:
            body, joint = parents[body]
            chain.append(joint)
        assert name in chain


def test_patterns_select_leg_joints_and_refuse_the_rest(desc, profile, names):
    assert len(names) == 2 and list(names) == [n for n in desc.joints if n in names]
    assert compliance.resolve_tilt(desc, profile, [names[0], names[0]]) == (names[0],)
    with pytest.raises(ValueError, match="selects no joint"):
        compliance.resolve_tilt(desc, profile, ["no_such_joint"])
    hinge = desc.linkages[0].joints[0]  # moved by cranks
    with pytest.raises(ValueError, match="selects no joint"):
        compliance.resolve_tilt(desc, profile, [hinge])
    waist = next(n for n in compliance.give_candidates(desc) if n not in compliance.tilt_legs(desc, profile))
    with pytest.raises(ValueError, match="selects no joint"):
        compliance.resolve_tilt(desc, profile, [waist])
    pytest.importorskip("pxr")
    from humanoid_sim.model.usd import _checked_tilt

    with pytest.raises(ValueError, match="cannot carry it"):
        _checked_tilt(desc, profile, [waist])


def test_the_rigid_asset_is_untouched_and_the_variant_adds_one_tilt_per_joint(desc, profile, names, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    rigid = build_asset(desc, profile, tmp_path)
    assert rigid.parent == asset_dir(desc, profile, tmp_path) == asset_dir(desc, profile, tmp_path, load_tilt=())
    variant = build_asset(desc, profile, tmp_path, load_tilt=names)
    assert variant.parent == asset_dir(desc, profile, tmp_path, load_tilt=names) != rigid.parent
    assert asset_dir(desc, profile, tmp_path, load_tilt=names[::-1]) == variant.parent
    assert asset_dir(desc, profile, tmp_path, load_tilt=names[:1]) not in (variant.parent, rigid.parent)
    # Its own asset, neither the give's nor the same joints' give.
    assert asset_dir(desc, profile, tmp_path, joint_give=names) not in (variant.parent, rigid.parent)
    assert asset_dir(desc, profile, tmp_path, joint_give=names, load_tilt=names) not in (
        variant.parent, rigid.parent, asset_dir(desc, profile, tmp_path, joint_give=names))

    rigid_stage = Usd.Stage.Open(str(rigid))
    assert not [p for p in rigid_stage.Traverse() if p.GetName().endswith(compliance.TILT_SUFFIX)]
    assert "load_tilt" not in json.loads((rigid.parent / "asset.json").read_text())
    assert tilt_springs(rigid) == {}

    stage = Usd.Stage.Open(str(variant))
    assert tilt_springs(variant) == {n: compliance.tilt_joint(n) for n in names}
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

    for name in names:
        parent, link = desc.usd_model.joint_bodies(name)
        output = compliance.tilt_output_body(name)
        drive = UsdPhysics.RevoluteJoint(prims[name])
        assert ends(drive) == [[parent], [output]]  # the joint keeps its name and parent, and turns the new body
        rigid_drive = UsdPhysics.RevoluteJoint(rigid_prims[name])
        assert [drive.GetLowerLimitAttr().Get(), drive.GetUpperLimitAttr().Get()] == [
            rigid_drive.GetLowerLimitAttr().Get(), rigid_drive.GetUpperLimitAttr().Get()]
        assert prims[output].HasAPI(UsdPhysics.RigidBodyAPI)
        np.testing.assert_allclose(pose(output), pose(link), atol=1e-9)
        tilt = UsdPhysics.RevoluteJoint(prims[compliance.tilt_joint(name)])
        assert ends(tilt) == [[output], [link]] and tilt.GetAxisAttr().Get() == drive.GetAxisAttr().Get()
        world = [pose(output) @ frame(tilt, 0), pose(link) @ frame(tilt, 1), pose(output) @ frame(drive, 1)]
        for other in world[1:]:
            np.testing.assert_allclose(other, world[0], atol=1e-6)
    meta = json.loads((variant.parent / "asset.json").read_text())
    excluded = {frozenset(path.split("/")[-1] for path in pair) for pair in meta["self_collision_excludes"]}
    for name in names:
        assert frozenset(desc.usd_model.joint_bodies(name)) in excluded, name
    closures = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                and UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
    assert closures == {n for n, j in desc.usd_model.joints.items() if j.GetExcludeFromArticulationAttr().Get()}


def test_a_joint_can_carry_give_and_tilt_in_series(desc, profile, names, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    both = build_asset(desc, profile, tmp_path, joint_give=names, load_tilt=names)
    stage = Usd.Stage.Open(str(both))
    prims = {p.GetName(): p for p in stage.Traverse()}

    def ends(name):
        joint = UsdPhysics.Joint(prims[name])
        return [[t.name for t in rel.GetTargets()] for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]

    for name in names:
        parent, link = desc.usd_model.joint_bodies(name)
        give, tilt = compliance.give_joint(name), compliance.tilt_joint(name)
        # parent -(drive)- output -(give)- tilt body -(tilt)- link: three joints in series on one axis.
        assert ends(name) == [[parent], [compliance.output_body(name)]]
        assert ends(give) == [[compliance.output_body(name)], [compliance.tilt_output_body(name)]]
        assert ends(tilt) == [[compliance.tilt_output_body(name)], [link]]


def test_tilt_gains_come_from_the_profile_and_are_required(tmp_path, names):
    from humanoid_sim.physics.isaaclab import compliance_gains, tilt_gains

    usd = tmp_path / "robot.usda"
    usd.write_text("")
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda", "load_tilt": {
        "springs": {n: compliance.tilt_joint(n) for n in names}}}))
    physics = {"load_tilt": {"stiffness_nm_per_rad": 8000.0, "damping_nm_s_per_rad": 12.0}}
    expected = {compliance.tilt_joint(n): (8000.0, 12.0) for n in names}
    assert tilt_gains(usd, physics) == expected
    desc = Description.load("ethrc_v1")
    assert compliance_gains(usd, physics, desc) == expected
    for bad in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="stiffness_nm_per_rad"):
            tilt_gains(usd, {"load_tilt": {"stiffness_nm_per_rad": bad, "damping_nm_s_per_rad": 1.0}})
    with pytest.raises(ValueError, match="damping_nm_s_per_rad"):
        tilt_gains(usd, {"load_tilt": {"stiffness_nm_per_rad": 8000.0}})
    # An asset without tilt needs nothing; the profile's defaults are tilt off and a saturation load the caller gives.
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda"}))
    assert tilt_gains(usd, load_profile("ethrc_v1")["physics"]) == {}
    block = load_profile("ethrc_v1")["physics"]["load_tilt"]
    assert block["angle_rad"] == 0.0 and block["saturation_load_n"] is None
    assert block["stiffness_nm_per_rad"] > 0 and block["damping_nm_s_per_rad"] > 0


def test_a_simcore_refuses_an_asset_with_tilt(desc, profile, names, tmp_path):
    pytest.importorskip("pxr")
    from humanoid_sim.sim import SimCore

    usd = build_asset(desc, profile, tmp_path, load_tilt=names)
    with pytest.raises(ValueError, match="load tilt"):
        SimCore(usd, device="cpu")


def test_tilt_knobs_select_joints_and_draw_per_env_at_reset():
    spec = ParamSpec.from_dict({"param": "load_tilt_angle", "tilt_joints": "knee_l", "dist": "uniform",
                                "range": [0.0, 0.08], "op": "set"})
    assert (spec.kind, spec.startup, spec.select) == ("device", False, "knee_l")
    with pytest.raises(ValueError, match="does not apply to a tilt parameter"):
        ParamSpec.from_dict({"param": "load_tilt_saturation_load", "joints": ".*", "dist": "uniform",
                             "range": [200.0, 600.0], "op": "set"})
    joints = ["hip_l", "knee_l", "knee_r"]
    nominal = {"load_tilt_angle": torch.zeros(32, 3)}
    r = Randomizer([spec], nominal, {"tilt": joints}, num_envs=32)
    v = r.values["load_tilt_angle"]
    assert (v[:, [0, 2]] == 0).all() and ((v[:, 1] >= 0) & (v[:, 1] <= 0.08)).all() and v[:, 1].unique().numel() > 1
    before = v[:, 1].clone()
    r.draw_reset(torch.arange(8))  # a reset redraws only the reset envs
    assert (r.values["load_tilt_angle"][8:, 1] == before[8:]).all()
    assert (r.values["load_tilt_angle"][:8, 1] != before[:8]).any()
    with pytest.raises(ValueError, match="selects no tilt"):
        Randomizer([ParamSpec.from_dict({"param": "load_tilt_saturation_load", "tilt_joints": "elbow.*",
                                         "dist": "uniform", "range": [200.0, 600.0], "op": "set"})],
                   {"load_tilt_saturation_load": torch.ones(4, 3)}, {"tilt": joints}, num_envs=4)


def test_the_least_saturation_load_the_specs_can_draw_is_known_without_drawing():
    from humanoid_sim.train.plant import tilt_load_floor

    def spec(op, lo, hi, dist="uniform"):
        return ParamSpec("load_tilt_saturation_load", dist, (lo, hi), op)

    assert tilt_load_floor([], 300.0) == 300.0
    assert tilt_load_floor([spec("set", 200.0, 900.0)], 300.0) == 200.0
    assert tilt_load_floor([spec("scale", 0.5, 2.0)], 300.0) == 150.0
    assert tilt_load_floor([spec("add", -100.0, 50.0)], 300.0) == 200.0
    assert tilt_load_floor([spec("scale", 0.5, 2.0), spec("add", -100.0, 0.0)], 300.0) == 50.0
    assert tilt_load_floor([spec("set", -1.0, 900.0)], 300.0) < 0
    with pytest.raises(ValueError, match="bounded"):
        tilt_load_floor([spec("set", 300.0, 30.0, "normal")], 300.0)
