"""Ankle crank backlash: the play's statics, the plant reporting the output shaft, the asset variant that carries the
play, and its knobs."""
import json
import math

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.model.asset import asset_dir, backlash_joints, build_asset, compliance_springs, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.usd import BACKLASH_SCHEMA
from humanoid_sim.randomization import ParamSpec, Randomizer
from humanoid_sim.transmissions import backlash, compliance
from humanoid_sim.transmissions.drive_map import native_joint_names


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def _cranks(desc):
    return [d for d, drive in desc.drives.items() if drive.crank is not None]


def test_every_crank_drive_has_one_backlash_joint(desc):
    joints = backlash.backlash_joints(desc)
    assert list(joints) == _cranks(desc)
    from humanoid_sim.model.usd import crank_joints

    assert joints == {d: crank_joints(desc)[d] + backlash.BACKLASH_SUFFIX for d in _cranks(desc)}


def test_the_stops_bound_a_band_of_the_width_and_load_rests_the_shaft_on_one():
    width = torch.tensor([0.0074, 0.0085, 0.01], dtype=torch.float64)
    stops = backlash.limits(width)
    torch.testing.assert_close(stops[:, 1] - stops[:, 0], width)
    torch.testing.assert_close(stops.sum(-1), torch.zeros(3, dtype=torch.float64))
    gap = torch.tensor([-0.02, 0.001, 0.009], dtype=torch.float64)
    torch.testing.assert_close(backlash.dead_band(gap, width), torch.tensor([-0.0163, 0.0, 0.004],
                                                                          dtype=torch.float64))
    # Pushing forward, the motor side leads: the shaft rests on its lower stop, and motor minus output steps by the
    # width where the torque changes sign.
    torque = torch.tensor([2.0, -2.0, 0.0], dtype=torch.float64)
    play = backlash.static_play(torque, 0.008)
    torch.testing.assert_close(play, torch.tensor([-0.004, 0.004, 0.0], dtype=torch.float64))
    assert float(-play[0] - -play[1]) == pytest.approx(0.008)


def _native_core(desc, envs=3):
    return PlantCore(desc, list(desc.joints), envs, PlantCfg(dtype=torch.float64),
                     solver_joint_names=native_joint_names(desc))


def test_the_plant_reports_the_output_shaft_and_regulates_the_motor_side(desc):
    core = _native_core(desc)
    generator = torch.Generator().manual_seed(0)
    n = len(core.solver_map.joint_names)
    q = 0.2 * (torch.rand(3, n, generator=generator, dtype=torch.float64) - 0.5)
    qd = torch.rand(3, n, generator=generator, dtype=torch.float64) - 0.5
    rigid = core.feedback(q, qd)
    rigid = [t.clone() for t in (rigid.position_rad, rigid.velocity_rad_s)]
    assert core.backlash_rad is None

    core.use_backlash()
    cranks = [core.map.drive_names.index(d) for d in _cranks(desc)]
    assert core.crank_drives.tolist() == cranks
    torch.testing.assert_close(core.feedback(q, qd).position_rad, rigid[0])  # no play yet
    joint = 0.004 * (torch.rand(3, len(cranks), generator=generator, dtype=torch.float64) - 0.5)
    core.write_backlash(joint)
    sign = torch.tensor([float(desc.drives[d].sign) for d in _cranks(desc)], dtype=torch.float64)
    torch.testing.assert_close(core.backlash_rad, joint * sign)
    fb = core.feedback(q, qd)
    expected = rigid[0].clone()
    expected[:, cranks] += joint * sign
    torch.testing.assert_close(fb.position_rad, expected)
    torch.testing.assert_close(fb.velocity_rad_s, rigid[1])  # the motor side's
    # The drive bank works from the motor side: the state kept for the next step has no play in it.
    torch.testing.assert_close(core.post_state[0], rigid[0])

    # A PD to the reported positions pulls the motor side by the play: the drive regulates its own encoder.
    stiffness = torch.full((len(core.map.drive_names),), 10.0, dtype=torch.float64)
    core.submit(DriveDemand.pd(fb.position_rad.clone(), stiffness, torch.zeros_like(stiffness)),
                torch.zeros(3, dtype=torch.int64))
    core.pre_physics(0, q, torch.zeros_like(qd), None)
    torch.testing.assert_close(core.last.demanded[:, cranks], 10.0 * joint * sign)

    # A dual-loop drive reads the output shaft: the same demand is already met, the play inside its loop.
    core.use_backlash("output")
    torch.testing.assert_close(core.backlash_rad, joint * sign)  # switching the loop keeps the play
    core.pre_physics(0, q, torch.zeros_like(qd), None)
    torch.testing.assert_close(core.last.demanded[:, cranks], torch.zeros(3, len(cranks), dtype=torch.float64))
    with pytest.raises(ValueError, match="loop_encoder"):
        core.use_backlash("rotor")


def test_the_loop_encoder_is_a_named_profile_knob():
    assert backlash.loop_encoder({}) == "motor"
    assert backlash.loop_encoder({"ankle_backlash": {"loop_encoder": "output"}}) == "output"
    assert backlash.loop_encoder(load_profile("ethrc_v1")["physics"]) in backlash.LOOP_ENCODERS
    with pytest.raises(ValueError, match="loop_encoder"):
        backlash.loop_encoder({"ankle_backlash": {"loop_encoder": "both"}})


def test_rigid_asset_is_untouched_and_the_variant_puts_play_behind_each_crank_drive(desc, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from humanoid_sim.model.usd import crank_joints

    profile = load_profile(desc.robot)
    rigid = build_asset(desc, profile, tmp_path)
    assert asset_dir(desc, profile, tmp_path, ankle_backlash=False) == rigid.parent
    variant = build_asset(desc, profile, tmp_path, ankle_backlash=True)
    both = build_asset(desc, profile, tmp_path, ankle_backlash=True, ankle_compliance=True)
    assert variant.parent == asset_dir(desc, profile, tmp_path, ankle_backlash=True) != rigid.parent
    assert both.parent == asset_dir(desc, profile, tmp_path, ankle_backlash=True, ankle_compliance=True)
    assert len({rigid.parent, variant.parent, both.parent,
                asset_dir(desc, profile, tmp_path, ankle_compliance=True)}) == 4

    rigid_stage = Usd.Stage.Open(str(rigid))
    assert not [p for p in rigid_stage.Traverse()
                if p.GetName().endswith((backlash.BACKLASH_SUFFIX, backlash.MOTOR_SUFFIX))]
    assert backlash_joints(rigid) == {}
    meta = json.loads((variant.parent / "asset.json").read_text())
    assert "ankle_compliance" not in meta and meta["ankle_backlash"]["schema"] == BACKLASH_SCHEMA
    assert backlash_joints(variant) == backlash.backlash_joints(desc) == backlash_joints(both)
    assert compliance_springs(both) == compliance.spring_joints(desc)

    def frame(joint, side):
        pos = getattr(joint, f"GetLocalPos{side}Attr")().Get()
        rot = getattr(joint, f"GetLocalRot{side}Attr")().Get()
        m = Gf.Matrix4d(1.0)
        m.SetRotate(Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary())))
        m.SetTranslateOnly(Gf.Vec3d(*pos))
        return np.asarray(m).T

    for usd, springs in ((variant, False), (both, True)):
        stage = Usd.Stage.Open(str(usd))
        prims = {p.GetName(): p for p in stage.Traverse()}
        xforms = UsdGeom.XformCache()

        def pose(name, prims=prims, xforms=xforms):
            return np.asarray(xforms.GetLocalToWorldTransform(prims[name])).T

        for drive, play in backlash_joints(usd).items():
            name = crank_joints(desc)[drive]
            crank = UsdPhysics.RevoluteJoint(prims[name])
            source = desc.usd_model.joints[name]
            shank, arm = desc.usd_model.joint_bodies(name)
            motor = backlash.motor_body(name)
            shaft = compliance.output_body(name) if springs else arm
            # The drive's joint keeps its shank, frames, axis and stops, and now turns the motor side.
            assert [t.name for t in crank.GetBody0Rel().GetTargets()] == [shank]
            assert [t.name for t in crank.GetBody1Rel().GetTargets()] == [motor]
            assert crank.GetLocalPos1Attr().Get() == source.GetLocalPos1Attr().Get()
            assert crank.GetLowerLimitAttr().Get() < crank.GetUpperLimitAttr().Get()
            assert prims[motor].HasAPI(UsdPhysics.RigidBodyAPI)
            assert UsdPhysics.MassAPI(prims[motor]).GetMassAttr().Get() > 0
            meta = json.loads((usd.parent / "asset.json").read_text())["ankle_backlash"]
            np.testing.assert_allclose(UsdPhysics.MassAPI(prims[motor]).GetDiagonalInertiaAttr().Get(),
                                       [meta["motor_inertia_kg_m2"]] * 3, rtol=1e-6)
            if springs:  # the flange the play frees takes the profile's inertia, recorded for the stability check
                flange = profile["physics"]["ankle_backlash"]["output_inertia_kg_m2"]
                assert meta["output_inertia_kg_m2"] == flange
                np.testing.assert_allclose(UsdPhysics.MassAPI(prims[shaft]).GetDiagonalInertiaAttr().Get(),
                                           [flange] * 3, rtol=1e-6)
            else:
                assert "output_inertia_kg_m2" not in meta
            np.testing.assert_allclose(pose(motor), pose(shaft), atol=1e-9)
            gap = UsdPhysics.RevoluteJoint(prims[play])
            assert [t.name for t in gap.GetBody0Rel().GetTargets()] == [motor]
            assert [t.name for t in gap.GetBody1Rel().GetTargets()] == [shaft]
            assert gap.GetAxisAttr().Get() == crank.GetAxisAttr().Get()
            assert not gap.GetExcludeFromArticulationAttr().Get()
            # Authored all but closed, yet not empty (MuJoCo compiles no empty range); the spawn opens it to the
            # profile's width.
            lower, upper = (math.radians(a.Get()) for a in (gap.GetLowerLimitAttr(), gap.GetUpperLimitAttr()))
            assert lower == pytest.approx(-backlash.AUTHORED_HALF_WIDTH_RAD, rel=1e-4)
            assert upper == pytest.approx(backlash.AUTHORED_HALF_WIDTH_RAD, rel=1e-4)
            assert tuple(gap.GetPrim().GetAttribute("mjc:solreflimit").Get()) == pytest.approx(
                profile["physics"]["mjwarp"]["ankle_backlash_solref"])
            world = [pose(motor) @ frame(gap, 0), pose(shaft) @ frame(gap, 1), pose(motor) @ frame(crank, 1)]
            for other in world[1:]:
                np.testing.assert_allclose(other, world[0], atol=1e-6)
            if springs:  # motor, play, output flange, spring, crank arm
                spring = UsdPhysics.RevoluteJoint(prims[compliance_springs(usd)[drive]])
                assert [t.name for t in spring.GetBody0Rel().GetTargets()] == [shaft]
                assert [t.name for t in spring.GetBody1Rel().GetTargets()] == [arm]
        closures = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                    and UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
        assert closures == {n for n, j in desc.usd_model.joints.items() if j.GetExcludeFromArticulationAttr().Get()}


def test_the_backlash_solref_is_part_of_the_variant_key_only(desc, tmp_path):
    from humanoid_sim.model.usd import asset_key

    profile = load_profile(desc.robot)
    other = load_profile(desc.robot, physics_overrides={"mjwarp": {"ankle_backlash_solref": [0.004, 1.0]}})
    assert asset_key(desc, profile) == asset_key(desc, other)
    assert asset_key(desc, profile, ankle_backlash=True) != asset_key(desc, other, ankle_backlash=True)
    # The flange's inertia: only the variant with springs and play has a free flange.
    heavy = load_profile(desc.robot, physics_overrides={"ankle_backlash": {
        "output_inertia_kg_m2": 2 * profile["physics"]["ankle_backlash"]["output_inertia_kg_m2"]}})
    for flags in ({}, {"ankle_backlash": True}, {"ankle_compliance": True}):
        assert asset_key(desc, profile, **flags) == asset_key(desc, heavy, **flags)
    both = {"ankle_backlash": True, "ankle_compliance": True}
    assert asset_key(desc, profile, **both) != asset_key(desc, heavy, **both)


def test_widths_come_from_the_profile_and_are_required(tmp_path):
    from humanoid_sim.physics.isaaclab import backlash_widths

    usd = tmp_path / "robot.usda"
    usd.write_text("")
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda", "ankle_backlash": {
        "joints": {"A": "a_backlash", "B": "b_backlash"}}}))
    assert backlash_widths(usd, {"ankle_backlash": {"width_rad": 0.008}}) == {"a_backlash": 0.008,
                                                                            "b_backlash": 0.008}
    assert backlash_widths(usd, {"ankle_backlash": {"width_rad": {"A": 0.007, "B": 0.009}}}) == {
        "a_backlash": 0.007, "b_backlash": 0.009}
    for bad in (None, 0.0, -0.001, float("nan"), {"A": 0.007}):
        with pytest.raises(ValueError, match="width_rad"):
            backlash_widths(usd, {"ankle_backlash": {"width_rad": bad}})
    # An asset without play needs nothing, and the defaults ask for nothing.
    (tmp_path / "asset.json").write_text(json.dumps({"usd": "robot.usda"}))
    assert backlash_widths(usd, load_profile("ethrc_v1")["physics"]) == {}


def test_the_defaults_have_no_play():
    assert load_profile("ethrc_v1")["physics"]["ankle_backlash"]["width_rad"] is None


def test_a_spring_on_the_freed_flange_is_checked_against_the_step():
    k, b, dt = 900.0, 0.4, 0.002  # fixture values, not the robot's
    bound = (k * dt**2 + 2 * b * dt) / backlash.STABLE_MEASURE  # the lightest stable flange
    backlash.check_output_stability(torch.full((3, 4), k), torch.full((3, 4), b), 1.01 * bound, dt)
    stiff = torch.full((3, 4), k)
    stiff[2, 1] = 1.1 * k  # one env's crank, as a randomized draw
    with pytest.raises(ValueError, match="output_inertia_kg_m2"):
        backlash.check_output_stability(stiff, torch.full((3, 4), b), 1.01 * bound, dt)
    with pytest.raises(ValueError, match="unstable"):
        backlash.check_output_stability(torch.tensor(k), torch.tensor(b), 0.99 * bound, dt)


def test_the_flange_inertia_is_a_positive_build_knob(desc):
    from humanoid_sim.model.usd import output_inertia

    for bad in (None, 0.0, -1e-4, float("nan"), True):
        with pytest.raises(ValueError, match="output_inertia_kg_m2"):
            output_inertia(load_profile(desc.robot, physics_overrides={"ankle_backlash": {"output_inertia_kg_m2": bad}}))
    assert output_inertia(load_profile(desc.robot, physics_overrides={
        "ankle_backlash": {"output_inertia_kg_m2": 7e-4}})) == 7e-4


def test_the_width_knob_selects_cranks_and_draws_per_env():
    spec = ParamSpec.from_dict({"param": "ankle_backlash", "cranks": ".*R_.*", "dist": "uniform",
                                "range": [0.005, 0.01], "op": "set"})
    assert (spec.kind, spec.startup, spec.select) == ("physics", True, ".*R_.*")
    with pytest.raises(ValueError, match="does not apply to a crank"):
        ParamSpec.from_dict({"param": "ankle_backlash", "drives": ".*", "dist": "uniform", "range": [0.005, 0.01],
                             "op": "set"})
    cranks = ["HEJ50L_ANK1", "HEJ50L_ANK2", "HEJ50R_ANK1", "HEJ50R_ANK2"]
    nominal = {"ankle_backlash": torch.full((32, 4), 0.008)}
    v = Randomizer([spec], nominal, {"crank": cranks}, num_envs=32).values["ankle_backlash"]
    assert ((v[:, 2:] >= 0.005) & (v[:, 2:] <= 0.01)).all() and (v[:, :2] == 0.008).all()
    assert v[:, 2].unique().numel() > 1
