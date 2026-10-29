import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Usd, UsdPhysics

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.model.asset import asset_dir, asset_meta, build_asset, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import BASE_JOINTS
from humanoid_sim.model.usd import crank_joints
from humanoid_sim.physics.joint_properties import joint_properties


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    desc = Description.load()
    root = tmp_path_factory.mktemp("native-usd")
    profile = load_profile(desc.robot)
    path = build_asset(desc, profile, root)
    return desc, profile, root, path


def test_build_preserves_closed_chain_and_inertias(native):
    desc, _, _, path = native
    source = Usd.Stage.Open(str(desc.usd_path))
    built = Usd.Stage.Open(str(path))
    closures = []
    for prim in source.Traverse():
        other = built.GetPrimAtPath(prim.GetPath())
        if prim.HasAPI(UsdPhysics.MassAPI):
            for name in ("mass", "centerOfMass", "diagonalInertia", "principalAxes"):
                assert other.GetAttribute(f"physics:{name}").Get() == prim.GetAttribute(f"physics:{name}").Get()
        if prim.IsA(UsdPhysics.SphericalJoint):
            assert other.IsA(UsdPhysics.SphericalJoint)
            for name in ("localPos0", "localPos1", "localRot0", "localRot1", "excludeFromArticulation"):
                assert other.GetAttribute(f"physics:{name}").Get() == prim.GetAttribute(f"physics:{name}").Get()
            if UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
                closures.append(str(prim.GetPath()))
    assert closures
    assert asset_meta(path)["loop_closures"] == closures


def test_native_cranks_have_drive_stops_and_no_authored_controller(native):
    desc, _, _, path = native
    stage = Usd.Stage.Open(str(path))
    joints = {prim.GetName(): prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
    cranks = crank_joints(desc)
    assert set(joints) == set(desc.joints) | set(cranks.values())
    for drive, name in cranks.items():
        joint = UsdPhysics.RevoluteJoint(joints[name])
        actual = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
        assert actual == pytest.approx([math.degrees(value) for value in desc.drives[drive].position_limit_device])
        assert not joints[name].HasAPI(UsdPhysics.DriveAPI, "angular")


def test_native_sole_metadata_is_usable_by_runtime(native):
    from humanoid_sim.runtime.host import RuntimeHost

    _, _, _, path = native
    boxes = asset_meta(path)["sole_boxes"]
    assert {box["side"] for box in boxes} == {"l", "r"}
    stage = Usd.Stage.Open(str(path))
    names = [prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    host = SimpleNamespace(robot=SimpleNamespace(body_names=names), device="cpu")
    soles = RuntimeHost._sole_geometry(host, boxes)
    assert len(soles) == len(boxes)
    for sole, box in zip(soles, boxes, strict=True):
        assert torch.isfinite(sole["corners"]).all()
        assert (sole["half"] > 0).all()
        assert box["replaces_mesh"] is None


def test_native_asset_applies_profile_contact_excludes(native):
    _, profile, _, path = native
    stage = Usd.Stage.Open(str(path))
    bodies = {prim.GetName(): prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    for first, second in profile["model"]["contact_excludes"]:
        targets = UsdPhysics.FilteredPairsAPI(bodies[first]).GetFilteredPairsRel().GetTargets()
        assert bodies[second].GetPath() in targets
    assert asset_meta(path)["contact_excludes"] == [
        [str(bodies[name].GetPath()) for name in pair] for pair in profile["model"]["contact_excludes"]]


def test_base_variant_keeps_the_same_mechanism(native):
    desc, profile, root, path = native
    base_path = build_asset(desc, profile, root, base_joints=True)
    assert base_path != path
    stage = Usd.Stage.Open(str(base_path))
    names = {prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)}
    assert {name for name, _ in BASE_JOINTS} <= names
    assert asset_meta(base_path)["loop_closures"] == asset_meta(path)["loop_closures"]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass = UsdPhysics.MassAPI(prim)
        assert np.isfinite(mass.GetCenterOfMassAttr().Get()).all()
        assert mass.GetPrincipalAxesAttr().Get().GetLength() == pytest.approx(1)
    assert path.parent == asset_dir(desc, profile, root)


def test_native_drive_torque_and_feedback_use_the_crank_shafts(native):
    desc, _, _, _ = native
    names = list(desc.physics_joints)
    plant = PlantCore(desc, names, 1, PlantCfg(dtype=torch.float64))
    assert not plant.map.ankles
    positions = torch.zeros(1, len(names), dtype=torch.float64)
    velocities = torch.zeros_like(positions)
    cranks = crank_joints(desc)
    for drive, joint in cranks.items():
        positions[0, names.index(joint)] = 0.12
        velocities[0, names.index(joint)] = 0.23
    feedback = plant.feedback(positions, velocities)
    for drive in cranks:
        index = plant.map.drive_names.index(drive)
        assert feedback.position_rad[0, index] == pytest.approx(0.12)
        assert feedback.velocity_rad_s[0, index] == pytest.approx(0.23)
    demand = DriveDemand.pd(torch.zeros_like(feedback.position_rad), plant.drive_stiffness,
                           plant.drive_damping)
    plant.submit(demand, torch.tensor([0]))
    plant.pre_physics(0, positions, velocities)
    torque = plant.map.joint_torque(plant.last.applied, [])
    props = joint_properties(plant, names)
    for linkage in desc.linkages:
        for name in linkage.joints:
            index = names.index(name)
            assert torque[0, index] == 0
            assert props.stiffness[index] == 0
            assert props.armature[index] == 0
        for drive in (linkage.upper, linkage.lower):
            index = names.index(cranks[drive])
            assert torque[0, index] == plant.last.applied[0, plant.map.drive_names.index(drive)]
            assert props.armature[index] > 0
            assert props.effort_limit[index] == desc.drives[drive].effort_peak


def test_native_reset_closes_rods_in_newton_forward_kinematics(native):
    pytest.importorskip("newton")
    from humanoid_sim.model.kinematics import RobotKinematics, _rotate

    desc, _, _, path = native
    names = list(desc.joints)
    kin = RobotKinematics(path, names)
    model = kin.model
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    parent_frames, child_frames = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    generator = torch.Generator().manual_seed(10)
    for _ in range(5):
        positions = torch.zeros(1, len(names))
        for linkage in desc.linkages:
            indices = [names.index(name) for name in linkage.joints]
            positions[:, indices] = (torch.rand(1, 2, generator=generator) - 0.5) * 0.1
        poses = kin.body_poses(positions)
        for closure in asset_meta(path)["loop_closures"]:
            index = labels.index(closure.rsplit("/", 1)[-1])
            parent, child = poses[parents[index]], poses[children[index]]
            start = parent[:3] + _rotate(parent[3:], parent_frames[index, :3])
            end = child[:3] + _rotate(child[3:], child_frames[index, :3])
            assert np.linalg.norm(start - end) < 2e-6


def test_native_reset_velocities_follow_pose_derivatives(native):
    from humanoid_sim.model.native_state import NativeState

    desc, _, _, _ = native
    names = list(desc.joints)
    state = NativeState(desc, names, dtype=torch.float64)
    positions = torch.full((2, len(names)), 0.02, dtype=torch.float64)
    velocities = torch.full_like(positions, 0.1)
    values = state.state(positions, velocities)
    assert all(torch.isfinite(position).all() and torch.isfinite(velocity).all()
               for position, velocity in values.values())
    before = state.coordinates(positions - 1e-6 * velocities)
    after = state.coordinates(positions + 1e-6 * velocities)
    for _, crank_names, _ in state.transmissions.values():
        for name in crank_names:
            torch.testing.assert_close(values[name][1], (after[name] - before[name]) / 2e-6, atol=1e-8, rtol=0)
    for name, *_ in state.rods:
        previous, following = before[name].numpy(), after[name].numpy()
        relative_vector = (previous[:, 3:] * following[:, :3] - following[:, 3:] * previous[:, :3]
                           + np.cross(previous[:, :3], following[:, :3]))
        angular = torch.as_tensor(relative_vector / 1e-6)
        reported = torch.stack([values[f"{name}:{axis}"][1] for axis in range(3)], dim=-1)
        torch.testing.assert_close(reported, angular, atol=1e-8, rtol=0)
