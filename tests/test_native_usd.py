"""The native plant preserves CAD loops and measures the physical shafts."""
import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.asset import build_asset, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.native_state import NativeState
from humanoid_sim.physics.joint_properties import joint_properties
from humanoid_sim.transmissions.drive_map import native_joint_names


def test_fixed_body_contacts_are_excluded_transitively_without_excluding_movable_neighbors():
    from humanoid_sim.model.usd import fixed_body_excludes

    assert fixed_body_excludes([('palm', 'finger'), ('finger', 'tip'), ('wrist', 'bracket')]) == [
        ('bracket', 'wrist'), ('finger', 'palm'), ('finger', 'tip'), ('palm', 'tip')]


def test_fixed_joint_collapse_preserves_native_mass_geometry_and_moving_joints(desc, tmp_path):
    newton = pytest.importorskip("newton")
    import warp as wp

    from humanoid_sim.physics.model_build import _FixedJointCollapse

    asset = build_asset(desc, load_profile(desc.robot), tmp_path)
    builder = newton.ModelBuilder()
    builder.add_usd(str(asset), floating=True)

    def snapshot():
        total_mass = sum(builder.body_mass)
        first_moment = np.zeros(3)
        world_inertia = np.zeros((3, 3))
        for pose, mass, com, inertia in zip(builder.body_q, builder.body_mass, builder.body_com,
                                           builder.body_inertia, strict=True):
            p = np.asarray(wp.transform_point(pose, wp.vec3(com)), dtype=float)
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(pose)), dtype=float).reshape(3, 3)
            first_moment += mass * p
            world_inertia += rotation @ np.asarray(inertia).reshape(3, 3) @ rotation.T
            world_inertia += mass * (np.dot(p, p) * np.eye(3) - np.outer(p, p))
        shapes = {label: np.asarray(wp.transform_multiply(builder.body_q[body], local))
                  for label, body, local in zip(builder.shape_label, builder.shape_body,
                                                builder.shape_transform, strict=True) if body >= 0}
        joints = {label: (int(kind), np.asarray(wp.transform_multiply(builder.body_q[child], frame)))
                  for label, kind, child, frame in zip(builder.joint_label, builder.joint_type,
                                                       builder.joint_child, builder.joint_X_c, strict=True)
                  if kind != newton.JointType.FIXED}
        return total_mass, first_moment, world_inertia, shapes, joints

    before = snapshot()
    original_bodies = builder.body_count
    has_fixed_joints = any(kind == newton.JointType.FIXED for kind in builder.joint_type)
    hook = _FixedJointCollapse()
    hook.world(builder, 0, None, None)
    hook.finalize(None)
    after = snapshot()
    assert builder.body_count <= original_bodies
    if has_fixed_joints:
        assert builder.body_count < original_bodies
    for a, b in zip(before[:3], after[:3], strict=True):
        np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-6)
    assert before[3].keys() == after[3].keys()
    def assert_pose(a, b):
        a = a.copy()
        if np.dot(a[3:], b[3:]) < 0:
            a[3:] *= -1  # q and -q encode the same rotation
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=0)

    for label, pose in before[3].items():
        assert_pose(after[3][label], pose)
    assert before[4].keys() == after[4].keys()
    for label, (kind, pose) in before[4].items():
        assert after[4][label][0] == kind
        assert_pose(after[4][label][1], pose)


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def test_reset_closes_every_delivered_loop_and_velocity(desc):
    source, state = desc.usd_model, NativeState(desc)
    q = torch.zeros(3, len(desc.joints), dtype=torch.float64)
    for lk in desc.linkages:
        cols = [list(desc.joints).index(n) for n in lk.joints]
        q[:, cols] = torch.tensor([[0, 0], [-.15, .05], [-.1, -.06]], dtype=q.dtype)
        assert desc.ankle_model.feasible(q[:, cols].numpy(), lk.side).all()
    poses = state.body_transforms(q)
    for name, joint in source.joints.items():
        if joint.GetExcludeFromArticulationAttr().Get():
            a, b = source.joint_bodies(name)
            fa, fb = source.joint_frames(name)
            pa = poses[a] @ torch.tensor(fa)
            pb = poses[b] @ torch.tensor(fb)
            torch.testing.assert_close(pa[:, :3, 3], pb[:, :3, 3], atol=2e-6, rtol=0)
    qd = torch.full_like(q, .07)
    values, rates = state.dofs(q, qd)
    assert all(torch.isfinite(v).all() for v in [*values.values(), *rates.values()])
    h = 10_000 * 1e-9
    ahead, _ = state.coordinates(q + h * qd)
    behind, _ = state.coordinates(q - h * qd)
    for n in native_joint_names(desc):
        torch.testing.assert_close(rates[n], (ahead[n] - behind[n]) / (2 * h), atol=1e-8, rtol=1e-6)


def test_physical_crank_feedback_does_not_reconstruct_from_hinges(desc):
    names = native_joint_names(desc)
    core = PlantCore(desc, list(desc.joints), 2, PlantCfg(), solver_joint_names=names)
    q = torch.zeros(2, len(names))
    for spec in desc.ankle_model.payload["sides"].values():
        for role, drive in zip(("upper", "lower"), spec["motor_names"], strict=True):
            q[:, names.index(spec["usd_joints"][role])] = .123
            i = core.map.drive_names.index(drive)
            d = desc.drives[drive]
            expected = torch.tensor(d.sign * (.123 - d.shaft_zero_rad), dtype=q.dtype)
            torch.testing.assert_close(core.feedback(q, q * 0).position_rad[:, i], expected.expand(2))
    props = joint_properties(core, names)
    for lk in desc.linkages:
        assert all(props.stiffness[names.index(n)] == props.armature[names.index(n)] == 0 for n in lk.joints)
    hinge = names.index(desc.linkages[0].joints[0])
    core.external_torque[:, hinge] = 1.25
    assert torch.isfinite(core.pre_physics(0, q, q * 0)).all()
    assert torch.all(core.passive_effort[:, hinge] == 1.25)
    assert torch.all(core.passive_effort[:, core.actuated] == 0)


def test_native_asset_preserves_closures_and_authors_shaft_limits(desc, tmp_path):
    from pxr import Usd, UsdPhysics

    source_hash = desc.usd_model.digest()
    profile = load_profile(desc.robot)
    path = build_asset(desc, profile, tmp_path)
    stage = Usd.Stage.Open(str(path))
    source = desc.usd_model
    source_closures = {n for n, j in source.joints.items() if j.GetExcludeFromArticulationAttr().Get()}
    closures = {p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)
                and UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()}
    assert closures == source_closures and closures
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    for pair in profile["model"].get("contact_excludes", []):
        a, b = sorted(pair)
        targets = UsdPhysics.FilteredPairsAPI(bodies[a]).GetFilteredPairsRel().GetTargets()
        assert bodies[b].GetPath() in targets
    for spec in desc.ankle_model.payload["sides"].values():
        for role, drive in zip(("upper", "lower"), spec["motor_names"], strict=True):
            name = spec["usd_joints"][role]
            p = next(p for p in stage.Traverse() if p.GetName() == name)
            joint = UsdPhysics.RevoluteJoint(p)
            assert np.radians([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]) == pytest.approx(
                desc.drives[drive].shaft_limits, abs=1e-6)
    assert source.digest() == source_hash


@pytest.mark.physics
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_native_runtime_steps_and_reads_physical_shafts(desc, tmp_path, cuda_graph):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native runtime smoke requires CUDA")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.sim import SimCore

    path = build_asset(desc, load_profile(desc.robot), tmp_path)
    sim = SimCore(path, device="cuda:0", fixed_base=True, cuda_graph=cuda_graph)
    try:
        core = sim.plant
        q, qd = sim.joint_state()
        sim.write_joint_state(q, torch.zeros_like(qd))
        physical, velocity = sim.solver_state()
        held, _, _ = core.solver_map.drive_state(physical, velocity)
        core.submit(DriveDemand.pd(held, core.drive_stiffness[None], core.drive_damping[None]),
                    torch.zeros(1, device="cuda:0", dtype=torch.long))
        for _ in range(8):
            feedback = sim.step()
        assert torch.isfinite(feedback.applied_torque_nm).all()
        physical, velocity = sim.solver_state()
        expected, _, _ = core.solver_map.drive_state(physical, velocity)
        torch.testing.assert_close(feedback.position_rad, expected)
        assert sim.t_ns == 8 * sim.period_ns
    finally:
        sim.close()


@pytest.mark.physics
def test_native_host_publishes_feedback_before_the_first_step(desc, tmp_path):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native runtime smoke requires CUDA")
    from humanoid_sim.runtime.host import RuntimeHost

    path = build_asset(desc, load_profile(desc.robot), tmp_path / "asset")
    host = RuntimeHost(tmp_path / "plant", path, device="cuda:0", support="free", compile=False)
    try:
        host.sim.feedback = None
        host._publish()
        record = host.seg.regions["feedback"].read()
        assert record is not None
        physical, velocity = host.sim.solver_state()
        expected, _, _ = host.sim.plant.solver_map.drive_state(physical, velocity)
        np.testing.assert_allclose(
            record[1]["drive"]["position_rad"][:len(host.drives)], expected[0].cpu().numpy())
    finally:
        host.close()


def test_fixed_joint_collapse_keeps_the_world_anchor_body():
    newton = pytest.importorskip("newton")
    import warp as wp

    from humanoid_sim.physics.model_build import collapse_fixed_joints

    builder = newton.ModelBuilder()
    root = builder.add_link(mass=2.0, inertia=wp.mat33(1.0), label="anchored_root")
    child = builder.add_link(mass=1.0, inertia=wp.mat33(0.5), label="rigid_child")
    anchor = builder.add_joint_fixed(-1, root, label="world_anchor")
    rigid = builder.add_joint_fixed(root, child, label="rigid_assembly")
    builder.add_articulation([anchor, rigid])
    collapse_fixed_joints(builder)
    assert builder.body_label == ["anchored_root"]
    assert builder.joint_label == ["world_anchor"]
    assert builder.joint_parent == [-1]
    assert builder.body_mass == [3.0]
