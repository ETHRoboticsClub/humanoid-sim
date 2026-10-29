import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand


@pytest.fixture(autouse=True)
def no_contact_overflow(capfd):
    yield
    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "exceeded MJWarp limit" not in output
    assert "Contact buffer overflowed" not in output
    assert "possibly invalid inertia tensor" not in output


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_native_ankle_drives_step_on_cuda(request, solver, cuda_graph):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native ankle qualification requires CUDA")
    from humanoid_sim.model.usd import crank_joints
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd("ethrc_v1"), device="cuda:0", num_envs=2,
                  fixed_base=True, solver=solver, cuda_graph=cuda_graph)
    request.addfinalizer(sim.close)
    assert sim.native is not None
    positions, velocities = sim.joint_state()
    for linkage in sim.desc.linkages:
        ids = [sim.plant.map.joint_names.index(name) for name in linkage.joints]
        positions[1, ids] += 0.02
    sim.write_joint_state(positions, torch.zeros_like(velocities))
    feedback = sim.drive_feedback()
    targets = feedback.position_rad.clone()
    sim.plant.submit(DriveDemand.pd(targets, sim.plant.drive_stiffness, sim.plant.drive_damping),
                     torch.zeros(2, dtype=torch.int64, device="cuda:0"))
    for _ in range(100):
        feedback = sim.step()
    positions, velocities = sim.joint_state()
    assert torch.isfinite(positions).all() and torch.isfinite(velocities).all()
    assert torch.isfinite(feedback.applied_torque_nm).all()
    native_positions, native_velocities = sim.backend.joint_state()
    for drive, joint in crank_joints(sim.desc).items():
        drive_id = sim.plant.map.drive_names.index(drive)
        joint_id = sim.backend.joint_names.index(joint)
        torch.testing.assert_close(feedback.position_rad[:, drive_id], native_positions[:, joint_id])
        torch.testing.assert_close(feedback.velocity_rad_s[:, drive_id], native_velocities[:, joint_id])
        lower, upper = sim.desc.drives[drive].position_limit_device
        assert (feedback.position_rad[:, drive_id] >= lower - 0.02).all()
        assert (feedback.position_rad[:, drive_id] <= upper + 0.02).all()


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
def test_native_training_reset_randomization_and_feedback(request, solver):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native ankle qualification requires CUDA")
    from humanoid_sim.randomization import ParamSpec
    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    sim = SimCore(built_usd("ethrc_v1"), device="cuda:0", num_envs=2,
                  fixed_base=True, solver=solver, cuda_graph=False)
    request.addfinalizer(sim.close)
    plant = RobotPlant(sim.backend.robot, TrainPlantCfg(reporting=False, randomization=[
        ParamSpec("joint_armature", "uniform", (1.2, 1.2), "scale")]))
    positions, _ = plant.joint_state()
    untouched = positions[0].clone()
    reset = positions[1:].clone()
    for linkage in plant.desc.linkages:
        reset[:, [plant.joints.index(name) for name in linkage.joints]] += 0.02
    plant.reset(torch.tensor([1], device="cuda:0"), joint_pos=reset)
    actual, _ = plant.joint_state()
    torch.testing.assert_close(actual[0], untouched)
    torch.testing.assert_close(actual[1:], reset)
    physical = sim.backend.robot.data.joint_armature.torch[:, plant.native_ids]
    torch.testing.assert_close(physical, plant.params()["joint_armature"])
    feedback = plant.truth().drive
    plant.submit(DriveDemand.pd(feedback.position_rad, plant.core.drive_stiffness, plant.core.drive_damping),
                 torch.zeros(2, dtype=torch.int64, device="cuda:0"))
    for _ in range(20):
        plant.pre_physics()
        sim.backend.step()
    truth, measured = plant.truth(), plant.measured()
    assert torch.isfinite(truth.joint_pos).all()
    torch.testing.assert_close(truth.drive.position_rad, measured.drive.position_rad)


@pytest.mark.physics
def test_native_runtime_publishes_soles_and_drive_feedback(request, tmp_path):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native ankle qualification requires CUDA")
    from humanoid_sim.runtime.host import RuntimeHost

    host = RuntimeHost(tmp_path / "native-plant", built_usd("ethrc_v1"), device="cuda:0",
                       solver="mjwarp", compile=False)
    request.addfinalizer(host.close)
    for _ in range(5):
        host.step()
    assert len(host._soles) == 2
    assert host.seg.regions["feedback"].read() is not None
    assert host.seg.regions["truth"].read() is not None


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
def test_native_prescribed_base_replay_variant(request, solver):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native ankle qualification requires CUDA")
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd("ethrc_v1", base_joints=True), device="cuda:0", num_envs=2,
                  fixed_base=True, base_joints=True, solver=solver, cuda_graph=False)
    request.addfinalizer(sim.close)
    angles = torch.full((2, 3), 0.02, device="cuda:0")
    sim.set_base(angles, torch.zeros_like(angles), write_state=True)
    feedback = sim.drive_feedback()
    sim.plant.submit(DriveDemand.pd(feedback.position_rad, sim.plant.drive_stiffness, sim.plant.drive_damping),
                     torch.zeros(2, dtype=torch.int64, device="cuda:0"))
    for _ in range(20):
        feedback = sim.step()
    assert torch.isfinite(feedback.position_rad).all()
    positions, _ = sim.backend.joint_state()
    torch.testing.assert_close(positions[:, sim._base], angles, atol=1e-3, rtol=0)
