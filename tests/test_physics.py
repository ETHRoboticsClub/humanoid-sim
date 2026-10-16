"""Physics on the built robot asset through Isaac Lab / MJWarp (Warp CPU device). Runs in the Isaac Lab image."""

import json
from pathlib import Path

import pytest
import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"


def _asset(base_joints=False):
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS, base_joints=base_joints)
    if not (out / "asset.json").exists():
        flag = " --base-joints" if base_joints else ""
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset{flag}` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


@pytest.fixture(scope="module")
def usd():
    return _asset()


def test_kinematic_model_mass_and_bodies(usd):
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    from humanoid_sim.model.kinematics import RobotKinematics

    kin = RobotKinematics(usd, list(Description.load().joints))
    assert float(kin.model.body_mass.numpy().sum()) == pytest.approx(58.684, abs=0.01)
    assert kin.model.body_count == 23


@pytest.mark.physics
def test_holds_standing_on_fixed_base(usd, request):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.sim import SimCore

    sim = SimCore(usd, device="cpu", fixed_base=True)
    request.addfinalizer(sim.close)
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


@pytest.mark.physics
def test_base_joints_follow_prescribed_yaw(request):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.model.urdf import BASE_JOINTS
    from humanoid_sim.sim import SimCore

    sim = SimCore(_asset(base_joints=True), device="cpu", base_joints=True)
    request.addfinalizer(sim.close)
    q, qd = sim.joint_state()
    p = sim.plant
    qdrv, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.jvpt(qdrv, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    names = sim.backend.joint_names
    yaw = names.index(BASE_JOINTS[0][0])
    rate = 0.5  # rad/s
    sim.set_base(torch.zeros(1, 3), torch.tensor([[rate, 0.0, 0.0]]), write_state=True)
    for k in range(100):
        # Targets at the step start: with the velocity target the base lands on the step-end angle.
        sim.set_base(torch.tensor([[rate * k * 0.004, 0.0, 0.0]]), torch.tensor([[rate, 0.0, 0.0]]))
        sim.step()
    bq, _ = sim.backend.joint_state()
    assert bq[0, yaw].item() == pytest.approx(rate * 0.4, abs=2e-4)
    q1, _ = sim.joint_state()
    assert (q1 - q).abs().max().item() < 0.02  # the robot holds its pose while the base turns


@pytest.mark.physics
def test_imu_delivers_raw_samples_on_a_fixed_base(usd, request):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.sim import SimCore

    sim = SimCore(usd, device="cpu", fixed_base=True, imu=True, contacts=True)
    request.addfinalizer(sim.close)
    for _ in range(50):  # 0.2 s: ~73 samples at ~365/s, in bursts
        sim.step()
    s = sim.imu.samples
    n = int(s.count[0])
    assert 50 <= n <= 90
    assert s.specific_force[0, 0].norm().item() == pytest.approx(9.81, abs=0.1)  # at rest: gravity only
    assert s.gyro[0, 0].norm().item() < 0.01
    assert (s.stamp_ns[0, :8] >= s.sample_ns[0, :8]).all()
    assert sim.backend.contacts is not None


@pytest.mark.physics
def test_cuda_graph_step_matches_eager(usd):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.sim import SimCore

    def run(cuda_graph, imu=False, compile=False):
        sim = SimCore(usd, device="cuda:0", fixed_base=True, cuda_graph=cuda_graph, imu=imu, compile=compile)
        try:
            p = sim.plant
            q, qd = sim.joint_state()
            stand, _, _ = p.map.drive_state(q, qd)
            out = []
            for k in range(60):
                if k % 5 == 0:  # a new demand every 20 ms, mid-step arrival
                    p.submit(DriveDemand.jvpt(stand + 0.02 * (k % 3), p.drive_stiffness.expand(1, -1),
                                              p.drive_damping.expand(1, -1)),
                             torch.full((1,), sim.t_ns + 1_500_000, dtype=torch.long, device=q.device))
                fb = sim.step()
                out.append(torch.stack([fb.position_rad, fb.applied_torque_nm]).clone())
            return torch.stack(out), (sim.imu.samples.seq[0].clone(), sim.imu.samples.gyro[0].clone()) if imu else None
        finally:
            sim.close()

    eager, _ = run(False)
    graphed, _ = run(True)
    assert torch.allclose(eager, graphed, atol=1e-5)
    compiled, _ = run(True, compile=True)
    # Fused kernels reorder float operations; the difference feeds back through the physics.
    assert torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3)
    for compile in (False, True):
        _, (seq, gyro) = run(True, imu=True, compile=compile)
        assert (seq[:8].diff() == -1).all() and gyro[:8].std() > 0  # samples delivered, noise drawn per replay


def test_a_joint_pushed_into_its_stop_stays_at_it(usd):
    """MuJoCo's own limit constraint (profile joint_limits: mujoco): 40 Nm into the hip abduction stop sinks less
    than 2 mrad past it (Newton's force-space default lets it sink 11 mrad)."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.sim import SimCore

    sim = SimCore(usd, device="cpu", fixed_base=True)
    try:
        j = list(sim.desc.joints).index("leg_abduction_l")
        lo, hi = sim.desc.calibration_travel("leg_abduction_l")
        q, qd = sim.joint_state()
        q = q.clone()
        q[0, j] = hi - 0.01
        sim.write_joint_state(q, torch.zeros_like(qd))
        sim.plant.external_torque[0, j] = 40.0
        for _ in range(150):
            sim.step()
        past = float(sim.joint_state()[0][0, j]) - hi
        assert -0.001 < past < 0.002, past
    finally:
        sim.close()
