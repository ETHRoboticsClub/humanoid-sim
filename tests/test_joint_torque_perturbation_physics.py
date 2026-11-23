"""A joint torque perturbation on the V1 deflects the joint the way the torque points, on both solvers.

The robot hangs by its base without gravity with every drive's gains at zero, so a joint moves only under what is put
on it (and its own passive damping and friction, which the check keeps out of the way). The perturbation holds a
torque on one joint for the whole run; the joint turns in the torque's direction (positive about the joint's axis is
positive joint angle), the pair is read back from the running model (parent, child and axis agree with the
description).
"""
import pytest
import torch
from conftest import PARTS, built_usd

from humanoid_sim.devices.demand import DriveDemand

STEPS = 200  # 0.2 s at the V1's 1 ms
ENVS = 8
MAX_NM = 6.0  # test input: what one draw can reach on the joint
STRONG_NM = 1.5  # the draws that count: well past the joint's passive friction


def _cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("physics checks run on CUDA")


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_a_constant_perturbation_deflects_the_joint_in_the_torques_direction(solver):
    _cuda()
    from humanoid_sim.sim import SimCore
    from humanoid_sim.world.disturbances import JointTorquePerturbation, newton_joint_geometry

    joint = PARTS["ethrc_v1"]["elbow"]
    sim = SimCore(built_usd("ethrc_v1"), device="cuda:0", num_envs=ENVS, fixed_base=True, solver=solver,
                  gravity_m_s2=(0.0, 0.0, 0.0))
    try:
        desc, core, robot = sim.desc, sim.plant, sim.backend.robot
        cols = [core.solver_map.joint_names.index(joint)]
        core.friction_relief[:, cols] = 0.0
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.full((ENVS, 1), 1e-6, device="cuda:0"), joint_ids=sim._to_solver[cols])
        sim.write_joint_state(*sim.joint_state())
        feedback = core.feedback(*sim.solver_state())
        core.submit(DriveDemand.pd(feedback.position_rad.clone(), 0.0, 0.0),
                    torch.zeros(ENVS, dtype=torch.int64, device="cuda:0"))
        # The running model's joint agrees with the description: the bodies and the axis in the world.
        geometry = newton_joint_geometry(robot, [joint])
        bodies = list(robot.body_names)
        parent, child = desc.usd_model.joint_bodies(joint)
        assert (bodies[int(geometry.parent[0])], bodies[int(geometry.child[0])]) == (parent, child)
        from humanoid_sim.world.disturbances import _rotate

        pose = robot.data.body_link_pose_w.torch
        frame0, _ = desc.usd_model.joint_frames(joint)
        from_parent = _rotate(pose[:, geometry.parent[0], 3:],
                              torch.tensor(frame0[:3, :3] @ desc.usd_model.joint_axis(joint), device="cuda:0",
                                           dtype=torch.float32).expand(ENVS, 3))
        from_child = _rotate(pose[:, geometry.child[0], 3:], geometry.axis[0].expand(ENVS, 3))
        torch.testing.assert_close(from_child, from_parent, rtol=0, atol=1e-4)

        perturbation = JointTorquePerturbation(robot, joint, max_nm=MAX_NM, hold_s=(100.0, 100.0), active_share=1.0,
                                               seed=5)
        perturbation.reset(torch.arange(ENVS, device="cuda:0"), sim.t_ns)
        before = sim.joint_state()[0][:, list(desc.joints).index(joint)].clone()
        for _ in range(STEPS):
            perturbation.step(sim.t_ns)
            sim.step()
        after = sim.joint_state()[0][:, list(desc.joints).index(joint)]
        torque = perturbation.torque_nm[:, 0]
        deflection = after - before
        strong = torque.abs() > STRONG_NM
        assert int(strong.sum()) >= 3, torque
        assert (torch.sign(deflection[strong]) == torch.sign(torque[strong])).all(), (torque, deflection)
        assert deflection[strong].abs().min() > 1e-3
    finally:
        sim.close()
