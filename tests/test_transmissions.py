import torch

from humanoid_sim.transmissions import CoupledTransmission, DirectTransmission


def test_direct_sign_offset_and_virtual_work():
    model = DirectTransmission(torch.tensor([-1.0, 1.0]), torch.tensor([0.2, -0.4]))
    q = torch.tensor([[0.5, 0.6]])
    dq = torch.tensor([[2.0, -3.0]])
    effort = torch.tensor([[4.0, 5.0]])
    motor_q, motor_dq = model.device_state(q, dq)
    torch.testing.assert_close(motor_q, torch.tensor([[-0.3, 0.2]]))
    torch.testing.assert_close((motor_dq * effort).sum(), (dq * model.joint_effort(q, effort)).sum())


def test_nonlinear_coupling_jacobian_and_power():
    def kinematics(q):
        a, b = q.unbind(-1)
        angles = torch.stack((a.sin() + b, a - b.sin()), -1)
        one = torch.ones_like(a)
        jacobian = torch.stack((a.cos(), one, one, -b.cos()), -1).reshape(-1, 2, 2)
        return angles, jacobian

    mapping = CoupledTransmission(kinematics)
    q = torch.tensor([[0.1, 0.4], [-0.3, 0.2]], dtype=torch.float64)
    dq = torch.tensor([[1.0, -2.0], [0.3, 0.4]], dtype=torch.float64)
    efforts = torch.tensor([[3.0, 4.0], [2.0, -1.0]], dtype=torch.float64)
    _, mdq = mapping.device_state(q, dq)
    finite_difference = (kinematics(q + 1e-6 * dq)[0] - kinematics(q - 1e-6 * dq)[0]) / 2e-6
    torch.testing.assert_close(mdq, finite_difference)
    torch.testing.assert_close((mdq * efforts).sum(-1), (dq * mapping.joint_effort(q, efforts)).sum(-1))
