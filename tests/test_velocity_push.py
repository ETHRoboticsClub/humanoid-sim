from types import SimpleNamespace

import torch

from humanoid_sim.world.disturbances import VelocityPush


def test_velocity_push_sets_only_xy_when_due_and_respects_individual_resets():
    class Robot:
        num_instances = 2
        device = 'cpu'
        def __init__(self):
            self.value = torch.tensor([[1., 2., 3., 4., 5., 6.], [7., 8., 9., 10., 11., 12.]])
            self.data = SimpleNamespace(root_com_vel_w=SimpleNamespace(torch=self.value))
        def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
            self.value[env_ids] = root_velocity
    robot = Robot()
    initial = robot.value.clone()
    push = VelocityPush(robot, interval_ns=500, velocity_range_m_s=(-.5, .5), seed=4)
    push.step(499)
    torch.testing.assert_close(robot.value, initial)
    push.reset(torch.tensor([1]), 250)
    push.step(500)
    assert (robot.value[0, :2].abs() <= .5).all()
    torch.testing.assert_close(robot.value[1], initial[1])
    torch.testing.assert_close(robot.value[:, 2:], initial[:, 2:])
    before = robot.value.clone()
    push.step(750)
    torch.testing.assert_close(robot.value[0], before[0])
    assert (robot.value[1, :2].abs() <= .5).all()
