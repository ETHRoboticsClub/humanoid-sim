"""Gantry supports: programs (CPU) and ropes on the robot (Isaac Lab image)."""

import pytest
import torch
from conftest import GRAVITY as G
from conftest import PARTS, PHYSICS_ROBOTS, built_usd

from humanoid_sim.world.supports import Gantry, interp_program


def test_program_is_linear_between_breakpoints_and_held_beyond():
    times = torch.tensor([[100, 200, 300], [0, 1000, 1001]])
    values = torch.tensor([[1.0, 3.0, 3.0], [5.0, 7.0, 7.0]])
    at = lambda t: interp_program(torch.tensor([t, t]), times, values)  # noqa: E731
    assert at(0).tolist() == [1.0, 5.0]
    assert at(150).tolist() == [2.0, pytest.approx(5.3)]
    assert at(5000).tolist() == [3.0, 7.0]


class _Composer:
    def __init__(self):
        self.forces = []

    def add_forces_and_torques_index(self, *, forces, torques, positions, body_ids, env_ids, is_global):
        self.forces.append(forces[:, 0].clone())


class _Robot:
    """The articulation surface `Gantry` uses: two envs, base at rest 1 m up, or moving at `velocity`."""

    def __init__(self, velocity=(0.0, 0.0, 0.0)):
        self.num_instances, self.device, self.body_names = 2, "cpu", ["base"]
        pose = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).expand(2, 1, 7)
        vel = torch.tensor([[*velocity, 0.0, 0.0, 0.0]]).expand(2, 1, 6)
        self.data = type("Data", (), {"body_link_pose_w": type("T", (), {"torch": pose})(),
                                      "body_link_vel_w": type("T", (), {"torch": vel})()})()
        self.instantaneous_wrench_composer = _Composer()


def _cables(gantry, swing=0.0):
    """Anchors 1 m above the harness point (1.3 m up); env 0 keeps 0.98 m of rope, env 1 pays out to 1.5 m
    from 2 s on."""
    gantry.hang_cables(torch.arange(2), 1000, attachment_b=[0.0, 0.0, 0.3],
                       anchor_w=torch.tensor([[0.0, 0.0, 2.3], [0.0, 0.0, 2.3]]),
                       times_ns=torch.tensor([[0, 1], [0, 2_000_000_000]]),
                       lengths_m=torch.tensor([[0.98, 0.98], [0.98, 1.5]]),
                       stiffness_n_m=torch.tensor([1000.0, 2000.0]), damping_n_s_m=torch.zeros(2),
                       swing_damping_n_s_m=torch.full((2,), swing))


def test_cables_follow_each_envs_own_program_and_stiffness():
    robot = _Robot()
    gantry = Gantry(robot)
    _cables(gantry)
    gantry.apply(1000)  # 2 cm stretched in both envs
    assert gantry.tension_n.tolist() == [pytest.approx(20.0, rel=1e-4), pytest.approx(40.0, rel=1e-4)]
    assert robot.instantaneous_wrench_composer.forces[0][:, 2].tolist() == gantry.tension_n.tolist()
    gantry.apply(1000 + 2_000_000_000)  # env 1 paid out past the 1 m to its anchor: slack
    assert gantry.tension_n.tolist() == [pytest.approx(20.0, rel=1e-4), 0.0]


def test_swing_damping_resists_motion_across_a_taut_cable_only():
    robot = _Robot(velocity=(0.5, 0.0, 0.0))
    gantry = Gantry(robot)
    _cables(gantry, swing=10.0)
    gantry.apply(1000 + 2_000_000_000)
    force = robot.instantaneous_wrench_composer.forces[0]
    assert force[0].tolist() == [pytest.approx(-5.0), 0.0, pytest.approx(20.0, rel=1e-4)]
    assert force[1].tolist() == [0.0, 0.0, 0.0]  # slack: no swing damping either


def _sim(robot, request, height):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd(robot), device="cpu", robot=robot, base_height_m=height)
    request.addfinalizer(sim.close)
    return sim


def _com(robot):
    m = robot.data.body_mass.torch[0]
    return (m[:, None] * robot.data.body_com_pos_w.torch[0]).sum(0) / m.sum(), float(m.sum())


@pytest.mark.physics
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_tension_equal_to_the_weight_holds_the_centre_of_mass(request, robot):
    sim = _sim(robot, request, 2.5)  # in the air, drives off
    art = sim.backend.robot
    com0, mass = _com(art)
    sim.gantry.hang_tension(torch.tensor([0]), sim.t_ns, [(0.0, mass * G)],
                            attachment_bodies=list(PARTS[robot]["shoulders"]))
    for _ in range(100):  # 0.4 s: free fall would drop 0.78 m
        sim.step()
    com1, _ = _com(art)
    assert sim.gantry.tension_n.item() == pytest.approx(mass * G)
    assert abs((com1 - com0)[2].item()) < 0.005


@pytest.mark.physics
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_cable_carries_the_robot_at_its_stretched_length(request, robot):
    from humanoid_sim.devices.demand import DriveDemand

    sim = _sim(robot, request, 2.5)
    p = sim.plant
    q, qd = sim.joint_state()
    stand, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.pd(stand, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    _, mass = _com(sim.backend.robot)
    k, length = 4e4, 0.5
    sim.gantry.hang_cable(torch.tensor([0]), sim.t_ns, attachment_b=[0.0, 0.0, 0.3], overhead_m=length,
                          length_breakpoints_s_m=[(0.0, length)], stiffness_n_m=k, damping_n_s_m=1500.0)
    tension = []
    for i in range(500):  # 2 s
        sim.step()
        if i >= 400:
            tension.append(sim.gantry.tension_n.item())
    assert sum(tension) / len(tension) == pytest.approx(mass * G, rel=0.05)
    assert sim.gantry.distance_m.item() == pytest.approx(length + mass * G / k, abs=0.003)
    sim.gantry.release(torch.tensor([0]))
    sim.step()
    assert sim.gantry.tension_n.item() == 0.0 and not sim.gantry.active
