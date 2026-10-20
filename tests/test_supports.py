"""Gantry supports: programs (CPU) and ropes on the robot (Isaac Lab image)."""

import pytest
import torch
from conftest import GRAVITY as G
from conftest import PARTS, PHYSICS_ROBOTS, built_usd

from humanoid_sim.world.supports import interp_program


def test_program_is_linear_between_breakpoints_and_held_beyond():
    times = torch.tensor([[100, 200, 300], [0, 1000, 1001]])
    values = torch.tensor([[1.0, 3.0, 3.0], [5.0, 7.0, 7.0]])
    at = lambda t: interp_program(torch.tensor([t, t]), times, values)  # noqa: E731
    assert at(0).tolist() == [1.0, 5.0]
    assert at(150).tolist() == [2.0, pytest.approx(5.3)]
    assert at(5000).tolist() == [3.0, 7.0]


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


@pytest.mark.physics
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_a_cable_on_another_body_hangs_the_robot_from_that_body(request, robot):
    """The rope's point is fixed in the named body's frame: the stretched rope reaches from the anchor to that body's
    point, not to the same coordinates in the base frame, and it carries the whole robot. A shoulder: its frame sits
    well away from the base's, unlike a torso frame that can share the base's origin (the V1's CAD frames)."""
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.world.supports import _quat_rotate

    sim = _sim(robot, request, 2.5)
    p, art = sim.plant, sim.backend.robot
    q, qd = sim.joint_state()
    stand, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.pd(stand, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    _, mass = _com(art)
    body = PARTS[robot]["shoulders"][0]
    b = art.body_names.index(body)
    local = torch.tensor([0.0, 0.0, 0.1])
    k, length = 4e4, 0.5
    sim.gantry.hang_cable(torch.tensor([0]), sim.t_ns, attachment_b=local.tolist(), overhead_m=length,
                          length_breakpoints_s_m=[(0.0, length)], stiffness_n_m=k, damping_n_s_m=1500.0, body=body)
    anchor = sim.gantry.anchor_w[0].clone()
    tension = []
    for i in range(500):  # 2 s
        sim.step()
        if i >= 400:
            tension.append(sim.gantry.tension_n.item())
    assert sum(tension) / len(tension) == pytest.approx(mass * G, rel=0.05)

    def reach(index):  # anchor to the rope point fixed in body `index`'s frame, from the body's own pose
        pose = art.data.body_link_pose_w.torch[0, index]
        return (anchor - (pose[:3] + _quat_rotate(pose[3:], local))).norm().item()

    assert sim.gantry.distance_m.item() == pytest.approx(reach(b), abs=1e-4)
    assert sim.gantry.distance_m.item() == pytest.approx(length + mass * G / k, abs=0.003)
    assert abs(reach(0) - reach(b)) > 0.01  # the base's frame would put the rope elsewhere


def test_a_rope_on_an_unknown_body_is_refused():
    """Checked before anything is hung (no physics needed: the refusal precedes every tensor write)."""
    from types import SimpleNamespace

    from humanoid_sim.world.supports import Gantry

    robot = SimpleNamespace(num_instances=1, device="cpu", body_names=["pelvis", "torso_link"])
    g = Gantry(robot)
    with pytest.raises(ValueError, match="no body"):
        g.hang_cable(torch.tensor([0]), 0, attachment_b=[0, 0, 0], overhead_m=1.0, length_breakpoints_s_m=[(0, 1.0)],
                     stiffness_n_m=1.0, damping_n_s_m=0.0, body="head_link")
    assert not g.active and g.mode.tolist() == [0]
