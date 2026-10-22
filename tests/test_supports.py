"""Gantry supports: programs (CPU) and ropes on the robot (Isaac Lab image)."""

import math
import xml.etree.ElementTree as ET

import pytest
import torch
from conftest import GRAVITY as G
from conftest import PHYSICS_ROBOTS, built_usd

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.world.supports import interp_program


def test_program_is_linear_between_breakpoints_and_held_beyond():
    times = torch.tensor([[100, 200, 300], [0, 1000, 1001]])
    values = torch.tensor([[1.0, 3.0, 3.0], [5.0, 7.0, 7.0]])
    at = lambda t: interp_program(torch.tensor([t, t]), times, values)  # noqa: E731
    assert at(0).tolist() == [1.0, 5.0]
    assert at(150).tolist() == [2.0, pytest.approx(5.3)]
    assert at(5000).tolist() == [3.0, 7.0]


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_the_harness_hooks_on_kept_bodies_with_the_shoulders_fixed_on_it(robot):
    """The asset merges a link behind a fixed joint into its parent, so the harness body and both shoulders sit
    behind moving joints; each shoulder's joint hangs off the harness body through fixed joints only, so the hook
    between them stays put on the harness whatever the other joints do."""
    harness = load_profile(robot)["support"]["harness"]
    urdf = ET.parse(Description.load(robot).urdf_path).getroot()
    joint_of = {j.find("child").get("link"): j for j in urdf.iter("joint") if j.find("child") is not None}

    def moving(link):
        return link in joint_of and joint_of[link].get("type") != "fixed"

    def body_of(link):  # the asset body the link is merged into
        while link in joint_of and not moving(link):
            link = joint_of[link].find("parent").get("link")
        return link

    assert moving(harness["body"])
    assert len(harness["shoulders"]) == 2
    for shoulder in harness["shoulders"]:
        assert moving(shoulder)
        assert body_of(joint_of[shoulder].find("parent").get("link")) == harness["body"]


class _Robot:
    """What the gantry's configuration reads of an articulation: body names and link poses ``[E, B, 7]`` (xyzw)."""

    def __init__(self, poses):
        self.body_names = ["pelvis", "torso", "shoulder_l", "shoulder_r"]
        self.num_instances, self.device = 1, "cpu"
        self.data = type("Data", (), {})()
        self.data.body_link_pose_w = type("Poses", (), {"torch": torch.tensor([poses], dtype=torch.float32)})()


def _fake_gantry():
    from humanoid_sim.world.supports import Gantry

    s = math.sqrt(0.5)
    # The torso 1 m up, turned 90 deg about z; the shoulders 0.2 m to either side of it, 0.3 m higher.
    poses = [[0, 0, 0.8, 0, 0, 0, 1], [0, 0, 1.0, 0, 0, s, s], [-0.2, 0, 1.3, 0, 0, s, s], [0.2, 0, 1.3, 0, 0, s, s]]
    return Gantry(_Robot(poses), harness="torso")


def _cable(gantry, **kw):
    gantry.hang_cable(torch.tensor([0]), 0, overhead_m=1.0, length_breakpoints_s_m=[(0.0, 0.98)],
                      stiffness_n_m=1e4, damping_n_s_m=10.0, **kw)


def test_a_cable_on_the_shoulders_hooks_on_the_torso_between_them():
    g = _fake_gantry()
    _cable(g, attachment_bodies=["shoulder_l", "shoulder_r"])
    assert g.body_id == 1
    # The hook, midway between the shoulders (0, 0, 1.3) in the world, is (0, 0, 0.3) in the turned torso's frame.
    assert g.points_b[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.3], abs=1e-6)
    assert g.anchor_w[0].tolist() == pytest.approx([0.0, 0.0, 2.3], abs=1e-6)


def test_a_base_frame_rope_hangs_on_the_base():
    g = _fake_gantry()
    g.hang_tension(torch.tensor([0]), 0, [(0.0, 100.0)], attachment_b=[0.0, 0.0, 0.2])
    assert g.body_id == 0 and g.points_b[0, 0].tolist() == pytest.approx([0.0, 0.0, 0.2])


@pytest.mark.parametrize("bad", [
    {"length_breakpoints_s_m": [(1.0, 0.98), (0.0, 0.9)]},  # times not increasing
    {"attachment_bodies": ["shoulder_l", "hand"]},  # no such body
    {"attachment_b": [[0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]]},  # more points than a rope takes
])
def test_a_rejected_rope_leaves_the_previous_one_hanging(bad):
    g = _fake_gantry()
    _cable(g, attachment_bodies=["shoulder_l", "shoulder_r"])
    before = (g.body_id, g.points_b.clone(), g.anchor_w.clone(), g.program.clone(), g.mode.clone())
    args = {"attachment_b": [0.0, 0.0, 0.2]} | bad
    with pytest.raises(ValueError):
        g.hang_cable(torch.tensor([0]), 5, overhead_m=1.0, stiffness_n_m=1e4, damping_n_s_m=10.0,
                     **{"length_breakpoints_s_m": [(0.0, 0.9)]} | args)
    after = (g.body_id, g.points_b, g.anchor_w, g.program, g.mode)
    assert after[0] == before[0] and all(torch.equal(a, b) for a, b in zip(after[1:], before[1:], strict=True))


def test_a_dry_run_checks_without_hanging():
    g = _fake_gantry()
    g.hang_tension(torch.tensor([0]), 0, [(0.0, 100.0)], attachment_bodies=["shoulder_l"], dry_run=True)
    assert not g.active and g.body_id == 0 and int(g.mode[0]) == 0
    with pytest.raises(ValueError, match="non-negative"):
        g.hang_tension(torch.tensor([0]), 0, [(0.0, -1.0)], attachment_bodies=["shoulder_l"], dry_run=True)


def test_ropes_of_all_envs_pull_on_one_body():
    from humanoid_sim.world.supports import Gantry

    robot = _Robot([[0, 0, 1, 0, 0, 0, 1]] * 4)
    robot.num_instances = 2
    robot.data.body_link_pose_w.torch = robot.data.body_link_pose_w.torch.expand(2, -1, -1).clone()
    g = Gantry(robot, harness="torso")
    g.hang_tension(torch.tensor([0]), 0, [(0.0, 100.0)], attachment_b=[0.0, 0.0, 0.2])
    with pytest.raises(ValueError, match="one body"):
        g.hang_tension(torch.tensor([1]), 0, [(0.0, 100.0)], attachment_bodies=["shoulder_l"])
    g.hang_tension(torch.tensor([0, 1]), 0, [(0.0, 100.0)], attachment_bodies=["shoulder_l"])  # both move over
    assert g.body_id == 1


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
                            attachment_bodies=load_profile(robot)["support"]["harness"]["shoulders"])
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
