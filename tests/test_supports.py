"""Gantry supports: programs (CPU) and ropes on the robot (Isaac Lab image)."""

import json
from pathlib import Path

import pytest
import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.world.supports import interp_program

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"
G = 9.81


def test_program_is_linear_between_breakpoints_and_held_beyond():
    times = torch.tensor([[100, 200, 300], [0, 1000, 1001]])
    values = torch.tensor([[1.0, 3.0, 3.0], [5.0, 7.0, 7.0]])
    at = lambda t: interp_program(torch.tensor([t, t]), times, values)  # noqa: E731
    assert at(0).tolist() == [1.0, 5.0]
    assert at(150).tolist() == [2.0, pytest.approx(5.3)]
    assert at(5000).tolist() == [3.0, 7.0]


@pytest.fixture(scope="module")
def usd():
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def _sim(usd, request, height):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.sim import SimCore

    sim = SimCore(usd, device="cpu", base_height_m=height)
    request.addfinalizer(sim.close)
    return sim


def _com(robot):
    m = robot.data.body_mass.torch[0]
    return (m[:, None] * robot.data.body_com_pos_w.torch[0]).sum(0) / m.sum(), float(m.sum())


@pytest.mark.physics
def test_tension_equal_to_the_weight_holds_the_centre_of_mass(usd, request):
    sim = _sim(usd, request, 2.5)  # in the air, drives off
    robot = sim.backend.robot
    com0, mass = _com(robot)
    sim.gantry.hang_tension(torch.tensor([0]), sim.t_ns, [(0.0, mass * G)],
                            attachment_bodies=["shoulder_connector_l_1", "shoulder_connector_r_1"])
    for _ in range(100):  # 0.4 s: free fall would drop 0.78 m
        sim.step()
    com1, _ = _com(robot)
    assert sim.gantry.tension_n.item() == pytest.approx(mass * G)
    assert abs((com1 - com0)[2].item()) < 0.005


@pytest.mark.physics
def test_cable_carries_the_robot_at_its_stretched_length(usd, request):
    from humanoid_sim.devices.demand import DriveDemand

    sim = _sim(usd, request, 2.5)
    p = sim.plant
    q, qd = sim.joint_state()
    stand, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.jvpt(stand, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
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
