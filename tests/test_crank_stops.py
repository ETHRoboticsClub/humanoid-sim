"""The ankle cranks' stops as spatial tendons (physics/crank_stops.py): geometry checked natively against
humanoid-urdf's ``ankle_model``, which decides feasibility from the crank angles of its own analytic inverse
kinematics. The tendon lengths here come from the URDF's joint chain, the sites from ``crank_stops``."""

import numpy as np
import pytest
import torch
from conftest import GRAVITY, built_usd

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import PreparedUrdf
from humanoid_sim.physics.crank_stops import (
    AnkleChain,
    ankle_stops_setting,
    crank_stops,
    tendon_length_m,
)

ROBOT = "ethrc_v1"
# Hinge angles probed: the hinges' travel widened by this much on every side, far past where any crank reaches.
PROBE_MARGIN_RAD = 0.5
PROBE_STEPS = 161  # per hinge
# A pose with a crank at its stop puts that stop's tendon at the rod's length, up to the transmission's Newton
# solve (float64, 8 steps) and the URDF's rounding of the hinge axes.
AT_STOP_TOLERANCE_M = 1e-7


@pytest.fixture(scope="module")
def setup():
    desc = Description.load(ROBOT)
    urdf = PreparedUrdf(desc, load_profile(ROBOT))
    return desc, urdf, crank_stops(desc, urdf)


def _inside_by_tendons(stops, chain, side, qp, qr):
    inside = np.ones(np.shape(qp), dtype=bool)
    for stop in (s for s in stops if s.side == side):
        length = tendon_length_m(chain, stop, qp, qr)
        lo, hi = stop.length_range_m()
        inside &= (length >= lo) & (length <= hi)
    return inside


def test_four_stops_per_ankle_on_its_shank_and_foot(setup):
    desc, urdf, stops = setup
    assert len(stops) == 4 * len(desc.linkages)
    for lk in desc.linkages:
        chain = AnkleChain.from_urdf(urdf, *lk.joints)
        own = [s for s in stops if s.side == lk.side]
        assert {(s.drive, s.bound) for s in own} == {(d, b) for d in (lk.upper, lk.lower) for b in ("lower", "upper")}
        assert all(s.shank == chain.shank and s.foot == chain.foot for s in own)
        for s in own:
            assert s.stop_rad == desc.drives[s.drive].position_limit_device[("lower", "upper").index(s.bound)]


def test_tendons_admit_exactly_the_poses_the_ankle_model_admits(setup):
    """Over the hinges' travel and well past it, a pose keeps every stop tendon in its range exactly when
    humanoid-urdf's ankle_model finds it feasible (every crank within its stops, every rod reaching)."""
    desc, urdf, stops = setup
    model = desc.ankle_model
    for lk in desc.linkages:
        chain = AnkleChain.from_urdf(urdf, *lk.joints)
        (plo, phi), (rlo, rhi) = (desc.calibration_travel(j) for j in lk.joints)
        qp, qr = np.meshgrid(np.linspace(plo - PROBE_MARGIN_RAD, phi + PROBE_MARGIN_RAD, PROBE_STEPS),
                             np.linspace(rlo - PROBE_MARGIN_RAD, rhi + PROBE_MARGIN_RAD, PROBE_STEPS))
        feasible = model.feasible(np.stack([qp, qr], -1), lk.side)
        assert 0 < feasible.sum() < feasible.size
        assert np.array_equal(_inside_by_tendons(stops, chain, lk.side, qp, qr), feasible)


def test_a_crank_at_its_stop_puts_its_tendon_at_the_rod_length(setup):
    """Hinge angles that put one crank exactly at its stop (the transmission's inverse, from the linkage
    neutral) put that stop's tendon at the rod's length: the stop is the tendon's limit, not near it."""
    desc, urdf, stops = setup
    for lk in desc.linkages:
        chain = AnkleChain.from_urdf(urdf, *lk.joints)
        tx = desc.ankle_transmission(lk.side, dtype=torch.float64)
        neutral = torch.tensor(desc.ankle_neutral(lk.side), dtype=torch.float64)
        for s in (s for s in stops if s.side == lk.side):
            k = (lk.upper, lk.lower).index(s.drive)
            theta = torch.zeros(2, dtype=torch.float64)
            theta[k] = s.stop_rad
            q = tx.joints_from_cranks(theta, neutral)
            assert torch.allclose(tx.cranks(q), theta, atol=1e-10)
            assert abs(float(tendon_length_m(chain, s, float(q[0]), float(q[1]))) - s.rod_m) < AT_STOP_TOLERANCE_M


def test_the_hinge_box_alone_lets_cranks_past_their_stops(setup):
    """Why the stops are needed: the hinges' box is the axis hull of the feasible set, and some of its corners
    need a crank past its stop."""
    desc, _, _ = setup
    for lk in desc.linkages:
        (plo, phi), (rlo, rhi) = (desc.calibration_travel(j) for j in lk.joints)
        corners = np.array([[p, r] for p in (plo, phi) for r in (rlo, rhi)])
        assert not desc.ankle_model.feasible(corners, lk.side).all()


def test_ankle_stops_setting_is_explicit():
    physics = load_profile(ROBOT)["physics"]
    assert ankle_stops_setting(physics, "mjwarp") == "cranks"
    assert ankle_stops_setting(physics, "kamino") == "hinges"  # Kamino has no tendons
    with pytest.raises(ValueError, match="ankle_stops"):
        ankle_stops_setting({**physics, "mjwarp": {**physics["mjwarp"], "ankle_stops": "cone"}}, "mjwarp")


def test_a_robot_without_cranks_has_no_stops():
    desc = Description.load(ROBOT)
    desc.linkages = []
    assert crank_stops(desc) == []


# -- physics (Isaac Lab image) -----------------------------------------------------------------------------------

# How far a crank may pass its stop: the stop is MuJoCo's soft limit constraint (profile ankle_stop_solref, time
# constant and damping ratio 1), which yields under load and takes a time constant to stop a crank arriving at speed.
# Under a steady push, a crank driven with its whole baseline spring a full radian past the stop (about 22 Nm,
# three quarters of its 30 Nm peak) rests about 13 mrad past it: STOP_REST_RAD bounds that rest. Arriving at speed
# v, a critically damped stop of time constant tc lets it run at most v tc / e further, so a crank never passes its
# stop by more than STOP_REST_RAD plus its fastest speed times tc. The hinges' box alone lets it run 0.5 rad past.
STOP_REST_RAD = 0.02
# A robot falling with its drives off lands on its feet with the cranks free: the body's weight and the impact load
# the stops on top of the crank's speed, and MuJoCo's soft constraint lets part of that load through (its
# impedance is below 1). Eight such falls (test_a_robot_falling_onto_its_ankle_stops_gains_no_energy) overran by at
# most 0.13 rad at crank speeds up to 10 rad/s; with the hinges' box alone the cranks run 0.5 rad past.
STOP_IMPACT_RAD = 0.15
DRIVEN_PAST_RAD = 1.0  # how far past its stop each crank's target is put
# Sole boxes sink into the ground by the contact's soft penetration under the robot's weight: millimetres. The
# ankle stops write no state, so nothing pushes a foot into the ground. A foot whose ankle is driven into a corner
# of its stops rolls onto its sole's edge, which carries the weight on a line and sinks further.
SOLE_DEPTH_M = 0.010
SOLE_EDGE_DEPTH_M = 0.020
# Energy a fall may appear to gain, as a height times the robot's weight: what a contact's soft penetration of a
# few millimetres stores (test_world's heightfield energy test).
ENERGY_ALLOWANCE_M = 0.01
# Directions in a linkage's crank plane (upper, lower) the cranks are driven in, at the stop and the corners.
CRANK_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))


def _overrun(desc, joints, q, qd=None):
    """``[E]`` the most any crank of ``q`` (URDF joint state) lies past one of its stops (negative inside), from
    the transmission's crank angles (humanoid-urdf ``ankle_model``'s inverse kinematics) and the stops in the
    description; with ``qd``, also ``[E]`` the cranks' fastest speed."""
    worst = torch.full((q.shape[0],), -torch.inf, dtype=torch.float64)
    speed = torch.zeros(q.shape[0], dtype=torch.float64)
    for lk in desc.linkages:
        cols = [joints.index(j) for j in lk.joints]
        tx = desc.ankle_transmission(lk.side, dtype=torch.float64)
        theta = tx.cranks(q[:, cols].double())
        if qd is not None:
            rate = (tx.jacobian(q[:, cols].double()) @ qd[:, cols].double().unsqueeze(-1)).squeeze(-1)
            speed = torch.maximum(speed, rate.abs().amax(-1))
        for c, drive in enumerate((lk.upper, lk.lower)):
            lo, hi = desc.drives[drive].position_limit_device
            worst = torch.maximum(worst, torch.maximum(theta[:, c] - hi, lo - theta[:, c]))
    return worst if qd is None else (worst, speed)


def _time_constant_s():
    return float(load_profile(ROBOT)["physics"]["mjwarp"]["ankle_stop_solref"][0])


@pytest.mark.physics
def test_cranks_driven_past_their_stops_stop_at_them(request):
    """On a welded base, each env drives both cranks of every ankle a radian past their stops in one of eight
    directions of the crank plane (into a stop, into a corner). The solver holds each crank at its stop, within the
    soft constraint's give; no joint state is written."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.physics.crank_stops import check_crank_stops
    from humanoid_sim.sim import SimCore

    n = len(CRANK_DIRECTIONS)
    sim = SimCore(built_usd(ROBOT), device="cpu", robot=ROBOT, fixed_base=True, num_envs=n)
    request.addfinalizer(sim.close)
    assert check_crank_stops(ROBOT, sim.profile["physics"]) == 4 * len(sim.desc.linkages)
    p, desc, joints = sim.plant, sim.desc, list(sim.desc.joints)
    q, qd = sim.joint_state()
    target, _, _ = p.map.drive_state(q, qd)
    names = p.map.drive_names
    directions = torch.tensor(CRANK_DIRECTIONS, dtype=target.dtype)
    for lk in desc.linkages:
        for k, drive in enumerate((lk.upper, lk.lower)):
            i = names.index(drive)
            lo, hi = desc.drives[drive].position_limit_device
            d = directions[:, k]
            target[:, i] = torch.where(d > 0, hi + DRIVEN_PAST_RAD, torch.where(d < 0, lo - DRIVEN_PAST_RAD,
                                                                              target[:, i]))
    p.submit(DriveDemand.pd(target, p.drive_stiffness.expand(n, -1), p.drive_damping.expand(n, -1)),
             torch.zeros(n, dtype=torch.long))
    worst = torch.full((n,), -torch.inf, dtype=torch.float64)
    fastest = torch.zeros(n, dtype=torch.float64)
    for _ in range(250):  # 1 s
        sim.step()
        past, speed = _overrun(desc, joints, *sim.joint_state())
        worst, fastest = torch.maximum(worst, past), torch.maximum(fastest, speed)
    rest = _overrun(desc, joints, sim.joint_state()[0])
    assert (rest > -STOP_REST_RAD).all(), rest  # every env rests on a stop
    assert (rest < STOP_REST_RAD).all(), rest
    assert (worst < STOP_REST_RAD + fastest * _time_constant_s()).all(), (worst, fastest)


def _flat_scene(n):
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg

    sim = sim_utils.SimulationContext(robot_sim_cfg(ROBOT, device="cpu"))
    with robot_build(ROBOT):
        scene = instantiate(robot_scene_cfg(n, built_usd(ROBOT), robot=ROBOT))
        sim.reset()
    return sim, scene


def _step(sim, scene, plant, n):
    for _ in range(n):
        plant.pre_physics()
        scene.write_data_to_sim()
        sim.step()
        scene.update(plant.period_ns / 1e9)


@pytest.mark.physics
def test_standing_feet_stay_on_the_ground_inside_their_stops():
    """Robots holding their home pose on the flat ground, some with their ankles' targets a radian past the stops
    (the stance a policy at zero action can ask for): the soles rest on the ground within a contact's soft
    penetration while the robot is up, and every crank stays at or inside its stops. (At the baseline gains the V1
    holding its home pose topples after about 1.5 s: test_world's toppling test.) The reverted per-step state write
    that stopped the ankle at a cone sank the feet 5 to 10 cm."""
    pytest.importorskip("isaaclab_newton")
    import isaaclab.sim as sim_utils

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.model.asset import asset_meta
    from humanoid_sim.model.kinematics import _sole_corners
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    n = 1 + len(CRANK_DIRECTIONS)  # env 0 holds the home pose; the others drive their cranks past the stops
    sim, scene = _flat_scene(n)
    try:
        art = scene["robot"]
        plant = RobotPlant(art, TrainPlantCfg(robot=ROBOT), imu=None, contacts=scene["contacts"])
        desc, joints, core = plant.desc, plant.joints, plant.core
        stand = art.data.default_root_pose.torch[0, 2].item()
        default = art.data.default_joint_pos.torch
        q = default[:, [list(art.joint_names).index(j) for j in joints]]
        target, _, _ = core.map.drive_state(q, torch.zeros_like(q))
        names = core.map.drive_names
        directions = torch.tensor(((0, 0),) + CRANK_DIRECTIONS, dtype=target.dtype)
        for lk in desc.linkages:
            for k, drive in enumerate((lk.upper, lk.lower)):
                i = names.index(drive)
                lo, hi = desc.drives[drive].position_limit_device
                d = directions[:, k]
                target[:, i] = torch.where(d > 0, hi + DRIVEN_PAST_RAD,
                                           torch.where(d < 0, lo - DRIVEN_PAST_RAD, target[:, i]))
        plant.reset(torch.arange(n), demand=DriveDemand.pd(target, core.drive_stiffness.expand(n, -1),
                                                           core.drive_damping.expand(n, -1)))
        boxes = asset_meta(built_usd(ROBOT))["sole_boxes"]
        soles = [art.body_names.index(b["body"]) for b in boxes]
        ground = scene.env_origins[:, 2]
        worst = torch.full((n,), -torch.inf, dtype=torch.float64)
        fastest = torch.zeros(n, dtype=torch.float64)
        deepest = torch.zeros(n)
        for _ in range(int(2.0e9 / plant.period_ns)):  # 2 s
            _step(sim, scene, plant, 1)
            truth = plant.truth()
            past, speed = _overrun(desc, joints, truth.joint_pos, truth.joint_vel)
            worst, fastest = torch.maximum(worst, past), torch.maximum(fastest, speed)
            poses = art.data.body_link_pose_w.torch.numpy()
            up = art.data.root_link_pose_w.torch[:, 2] - ground > 0.8 * stand
            for e in range(n):
                if up[e]:
                    low = min(float(_sole_corners(poses[e, b], box)[:, 2].min()) for b, box in
                              zip(soles, boxes, strict=True))
                    deepest[e] = min(deepest[e].item(), low - ground[e].item())
        assert (worst < STOP_REST_RAD + fastest * _time_constant_s()).all(), (worst, fastest)
        assert deepest[0] > -SOLE_DEPTH_M, deepest
        assert (deepest > -SOLE_EDGE_DEPTH_M).all(), deepest
    finally:
        sim_utils.SimulationContext.clear_instance()


@pytest.mark.physics
def test_a_robot_falling_onto_its_ankle_stops_gains_no_energy():
    """Drives off, robots tipped over onto the flat ground: their feet flop into the crank stops and the stops
    must only take energy out. The centre-of-mass kinetic plus potential energy (a lower bound of the total) never
    exceeds its start by more than a contact's soft penetration stores, as in the heightfield test."""
    pytest.importorskip("isaaclab_newton")
    import isaaclab.sim as sim_utils

    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    n = 8
    sim, scene = _flat_scene(n)
    try:
        art = scene["robot"]
        plant = RobotPlant(art, TrainPlantCfg(robot=ROBOT), imu=None, contacts=None)
        stand = art.data.default_root_pose.torch[0, 2].item()
        gen = torch.Generator().manual_seed(0)
        tilt = (torch.rand(n, 2, generator=gen) - 0.5) * 0.6  # up to 0.3 rad about x and y: every robot falls
        half = 0.5 * tilt
        quat = torch.stack([torch.sin(half[:, 0]), torch.sin(half[:, 1]), torch.zeros(n),
                            torch.cos(half[:, 0]) * torch.cos(half[:, 1])], 1)
        quat = quat / quat.norm(dim=1, keepdim=True)
        root = torch.cat([scene.env_origins + torch.tensor([0.0, 0.0, stand + 0.05]), quat], 1)
        plant.reset(torch.arange(n), root_pose=root, root_vel=torch.zeros(n, 6))
        mass = art.data.body_mass.torch
        ground = scene.env_origins[:, 2:3]

        def energy():
            z = art.data.body_com_pos_w.torch[..., 2] - ground
            v = art.data.body_com_lin_vel_w.torch
            return (mass * (GRAVITY * z + 0.5 * (v**2).sum(-1))).sum(1)

        _step(sim, scene, plant, 1)
        start, peak = energy(), energy()
        worst = torch.full((n,), -torch.inf, dtype=torch.float64)
        fastest = torch.zeros(n, dtype=torch.float64)
        for _ in range(int(1.5e9 / plant.period_ns)):  # 1.5 s: it tips over and lands
            _step(sim, scene, plant, 1)
            peak = torch.maximum(peak, energy())
            truth = plant.truth()
            past, speed = _overrun(plant.desc, plant.joints, truth.joint_pos, truth.joint_vel)
            worst, fastest = torch.maximum(worst, past), torch.maximum(fastest, speed)
        weight = mass.sum(1) * GRAVITY
        assert (peak <= start + weight * ENERGY_ALLOWANCE_M).all(), (peak - start) / weight
        assert (worst < STOP_IMPACT_RAD).all(), (worst, fastest)
    finally:
        sim_utils.SimulationContext.clear_instance()
