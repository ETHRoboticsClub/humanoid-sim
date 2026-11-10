"""Kamino's soft crank stops (physics/crank_stops.py): the profile block, the torque law, the explicit-step limit and
the torque pair on a stand-in articulation. The solver side is in test_kamino_crank_stops_physics.py."""

import types

import pytest
import torch

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.physics.crank_stops import (
    STABLE_MEASURE,
    KaminoCrankStops,
    kamino_crank_stops,
    kamino_stop_settings,
    stability_measure,
    stop_torque,
)

ROBOT = "ethrc_v1"
PERIOD_S = 1e-3
ARMATURE = 0.0056


def _physics(**crank_stops):
    return load_profile(ROBOT, physics_overrides={"kamino": {"crank_stops": crank_stops}} if crank_stops else None)["physics"]


# -- the profile block ---------------------------------------------------------------------------------------------

def test_the_default_block_keeps_the_hard_limits_and_names_a_stiff_nominal():
    settings = kamino_stop_settings(_physics())
    assert settings.soft is False
    assert settings.stiffness > 0 and settings.damping > 0 and settings.backstop_rad > 0
    # Stable at the V1's step on its crank armature, with room for a randomized ratio of a few.
    assert stability_measure(torch.tensor(settings.stiffness * 3), torch.tensor(settings.damping * 1.25),
                             torch.tensor(ARMATURE), PERIOD_S) < STABLE_MEASURE


def test_overrides_reach_the_settings():
    settings = kamino_stop_settings(_physics(soft=True, stiffness_nm_per_rad=500.0))
    assert settings.soft and settings.stiffness == 500.0


@pytest.mark.parametrize("bad", [{"soft": 1}, {"soft": "yes"}, {"stiffness_nm_per_rad": 0.0},
                                 {"damping_nm_s_per_rad": -1.0}, {"backstop_rad": float("nan")},
                                 {"stiffness_nm_per_rad": None}, {"backstop_rad": True}])
def test_a_bad_block_raises(bad):
    physics = _physics()
    physics["kamino"]["crank_stops"] = {**physics["kamino"]["crank_stops"], **bad}
    with pytest.raises(ValueError, match="crank_stops"):
        kamino_stop_settings(physics)


def test_a_block_with_other_keys_raises():
    physics = _physics()
    physics["kamino"]["crank_stops"]["stiffness"] = 1.0
    with pytest.raises(ValueError, match="exactly"):
        kamino_stop_settings(physics)
    del physics["kamino"]["crank_stops"]
    with pytest.raises(ValueError, match="exactly"):
        kamino_stop_settings(physics)


# -- the torque law ------------------------------------------------------------------------------------------------

LOWER, UPPER = torch.tensor(-0.3), torch.tensor(0.4)


def _torque(q, qd, k=1000.0, b=10.0):
    return stop_torque(torch.tensor(q), torch.tensor(qd), LOWER, UPPER, torch.tensor(k), torch.tensor(b))


def test_no_torque_inside_the_window_whatever_the_speed():
    for q in (-0.3, 0.0, 0.4):
        for qd in (-5.0, 0.0, 5.0):
            assert float(_torque(q, qd)) == 0.0


def test_a_spring_and_damper_past_each_bound():
    assert float(_torque(0.41, 0.0)) == pytest.approx(-10.0)  # k x
    assert float(_torque(0.41, 0.5)) == pytest.approx(-10.0 - 5.0)  # moving deeper: + b qd
    assert float(_torque(-0.32, 0.0)) == pytest.approx(20.0)
    assert float(_torque(-0.32, -0.5)) == pytest.approx(20.0 + 5.0)


def test_a_stop_never_pulls_the_joint_back_in_nor_holds_it_out():
    # Leaving fast just past the bound: the damper would pull outward, the stop gives zero instead.
    assert float(_torque(0.4005, -5.0)) == 0.0
    assert float(_torque(-0.3005, 5.0)) == 0.0
    # Leaving slowly deep in: still pushed back in, by less.
    assert float(_torque(0.45, -1.0)) == pytest.approx(-(1000 * 0.05 - 10.0))


def test_the_torque_is_per_env_and_per_crank():
    q = torch.tensor([[0.5, 0.0], [0.5, 0.5]])
    k = torch.tensor([[100.0], [400.0]])
    torque = stop_torque(q, torch.zeros(2, 2), LOWER, UPPER, k, torch.zeros(2, 1))
    torch.testing.assert_close(torque, torch.tensor([[-10.0, 0.0], [-40.0, -40.0]]))


def test_the_stability_measure_is_jurys_bound():
    # k dt²/I + 2 b dt/I < 4 and b dt/I < 2: simulate the explicit symplectic step and compare.
    def stable(k, b, inertia=ARMATURE, dt=PERIOD_S):
        x, v = 1.0, 0.0
        for _ in range(4000):
            v += dt * (-k * x - b * v) / inertia
            x += dt * v
            if abs(x) > 1e6:
                return False
        return True

    for k, b in ((2000.0, 2.0), (6000.0, 3.0), (21000.0, 0.0), (23000.0, 0.0), (10000.0, 30.0), (1000.0, 12.0)):
        m = float(stability_measure(torch.tensor(k), torch.tensor(b), torch.tensor(ARMATURE), PERIOD_S))
        assert stable(k, b) == (m < 4.0), (k, b, m)
    assert STABLE_MEASURE == 2.0


# -- the torque pair on a stand-in articulation -------------------------------------------------------------------

class _Composer:
    def __init__(self):
        self.calls = []

    def add_forces_and_torques_index(self, **kw):
        self.calls.append(kw)


def _quat_about_z(angle):
    return torch.tensor([0.0, 0.0, float(torch.sin(angle / 2)), float(torch.cos(angle / 2))])


def _matrix_to_quat(m):
    w = torch.sqrt(torch.clamp(1 + m[0, 0] + m[1, 1] + m[2, 2], min=1e-12)) / 2
    return torch.stack([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w), w])


class _Robot:
    """The articulation's surface the stops use, with a state that keeps each crank child on its joint's frames."""

    def __init__(self, desc, num_envs=3, compliance=False, q=None):
        from humanoid_sim.model.usd import crank_joints
        from humanoid_sim.physics.crank_stops import _about
        from humanoid_sim.transmissions.compliance import output_body

        self.device = "cpu"
        self.num_instances = num_envs
        joints = list(crank_joints(desc).values())
        names = ["root"]
        for joint in joints:
            parent, child = desc.usd_model.joint_bodies(joint)
            names += [parent, output_body(joint) if compliance else child]
        self.body_names = list(dict.fromkeys(names))
        self.joint_names = ["other", *joints]
        g = torch.Generator().manual_seed(0)
        pose = torch.zeros(num_envs, len(self.body_names), 7)
        pose[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        qs = torch.zeros(num_envs, len(self.joint_names))
        parents = {}
        qs[:, 1:] = (torch.rand(num_envs, len(joints), generator=g) - 0.5) if q is None else q
        for c, joint in enumerate(joints):
            parent, child = desc.usd_model.joint_bodies(joint)
            child = output_body(joint) if compliance else child
            r0, r1 = (torch.tensor(f[:3, :3], dtype=torch.float32) for f in desc.usd_model.joint_frames(joint))
            axis = desc.usd_model.joint_axis(joint)
            for e in range(num_envs):
                if (e, parent) not in parents:
                    rp = torch.linalg.qr(torch.randn(3, 3, generator=g))[0]
                    parents[e, parent] = rp * torch.sign(torch.det(rp))
                rp = parents[e, parent]
                rc = rp @ r0 @ _about(axis, qs[e, 1 + c]) @ r1.T
                pose[e, self.body_names.index(parent), 3:] = _matrix_to_quat(rp)
                pose[e, self.body_names.index(child), 3:] = _matrix_to_quat(rc)
        self.data = types.SimpleNamespace(
            body_link_pose_w=types.SimpleNamespace(torch=pose),
            joint_pos=types.SimpleNamespace(torch=qs), joint_vel=types.SimpleNamespace(torch=torch.zeros_like(qs)),
            joint_armature=types.SimpleNamespace(torch=torch.full_like(qs, ARMATURE)))
        self.instantaneous_wrench_composer = _Composer()
        self.limit_writes = []

    def write_joint_position_limit_to_sim_index(self, **kw):
        self.limit_writes.append(kw)


@pytest.fixture(scope="module")
def desc():
    return Description.load(ROBOT)


def _stops(desc, robot, **crank_stops):
    settings = kamino_stop_settings(_physics(soft=True, **crank_stops))
    return KaminoCrankStops(robot, desc, settings, PERIOD_S)


@pytest.mark.parametrize("compliance", [False, True])
def test_building_widens_each_cranks_hard_limit_to_a_backstop(desc, compliance):
    robot = _Robot(desc, compliance=compliance)
    stops = _stops(desc, robot)
    (write,) = robot.limit_writes
    limits = write["limits"]
    assert limits.shape == (3, 4, 2)
    pad = stops.settings.backstop_rad
    for c, drive in enumerate(stops.drives):
        lo, hi = desc.drives[drive].shaft_limits
        torch.testing.assert_close(limits[:, c, 0], torch.full((3,), lo - pad))
        torch.testing.assert_close(limits[:, c, 1], torch.full((3,), hi + pad))
    assert write["joint_ids"].tolist() == [robot.joint_names.index(j) for j in stops.joints]


def test_a_wrong_axis_or_frame_is_caught_at_build(desc):
    robot = _Robot(desc)
    robot.data.joint_pos.torch[:, 1:] *= -1.0  # a state the joints' axes do not produce
    with pytest.raises(RuntimeError, match="stop torque's axis"):
        _stops(desc, robot)


@pytest.mark.parametrize("compliance", [False, True])
def test_the_torque_pair_is_equal_and_opposite_about_the_joint_axis(desc, compliance):
    from humanoid_sim.model.usd import crank_joints

    mid = torch.tensor([[sum(desc.drives[d].shaft_limits) / 2 for d in crank_joints(desc)]]).expand(3, -1)
    robot = _Robot(desc, compliance=compliance, q=mid)
    stops = _stops(desc, robot, stiffness_nm_per_rad=1000.0, damping_nm_s_per_rad=2.0)
    # Env 1's first crank is 0.05 rad past its upper bound, env 2's second 0.02 past its lower, the bodies kept on
    # the joints' frames at the moved state.
    q = mid.clone()
    q[1, 0] = stops.upper[0] + 0.05
    q[2, 1] = stops.lower[1] - 0.02
    robot.data.joint_pos.torch[:, 1:] = q
    robot.data.joint_vel.torch[1, 1] = 0.5
    robot.data.body_link_pose_w.torch.copy_(_Robot(desc, compliance=compliance, q=q).data.body_link_pose_w.torch)
    stops.apply()
    expected = torch.tensor([[0.0] * 4, [-(1000 * 0.05 + 2 * 0.5), 0, 0, 0], [0, 1000 * 0.02, 0, 0]])
    torch.testing.assert_close(stops.torque, expected)
    (call,) = robot.instantaneous_wrench_composer.calls
    assert call["is_global"] and call["env_ids"].tolist() == [0, 1, 2]
    assert call["forces"].abs().max() == 0
    torques = call["torques"]
    bodies = call["body_ids"].tolist()
    assert len(bodies) == len(set(bodies))
    # A pure joint torque: no net torque on the pair, and none at all with the cranks inside their windows.
    torch.testing.assert_close(torques.sum(dim=1), torch.zeros(3, 3), atol=1e-4, rtol=0)
    assert torques[0].abs().max() == 0
    # Env 1: magnitude of the first crank's torque on its child body, along the joint's world axis.
    from humanoid_sim.physics.crank_stops import _rotate

    axis_w = _rotate(robot.data.body_link_pose_w.torch[1, stops.parent[0], 3:], stops.axis[0])  # one crank
    child = bodies.index(int(stops.child[0]))
    torch.testing.assert_close(torques[1, child], expected[1, 0] * axis_w, atol=1e-4, rtol=1e-4)
    parent = bodies.index(int(stops.parent[0]))
    assert parent != child


def test_the_scales_are_per_env_and_checked(desc):
    robot = _Robot(desc)
    stops = _stops(desc, robot)
    s, c = torch.tensor([[1.0], [0.1], [3.0]]), torch.tensor([[1.0], [2.0], [0.5]])
    stops.set_scales(s, c)
    torch.testing.assert_close(stops.stiffness, 2000.0 * s)
    torch.testing.assert_close(stops.damping, 2.0 * c)
    stops.set_scales()
    torch.testing.assert_close(stops.stiffness, torch.full((3, 1), 2000.0))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="crank_stop_stiffness"):
            stops.set_scales(torch.full((3, 1), bad), None)
        with pytest.raises(ValueError, match="crank_stop_damping"):
            stops.set_scales(None, torch.full((3, 1), bad))


def test_a_stop_the_step_cannot_carry_explicitly_is_refused_with_the_limit(desc):
    robot = _Robot(desc)
    stops = _stops(desc, robot)
    k_max = STABLE_MEASURE * ARMATURE / PERIOD_S**2  # at no damping
    stops.set_scales(torch.full((3, 1), 0.99 * k_max / 2000.0), torch.full((3, 1), 1e-3))
    with pytest.raises(ValueError, match="explicit stop carries at the 1 ms step"):
        stops.set_scales(torch.full((3, 1), 1.1 * k_max / 2000.0), torch.full((3, 1), 1e-3))
    with pytest.raises(ValueError, match="armature of 0.0056"):
        stops.set_scales(torch.ones(3, 1), torch.full((3, 1), 30.0))  # damping 60 N m s/rad: 2 b dt/I = 21
    # One env past the limit is enough, and the earlier realized values stay.
    wide = torch.ones(3, 1)
    wide[2] = 1e3
    with pytest.raises(ValueError, match="explicit stop"):
        stops.set_scales(wide, None)
    torch.testing.assert_close(stops.stiffness, torch.full((3, 1), 0.99 * k_max))


def test_no_armature_means_no_explicit_guarantee(desc):
    robot = _Robot(desc)
    robot.data.joint_armature.torch[0, 1] = 0.0
    with pytest.raises(ValueError, match="armature"):
        _stops(desc, robot)


def test_the_stops_exist_only_when_asked_for_and_once_per_articulation(desc):
    robot = _Robot(desc)
    assert kamino_crank_stops(robot, desc, _physics(), PERIOD_S) is None  # soft: false, nothing requires them
    assert robot.limit_writes == []
    required = kamino_crank_stops(robot, desc, _physics(), PERIOD_S, require=True)
    assert required is not None and len(robot.limit_writes) == 1
    assert kamino_crank_stops(robot, desc, _physics(), PERIOD_S) is required
    assert kamino_crank_stops(robot, desc, _physics(), PERIOD_S, require=True) is required
    assert len(robot.limit_writes) == 1
    other = _Robot(desc)
    assert kamino_crank_stops(other, desc, _physics(soft=True), PERIOD_S) is not None


def test_a_robot_without_a_native_ankle_has_no_kamino_stops():
    g1 = Description.load("g1_29dof_rev_1_0")
    assert kamino_crank_stops(types.SimpleNamespace(), g1, load_profile("g1_29dof_rev_1_0")["physics"], PERIOD_S,
                              require=True) is None
