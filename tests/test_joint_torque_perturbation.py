"""JointTorquePerturbation and its composition with ForcePush, on a fake articulation (no simulator)."""

from types import SimpleNamespace

import pytest
import torch

from humanoid_sim.timing import NS_PER_MS
from humanoid_sim.world import disturbances
from humanoid_sim.world.disturbances import ForcePush, JointGeometry, JointTorquePerturbation

BODIES = ["base", "thigh", "shank", "foot", "arm"]
# name: (parent, child, axis in the child frame): thigh and shank are both a child and a parent
JOINTS = {"hip": ("base", "thigh", (0.0, 1.0, 0.0)), "knee": ("thigh", "shank", (0.0, 1.0, 0.0)),
          "ankle": ("shank", "foot", (1.0, 0.0, 0.0)), "shoulder": ("base", "arm", (0.0, 0.0, 1.0))}


class FakeComposer:
    """The permanent wrench composer's contract: a set zeroes every body of the given envs, then writes."""

    def __init__(self, e, b):
        self.force, self.torque = torch.zeros(e, b, 3), torch.zeros(e, b, 3)
        self.calls = []

    def set_forces_and_torques_index(self, forces, torques, body_ids, env_ids, is_global):
        assert is_global
        self.calls.append(list(body_ids.tolist()))
        self.force[env_ids] = 0.0
        self.torque[env_ids] = 0.0
        self.force[env_ids[:, None], body_ids[None, :]] = forces
        self.torque[env_ids[:, None], body_ids[None, :]] = torques


def _quats(e, b, seed=0):
    q = torch.randn(e, b, 4, generator=torch.Generator().manual_seed(seed))
    return q / q.norm(dim=-1, keepdim=True)


def _matrix(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).unflatten(-1, (3, 3))


@pytest.fixture
def robot(monkeypatch):
    e = 64
    pose = torch.cat([torch.zeros(e, len(BODIES), 3), _quats(e, len(BODIES))], dim=-1)
    robot = SimpleNamespace(num_instances=e, device="cpu", body_names=list(BODIES), joint_names=list(JOINTS),
                            data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=pose)),
                            permanent_wrench_composer=FakeComposer(e, len(BODIES)))

    def geometry(robot, names):
        rows = [JOINTS[n] for n in names]
        return JointGeometry(torch.tensor([BODIES.index(p) for p, _, _ in rows]),
                             torch.tensor([BODIES.index(c) for _, c, _ in rows]),
                             torch.tensor([a for _, _, a in rows]))

    monkeypatch.setattr(disturbances, "newton_joint_geometry", geometry)
    return robot


def _perturbation(robot, joints=".*", **kw):
    kw = {"max_nm": 10.0, "hold_s": (0.010, 0.030), "active_share": 1.0, **kw}
    p = JointTorquePerturbation(robot, joints, **kw)
    p.reset(torch.arange(robot.num_instances), 0)
    return p


def _expected_wrench(robot, p):
    """Per body ``[E, B, 3]``: +tau on each joint's child, -tau on its parent, about the joint axis in the world."""
    out = torch.zeros(robot.num_instances, len(BODIES), 3)
    for j, name in enumerate(p.joint_names):
        parent, child, axis = JOINTS[name]
        rot = _matrix(robot.data.body_link_pose_w.torch[:, BODIES.index(child), 3:])
        tau = p.torque_nm[:, j, None] * (rot @ torch.tensor(axis))
        out[:, BODIES.index(child)] += tau
        out[:, BODIES.index(parent)] -= tau
    return out


def test_draws_are_within_the_bounds_per_joint_and_a_share_of_them_is_zero(robot):
    p = _perturbation(robot, max_nm=[1.0, 2.0, 4.0, 8.0], active_share=0.4)
    draws = []
    for k in range(200):
        p.step(k * 5 * NS_PER_MS)
        draws.append(p.torque_nm.clone())
    draws = torch.stack(draws)  # [T, E, J]
    assert draws.shape[-1] == 4 and p.torque_nm.shape == (64, 4)
    assert (draws.abs() <= torch.tensor([1.0, 2.0, 4.0, 8.0])).all()
    assert (draws.abs().amax((0, 1)) > 0.8 * torch.tensor([1.0, 2.0, 4.0, 8.0])).all()
    assert (draws < 0).any() and (draws > 0).any()
    assert (draws != 0).float().mean().item() == pytest.approx(0.4, abs=0.05)


def test_a_share_of_zero_draws_nothing_and_a_share_of_one_always_draws(robot):
    quiet = _perturbation(robot, active_share=0.0)
    quiet.step(0)
    assert (quiet.torque_nm == 0).all() and (robot.permanent_wrench_composer.torque == 0).all()
    busy = _perturbation(robot, active_share=1.0)
    busy.step(0)
    assert (busy.torque_nm != 0).all()


def test_the_level_scales_the_draws_and_a_level_of_zero_is_off(robot):
    full, half = _perturbation(robot, seed=3), _perturbation(robot, seed=3, level=0.5)
    full.step(0)
    half.step(0)
    torch.testing.assert_close(half.torque_nm, 0.5 * full.torque_nm)
    off = _perturbation(robot, level=0.0)
    off.step(0)
    assert (off.torque_nm == 0).all()
    per_env = _perturbation(robot, seed=3, level=torch.linspace(0, 1, robot.num_instances))
    per_env.step(0)
    torch.testing.assert_close(per_env.torque_nm, full.torque_nm * torch.linspace(0, 1, 64)[:, None])


def test_holds_are_integer_ns_within_the_range_and_the_torque_is_constant_until_the_next_draw(robot):
    lo, hi = 10 * NS_PER_MS, 30 * NS_PER_MS
    p = _perturbation(robot, hold_s=(0.010, 0.030))
    p.step(0)
    assert p.next_ns.dtype == torch.int64
    assert ((p.next_ns >= lo) & (p.next_ns <= hi)).all()
    assert p.next_ns.unique().numel() > 100  # distinct holds at ns resolution
    held, due = p.torque_nm.clone(), p.next_ns.clone()
    for t in range(0, hi + 1, NS_PER_MS):
        p.step(t)
        redrawn = due <= t
        assert (p.torque_nm[~redrawn] == held[~redrawn]).all()
        if redrawn.any():
            assert (p.torque_nm[redrawn] != held[redrawn]).all()
            hold = p.next_ns[redrawn] - t
            assert ((hold >= lo) & (hold <= hi)).all()
            held, due = p.torque_nm.clone(), p.next_ns.clone()


def test_reset_zeroes_the_envs_and_redraws_them_at_the_next_step_only(robot):
    p = _perturbation(robot, hold_s=(1.0, 2.0))
    p.step(0)
    before = p.torque_nm.clone()
    p.reset(torch.tensor([3, 5]), 100 * NS_PER_MS)
    assert (p.torque_nm[[3, 5]] == 0).all()
    other = torch.tensor([i for i in range(64) if i not in (3, 5)])
    assert (p.torque_nm[other] == before[other]).all()
    p.step(100 * NS_PER_MS)
    assert (p.torque_nm[[3, 5]] != 0).all() and (p.torque_nm[other] == before[other]).all()
    assert (p.next_ns[[3, 5]] >= 1_100 * NS_PER_MS).all()


def test_the_torque_is_equal_and_opposite_on_child_and_parent_with_no_net_wrench(robot):
    p = _perturbation(robot)
    p.step(0)
    composer = robot.permanent_wrench_composer
    torch.testing.assert_close(composer.torque, _expected_wrench(robot, p), atol=1e-5, rtol=1e-5)
    assert (composer.force == 0).all()
    assert composer.torque.abs().sum() > 0
    net = composer.torque.sum(1)  # about the world axes; a pure torque is free of lever arms
    assert net.abs().max() < 1e-4


def test_one_joint_puts_plus_tau_on_the_child_and_minus_tau_on_the_parent_about_the_world_axis(robot):
    composer = robot.permanent_wrench_composer
    single = _perturbation(robot, "knee")
    single.step(0)
    out = composer.torque
    tau = single.torque_nm[:, 0]
    axis = _matrix(robot.data.body_link_pose_w.torch[:, BODIES.index("shank"), 3:]) @ torch.tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(out[:, BODIES.index("shank")], tau[:, None] * axis, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, BODIES.index("thigh")], -tau[:, None] * axis, atol=1e-5, rtol=1e-5)
    assert (out[:, [0, 3, 4]] == 0).all()


def test_the_perturbation_does_not_move_with_the_hold_but_follows_the_link_pose(robot):
    p = _perturbation(robot, "knee", hold_s=(1.0, 1.0))
    p.step(0)
    first = robot.permanent_wrench_composer.torque.clone()
    robot.data.body_link_pose_w.torch[..., 3:] = _quats(64, len(BODIES), seed=9)
    p.step(NS_PER_MS)
    assert not torch.allclose(robot.permanent_wrench_composer.torque, first)
    torch.testing.assert_close(robot.permanent_wrench_composer.torque, _expected_wrench(robot, p), atol=1e-5,
                               rtol=1e-5)


@pytest.mark.parametrize("push_first", [True, False])
def test_it_adds_to_a_force_push_on_a_body_of_the_joint_in_either_order(robot, push_first):
    push = ForcePush(robot, "shank", interval_s=(0.0, 0.0), duration_s=(1.0, 1.0), magnitude_n=(30.0, 30.0))
    push.reset(torch.arange(64), 0)
    p = _perturbation(robot, "knee|ankle")  # the shank is the knee's child and the ankle's parent
    for t in (0, NS_PER_MS):
        for source in ((push, p) if push_first else (p, push)):
            source.step(t)
    composer = robot.permanent_wrench_composer
    assert push.force.abs().sum() > 0 and p.torque_nm.abs().sum() > 0
    torch.testing.assert_close(composer.torque, _expected_wrench(robot, p), atol=1e-5, rtol=1e-5)
    expected = torch.zeros(64, len(BODIES), 3)
    expected[:, BODIES.index("shank")] = push.force
    torch.testing.assert_close(composer.force, expected)


def test_a_force_push_alone_writes_its_one_body_as_before(robot):
    push = ForcePush(robot, "shank", interval_s=(0.0, 0.0), duration_s=(1.0, 1.0), magnitude_n=(30.0, 30.0))
    push.reset(torch.arange(64), 0)
    push.step(0)
    composer = robot.permanent_wrench_composer
    assert composer.calls == [[BODIES.index("shank")]]
    torch.testing.assert_close(composer.force[:, BODIES.index("shank")], push.force)
    assert (composer.torque == 0).all()
    assert (composer.force[:, [0, 1, 3, 4]] == 0).all()


def test_two_force_pushes_on_the_articulation_add(robot):
    a = ForcePush(robot, "shank", interval_s=(0.0, 0.0), duration_s=(1.0, 1.0), magnitude_n=(30.0, 30.0), seed=1)
    b = ForcePush(robot, "shank", interval_s=(0.0, 0.0), duration_s=(1.0, 1.0), magnitude_n=(30.0, 30.0), seed=2)
    for x in (a, b):
        x.reset(torch.arange(64), 0)
        x.step(0)
    torch.testing.assert_close(robot.permanent_wrench_composer.force[:, BODIES.index("shank")], a.force + b.force)


def test_joints_are_selected_by_name_or_regex_in_the_order_given(robot):
    assert JointTorquePerturbation(robot, ["shoulder", "hip"], max_nm=1.0, hold_s=(0.1, 0.2),
                                   active_share=1.0).joint_names == ["shoulder", "hip"]
    assert JointTorquePerturbation(robot, "hip|k.*", max_nm=1.0, hold_s=(0.1, 0.2),
                                   active_share=1.0).joint_names == ["hip", "knee"]
    assert JointTorquePerturbation(robot, ["a.*", "ankle", "hip"], max_nm=1.0, hold_s=(0.1, 0.2),
                                   active_share=1.0).joint_names == ["ankle", "hip"]
    with pytest.raises(ValueError, match="no joint matches"):
        JointTorquePerturbation(robot, "elbow", max_nm=1.0, hold_s=(0.1, 0.2), active_share=1.0)


@pytest.mark.parametrize("kw", [{"max_nm": [1.0, 2.0]}, {"max_nm": -1.0}, {"max_nm": float("nan")},
                                {"hold_s": (0.0, 0.1)}, {"hold_s": (0.2, 0.1)}, {"active_share": 1.5},
                                {"active_share": -0.1}])
def test_invalid_arguments_raise(robot, kw):
    args = {"max_nm": 1.0, "hold_s": (0.1, 0.2), "active_share": 0.5, **kw}
    with pytest.raises(ValueError):
        JointTorquePerturbation(robot, ["hip", "knee", "ankle"], **args)


def test_the_same_seed_gives_the_same_torques(robot):
    runs = []
    for _ in range(2):
        p = _perturbation(robot, seed=7, active_share=0.5)
        for k in range(20):
            p.step(k * 5 * NS_PER_MS)
        runs.append(p.torque_nm.clone())
    assert torch.equal(*runs)
