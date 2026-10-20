"""PACE fit: parameter layout, decoding and PACE's bounds."""

import math

import numpy as np
import pytest

from humanoid_sim.sysid.pace import PaceSpace

J = ["a", "b"]
D = ["da", "db"]
START = {"armature": np.array([0.1, 0.2]), "damping": np.array([1.0, 2.0]), "friction": np.array([0.3, 0.4])}


def test_layout_is_paces():
    s = PaceSpace(J, D, START)
    assert s.dim == 4 * 2 + 1 + 2
    assert s.names[:2] == ["armature.a", "armature.b"]
    assert s.names[6:9] == ["bias.da", "bias.db", "delay.all"]
    assert s.names[-1] == "friction_ratio.b"
    assert PaceSpace(J, D, START, separate_dynamic_friction=False).dim == 4 * 2 + 1


def test_decode_starts_at_the_start_values_and_spans_the_bounds():
    s = PaceSpace(J, D, START, log_bound=math.log(10.0), bias_bound_rad=0.01, delay_bounds_ns=(0, 10_000_000))
    x0 = s.x0()
    assert np.allclose(x0[s.slices()["friction_ratio"]], 1.0)  # the description's ratio: no dynamic friction
    v = s.decode(np.stack([x0, np.ones(s.dim), -np.ones(s.dim), 3 * np.ones(s.dim)]))
    assert np.allclose(v["armature"][0], START["armature"]) and np.allclose(v["bias"][0], 0.0)
    assert v["delay"].tolist() == [5_000_000, 10_000_000, 0, 10_000_000]  # outside the box: at its edge
    assert np.allclose(v["damping"][1], 10 * START["damping"]) and np.allclose(v["friction"][2], START["friction"] / 10)
    assert np.allclose(v["friction_ratio"][0], 1.0) and np.allclose(v["friction_ratio"][2], 0.0)
    assert np.allclose(v["bias"][2], -0.01)
    r = PaceSpace(J, D, START | {"friction_ratio": np.array([0.25, 0.5])})
    assert np.allclose(r.decode(r.x0())["friction_ratio"], [0.25, 0.5])


def test_fixed_entries_stay_at_the_start():
    s = PaceSpace(J, D, START, fixed={"bias": np.array([False, True]), "damping": np.array([True, False]),
                                      "friction_ratio": np.array([True, False])})
    v = s.decode(-np.ones((1, s.dim)))
    assert v["bias"][0].tolist() == [-s.bias_bound_rad, 0.0]
    assert v["friction_ratio"][0].tolist() == [1.0, 0.0]  # fixed at the start, not at the middle
    assert np.allclose(v["damping"][0], [START["damping"][0], START["damping"][1] / 10])


def test_pace_bounds_and_params_put_the_delay_in_steps():
    s = PaceSpace(J, D, START, delay_bounds_ns=(0, 20_000_000))
    b = s.bounds(4_000_000)
    assert b.shape == (s.dim, 2)
    assert b[s.slices()["delay"]].tolist() == [[0.0, 5.0]]
    v = {k: x[0] for k, x in s.decode(s.x0()[None]).items()}
    v["delay"] = np.int64(10_700_000)
    p = s.pace_params(v, 4_000_000)
    assert p.shape == (s.dim,) and p[s.slices()["delay"]][0] == 3  # PACE's delay buffer takes whole steps
    assert ((b[:, 0] <= p) & (p <= b[:, 1])).all()


def test_refuses_bad_spaces():
    with pytest.raises(ValueError, match="positive"):
        PaceSpace(J, D, START | {"friction": np.array([0.0, 0.1])})
    with pytest.raises(ValueError, match="ratio"):
        PaceSpace(J, D, START, ratio_bounds=(0.5, 1.5))
    with pytest.raises(ValueError, match="fixed"):
        PaceSpace(J, D, START, fixed={"bias": np.array([True])})


def test_command_frames_hold_each_joint_until_it_is_commanded_again():
    from humanoid_sim.replay.exc import CommandTrace

    c = CommandTrace.from_frames([(20, {"a": 2.0}), (10, {"a": 1.0, "b": 5.0}), (20, {"b": 6.0}), (30, {})])
    assert c.t_ns.tolist() == [10, 20, 30] and c.joints == ["a", "b"]
    assert c.positions_rad.tolist() == [[1.0, 5.0], [2.0, 6.0], [2.0, 6.0]]


@pytest.fixture(scope="module")
def plant():
    import torch

    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.description import Description

    desc = Description.load()
    return PlantCore(desc, list(desc.joints), 1, PlantCfg(dtype=torch.float64))


def test_drive_targets_as_the_master_builds_them(plant):
    """1:1: sign · (q + standing offset); ankles: the cranks of the hinges at standing offset + q (the standing
    pose's cranks at q = 0)."""
    import torch

    from humanoid_sim.replay.exc import CommandTrace
    from humanoid_sim.replay.replay import drive_targets

    desc = plant.desc
    joints = list(desc.joints)
    q = np.zeros((2, len(joints)))
    q[1, joints.index("knee_l")] = 0.1
    q[1, joints.index("foot_pitch_l")] = 0.05
    targets, t = drive_targets(plant, CommandTrace(np.array([0, 20]), joints, q))
    drives = plant.map.drive_names
    knee = desc.joints["knee_l"]
    assert targets[1, drives.index(knee.device)] == pytest.approx(knee.sign * (0.1 + knee.standing_offset_rad))
    standing = torch.tensor([[desc.joints[j].standing_offset_rad for j in joints]], dtype=torch.float64)
    qdrv, _, _ = plant.map.drive_state(standing, torch.zeros_like(standing))
    assert np.allclose(targets[0], qdrv[0].numpy())
    up, lo = desc.ankle_cranks("l")
    assert not np.isclose(targets[1, drives.index(up)], targets[0, drives.index(up)])
    with pytest.raises(ValueError, match="never name"):
        drive_targets(plant, CommandTrace(np.array([0]), joints[1:], q[:1, 1:]))


def test_command_demand_rms_finds_the_shift(plant):
    from humanoid_sim.replay.exc import CommandTrace, DriveTrace, sample_hold
    from humanoid_sim.replay.replay import command_demand_rms, drive_targets

    joints = list(plant.desc.joints)
    t_cmd = np.arange(100, dtype=np.int64) * 20_000_000
    q = 0.05 * np.sin(np.arange(100)[:, None] / 7 + np.arange(len(joints))[None])
    cmds = CommandTrace(t_cmd, joints, q)
    targets, _ = drive_targets(plant, cmds)
    t = np.arange(0, 99 * 20_000_000, 5_000_000, dtype=np.int64)

    class Cap:
        def commands(self):
            return cmds

        def pdo(self):
            return {d: DriveTrace(d, t, {"position_demand_rad": sample_hold(t_cmd, targets[:, i], t - 6_000_000)})
                    for i, d in enumerate(plant.map.drive_names)}

    r = command_demand_rms(Cap(), plant, np.arange(0, 12_000_001, 1_000_000))
    assert r["best_shift_ns"] == 6_000_000 and max(r["rms_at_best_rad"].values()) < 1e-12


def test_excitation_capture_reads_its_commands(tmp_path):
    import json

    from humanoid_sim.replay.exc import FORMAT, ExcitationCapture

    (tmp_path / "manifest.json").write_text(json.dumps({"format": FORMAT, "complete": True, "dataset_id": "exc-x"}))
    (tmp_path / "commands.csv").write_text(
        "seq,t_frame_ros_ns,t_signal_s,joint_id,q_nominal_rad,q_mirror_rad\n"
        "1,100,0.0,knee_l,0.1,0.1\n1,100,0.0,knee_r,0.2,0.2\n2,120,0.02,knee_l,0.3,0.3\n")
    c = ExcitationCapture(tmp_path).commands()
    assert c.t_ns.tolist() == [100, 120] and c.positions_rad.tolist() == [[0.1, 0.2], [0.3, 0.2]]


def test_write_follows_paces_bias_sign_and_sets_the_relief(plant):
    """PACE's controller sees the truth minus the bias: the drive's reading offset is -bias. The relief is the
    static friction times one minus the ratio; the solver gets armature, damping and static friction."""
    import torch

    from humanoid_sim.sysid.pace import start_values, write

    class Robot:
        device, written = "cpu", {}

        def __init__(self, joints):
            self.joint_names = list(joints)

        def write_joint_armature_to_sim_index(self, armature, joint_ids):
            self.written["armature"] = armature

        def write_joint_viscous_friction_coefficient_to_sim_index(self, joint_viscous_friction_coeff, joint_ids):
            self.written["damping"] = joint_viscous_friction_coeff

        def write_joint_friction_coefficient_to_sim_index(self, joint_friction_coeff, joint_ids):
            self.written["friction"] = joint_friction_coeff

    class Sim:
        pass

    sim = Sim()
    sim.plant, sim.backend = plant, Sim()
    joints, drives = list(plant.map.joint_names), list(plant.map.drive_names)
    sim.backend.robot = Robot(reversed(joints))  # the articulation's own order
    space = PaceSpace(joints, drives, start_values(plant.desc, joints))
    x = space.x0()
    x[space.slices()["bias"]] = 0.5
    x[space.slices()["friction_ratio"]] = 0.0  # ratio 0.5
    v = space.decode(x[None])
    write(sim, space, v)
    assert torch.allclose(plant.bank.position_offset, torch.full_like(plant.bank.position_offset,
                                                                       -0.5 * space.bias_bound_rad))
    assert np.allclose(plant.friction_relief[0].numpy(), 0.5 * v["friction"][0])
    assert np.allclose(sim.backend.robot.written["friction"].numpy(), v["friction"])
    plant.bank.position_offset.zero_()
    plant.friction_relief.zero_()


def test_blocks_cover_every_entry_once(plant):
    from humanoid_sim.sysid.pace import blocks, start_values

    joints, drives = list(plant.map.joint_names), list(plant.map.drive_names)
    space = PaceSpace(joints, drives, start_values(plant.desc, joints), fixed={"bias": np.eye(len(drives))[0] > 0})
    parts = blocks(space, plant, list(range(len(drives))))
    entries = np.concatenate([idx for idx, _ in parts])
    assert len(entries) == len(set(entries.tolist())) == space.dim - 1  # every entry once, the fixed bias out
    assert len(parts) == 18 + 2 + 1  # 1:1 drives, ankles, the delay
    up, lo = plant.desc.ankle_cranks("l")
    ankle = next(ds for _, ds in parts if drives.index(up) in ds)
    assert sorted(ankle) == sorted([drives.index(up), drives.index(lo)])
    assert parts[-1][1] == list(range(len(drives)))  # the delay scores on everything
    skipped = blocks(space, plant, [d for d in range(len(drives)) if d != 3])
    assert len(skipped) == len(parts) - 1


def test_block_search_converges_on_separable_drives(monkeypatch):
    """Each drive's error depends only on its block: the block search finds every target."""
    import torch

    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.description import Description
    from humanoid_sim.sysid import pace

    desc = Description.load()
    plant = PlantCore(desc, list(desc.joints), 48, PlantCfg(dtype=torch.float64))
    joints, drives = list(plant.map.joint_names), list(plant.map.drive_names)
    space = PaceSpace(joints, drives, pace.start_values(desc, joints))
    parts = pace.blocks(space, plant, list(range(len(drives))))
    target = np.random.default_rng(1).uniform(-0.6, 0.6, space.dim)

    def evaluate(sim, captures, space, x, scored, **_):
        per = np.zeros((1, len(x), len(drives)))
        for idx, ds in parts[:-1]:
            per[0][:, ds] = np.sqrt(((x[:, idx] - target[idx]) ** 2).sum(axis=1))[:, None]
        per[0] += np.abs(x[:, parts[-1][0]] - target[parts[-1][0]])  # the delay: on every drive
        return per[0].mean(axis=1), per

    monkeypatch.setattr(pace, "evaluate", evaluate)

    class Sim:
        pass

    sim = Sim()
    sim.plant = plant
    r = pace.fit(sim, [None], space, list(range(len(drives))), generations=60, base_motion=None, log=lambda *a, **k: None)
    assert r["objective"] < 0.02 * r["objective_start"]
    assert np.abs(np.asarray(r["x"]) - target).max() < 0.05
