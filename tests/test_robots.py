"""Every robot in humanoid-urdf loads, builds and steps the plant: the ETHRC V1 and the Unitree G1 variants."""

import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description, robots
from humanoid_sim.model.mirror import mirror_map
from humanoid_sim.model.poses import home_pose
from humanoid_sim.model.urdf import PreparedUrdf

ROBOTS = robots()


def test_the_description_root_holds_the_v1_and_the_g1():
    assert {"ethrc_v1", "g1_29dof_rev_1_0", "g1_29dof_with_hand_rev_1_0"} <= set(ROBOTS)


def test_an_unknown_robot_names_the_known_ones():
    with pytest.raises(ValueError, match="ethrc_v1"):
        Description.load("no_such_robot")


@pytest.mark.parametrize("robot", ROBOTS)
def test_drives_cover_every_joint(robot):
    desc = Description.load(robot)
    one_to_one = {d.joints[0] for d in desc.drives.values() if d.crank is None}
    linked = {j for lk in desc.linkages for j in lk.joints}
    assert one_to_one | linked == set(desc.joints)
    assert not one_to_one & linked
    for d in desc.drives.values():
        assert d.baseline_stiffness > 0 and d.baseline_damping > 0 and d.effort_peak > 0
        assert desc.torque_speed(d.family)
    for j in desc.joints.values():
        assert j.armature > 0 and j.damping >= 0 and j.friction >= 0


def test_unitree_motors_keep_their_slots():
    desc = Description.load("g1_29dof_with_hand_rev_1_0")
    assert sorted(d.motor_index for d in desc.drives.values()) == list(range(29))
    assert desc.drives["left_hip_pitch_joint"].motor_index == 0
    assert not desc.linkages


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_profile_merges_over_the_defaults(robot):
    profile = load_profile(robot)
    assert profile["robot"] == robot
    assert profile["physics"]["solver"] in profile["physics"]
    assert profile["sensors"]["imu"]["sample_period_ns"] > 0


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_prepared_urdf_limits_every_joint(robot):
    desc = Description.load(robot)
    p = PreparedUrdf(desc, load_profile(robot)).build()
    movable = {j.get("name") for j in p.root.findall("joint") if j.get("type") != "fixed"}
    assert movable == set(desc.joints)


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_mirror_map_pairs_left_with_right(robot):
    m = mirror_map(Description.load(robot))
    for i, name in enumerate(m.joint_names):
        partner = m.joint_names[m.joint_perm[i]]
        assert m.joint_perm[m.joint_perm[i]] == i
        if name.startswith("left_") or name.endswith("_l"):
            assert partner != name


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_home_pose_lies_inside_the_travel(robot):
    desc = Description.load(robot)
    home = home_pose(desc, load_profile(robot))
    for name, q in home.items():
        lo, hi = desc.calibration_travel(name)
        assert lo <= q <= hi, name


def test_the_g1_home_pose_is_unitrees_standing_pose():
    desc = Description.load("g1_29dof_with_hand_rev_1_0")
    home = home_pose(desc, load_profile(desc.robot))
    assert home["left_knee_joint"] == pytest.approx(0.3)
    assert home["right_ankle_pitch_joint"] == pytest.approx(-0.2)
    assert home["waist_yaw_joint"] == 0.0


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_plant_holds_the_home_pose(robot):
    desc = Description.load(robot)
    joints = list(desc.joints)
    plant = PlantCore(desc, joints, 2, PlantCfg(robot=robot))
    home = home_pose(desc, load_profile(robot))
    q = torch.tensor([[home[n] for n in joints]] * 2)
    qd = torch.zeros_like(q)
    q_drive, _, _ = plant.map.drive_state(q, qd)
    hold = DriveDemand.jvpt(q_drive, plant.drive_stiffness.expand(2, -1), plant.drive_damping.expand(2, -1))
    plant.submit(hold, torch.zeros(2, dtype=torch.long))
    target = plant.pre_physics(0, q, qd)
    assert torch.isfinite(target).all()
    assert torch.allclose(target, q, atol=1e-4)  # at the target already: no torque, the target stays put
