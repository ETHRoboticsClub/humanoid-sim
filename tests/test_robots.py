"""Every robot in humanoid-urdf loads, builds and steps the plant: the ETHRC V1 and the Unitree G1 variants."""

import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description, robots
from humanoid_sim.model.mirror import mirror_map
from humanoid_sim.model.poses import home_pose, pose_joint_positions, start_pose
from humanoid_sim.model.urdf import PreparedUrdf
from humanoid_sim.transmissions.drive_map import DriveJointMap

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


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_prepared_urdf_limits_every_joint(robot):
    desc = Description.load(robot)
    p = PreparedUrdf(desc, load_profile(robot)).build()
    movable = {j.get("name") for j in p.root.findall("joint") if j.get("type") != "fixed"}
    assert movable == set(desc.joints)


@pytest.mark.parametrize("robot", ROBOTS)
def test_each_foot_has_a_sole_box(robot):
    p = PreparedUrdf(Description.load(robot), load_profile(robot)).build()
    assert sorted(s.side for s in p.soles) == ["l", "r"]
    for s in p.soles:
        assert s.half_extents[0] > 0.05 and s.half_extents[1] > 0.02  # a foot's footprint, not a point


def test_the_g1_sole_box_bounds_its_contact_spheres():
    p = PreparedUrdf(Description.load("g1_29dof_with_hand_rev_1_0"), load_profile("g1_29dof_with_hand_rev_1_0"))
    p.build()
    for s in p.soles:
        assert s.mesh is None  # describes the sole, replaces nothing
        bottom = s.center[2] - s.half_extents[2]
        assert bottom == pytest.approx(-0.035)  # sphere centers at z -0.03, radius 5 mm
        assert s.center[0] - s.half_extents[0] == pytest.approx(-0.055)
        assert s.center[0] + s.half_extents[0] == pytest.approx(0.125)
    assert not p.root.findall(".//collision[@name='sole_box_l']")


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_mirror_map_pairs_left_with_right(robot):
    m = mirror_map(Description.load(robot))
    for i, name in enumerate(m.joint_names):
        partner = m.joint_names[m.joint_perm[i]]
        assert m.joint_perm[m.joint_perm[i]] == i
        if name.startswith("left_") or name.endswith("_l"):
            assert partner != name


# humanoid-urdf c0ec54e set every standing offset to 0 without converting the standing-relative position limits
# (its ethrc_v1/AGENTS.md: unvalidated in the calibration frame until re-authored), so the V1's CAD export pose has
# its knees about 33 mrad past their limit until the URDF is re-exported at standing.
@pytest.mark.parametrize("robot", [pytest.param(r, marks=pytest.mark.xfail(
    strict=True, reason="humanoid-urdf limits not yet re-authored for zero standing offsets")) if r == "ethrc_v1"
    else r for r in ROBOTS])
def test_the_home_pose_lies_inside_the_travel(robot):
    desc = Description.load(robot)
    home = home_pose(desc, load_profile(robot))
    for name, q in home.items():
        lo, hi = desc.calibration_travel(name)
        assert lo <= q <= hi, name


@pytest.mark.parametrize("robot", ROBOTS)
def test_the_runtime_start_pose_lies_inside_every_drive_window(robot):
    """Controls arms a drive only inside its window: the joint's position_limit (standing-relative) for a 1:1 drive,
    the device-frame position_limit for an ankle crank."""
    desc, profile = Description.load(robot), load_profile(robot)
    pose = pose_joint_positions(desc, profile, start_pose(profile))
    joints = list(desc.joints)
    q = torch.tensor([[pose[n] for n in joints]], dtype=torch.float64)
    drive_map = DriveJointMap(desc, joints, dtype=torch.float64)
    at = dict(zip(drive_map.drive_names, drive_map.drive_state(q, torch.zeros_like(q))[0][0].tolist()))
    for name, d in desc.drives.items():
        if d.crank is None:
            entry = desc.table["joints"][d.joints[0]]
            lo, hi = (float(b) + float(entry["standing_offset_rad"]) for b in entry["position_limit"])
            assert lo < pose[d.joints[0]] < hi, name
        else:
            lo, hi = d.position_limit_device
            assert lo < at[name] < hi, name


def test_the_v1_starts_a_session_standing_and_the_g1_at_home():
    assert start_pose(load_profile("ethrc_v1")) == "standing"
    assert start_pose(load_profile("g1_29dof_rev_1_0")) == "home"
    with pytest.raises(ValueError, match="poses.start"):
        start_pose({"poses": {"start": "crouch"}})


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
    hold = DriveDemand.pd(q_drive, plant.drive_stiffness.expand(2, -1), plant.drive_damping.expand(2, -1))
    plant.submit(hold, torch.zeros(2, dtype=torch.long))
    target = plant.pre_physics(0, q, qd)
    assert torch.isfinite(target).all()
    assert torch.allclose(target, q, atol=1e-4)  # at the target already: no torque, the target stays put


def test_a_built_asset_opens_only_for_its_robot(tmp_path):
    import json

    from humanoid_sim.model.asset import check_asset_robot

    usd = tmp_path / "g1" / "usd" / "robot.usda"
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0\n")
    (tmp_path / "g1" / "asset.json").write_text(json.dumps({"robot": "g1_29dof_rev_1_0", "usd": "usd/robot.usda"}))
    check_asset_robot(usd, "g1_29dof_rev_1_0")
    with pytest.raises(ValueError, match="pass robot='g1_29dof_rev_1_0'"):
        check_asset_robot(usd, "ethrc_v1")
    loose = tmp_path / "loose.usda"
    loose.write_text("#usda 1.0\n")
    check_asset_robot(loose, "ethrc_v1")  # outside a built asset: not checked
