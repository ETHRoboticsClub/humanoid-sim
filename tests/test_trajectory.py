"""The recorded truth: round trip, interpolation, and the frames' world poses along it. No simulator."""

import math

import numpy as np
import pytest

from humanoid_sim.model.description import Description, robots
from humanoid_sim.model.frames import FrameTree, frame_poses, quat_matrix, write_tum
from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.runtime.trajectory import Trajectory, TrajectoryWriter
from humanoid_sim.sensors import rigs

JOINTS = ["a_joint", "b_joint"]


def truth(t_ns, x=0.0, yaw=0.0, q=(0.0, 0.0), hook=float("nan")):
    t = np.zeros((), abi.TRUTH)
    t["sim_time_ns"] = t_ns
    t["base_position_m"] = (x, 0.0, 1.0)
    t["base_quat_xyzw"] = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    t["joint_pos_rad"][:2] = q
    t["gantry_position_m"] = hook
    return t


def write(path, records):
    w = TrajectoryWriter(path, robot="r", description_digest="d", joints=JOINTS, period_ns=4_000_000, base_frame="base")
    written = [w.append(r, session_id=7) for r in records]
    w.close()
    return written


def test_a_trajectory_round_trips_and_skips_repeated_plant_times(tmp_path):
    path = tmp_path / "run.traj"
    written = write(path, [truth(0), truth(4_000_000, x=0.1), truth(4_000_000, x=9.0), truth(8_000_000, x=0.2)])
    assert written == [True, True, False, True]  # a truth after a command shares its step's time
    traj = Trajectory.load(path)
    assert traj.joints == JOINTS and traj.robot == "r" and traj.period_ns == 4_000_000
    assert list(traj.records["sim_time_ns"]) == [0, 4_000_000, 8_000_000]
    assert traj.records["session_id"][0] == 7
    assert traj.records["base_position_m"][1, 0] == pytest.approx(0.1)


def test_a_record_cut_off_by_a_kill_is_dropped(tmp_path):
    path = tmp_path / "run.traj"
    write(path, [truth(0), truth(4_000_000), truth(8_000_000)])
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    assert len(Trajectory.load(path).records) == 2


def test_interpolation_is_linear_in_position_and_joints_and_spherical_in_orientation(tmp_path):
    path = tmp_path / "run.traj"
    write(path, [truth(0, x=0.0, yaw=0.0, q=(0.0, 1.0)), truth(10_000_000, x=1.0, yaw=math.pi / 2, q=(1.0, 3.0))])
    at = Trajectory.load(path).at([0, 2_500_000, 10_000_000])
    np.testing.assert_allclose(at["base_position_m"][:, 0], [0.0, 0.25, 1.0])
    np.testing.assert_allclose(at["joint_pos_rad"][1], [0.25, 1.5])
    yaw = 2 * np.arctan2(at["base_quat_xyzw"][:, 2], at["base_quat_xyzw"][:, 3])
    np.testing.assert_allclose(yaw, [0.0, math.pi / 8, math.pi / 2], atol=1e-12)
    assert np.isnan(at["gantry_position_m"]).all()  # no crane stays NaN


def test_times_outside_the_recording_are_refused(tmp_path):
    path = tmp_path / "run.traj"
    write(path, [truth(0), truth(4_000_000)])
    with pytest.raises(ValueError, match="outside"):
        Trajectory.load(path).at(5_000_000)


URDF = """<robot name="r">
  <link name="base"/><link name="arm"/><link name="hand"/><link name="tool"/>
  <joint name="a_joint" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
  <joint name="b_joint" type="prismatic"><parent link="arm"/><child link="hand"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
  <joint name="tool_joint" type="fixed"><parent link="hand"/><child link="tool"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/></joint>
</robot>"""


def test_a_frame_follows_the_base_pose_through_revolute_prismatic_and_fixed_joints(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text(URDF)
    tree = FrameTree(urdf, JOINTS)
    # Base at (10, 0, 1) turned 90 deg about z; a_joint turns the arm a further 90 deg; b_joint slides 2 m along the arm.
    quat = [[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]]
    pos, q = tree.world_poses("tool", [[10.0, 0.0, 1.0]], quat, [[math.pi / 2, 2.0]])
    # arm origin: base + Rz(90)(1,0,0) = (10, 1, 1); arm x axis points along world -x; the slide 2 m puts the hand at
    # (8, 1, 1) and the tool 0.5 m above it.
    np.testing.assert_allclose(pos[0], [8.0, 1.0, 1.5], atol=1e-12)
    assert 2 * math.atan2(q[0][2], q[0][3]) == pytest.approx(math.pi)


def test_a_joint_on_the_path_that_is_not_given_is_refused(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text(URDF)
    with pytest.raises(ValueError, match="not among the given joints"):
        FrameTree(urdf, ["a_joint"]).world_poses("tool", [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]], [[0.0]])


@pytest.mark.parametrize("robot", robots())
def test_every_rig_frame_has_a_ground_truth_pose_on_the_real_description(robot, tmp_path):
    desc = Description.load(robot)
    path = tmp_path / "run.traj"
    root = FrameTree(desc.urdf_path, list(desc.joints)).root
    w = TrajectoryWriter(path, robot=robot, description_digest=desc.digest, joints=list(desc.joints),
                         period_ns=4_000_000, base_frame=root)
    for i in range(3):
        t = np.zeros((), abi.TRUTH)
        t["sim_time_ns"], t["base_quat_xyzw"] = i * 4_000_000, (0.0, 0.0, 0.0, 1.0)
        t["joint_pos_rad"][:len(desc.joints)] = 0.0
        w.append(t)
    w.close()
    traj = Trajectory.load(path)
    rig = rigs.load(robot, desc, "g1-cameras" if robot.startswith("g1") else None)
    for sensor in rig.sensors.values():
        poses = frame_poses(traj, desc.urdf_path, sensor.frame)
        assert poses["position_m"].shape == (3, 3) and np.isfinite(poses["position_m"]).all()
    out = tmp_path / "gt.tum"
    write_tum(poses, out)
    assert len(out.read_text().splitlines()) == 3


def imu_samples(seqs):
    """The feedback's IMU array, newest first, for the given sequence numbers."""
    s = np.zeros(len(seqs), abi.IMU_SAMPLE)
    s["seq"], s["sample_ns"], s["stamp_ns"] = seqs, np.array(seqs) * 1000, np.array(seqs) * 1000 + 5
    s["gyro_rad_s"][:, 0], s["specific_force_m_s2"][:, 2] = np.array(seqs) * 0.01, 9.81
    return s


def test_the_imu_file_holds_each_delivered_sample_once_oldest_first(tmp_path):
    from humanoid_sim.runtime.trajectory import ImuWriter, read_imu

    w = ImuWriter(tmp_path / "run.traj.imu", {"robot": "r", "description_digest": "d"}, {"frame": "imu_link"})
    assert w.append(imu_samples([2, 1])) == 2  # the feedback lists newest first
    assert w.append(imu_samples([4, 3, 2, 1])) == 2  # the same window one step later: only 3 and 4 are new
    assert w.append(imu_samples([4, 3])) == 0
    w.close()
    header, rec = read_imu(tmp_path / "run.traj.imu")
    assert header["robot"] == "r" and header["imu"] == {"frame": "imu_link"}
    assert list(rec["seq"]) == [1, 2, 3, 4] and list(rec["sample_ns"]) == [1000, 2000, 3000, 4000]
    assert rec["specific_force_m_s2"][0, 2] == pytest.approx(9.81)


def test_interpolation_takes_the_short_way_between_opposite_signs_of_a_quaternion(tmp_path):
    path = tmp_path / "run.traj"
    a, b = truth(0, yaw=0.2), truth(10_000_000, yaw=0.2)
    b["base_quat_xyzw"] = -b["base_quat_xyzw"]  # the same rotation, the other sign
    write(path, [a, b])
    q = Trajectory.load(path).at(5_000_000)["base_quat_xyzw"][0]
    assert abs(2 * np.arctan2(q[2], q[3])) == pytest.approx(0.2) and np.linalg.norm(q) == pytest.approx(1.0)


ROTATED = """<robot name="r">
  <link name="base"/><link name="mount"/><link name="sensor"/>
  <joint name="a_joint" type="revolute"><parent link="base"/><child link="mount"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
  <joint name="b_joint" type="fixed"><parent link="mount"/><child link="sensor"/>
    <origin xyz="0 1 0" rpy="0 0 1.5707963267948966"/></joint>
</robot>"""


def test_an_origin_rotation_composes_after_the_joints_above_it(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text(ROTATED)
    tree = FrameTree(urdf, ["a_joint"])
    # a_joint turns 90 deg about x, which takes the mount's y axis to z; the sensor sits 1 m along it, and its own
    # origin yaw turns its x axis to the mount's y, which that turn has taken to the world's z.
    pos, quat = tree.world_poses("sensor", [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]], [[math.pi / 2]])
    np.testing.assert_allclose(pos[0], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(quat_matrix(np.array(quat))[0] @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-12)


def test_tum_has_seconds_and_a_quaternion_that_does_not_flip_sign(tmp_path):
    t = np.array([0, 2_000_000_000, 4_000_000_000])
    poses = {"sim_time_ns": t, "position_m": np.arange(9.0).reshape(3, 3),
             "quat_xyzw": np.array([[0, 0, 0.6, 0.8], [0, 0, 0.8, 0.6], [0, 0, 0.9, 0.1]])}
    write_tum(poses, tmp_path / "a.tum")
    rows = np.loadtxt(tmp_path / "a.tum")
    np.testing.assert_allclose(rows[:, 0], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(rows[1, 1:4], [3.0, 4.0, 5.0])


def test_frame_poses_keep_the_quaternion_sign_continuous_and_refuse_another_base(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text(URDF)
    path = tmp_path / "run.traj"
    # The base turns through 2 pi: its quaternion crosses w = 0, where a w >= 0 convention flips the sign.
    write(path, [truth(i * 4_000_000, yaw=i * math.pi / 2) for i in range(5)])
    traj = Trajectory.load(path)
    traj.base_frame = "base"
    q = frame_poses(traj, urdf, "base")["quat_xyzw"]
    assert (np.sum(q[1:] * q[:-1], axis=1) > 0).all()
    traj.base_frame = "pelvis"
    with pytest.raises(ValueError, match="root link"):
        frame_poses(traj, urdf, "base")


def test_a_gap_in_the_delivered_imu_samples_is_logged(tmp_path, caplog):
    from humanoid_sim.runtime.trajectory import ImuWriter

    w = ImuWriter(tmp_path / "run.traj.imu", {}, {})
    w.append(imu_samples([2, 1]))
    with caplog.at_level("WARNING"):
        w.append(imu_samples([9, 8]))  # 3..7 left the feedback window unwritten
    assert "3..7" in caplog.text
    w.close()
