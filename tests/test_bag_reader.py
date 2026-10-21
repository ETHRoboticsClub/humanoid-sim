"""rosbag2 reader on a bag written here: the capture interface the replay reads."""

import numpy as np
import pytest

pytest.importorskip("rosbags")

from humanoid_sim.replay.bag import (  # noqa: E402
    COMMAND_TOPIC,
    FEEDBACK_TOPIC,
    GAINS_TOPIC,
    IMU_TOPIC,
    MSGS,
    BagCapture,
)

DRIVES = ["maxonHEJ90L1", "maxonHEJ50L_ANK1"]
T0 = 1_760_000_000_000_000_000
PERIOD = 5_000_000


def _store():
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    store = get_typestore(Stores.ROS2_HUMBLE)
    for name in ("DriveFeedback", "DriveGains", "BodyControlFrame"):
        store.register(get_types_from_msg((MSGS / f"{name}.msg").read_text(), f"msg_ethrc/msg/{name}"))
    return store


def _stamp(store, t):
    return store.types["builtin_interfaces/msg/Time"](sec=t // 1_000_000_000, nanosec=t % 1_000_000_000)


def _header(store, t):
    return store.types["std_msgs/msg/Header"](stamp=_stamp(store, t), frame_id="")


def _write(path, *, plugin="SQLITE3", gains=((320.0, 12.0), (480.0, 9.0)), change_at=None, n=100, defs=True):
    from rosbags.rosbag2 import StoragePlugin, Writer

    store = _store()
    T = store.types
    with Writer(path, version=8, storage_plugin=getattr(StoragePlugin, plugin)) as w:
        # Humble's sqlite3 bags carry no message definitions: the reader falls back to its copies.
        bare = {} if defs else {"msgdef": "", "rihs01": "RIHS01_" + "0" * 64}
        fb = w.add_connection(FEEDBACK_TOPIC, "msg_ethrc/msg/DriveFeedback", typestore=store, **bare)
        g = w.add_connection(GAINS_TOPIC, "msg_ethrc/msg/DriveGains", typestore=store, **bare)
        imu = w.add_connection(IMU_TOPIC, "sensor_msgs/msg/Imu", typestore=store)
        cmd = w.add_connection(COMMAND_TOPIC, "msg_ethrc/msg/BodyControlFrame", typestore=store, **bare)

        def frame(t, kind, joints, q):
            return T["msg_ethrc/msg/BodyControlFrame"](
                header=_header(store, t), body_mode=0, kind=kind, clear_latches=False, robot_id="", authority_runtime_id="",
                component_session_id="", grant_id="", source_id="", sequence=1, input_sequence=0, valid_for_ns=0,
                joint_ids=joints, positions_rad=np.asarray(q, dtype=np.float64), require_exact_positions=False,
                hold_joint_ids=[], stop_profile_id="")

        def gains_msg(t, pairs):
            f = np.array
            return T["msg_ethrc/msg/DriveGains"](
                header=_header(store, t), holder="", applied=True, failed=False, motors=DRIVES,
                kp_nm_per_rad=f([p[0] for p in pairs]), kd_nm_s_per_rad=f([p[1] for p in pairs]),
                baseline_kp_nm_per_rad=f([p[0] for p in pairs]), baseline_kd_nm_s_per_rad=f([p[1] for p in pairs]),
                motor_applied=np.ones(2, dtype=bool), failed_motors=[], target_kp_nm_per_rad=f([p[0] for p in pairs]),
                target_kd_nm_s_per_rad=f([p[1] for p in pairs]), ramp_remaining_s=0.0, kp_resolution_nm_per_rad=0.0,
                kd_resolution_nm_s_per_rad=0.0)

        w.write(g, T0 - 100, store.serialize_cdr(gains_msg(T0 - 100, gains), "msg_ethrc/msg/DriveGains"))
        for k in range(n):
            t = T0 + k * PERIOD
            q = np.array([0.1 * np.sin(k / 10), -0.2])
            m = T["msg_ethrc/msg/DriveFeedback"](
                header=_header(store, t + 1000), device=DRIVES, source_stamp_ns=np.array([t, t - 7], dtype=np.int64),
                position_rad=q, velocity_rad_s=np.zeros(2), status_word=np.zeros(2, dtype=np.uint16),
                torque_nm=np.ones(2), current_a=np.zeros(2), motor_position_rad=np.full(2, np.nan),
                position_demand_rad=q + 0.01, current_demand_a=np.zeros(2))
            w.write(fb, t + 2000, store.serialize_cdr(m, "msg_ethrc/msg/DriveFeedback"))
            if k % 2 == 0:  # repeated feedback: the same source stamp twice
                w.write(fb, t + 3000, store.serialize_cdr(m, "msg_ethrc/msg/DriveFeedback"))
            if change_at is not None and k == change_at:
                w.write(g, t, store.serialize_cdr(gains_msg(t, [(p[0] * 2, p[1]) for p in gains]),
                                                  "msg_ethrc/msg/DriveGains"))
            V, Q = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
            im = T["sensor_msgs/msg/Imu"](
                header=_header(store, t), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9), angular_velocity=V(x=0.0, y=0.0, z=0.1),
                angular_velocity_covariance=np.zeros(9), linear_acceleration=V(x=0.0, y=0.0, z=9.81),
                linear_acceleration_covariance=np.zeros(9))
            w.write(imu, t, store.serialize_cdr(im, "sensor_msgs/msg/Imu"))
            if k % 4 == 0:
                w.write(cmd, t, store.serialize_cdr(frame(t, 2, ["knee_l", "knee_r"], [0.01 * k, -0.01 * k]),
                                                    "msg_ethrc/msg/BodyControlFrame"))
                w.write(cmd, t + 1, store.serialize_cdr(frame(t + 1, 4, [], []), "msg_ethrc/msg/BodyControlFrame"))
    return path


@pytest.mark.parametrize("plugin,defs", [("SQLITE3", False), ("SQLITE3", True), ("MCAP", True)])
def test_reads_feedback_gains_and_imu(tmp_path, plugin, defs):
    cap = BagCapture(_write(tmp_path / "bag", plugin=plugin, defs=defs), drives=DRIVES)
    pdo = cap.pdo()
    assert set(pdo) == set(DRIVES)
    tr = pdo["maxonHEJ90L1"]
    assert len(tr.t_ns) == 100 and tr.t_ns[0] == T0 and np.all(np.diff(tr.t_ns) == PERIOD)  # duplicates dropped
    assert np.allclose(tr["position_demand_rad"] - tr["native_position_rad"], 0.01)
    assert np.isnan(tr["motor_position_rad"]).all()
    assert pdo["maxonHEJ50L_ANK1"].t_ns[0] == T0 - 7
    assert cap.time_span_ns() == (T0, T0 + 99 * PERIOD - 7)
    g = cap.gains()
    assert (g["maxonHEJ90L1"].stiffness, g["maxonHEJ50L_ANK1"].damping) == (320.0, 9.0)
    imu = cap.imu()
    assert len(imu.t_ns) == 100 and np.allclose(imu.angular_velocity[:, 2], 0.1)
    c = cap.commands()  # COMMAND frames only
    assert c.joints == ["knee_l", "knee_r"] and len(c.t_ns) == 25 and c.t_ns[1] - c.t_ns[0] == 4 * PERIOD
    assert c.positions_rad[1].tolist() == [0.04, -0.04]
    assert cap.dataset_id.startswith("bag-") and cap.dataset_id == BagCapture(tmp_path / "bag").dataset_id


def test_refuses_changing_gains_and_missing_drives(tmp_path):
    with pytest.raises(ValueError, match="gains changed"):
        BagCapture(_write(tmp_path / "a", change_at=50)).gains()
    with pytest.raises(ValueError, match="no feedback from drives"):
        BagCapture(_write(tmp_path / "b"), drives=DRIVES + ["maxonHEJ90R1"]).pdo()
    with pytest.raises(ValueError, match="not a rosbag2"):
        BagCapture(tmp_path)
