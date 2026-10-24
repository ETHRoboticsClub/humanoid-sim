"""view-runtime reads a runtime segment and records the robot and its signals. The segment is written here
through the ABI, so no physics runs; posing the robot needs Newton and the built asset."""

import threading
import time

import numpy as np
import pytest
from conftest import PHYSICS_ROBOTS, built_usd

from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi


def _publish(seg: abi.Segment, desc: Description, stop: threading.Event) -> None:
    """A stand-in host: a robot standing still on a gantry and one drive swinging, 4 ms of plant time per record."""
    joints, drives = list(desc.joints), list(desc.drives)
    q = np.array([desc.joints[j].standing_offset_rad for j in joints])
    t_ns = 0
    while not stop.is_set():
        t_ns += 4_000_000
        truth = np.zeros((), abi.TRUTH)
        truth["sim_time_ns"], truth["num_joints"] = t_ns, len(joints)
        truth["base_position_m"], truth["base_quat_xyzw"] = [0.0, 0.0, 0.95], [0.0, 0.0, 0.0, 1.0]
        truth["joint_pos_rad"][: len(joints)] = q
        # Hung from a spreader bar by two straps.
        truth["mass_kg"], truth["gantry_tension_n"] = 50.0, 400.0
        truth["gantry_top_m"], truth["gantry_harness_m"] = [0.0, 0.0, 2.5], [0.0, 0.0, 1.3]
        truth["gantry_strap_top_m"] = [[0.0, 0.2, 2.5], [0.0, -0.2, 2.5]]
        truth["gantry_strap_bottom_m"] = [[0.0, 0.2, 1.3], [0.0, -0.2, 1.3]]
        truth["gantry_position_m"] = truth["gantry_command_m"] = np.nan
        seg.regions["truth"].write(truth)
        fb = np.zeros((), abi.FEEDBACK)
        fb["sim_time_ns"], fb["num_drives"], fb["keeping_up"] = t_ns, len(drives), 1
        fb["drive"]["position_rad"][0] = np.sin(t_ns / 1e9)
        fb["imu_count"], fb["imu"]["gyro_rad_s"][0] = 1, [0.1, 0.2, 0.3]
        seg.regions["feedback"].write(fb)
        time.sleep(0.004)


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_view_runtime_records_frames_from_a_segment(tmp_path, robot):
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    pytest.importorskip("rerun")
    from humanoid_sim.runtime.view import view_runtime

    usd = built_usd(robot)
    desc = Description.load(robot)
    joints, drives = list(desc.joints), list(desc.drives)
    path = tmp_path / "plant"
    seg = abi.Segment.create(path, robot_id=desc.robot, description_digest=desc.digest, sim_revision="test",
                             num_drives=len(drives), num_joints=len(joints), physics_period_ns=4_000_000,
                             bus_period_ns=1_500_000, descriptor={"drives": drives, "joints": joints, "soles": []})
    stop = threading.Event()
    writer = threading.Thread(target=_publish, args=(seg, desc, stop), daemon=True)
    writer.start()
    try:
        rrd = tmp_path / "runtime.rrd"
        frames = view_runtime(path, usd, rrd=str(rrd), fps=50.0, seconds=1.0, wait_s=5.0, robot=robot)
    finally:
        stop.set()
        writer.join()
        seg.end()
        seg.close()
    assert frames > 10
    assert rrd.stat().st_size > 100_000  # the robot's meshes and the signals
