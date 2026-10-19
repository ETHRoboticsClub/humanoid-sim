"""view-runtime reads a runtime segment and records the robot and its signals. The segment is written here
through the ABI, so no physics runs; posing the robot needs Newton and the built asset."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"


@pytest.fixture
def usd():
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    pytest.importorskip("rerun")
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), ASSETS)
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim build-asset` in the Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def _publish(seg: abi.Segment, joints: list[str], drives: list[str], stop: threading.Event) -> None:
    """A stand-in host: a robot standing still and one drive swinging, 4 ms of plant time per record."""
    desc = Description.load()
    q = np.array([desc.joints[j].standing_offset_rad for j in joints])
    t_ns = 0
    while not stop.is_set():
        t_ns += 4_000_000
        truth = np.zeros((), abi.TRUTH)
        truth["sim_time_ns"], truth["num_joints"] = t_ns, len(joints)
        truth["base_position_m"], truth["base_quat_xyzw"] = [0.0, 0.0, 0.95], [0.0, 0.0, 0.0, 1.0]
        truth["joint_pos_rad"][: len(joints)] = q
        seg.regions["truth"].write(truth)
        fb = np.zeros((), abi.FEEDBACK)
        fb["sim_time_ns"], fb["num_drives"], fb["keeping_up"] = t_ns, len(drives), 1
        fb["drive"]["position_rad"][0] = np.sin(t_ns / 1e9)
        fb["imu_count"], fb["imu"]["gyro_rad_s"][0] = 1, [0.1, 0.2, 0.3]
        seg.regions["feedback"].write(fb)
        time.sleep(0.004)


def test_view_runtime_records_frames_from_a_segment(usd, tmp_path):
    from humanoid_sim.runtime.view import view_runtime

    desc = Description.load()
    joints, drives = list(desc.joints), list(desc.drives)
    path = tmp_path / "plant"
    seg = abi.Segment.create(path, robot_id=desc.robot, description_digest=desc.digest, sim_revision="test",
                             num_drives=len(drives), num_joints=len(joints), physics_period_ns=4_000_000,
                             bus_period_ns=1_500_000, descriptor={"drives": drives, "joints": joints, "soles": []})
    stop = threading.Event()
    writer = threading.Thread(target=_publish, args=(seg, joints, drives, stop), daemon=True)
    writer.start()
    try:
        rrd = tmp_path / "runtime.rrd"
        frames = view_runtime(path, usd, rrd=str(rrd), fps=50.0, seconds=1.0, wait_s=5.0)
    finally:
        stop.set()
        writer.join()
        seg.end()
        seg.close()
    assert frames > 10
    assert rrd.stat().st_size > 100_000  # the robot's meshes and the signals
