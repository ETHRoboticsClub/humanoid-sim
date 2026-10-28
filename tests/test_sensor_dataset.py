"""The rendered sensor dataset's writer and the offline render's schedule. No simulator."""

import csv
import json
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.runtime.offline import check_trajectory, render_times
from humanoid_sim.sensors.dataset import SCHEMA, DatasetWriter, png_bytes


def decode_png(data: bytes):
    """Width, height, bit depth, colour type and the unfiltered rows of a PNG that :func:`png_bytes` wrote."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, idat, header = 8, b"", None
    while at < len(data):
        n, kind = struct.unpack(">I4s", data[at:at + 8])
        body = data[at + 8:at + 8 + n]
        assert struct.unpack(">I", data[at + 8 + n:at + 12 + n])[0] == zlib.crc32(kind + body)
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        at += 12 + n
    w, h, depth, color = header[:4]
    raw = zlib.decompress(idat)
    stride = 1 + w * (depth // 8) * (3 if color == 2 else 1)
    rows = [raw[y * stride + 1:(y + 1) * stride] for y in range(h)]
    assert all(raw[y * stride] == 0 for y in range(h))
    return w, h, depth, color, rows


def test_png_round_trips_grey_depth_and_rgb():
    grey = (np.arange(12, dtype=np.uint8) * 20).reshape(3, 4)
    w, h, depth, color, rows = decode_png(png_bytes(grey))
    assert (w, h, depth, color) == (4, 3, 8, 0) and np.frombuffer(b"".join(rows), np.uint8).tolist() == grey.ravel().tolist()
    mm = np.array([[0, 1500], [65535, 40000]], dtype=np.uint16)
    w, h, depth, color, rows = decode_png(png_bytes(mm))
    assert (depth, color) == (16, 0) and np.frombuffer(b"".join(rows), ">u2").tolist() == mm.ravel().tolist()
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert decode_png(png_bytes(rgb))[2:4] == (8, 2)
    with pytest.raises(ValueError, match="cannot write"):
        png_bytes(np.zeros((2, 2), dtype=np.float32))


def test_a_dataset_indexes_every_frame_and_sweep_by_plant_time(tmp_path):
    w = DatasetWriter(tmp_path / "d", {"robot": "r"})
    w.describe_camera("cam", {"streams": ["left"]})
    for i in range(3):
        w.camera_frame("cam", "left", 33_333_333 * i, np.full((2, 2), i, dtype=np.uint8))
    sweep = SimpleNamespace(points_m=np.ones((5, 3)), hit=np.ones(5, bool), time_s=np.linspace(0, 0.1, 5),
                            start_ns=100_000_000, origin_w=np.zeros(3), rotation_w=np.eye(3))
    w.lidar_sweep("lidar", sweep)
    w.close()
    rows = list(csv.DictReader(open(tmp_path / "d/cameras/cam/left/index.csv")))
    assert [int(r["stamp_ns"]) for r in rows] == [0, 33_333_333, 66_666_666]
    assert (tmp_path / "d/cameras/cam/left" / rows[2]["file"]).exists()
    z = np.load(tmp_path / "d/lidars/lidar/000000.npz")
    assert z["points_m"].shape == (5, 3) and int(z["start_ns"]) == 100_000_000
    manifest = json.loads((tmp_path / "d/manifest.json").read_text())
    assert manifest["schema"] == SCHEMA and manifest["counts"] == {"cam/left": 3, "lidar": 1}


def test_render_times_are_exact_multiples_of_the_period_from_the_start():
    t = render_times(1_000, 1_000 + 100_000_000, 33_333_333)
    assert t.dtype == np.int64 and list(np.diff(t)) == [33_333_333] * 3 and t[0] == 1_000
    with pytest.raises(ValueError, match="no camera or lidar"):
        render_times(0, 10, 0)


def test_a_trajectory_of_another_robot_or_description_is_refused():
    desc = SimpleNamespace(robot="r", digest="abc", joints=["a", "b"])
    traj = SimpleNamespace(robot="r", header={"description_digest": "abc"}, joints=["b", "a", "c"])
    assert check_trajectory(traj, desc) == [1, 0]
    with pytest.raises(ValueError, match="not of"):
        check_trajectory(SimpleNamespace(robot="r", header={"description_digest": "zzz"}, joints=["a", "b"]), desc)
    with pytest.raises(ValueError, match="lacks joints"):
        check_trajectory(SimpleNamespace(robot="r", header={"description_digest": "abc"}, joints=["a"]), desc)


def fake_session(camera_name="zed", mounts=None):
    stream = SimpleNamespace(name="left", rate_hz=30.0)
    spec = SimpleNamespace(camera_id=f"sim-{camera_name}", streams=(stream,), product="p", calibration_id="c",
                           sensor_link="l")
    lidar = SimpleNamespace(sensor=SimpleNamespace(name="mid360", model="m", product="p", frame="f", range_min_m=0.1),
                            period_ns=100_000_000)
    mount = SimpleNamespace(link="torso_link", sensor_link="camera0_camera", translation_m=(0.1, 0.0, 0.2),
                            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    return SimpleNamespace(camera_specs=[spec], lidars=[lidar], cfg=SimpleNamespace(splat="room.usdc"),
                           mounts=mounts or {camera_name: mount, "mid360": mount})


def test_the_manifest_finds_a_camera_mount_by_its_rig_name_not_its_feed_id(tmp_path, monkeypatch):
    from humanoid_sim.runtime import offline
    from humanoid_sim.sensors import feeds

    monkeypatch.setattr(feeds, "descriptor", lambda spec, session: {"camera_id": spec.camera_id, "session": session})
    w = DatasetWriter(tmp_path / "d", {})
    offline.describe_sensors(w, fake_session(), ["head_link", "pelvis"])
    assert w.manifest["cameras"]["sim-zed"]["mount"]["body"] == "torso_link"
    assert w.manifest["lidars"]["mid360"]["stage"] == {"splat": True, "robot_links_traced": ["head_link", "pelvis"]}
    assert w.manifest["lidars"]["mid360"]["range_min_m"] == 0.1
    assert w.manifest["lidars"]["mid360"]["period_ns"] == 100_000_000
    with pytest.raises(KeyError, match="no mount"):
        offline.describe_sensors(DatasetWriter(tmp_path / "e", {}), fake_session(mounts={"other": None}))


def test_a_sensor_that_wrote_nothing_is_reported():
    from humanoid_sim.runtime.offline import missing_data

    session = fake_session()
    assert missing_data(session, {"sim-zed/left": 30, "mid360": 10}) == []
    assert missing_data(session, {"sim-zed/left": 30}) == ["mid360"]
    assert missing_data(session, {"sim-zed/left": 0, "mid360": 10}) == ["sim-zed/left"]


def test_a_second_pass_adds_its_sensors_to_the_dataset_of_the_same_run(tmp_path):
    run = {"robot": "r", "description_digest": "d", "trajectory": "run.traj", "start_ns": 0, "end_ns": 100}
    cameras = DatasetWriter(tmp_path / "d", run)  # the pass with the splat
    cameras.describe_camera("cam", {"streams": ["left"]})
    cameras.camera_frame("cam", "left", 0, np.zeros((2, 2), dtype=np.uint8))
    cameras.close()
    lidar = DatasetWriter(tmp_path / "d", run)  # the pass without it
    lidar.describe_lidar("mid360", {"period_ns": 100})
    lidar.lidar_sweep("mid360", SimpleNamespace(points_m=np.ones((2, 3)), hit=np.ones(2, bool), time_s=np.zeros(2),
                                                start_ns=0, origin_w=np.zeros(3), rotation_w=np.eye(3)))
    lidar.close()
    manifest = json.loads((tmp_path / "d/manifest.json").read_text())
    assert set(manifest["cameras"]) == {"cam"} and set(manifest["lidars"]) == {"mid360"}
    assert manifest["counts"] == {"cam/left": 1, "mid360": 1}
    with pytest.raises(ValueError, match="another run"):
        DatasetWriter(tmp_path / "d", {**run, "trajectory": "other.traj"})
