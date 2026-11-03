"""Live camera scenes must never become LiDAR geometry or duplicate publishers."""

from types import SimpleNamespace

from humanoid_sim.runtime.render_workers import render_configs
from humanoid_sim.sensors.session import SessionCfg


def rig(*kinds):
    return SimpleNamespace(sensors={str(i): SimpleNamespace(kind=kind, rtx_ready=True)
                                   for i, kind in enumerate(kinds)})


def test_mixed_rendering_keeps_the_splat_out_of_lidar_and_has_one_publisher_per_feed():
    cfg = SessionCfg(splat="room.usdc", scan_mesh="room.obj", lights="lights.json",
                     feeds="/feeds", lidar_domain=17, images="frames", rrd="sensors.rrd")
    workers = render_configs(cfg, rig("camera", "lidar"))
    assert set(workers) == {"cameras", "lidar"}
    camera, lidar = workers["cameras"], workers["lidar"]
    assert camera.cameras and not camera.lidar
    assert camera.feeds == "/feeds" and camera.lidar_domain is None
    assert camera.splat == "room.usdc"
    assert lidar.lidar and not lidar.cameras
    assert lidar.scan_mesh == "room.obj" and lidar.lidar_domain == 17
    assert lidar.splat is None and lidar.lights is None and lidar.feeds is None and lidar.images is None
    assert lidar.rrd != camera.rrd
    assert cfg.cameras and cfg.lidar and cfg.splat == "room.usdc"


def test_single_sensor_kinds_and_disabled_sensors_do_not_start_an_empty_worker():
    cfg = SessionCfg()
    assert render_configs(cfg, rig("camera")) == {"sensors": cfg}
    assert render_configs(cfg, rig("lidar")) == {"sensors": cfg}
    disabled = SessionCfg(cameras=False)
    assert render_configs(disabled, rig("camera", "lidar")) == {"sensors": disabled}


def test_unsupported_sensor_models_do_not_create_a_worker():
    cfg = SessionCfg()
    sensors = rig("camera", "lidar")
    sensors.sensors["1"].rtx_ready = False
    assert render_configs(cfg, sensors) == {"sensors": cfg}
