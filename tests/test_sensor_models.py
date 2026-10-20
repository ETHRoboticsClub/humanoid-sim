"""Named sensor models and every robot's rig."""

import math

import pytest

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description, robots
from humanoid_sim.sensors.imu import ImuDeliveryCfg
from humanoid_sim.sensors.models import CameraModel, ImuModel, LidarModel, SensorRig, load_catalog

ROBOTS = robots()


def test_the_catalog_names_every_sensor_on_the_robots():
    catalog = load_catalog()
    assert isinstance(catalog["mti-610"], ImuModel) and isinstance(catalog["unitree-g1-imu"], ImuModel)
    assert isinstance(catalog["oak-d-w"], CameraModel) and isinstance(catalog["livox-mid360"], LidarModel)
    assert all(m.source for m in catalog.values())


def test_the_mti_610_is_its_leaflet():
    m = load_catalog()["mti-610"]
    assert m.gyro_noise_density_rad_s_sqrt_hz == pytest.approx(math.radians(0.007), rel=1e-4)
    assert m.accel_noise_density_m_s2_sqrt_hz == pytest.approx(60e-6 * 9.80665, rel=1e-3)
    assert m.gyro_bias_instability_rad_s == pytest.approx(math.radians(8.0) / 3600.0, rel=1e-3)
    assert m.accel_bias_instability_m_s2 == pytest.approx(15e-6 * 9.80665, rel=1e-3)


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_rig_sits_on_its_description(robot):
    desc = Description.load(robot)
    rig = SensorRig.load(load_profile(robot))
    rig.check_frames(desc)
    assert rig.imu.frame == desc.imu_link()
    cfg = ImuDeliveryCfg.for_robot(desc, load_profile(robot))
    assert cfg.sample_period_ns > 0 and cfg.pending * cfg.sample_period_ns >= cfg.burst_period_ns[1]


def test_the_v1_carries_the_mti_610_and_an_oak_d_w():
    rig = SensorRig.load(load_profile("ethrc_v1"))
    assert {n: s.model.name for n, s in rig.sensors.items()} == {"imu": "mti-610", "camera0": "oak-d-w"}
    assert rig.unsimulated() == ["camera0"]
    cfg = ImuDeliveryCfg.from_sensor(rig.imu)
    m = rig.imu.model
    assert cfg.gyro_noise_density == m.gyro_noise_density_rad_s_sqrt_hz
    assert cfg.gyro_turn_on_bias == m.gyro_turn_on_bias_rad_s
    assert (cfg.sample_period_ns, cfg.burst_period_ns) == (2_740_000, (13_900_000, 17_600_000))


@pytest.mark.parametrize("robot", ["g1_29dof_rev_1_0", "g1_29dof_with_hand_rev_1_0"])
def test_the_g1_carries_its_body_imu_and_a_mid360(robot):
    rig = SensorRig.load(load_profile(robot))
    assert {n: s.model.name for n, s in rig.sensors.items()} == {"imu": "unitree-g1-imu", "mid360": "livox-mid360"}
    assert rig.unsimulated() == ["mid360"]


def _profile(**sensors):
    return {"robot": "ethrc_v1", "sensors": sensors}


IMU = {"model": "mti-610", "frame": "imu_link", "sample_period_ns": 2_500_000,
       "burst_period_ns": [2_500_000, 2_500_000], "in_burst_spacing_ns": 0}


def test_an_unknown_model_names_the_known_ones():
    with pytest.raises(ValueError, match="mti-610"):
        SensorRig.load(_profile(imu={**IMU, "model": "bmi088"}))


def test_a_rig_entry_takes_only_its_kinds_keys():
    with pytest.raises(ValueError, match="gyro_noise_density"):
        SensorRig.load(_profile(imu={**IMU, "gyro_noise_density_rad_s_sqrt_hz": 0.0}))  # a product figure
    with pytest.raises(ValueError, match="missing"):
        SensorRig.load(_profile(imu={k: v for k, v in IMU.items() if k != "in_burst_spacing_ns"}))
    with pytest.raises(ValueError, match="no frame"):
        SensorRig.load(_profile(imu={k: v for k, v in IMU.items() if k != "frame"}))


def test_frames_are_checked_against_the_description():
    desc = Description.load("ethrc_v1")
    with pytest.raises(ValueError, match="no link"):
        SensorRig.load(_profile(imu={**IMU, "frame": "no_such_link"})).check_frames(desc)
    with pytest.raises(ValueError, match="imu_joint"):
        SensorRig.load(_profile(imu={**IMU, "frame": "base_link"})).check_frames(desc)


def test_a_rig_without_an_imu_says_what_it_has():
    rig = SensorRig.load(_profile(mid360={"model": "livox-mid360", "frame": "base_link"}))
    with pytest.raises(ValueError, match="mid360"):
        rig.imu


def test_values_are_range_checked():
    with pytest.raises(ValueError, match="sample_period_ns"):
        SensorRig.load(_profile(imu={**IMU, "sample_period_ns": 0}))
    with pytest.raises(ValueError, match="sample_period_ns"):
        SensorRig.load(_profile(imu={**IMU, "sample_period_ns": 2.5e6}))  # a float is not integer ns
    with pytest.raises(ValueError, match="burst_period_ns"):
        SensorRig.load(_profile(imu={**IMU, "burst_period_ns": [3_000_000, 2_000_000]}))
    with pytest.raises(ValueError, match="expected a mapping"):
        SensorRig.load(_profile(imu=None))
    camera = {"model": "oak-d-w", "frame": "camera0_camera", "width": 640, "height": 400,
              "frame_period_ns": 33_333_333, "intrinsics": {"thermal": [1.0, 1.0, 0.0, 0.0]}}
    with pytest.raises(ValueError, match="thermal"):
        SensorRig.load(_profile(camera0=camera))
    with pytest.raises(ValueError, match=r"\[fx, fy, cx, cy\]"):
        SensorRig.load(_profile(camera0={**camera, "intrinsics": {"rgb": [1.0, 1.0]}}))
    m = load_catalog()["mti-610"]
    with pytest.raises(ValueError, match="bias_correlation_time_ns"):
        ImuModel(**{**m.__dict__, "bias_correlation_time_ns": 0})
    with pytest.raises(ValueError, match="gyro_noise_density"):
        ImuModel(**{**m.__dict__, "gyro_noise_density_rad_s_sqrt_hz": -1.0})
