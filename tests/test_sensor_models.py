"""Sensor models and every robot's rig."""

import math

import pytest

from humanoid_sim.model.description import Description, robots
from humanoid_sim.sensors import rigs
from humanoid_sim.sensors.models import (
    Camera,
    Imu,
    LivoxMid360,
    Mti610,
    OakDW,
    RealSenseD435i,
    Rig,
    UnitreeG1Imu,
    ZedMini,
)

ROBOTS = robots()


def test_the_mti_610_is_its_leaflet():
    assert Mti610.gyro_noise_density_rad_s_sqrt_hz == pytest.approx(1.2217e-4, rel=1e-4)
    assert Mti610.accel_noise_density_m_s2_sqrt_hz == pytest.approx(5.884e-4, rel=1e-3)
    assert Mti610.gyro_bias_instability_rad_s == pytest.approx(3.878e-5, rel=1e-3)
    assert Mti610.accel_bias_instability_m_s2 == pytest.approx(1.471e-4, rel=1e-3)
    assert LivoxMid360.elevation_max_rad == pytest.approx(math.radians(52.0))


def test_an_imu_without_figures_is_ideal():
    cfg = UnitreeG1Imu("imu_link", sample_period_ns=2_000_000).delivery_cfg()
    assert cfg.gyro_noise_density == cfg.gyro_turn_on_bias == cfg.gyro_bias_instability == 0.0
    assert cfg.burst_period_ns == (2_000_000, 2_000_000) and cfg.in_burst_spacing_ns == 0


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_rig_sits_on_its_description(robot):
    desc = Description.load(robot)
    imu = rigs.imu(desc)
    assert imu.frame == desc.imu_link()
    cfg = imu.delivery_cfg()
    assert cfg.pending * cfg.sample_period_ns >= cfg.burst_period_ns[1]


def test_the_v1_carries_the_mti_610_on_can_and_an_oak_d_w():
    rig = rigs.rig("ethrc_v1")
    assert {n: type(s) for n, s in rig.sensors.items()} == {"imu": Mti610, "camera0": OakDW}
    assert rig.unsimulated() == ["camera0"] and rig.unsimulated(rtx=True) == ["camera0"]
    cfg = rig.imu.delivery_cfg()
    assert cfg.gyro_turn_on_bias == Mti610.gyro_turn_on_bias_rad_s
    assert (cfg.sample_period_ns, cfg.burst_period_ns) == (2_500_000, (2_500_000, 2_500_000))  # 400 Hz, no bursts


@pytest.mark.parametrize("robot", ["g1_29dof_rev_1_0", "g1_29dof_with_hand_rev_1_0"])
def test_the_g1_carries_its_body_imu_and_a_mid360(robot):
    rig = rigs.rig(robot)
    assert {n: type(s) for n, s in rig.sensors.items()} == {"imu": UnitreeG1Imu, "mid360": LivoxMid360}
    assert rig.unsimulated() == ["mid360"] and rig.unsimulated(rtx=True) == []


def test_an_extra_rig_adds_sensors_and_unknown_or_clashing_ones_are_refused():
    desc = Description.load("g1_29dof_with_hand_rev_1_0")
    rig = rigs.load(desc.robot, desc, "g1-cameras")
    assert {n: type(s) for n, s in rig.sensors.items()} == {
        "imu": UnitreeG1Imu, "mid360": LivoxMid360, "zed": ZedMini, "d435": RealSenseD435i}
    with pytest.raises(ValueError, match="g1-cameras"):
        rigs.load(desc.robot, desc, "g1-lidars")
    with pytest.raises(ValueError, match="already in the rig"):
        rigs.rig(desc.robot).extended(Rig(mid360=LivoxMid360("mid360_link")))


def test_a_robot_without_a_rig_has_no_imu():
    with pytest.raises(ValueError, match="no IMU"):
        rigs.rig("no_such_robot").imu


def test_a_sensor_takes_one_name():
    imu = Mti610("imu_link", sample_period_ns=2_500_000)
    Rig(imu=imu)
    with pytest.raises(ValueError, match="already named 'imu'"):
        Rig(body_imu=imu)


def test_frames_are_checked_against_the_description():
    desc = Description.load("ethrc_v1")
    with pytest.raises(ValueError, match="no link"):
        Rig(imu=Mti610("no_such_link", sample_period_ns=2_500_000)).check_frames(desc)
    with pytest.raises(ValueError, match="imu_joint"):
        Rig(imu=Mti610("base_link", sample_period_ns=2_500_000)).check_frames(desc)


def _d435(**kw):
    intrinsics = dict.fromkeys(RealSenseD435i.streams, (300.0, 300.0, 320.0, 240.0))
    return RealSenseD435i("d435_link", **{"width": 640, "height": 480, "frame_period_ns": 33_333_333,
                                          "intrinsics": intrinsics, **kw})


def test_a_camera_unit_places_its_streams_over_the_nominal_ones():
    assert _d435().stream_offsets_m["mono_right"] == (0.0, -0.050, 0.0)
    assert _d435(stream_offsets_m={"mono_right": (0.0, -0.0502, 0.0)}).stream_offsets_m["mono_right"] == (
        0.0, -0.0502, 0.0)
    assert _d435().rtx_ready
    assert not _d435(intrinsics={"rgb": (300.0, 300.0, 320.0, 240.0)}).rtx_ready  # streams without intrinsics


def test_values_are_checked():
    with pytest.raises(ValueError, match="sample_period_ns"):
        Mti610("imu_link", sample_period_ns=0)
    with pytest.raises(ValueError, match="sample_period_ns"):
        Mti610("imu_link", sample_period_ns=2.5e6)  # a float is not integer ns
    with pytest.raises(ValueError, match="low > high"):
        Mti610("imu_link", sample_period_ns=2_500_000, burst_period_ns=(3_000_000, 2_000_000))
    with pytest.raises(ValueError, match="thermal"):
        OakDW("camera0_camera", width=640, height=400, frame_period_ns=33_333_333,
              intrinsics={"thermal": (1.0, 1.0, 0.0, 0.0)})
    with pytest.raises(ValueError, match="coefficients"):
        _d435(distortion={"rgb": ("plumb_bob", (0.1, 0.0))})
    with pytest.raises(ValueError, match="coefficients"):
        _d435(distortion={"rgb": ("rational_polynomial", (0.0,) * 8)})
    with pytest.raises(ValueError, match="does not have"):
        _d435(stream_offsets_m={"thermal": (0.0, 0.0, 0.0)})
    with pytest.raises(ValueError, match="bias_correlation_time_ns"):
        type("Bad", (Imu,), {"bias_correlation_time_ns": 0})
    with pytest.raises(ValueError, match="gyro_noise_density"):
        type("Bad", (Imu,), {"gyro_noise_density_rad_s_sqrt_hz": -1.0})
    with pytest.raises(ValueError, match="stream_kinds"):
        type("Bad", (Camera,), {"streams": ("left",), "stream_kinds": {"left": "thermal"}})
    with pytest.raises(ValueError, match="depth_range_m"):
        type("Bad", (Camera,), {"streams": ("depth",), "depth_range_m": (2.0, 1.0)})


def test_review_regressions_integer_ns_shapes_frozen_defaults_and_names():
    with pytest.raises(ValueError, match="in_burst_spacing_ns"):
        Mti610("imu_link", sample_period_ns=2_500_000, in_burst_spacing_ns=13.5)
    with pytest.raises(ValueError, match="expected \\(low, high\\)"):
        Mti610("imu_link", sample_period_ns=2_500_000, burst_period_ns=(1, 2, 3))
    with pytest.raises(ValueError, match="resolution"):
        _d435(width=640.5)
    with pytest.raises(ValueError, match="coefficients"):
        _d435(distortion={"rgb": "equidistant"})
    with pytest.raises(TypeError):
        RealSenseD435i.stream_kinds["rgb"] = "depth"  # product figures are read-only
    assert type("StereoBase", (Camera,), {})  # an intermediate base without figures
    a, b = Mti610("imu_link", sample_period_ns=2_500_000), LivoxMid360("mid360_link")
    with pytest.raises(ValueError, match="two names"):
        Rig(imu=a, mid360=b, spare=a)
    assert (a.name, b.name) == ("", "")  # nothing named before the rig is accepted
