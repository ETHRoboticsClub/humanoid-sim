"""Where the G1's cameras and lidar sit and what they see (``sensors/geometry.py``), built from the sensor models and
the rig: mounts against the numbers humanoid-urdf's ``PROVENANCE.md`` records for the lab unit, each lens against
projections worked out by hand for points at known angles, and the lidar's pattern against its model's figures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanoid_sim.model.description import Description
from humanoid_sim.sensors import geometry, rigs
from humanoid_sim.sensors.models import LivoxMid360, OakDW

G1 = "g1_29dof_with_hand_rev_1_0"
CAMERAS = "g1-cameras"  # sensors/rigs.py EXTRAS


@pytest.fixture(scope="module")
def desc():
    return Description.load(G1)


@pytest.fixture(scope="module")
def rig(desc):
    return rigs.load(G1, desc, CAMERAS)


def test_the_mounts_are_the_lab_units(desc, rig):
    m = {s.frame: geometry.Mount.from_urdf(desc.urdf_path, s.frame) for s in rig.sensors.values() if s.kind != "imu"}
    assert {k: v.link for k, v in m.items()} == dict.fromkeys(("camera0_camera", "d435_link", "mid360_link"),
                                                              "torso_link")
    # PROVENANCE.md, "ZED Mini mount": the left optical centre at (0.1068, 0.0406, 0.457) m in torso_link, level.
    np.testing.assert_allclose(m["camera0_camera"].translation_m, (0.1068, 0.0406, 0.457))
    np.testing.assert_allclose(m["camera0_camera"].R, np.eye(3), atol=1e-12)
    # PROVENANCE.md, "MID-360 mount": Unitree's roll pi (upside down) with the measured pitch 0.0836 rad. The sensor's
    # z axis then points down and back by the pitch: Ry(0.0836) Rx(pi) applied to +z.
    z = m["mid360_link"].R @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(z, (-math.sin(0.0836), 0.0, -math.cos(0.0836)), atol=1e-9)
    # Unitree's D435 looks forward and down: its x axis pitched 0.8307767 rad below the torso's horizon.
    x = m["d435_link"].R @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(x, (math.cos(0.8307767239493009), 0.0, -math.sin(0.8307767239493009)), atol=1e-9)


def test_a_mount_composes_fixed_joints_to_the_kept_body(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text("""<robot name="r">
      <link name="base"/><link name="arm"/><link name="bracket"/><link name="cam"/>
      <joint name="j" type="revolute"><parent link="base"/><child link="arm"/></joint>
      <joint name="b" type="fixed"><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
        <parent link="arm"/><child link="bracket"/></joint>
      <joint name="c" type="fixed"><origin xyz="0 2 0" rpy="0 0 0"/><parent link="bracket"/><child link="cam"/></joint>
    </robot>""")
    m = geometry.Mount.from_urdf(urdf, "cam")
    # cam is 2 m along the bracket's y, which the bracket's quarter turn points along -x of the arm: (1 - 2, 0, 0).
    assert m.link == "arm"
    np.testing.assert_allclose(m.translation_m, (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(m.R @ [1.0, 0.0, 0.0], (0.0, 1.0, 0.0), atol=1e-12)


def test_a_pinhole_maps_angles_to_pixels():
    lens = geometry.Lens.from_fov(640, 480, 90.0)
    # A 90 deg field across 640 px: f = 320 px. A point 45 deg right of the axis lands on the right edge; 0.5 m up
    # at 1 m (optical y down) at v = 240 - 320 * 0.5.
    uv = lens.project([[0, 0, 2.0], [1.0, 0, 1.0], [0, -0.5, 1.0]])
    np.testing.assert_allclose(uv, [[320, 240], [640, 240], [320, 80]], atol=1e-9)


def test_the_zed_minis_streams_are_its_rig_calibration(rig):
    spec = geometry.camera_spec(rig.sensors["zed"])
    assert (spec.camera_id, spec.product, spec.sensor_link) == ("sim-zed", "Stereolabs ZED Mini", "camera0_camera")
    # Grey eyes at HD720 and 30 Hz, a stereo pair, as Controls' zed backend publishes them.
    assert [(s.name, s.kind, s.encoding, s.sync_group) for s in spec.streams] == [
        ("left", "mono", "mono8", "stereo"), ("right", "mono", "mono8", "stereo")]
    left, right = spec.streams
    assert (left.lens.width_px, left.lens.height_px, left.rate_hz) == (1280, 720, pytest.approx(30.0))
    # The unit's [STEREO] calibration places the right eye, not the model's nominal 63 mm.
    np.testing.assert_allclose(right.translation_m, (0.000233734, -0.0628272, 0.000181381))
    np.testing.assert_allclose(left.translation_m, (0.0, 0.0, 0.0))
    k1, k2, k3, k4 = left.lens.distortion
    for deg in (5.0, 20.0, 40.0):
        th = math.radians(deg)
        # OpenCV's equidistant model: radius in the image fx * th * (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8).
        r_px = 771.4 * th * (1 + k1 * th**2 + k2 * th**4 + k3 * th**6 + k4 * th**8)
        (u, v), = left.lens.project([[math.tan(th), 0.0, 1.0]])
        assert u == pytest.approx(629.115 + r_px) and v == pytest.approx(353.0685)


def test_the_d435is_streams_are_its_models(rig):
    spec = geometry.camera_spec(rig.sensors["d435"])
    assert [(s.name, s.kind, s.encoding, s.sync_group, s.depth_source) for s in spec.streams] == [
        ("rgb", "color", "bgr8", "", ""), ("mono_left", "mono", "mono8", "stereo", ""),
        ("mono_right", "mono", "mono8", "stereo", ""), ("depth", "depth", "16UC1", "", "stereo")]
    s = {st.name: st for st in spec.streams}
    # Nominal places from the catalog: the right imager one 50 mm baseline right of the left, colour 15 mm.
    np.testing.assert_allclose(s["mono_right"].translation_m, (0.0, -0.050, 0.0))
    np.testing.assert_allclose(s["rgb"].translation_m, (0.0, -0.015, 0.0))
    assert s["depth"].range_m == (0.28, 10.0) and s["rgb"].range_m == (0.0, math.inf)
    # The datasheet's colour field, 69 deg across 640 px: a point 34.5 deg right lands on the image's edge.
    (u, _), = s["rgb"].lens.project([[math.tan(math.radians(34.5)), 0.0, 1.0]])
    assert u == pytest.approx(640.0, abs=1e-3)


def test_a_point_ahead_of_each_camera_lands_on_its_principal_point(desc, rig):
    for sensor in (rig.sensors["zed"], rig.sensors["d435"]):
        mount = geometry.Mount.from_urdf(desc.urdf_path, sensor.frame)
        for stream in geometry.camera_spec(sensor).streams:
            # The optical frame in the torso, and a point 3 m along its optical axis there.
            T = mount.matrix() @ stream.optical_in_device()
            p_torso = T[:3, :3] @ [0.0, 0.0, 3.0] + T[:3, 3]
            p_opt = T[:3, :3].T @ (p_torso - T[:3, 3])
            (u, v), = stream.lens.project([p_opt])
            assert (u, v) == pytest.approx((stream.lens.cx_px, stream.lens.cy_px))
            # The optical z axis is the camera's x axis (forward), the optical x the camera's -y (right).
            np.testing.assert_allclose(stream.optical_in_device()[:3, :3] @ [0, 0, 1], (1, 0, 0), atol=1e-12)
            np.testing.assert_allclose(stream.optical_in_device()[:3, :3] @ [1, 0, 0], (0, -1, 0), atol=1e-12)


def test_a_camera_without_every_stream_placed_is_refused(rig):
    # The V1's OAK-D-W: every stream relative to the left mono, depth aligned to it, the right mono a baseline away.
    v1 = rigs.rig("ethrc_v1")
    spec = geometry.camera_spec(v1.sensors["camera0"])
    places = {st.name: st.translation_m for st in spec.streams}
    assert places["mono_left"] == places["depth"] == (0.0, 0.0, 0.0)
    assert places["mono_right"] == (0.0, -OakDW.stereo_baseline_m, 0.0)
    assert {st.name: st.kind for st in spec.streams} == dict(OakDW.stream_kinds)
    assert v1.unsimulated(rtx=True) == []
    # The G1's Mid-360 is simulated with the RTX sensors; with the example cameras so are they.
    assert rig.unsimulated() == ["mid360", "zed", "d435"] and rig.unsimulated(rtx=True) == []


def test_the_scan_pattern_covers_its_field_and_never_repeats():
    model = LivoxMid360
    d0, t0 = geometry.scan_pattern(model, 0)
    d1, _ = geometry.scan_pattern(model, 1)
    # 200,000 points a second in 100 ms frames.
    assert len(d0) == 20_000
    np.testing.assert_allclose(np.linalg.norm(d0, axis=1), 1.0, atol=1e-12)
    el = np.arcsin(d0[:, 2])
    lo, hi = model.elevation_min_rad, model.elevation_max_rad
    assert el.min() >= lo - 1e-12 and el.max() <= hi + 1e-12
    assert np.all((t0 > 0) & (t0 < 0.1)) and np.all(np.diff(t0) > 0)
    # Uniform over the band's solid angle: half the rays above the elevation that halves its area.
    mid = math.asin((math.sin(lo) + math.sin(hi)) / 2)
    assert abs(np.mean(el > mid) - 0.5) < 0.01
    az = np.arctan2(d0[:, 1], d0[:, 0])
    assert np.histogram(az, bins=8, range=(-math.pi, math.pi))[0].min() > 0.95 * len(d0) / 8
    assert not np.isin(np.round(d1, 9).view([("", float)] * 3), np.round(d0, 9).view([("", float)] * 3)).any()
