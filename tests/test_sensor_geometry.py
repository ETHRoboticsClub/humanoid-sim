"""Where the G1's cameras and lidar sit and what they see (``sensors/geometry.py``), built from the sensor models and
the rig: mounts on the bodies of the G1 URDF, each lens against projections worked out by hand for points at known
angles, an image resampled from one lens onto another, and the lidar's pattern against its model's figures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import PreparedUrdf
from humanoid_sim.sensors import geometry, rigs
from humanoid_sim.sensors.models import LivoxMid360, OakDW

G1 = "g1_29dof_with_hand_rev_1_0"
CAMERAS = "g1-cameras-d435"  # sensors/rigs.py EXTRAS


@pytest.fixture(scope="module")
def desc():
    return Description.load(G1)


@pytest.fixture(scope="module")
def rig(desc):
    return rigs.load(G1, desc, CAMERAS)


def test_every_mount_sits_on_a_body_of_the_robot(desc, rig):
    bodies = PreparedUrdf(desc, load_profile(G1)).build().body_tree()
    for s in rig.sensors.values():
        if s.kind == "imu":
            continue
        m = geometry.Mount.from_urdf(desc.urdf_path, s.frame)
        assert m.link in bodies, s.frame
        np.testing.assert_allclose(m.R.T @ m.R, np.eye(3), atol=1e-9)


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


RATIONAL = (0.3, -0.04, 0.0012, -0.0007, -0.0015, 0.65, -0.02, -0.01)  # k1 k2 p1 p2 k3 k4 k5 k6, a wide lens


def test_a_rational_lens_is_opencvs_rational_model():
    k1, k2, p1, p2, k3, k4, k5, k6 = RATIONAL
    lens = geometry.Lens(640, 400, 280.0, 281.0, 320.0, 200.0, "rational_polynomial", RATIONAL)
    for x, y in ((0.2, 0.0), (0.0, -0.5), (0.8, 0.6), (-1.1, 0.3)):
        r2 = x * x + y * y
        radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        (u, v), = lens.project([[2 * x, 2 * y, 2.0]])
        assert (u, v) == (pytest.approx(280.0 * xd + 320.0), pytest.approx(281.0 * yd + 200.0))
    # Without its denominator it is plumb_bob.
    plumb = geometry.Lens(640, 400, 280.0, 281.0, 320.0, 200.0, "plumb_bob", RATIONAL[:5])
    flat = geometry.Lens(640, 400, 280.0, 281.0, 320.0, 200.0, "rational_polynomial", (*RATIONAL[:5], 0, 0, 0))
    points = [[0.3, -0.2, 1.0], [-0.9, 0.5, 1.0]]
    np.testing.assert_allclose(flat.project(points), plumb.project(points))


def test_an_image_resampled_onto_a_pinhole_is_what_the_pinhole_sees():
    lens = geometry.Lens(640, 400, 280.0, 281.0, 320.0, 200.0, "rational_polynomial", RATIONAL)
    onto = geometry.Lens(640, 400, 260.0, 260.0, 322.0, 198.0)
    v, u = np.mgrid[0:400, 0:640].astype(float)
    out = geometry.resample(np.stack([u, v], axis=-1), lens, onto)  # each pixel holds its own place in ``lens``
    for x, y in ((322, 198), (100, 50), (600, 380)):
        (eu, ev), = lens.project([[(x - 322.0) / 260.0, (y - 198.0) / 260.0, 1.0]])
        np.testing.assert_allclose(out[y, x], (eu, ev), atol=1e-9)  # bilinear on a linear ramp is exact
    same = geometry.Lens(640, 400, 260.0, 260.0, 322.0, 198.0)
    np.testing.assert_allclose(geometry.resample(u, same, onto), u, atol=1e-9)
    with pytest.raises(ValueError, match="distortion-free"):
        geometry.resample(u, onto, lens)


def test_a_streams_calibrated_orientation_turns_its_optical_frame_and_its_rtx_camera():
    from humanoid_sim.sensors.rtx import stream_offset

    yaw = math.radians(2.0)  # the depth view turned 2 deg about the camera's up axis, towards its left
    turn = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    q = geometry.quat_xyzw(turn @ geometry.LINK_FROM_OPTICAL)
    pinhole = (300.0, 300.0, 320.0, 200.0)
    camera = OakDW("camera_link", width=640, height=400, frame_period_ns=33_333_333,
                   intrinsics={s: pinhole for s in OakDW.streams}, stream_rotations_xyzw={"depth": q})
    camera.name = "cam"
    spec = geometry.camera_spec(camera)
    depth, left = spec.stream("depth"), spec.stream("mono_left")
    np.testing.assert_allclose(geometry.quat_matrix(depth.rotation_xyzw), turn @ geometry.LINK_FROM_OPTICAL,
                               atol=1e-12)
    np.testing.assert_allclose(geometry.quat_matrix(left.rotation_xyzw), geometry.LINK_FROM_OPTICAL, atol=1e-12)
    # On a body turned a quarter about z, the RTX camera of the nominal stream looks along the body's y, the turned
    # one 2 deg further round.
    mount = geometry.Mount("camera_link", "body", (0.0, 0.0, 0.0), ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    for stream, angle in ((left, 0.0), (depth, yaw)):
        _, q_rtx = stream_offset(mount, stream)
        forward = geometry.quat_matrix(q_rtx) @ [1.0, 0.0, 0.0]
        np.testing.assert_allclose(forward, (-math.sin(angle), math.cos(angle), 0.0), atol=1e-12)


def test_a_turned_view_samples_where_its_rays_land():
    lens = geometry.Lens(640, 400, 280.0, 281.0, 320.0, 200.0, "rational_polynomial", RATIONAL)
    onto = geometry.Lens(640, 400, 260.0, 260.0, 322.0, 198.0)
    a = math.radians(1.0)
    rotation = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    v, u = np.mgrid[0:400, 0:640].astype(float)
    out = geometry.resample(np.stack([u, v], axis=-1), lens, onto, rotation)
    for x, y in ((322, 198), (100, 50), (600, 380)):
        (eu, ev), = lens.project([rotation @ [(x - 322.0) / 260.0, (y - 198.0) / 260.0, 1.0]])
        np.testing.assert_allclose(out[y, x], (eu, ev), atol=1e-9)


def test_the_v1s_raw_streams_have_rational_lenses_and_its_depth_none():
    spec = geometry.camera_spec(rigs.rig("ethrc_v1").sensors["camera0"])
    models = {s.name: (s.lens.distortion_model, len(s.lens.distortion)) for s in spec.streams}
    assert models == {"rgb": ("rational_polynomial", 8), "mono_left": ("rational_polynomial", 8),
                      "mono_right": ("rational_polynomial", 8), "depth": ("none", 0)}


def test_the_zed_minis_streams_are_its_rig_calibration(rig):
    spec = geometry.camera_spec(rig.sensors["zed"])
    assert (spec.camera_id, spec.product, spec.sensor_link) == ("sim-zed", "Stereolabs ZED Mini", "camera0_camera")
    # Grey eyes at HD720 and 30 Hz, a stereo pair, as Controls' zed backend publishes them.
    assert [(s.name, s.kind, s.encoding, s.sync_group, s.shutter) for s in spec.streams] == [
        ("left", "mono", "mono8", "stereo", "rolling"), ("right", "mono", "mono8", "stereo", "rolling")]
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
    # Controls' realsense backend: the infrared pair is global shutter, the colour imager rolling.
    assert {n: st.shutter for n, st in s.items()} == {"rgb": "rolling", "mono_left": "global",
                                                       "mono_right": "global", "depth": "global"}
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
    # The unit's calibration places the right mono about a baseline to the right (y right is negative).
    assert places["mono_right"][1] < 0
    assert np.linalg.norm(places["mono_right"]) == pytest.approx(OakDW.stereo_baseline_m, abs=0.001)
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
