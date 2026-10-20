"""The emulated Unitree Mid-360 cloud (``sensors/unitree_lidar.py``) parsed by Controls' own reader of Unitree's
sweeps (``ethrc_cameras.lidar.sweep_from_cloud`` with ``UNITREE_MID360``, from the checkout ``HUMANOID_SIM_CONTROLS``
names): the points that returned come back in the contract layout, the ray without a return is dropped, and each
point's time is its firing time."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from humanoid_sim.sensors import unitree_lidar as ul

POINTS = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [4.0, 0.0, -1.0]])
HIT = np.array([True, False, True])
TIMES_S = np.array([0.0001, 0.05, 0.1003])
STAMP_NS = 1_700_000_000_123_456_789


def test_controls_reads_the_sweep(controls_cameras):
    lidar = importlib.import_module("ethrc_cameras.lidar")
    assert "rt" + lidar.UNITREE_MID360.topic == ul.TOPIC
    sweep = ul.pack_sweep(POINTS, HIT, TIMES_S, STAMP_NS)
    fields = {name: (offset, datatype, 1) for name, offset, datatype in ul.FIELDS}
    parsed = lidar.sweep_from_cloud(lidar.UNITREE_MID360, sweep.stamp_ns, fields, ul.POINT_STEP, sweep.width, 1,
                                    sweep.row_step, False, sweep.data, ul.FLOAT32)
    assert parsed.device_stamp_ns == STAMP_NS
    np.testing.assert_allclose(np.stack([parsed.points[k] for k in "xyz"], axis=1), POINTS[HIT])
    np.testing.assert_allclose(parsed.points["t"], TIMES_S[HIT], rtol=1e-6)
    assert parsed.duration_ns == pytest.approx(TIMES_S.max() * 1e9, rel=1e-6)


def test_the_layout_without_controls():
    sweep = ul.pack_sweep(POINTS, HIT, TIMES_S, STAMP_NS)
    raw = np.frombuffer(sweep.data, dtype=ul.POINT_DTYPE)
    assert ul.POINT_DTYPE.itemsize == ul.POINT_STEP == 22 and len(sweep.data) == 3 * 22
    np.testing.assert_array_equal(raw["x"], [1.0, 0.0, 4.0])  # no return: a point at the origin
    np.testing.assert_array_equal(raw["ring"], [0, 1, 2])
    np.testing.assert_allclose(raw["time"], TIMES_S * 1e9, rtol=1e-6)


def test_the_message_round_trips_through_cdr():
    pytest.importorskip("cyclonedds")
    _, _, _, PointCloud2_ = ul.idl_types()
    pub = object.__new__(ul.UnitreeLidarPublisher)  # message building only: no DDS participant
    pub.Time_, pub.Header_, pub.PointField_, pub.PointCloud2_ = ul.idl_types()
    pub.frame_id = ul.FRAME_ID
    pub.fields = [pub.PointField_(name=n, offset=o, datatype=d, count=1) for n, o, d in ul.FIELDS]
    msg = pub.message(ul.pack_sweep(POINTS, HIT, TIMES_S, STAMP_NS))
    back = pub.PointCloud2_.deserialize(msg.serialize())
    assert back.header.frame_id == "livox_frame" and back.width == 3 and back.point_step == 22
    assert (back.header.stamp.sec, back.header.stamp.nanosec) == divmod(STAMP_NS, 1_000_000_000)
    assert bytes(back.data) == msg.data or list(back.data) == list(msg.data)
