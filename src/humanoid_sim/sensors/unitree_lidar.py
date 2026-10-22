"""The G1's Mid-360 as Unitree's lidar service publishes it, emulated on DDS: Controls' camera service
(``--unitree-lidar``) reads a simulated sweep exactly as it reads the robot's, and republishes it on its sensor
contract (``sensors/mid360/points`` and ``sensors/rig``). No ROS: the message is ROS 2's ``sensor_msgs/PointCloud2``
written straight onto DDS with CycloneDDS's Python binding, as Unitree's own SDK does.

What it matches, and where that comes from (Controls origin/master 44353e2b unless named otherwise):

* topic and type: ROS 2's ``/utlidar/cloud_livox_mid360`` (``ethrc_cameras/lidar.py``, ``UNITREE_MID360``), which is
  the DDS topic ``rt/utlidar/cloud_livox_mid360`` with type ``sensor_msgs::msg::dds_::PointCloud2_`` (ROS 2's
  mapping onto DDS, which rmw_cyclonedds and Unitree's SDK use);
* domain: Unitree's, 0 by default (``--unitree-domain`` in ``ethrc_cameras/main.py``);
* QoS: best effort, keep last 5. Controls reads with ROS's sensor-data profile (best effort, keep last 5,
  ``test_lidar_relay.py``), which accepts a best-effort or reliable writer alike; what Unitree's service writes with
  is not recorded;
* frame: ``livox_frame``, which Controls relabels ``mid360_link`` without rotation (the G1 URDF's
  ``mid360_joint``; the test double in ``test_lidar_relay.py``);
* point layout: x, y, z and intensity float32 at offsets 0, 4, 8 and 12, ring uint16 at 16 and time float32 at 18,
  22 bytes a point, one row (Controls' double of Unitree's sweep, ``test_lidar_relay.unitree_sweep``); a ray without
  a return is a point at the origin, which Controls drops (``lidar.sweep_from_cloud``);
* time: nanoseconds after the header stamp, the sweep's start; 0.1 to 100.3 ms across a 10 Hz sweep on the lab unit
  (``lidar.UNITREE_MID360``).

Not modelled: the Mid-360's own IMU (Unitree's service does not publish it either, ``ethrc_cameras/README.md``),
intensity (a constant), and multiple returns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from humanoid_sim.sensors.models import LivoxMid360

UNITREE_MODEL = LivoxMid360.model  # the lidar Unitree's service publishes
TOPIC = "rt/utlidar/cloud_livox_mid360"
FRAME_ID = "livox_frame"
DOMAIN_ID = 0
HISTORY_DEPTH = 5
# sensor_msgs/PointField datatypes
UINT16, FLOAT32 = 4, 7
FIELDS = (("x", 0, FLOAT32), ("y", 4, FLOAT32), ("z", 8, FLOAT32), ("intensity", 12, FLOAT32), ("ring", 16, UINT16),
          ("time", 18, FLOAT32))
POINT_STEP = 22
POINT_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "ring", "time"],
                        "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f4"],
                        "offsets": [0, 4, 8, 12, 16, 18], "itemsize": POINT_STEP})
RINGS = 4  # the Mid-360's four laser lines
INTENSITY = 100.0  # a constant reflectivity for every return


@dataclass(frozen=True)
class Sweep:
    """One sweep as the service publishes it: the header stamp (ns, the sweep's start) and the packed points."""

    stamp_ns: int
    data: bytes
    width: int

    @property
    def row_step(self) -> int:
        return self.width * POINT_STEP


def pack_sweep(points_m: np.ndarray, hit: np.ndarray, time_s: np.ndarray, stamp_ns: int) -> Sweep:
    """A sweep of ``points_m`` ``[N, 3]`` in the sensor frame, ``hit`` ``[N]`` marking rays that returned (the others
    become points at the origin) and each ray's time after the sweep's start ``time_s`` ``[N]``."""
    n = len(points_m)
    out = np.zeros(n, dtype=POINT_DTYPE)
    hit = np.asarray(hit, dtype=bool)
    xyz = np.where(hit[:, None], np.asarray(points_m, dtype=np.float32), 0.0)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["intensity"] = np.where(hit, INTENSITY, 0.0)
    out["ring"] = np.arange(n) % RINGS
    out["time"] = np.asarray(time_s, dtype=np.float64) * 1e9
    return Sweep(int(stamp_ns), out.tobytes(), n)


def idl_types():
    """The ROS 2 message types on DDS as CycloneDDS Python IDL structs (``Time_``, ``Header_``, ``PointField_``,
    ``PointCloud2_``, from ``sensors/ros2_idl.py``). Imports CycloneDDS."""
    from humanoid_sim.sensors import ros2_idl

    return ros2_idl.Time_, ros2_idl.Header_, ros2_idl.PointField_, ros2_idl.PointCloud2_


class UnitreeLidarPublisher:
    """Publishes sweeps on Unitree's DDS domain as its lidar service does. ``domain_id`` is Unitree's domain; the
    network interface follows CycloneDDS's own configuration (``CYCLONEDDS_URI``)."""

    def __init__(self, domain_id: int = DOMAIN_ID, topic: str = TOPIC, frame_id: str = FRAME_ID):
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        self.Time_, self.Header_, self.PointField_, self.PointCloud2_ = idl_types()
        self.frame_id = frame_id
        self.participant = DomainParticipant(domain_id)
        self.topic = Topic(self.participant, topic, self.PointCloud2_)
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(HISTORY_DEPTH))
        self.writer = DataWriter(self.participant, self.topic, qos=qos)
        self.fields = [self.PointField_(name=n, offset=o, datatype=d, count=1) for n, o, d in FIELDS]
        self.published = 0
        self.last_publish_s = float("nan")

    def message(self, sweep: Sweep):
        sec, nanosec = divmod(sweep.stamp_ns, 1_000_000_000)
        return self.PointCloud2_(header=self.Header_(stamp=self.Time_(sec=sec, nanosec=nanosec), frame_id=self.frame_id),
                                 height=1, width=sweep.width, fields=self.fields, is_bigendian=False,
                                 point_step=POINT_STEP, row_step=sweep.row_step, data=sweep.data, is_dense=False)

    def publish(self, sweep: Sweep) -> None:
        start = time.perf_counter()
        self.writer.write(self.message(sweep))
        self.last_publish_s = time.perf_counter() - start
        self.published += 1
