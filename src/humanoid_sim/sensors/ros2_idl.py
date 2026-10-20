"""ROS 2's ``sensor_msgs/PointCloud2`` and the types it holds, as they travel on DDS (ROS 2's ``<pkg>::msg::dds_``
mapping, which rmw_cyclonedds and Unitree's SDK use), as CycloneDDS Python IDL structs. Importing this module imports
CycloneDDS (the Isaac Sim image builds it; ``docker/Dockerfile.kit``). It has no ``from __future__ import
annotations``: CycloneDDS reads the field types from the annotations."""

from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl import types as t


@dataclass
class Time_(IdlStruct, typename="builtin_interfaces.msg.dds_.Time_"):
    sec: t.int32
    nanosec: t.uint32


@dataclass
class Header_(IdlStruct, typename="std_msgs.msg.dds_.Header_"):
    stamp: Time_
    frame_id: str


@dataclass
class PointField_(IdlStruct, typename="sensor_msgs.msg.dds_.PointField_"):
    name: str
    offset: t.uint32
    datatype: t.uint8
    count: t.uint32


@dataclass
class PointCloud2_(IdlStruct, typename="sensor_msgs.msg.dds_.PointCloud2_"):
    header: Header_
    height: t.uint32
    width: t.uint32
    fields: t.sequence[PointField_]
    is_bigendian: bool
    point_step: t.uint32
    row_step: t.uint32
    data: t.sequence[t.uint8]
    is_dense: bool
