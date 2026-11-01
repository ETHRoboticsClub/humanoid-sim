import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from humanoid_sim.sensors import rigs
from humanoid_sim.sensors.models import Mid360Imu, Rig
from humanoid_sim.sensors.geometry import Mount
from humanoid_sim.physics import isaaclab
from humanoid_sim.runtime import host

sensor_urdf = Path(os.environ["CAPTURE_SENSOR_URDF"])
robot_urdf = Path(os.environ["CAPTURE_ROBOT_URDF"])
sensor_tree = ET.parse(sensor_urdf).getroot()
robot_tree = ET.parse(robot_urdf).getroot()
for element in list(sensor_tree):
    if element.attrib.get("name") in {"mid360_imu_link", "mid360_imu_joint"}:
        sensor_tree.remove(element)
if ET.canonicalize(ET.tostring(sensor_tree, encoding="unicode"), strip_text=True) != ET.canonicalize(ET.tostring(robot_tree, encoding="unicode"), strip_text=True):
    raise RuntimeError("sensor description changes more than the additional IMU mount")

original_load = rigs.load
original_sensor_cfg = isaaclab.imu_sensor_cfg
original_descriptors = host.extra_imu_descriptors

def capture_rig():
    return rigs.g1_cameras().extended(Rig(mid360_imu=Mid360Imu("mid360_imu_link", sample_period_ns=5_000_000)))

rigs.EXTRAS["g1-capture"] = capture_rig

def load_rig(robot, desc, extra=None):
    if extra != "g1-capture":
        return original_load(robot, desc, extra)
    combined = rigs.rig(robot).extended(capture_rig())
    combined.check_frames(SimpleNamespace(urdf_path=sensor_urdf,
        link_names=lambda: {element.attrib["name"] for element in ET.parse(sensor_urdf).getroot().findall("link")},
        imu_link=desc.imu_link))
    return combined

def sensor_cfg(robot="ethrc_v1", frame=None, usd_path=None):
    if frame != "mid360_imu_link":
        return original_sensor_cfg(robot, frame, usd_path)
    from isaaclab.sensors import ImuCfg
    from humanoid_sim.sensors.rtx import body_paths
    mount = Mount.from_urdf(sensor_urdf, frame)
    return ImuCfg(prim_path=f"{isaaclab.ROBOT_PRIM}/{body_paths(usd_path, {mount.link})[mount.link]}",
                  offset=ImuCfg.OffsetCfg(pos=mount.translation_m, rot=mount.rotation_xyzw))

def descriptors(desc, extra_rig):
    if extra_rig != "g1-capture":
        return original_descriptors(desc, extra_rig)
    proxy = SimpleNamespace(robot=desc.robot, urdf_path=sensor_urdf, imu_link=desc.imu_link)
    result = original_descriptors(proxy, extra_rig)
    result["mid360_imu"]["mount"]["source_sha256"] = hashlib.sha256(sensor_urdf.read_bytes()).hexdigest()
    return result

rigs.load = load_rig
isaaclab.imu_sensor_cfg = sensor_cfg
host.extra_imu_descriptors = descriptors

if __name__ == "__main__":
    from humanoid_sim.cli import main
    raise SystemExit(main())
