"""Shared test helpers: the robots the physics tests run on, and their built assets."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description

ASSETS = Path(__file__).resolve().parents[1] / "outputs" / "assets"
GRAVITY = 9.81

# The robots the physics, world, training and runtime tests step: the ETHRC V1 and both G1 variants (the one with
# hands is the G1 Controls launches).
PHYSICS_ROBOTS = ("ethrc_v1", "g1_29dof_rev_1_0", "g1_29dof_with_hand_rev_1_0")

# Parts of each robot a test acts on by name. Test inputs, not expectations: what a test expects of them comes from
# the description.
#   torso: the upper-body link (a push's point of application)
#   vertical_joint: a joint turning about the vertical axis at home (gravity adds no torque of its own)
#   elbow: a 1:1 elbow drive's joint
#   ankle_pitch: the two ankle pitch hinges
#   shoulders: the two links a gantry harness hooks onto
_G1 = {"torso": "torso_link", "vertical_joint": "waist_yaw_joint", "elbow": "right_elbow_joint",
       "ankle_pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
       "shoulders": ("left_shoulder_pitch_link", "right_shoulder_pitch_link")}
PARTS = {
    "ethrc_v1": {"torso": "Component171_1", "vertical_joint": "waist_upperbody_yaw", "elbow": "elbow_flexion_r",
                 "ankle_pitch": ("foot_pitch_l", "foot_pitch_r"),
                 "shoulders": ("shoulder_connector_l_1", "shoulder_connector_r_1")},
    "g1_29dof_rev_1_0": _G1,
    "g1_29dof_with_hand_rev_1_0": _G1,
}


def built_usd(robot: str, *, base_joints: bool = False) -> Path:
    """The built asset's USD for ``robot``; skips the test when it is not built."""
    out = asset_dir(Description.load(robot), load_profile(robot), ASSETS, base_joints=base_joints)
    if not (out / "asset.json").exists():
        flag = " --base-joints" if base_joints else ""
        pytest.skip(f"robot asset not built ({out}); run `humanoid-sim --robot {robot} build-asset{flag}` in the "
                    "Isaac Lab image")
    return out / json.loads((out / "asset.json").read_text())["usd"]


def urdf_mass_kg(robot: str) -> float:
    """Total link mass read straight from the description's URDF (welded links included)."""
    root = ET.parse(Description.load(robot).urdf_path).getroot()
    return sum(float(m.get("value")) for m in root.findall("link/inertial/mass"))


def imu_samples_per_s(robot: str) -> float:
    """Mean IMU delivery rate of the robot's profile: one sample every ``sample_period_ns``."""
    return 1e9 / load_profile(robot)["sensors"]["imu"]["sample_period_ns"]


def gravity_torque(usd, joints, q, base_height_m):
    """Gravity torque ``[J]`` (Nm) on each joint at ``q`` with the base welded upright, from host forward
    kinematics alone: minus the gradient of the potential energy, by central differences."""
    from humanoid_sim.model.kinematics import RobotKinematics, _rotate

    kin = RobotKinematics(usd, joints, base_height_m=base_height_m)
    mass = kin.model.body_mass.numpy().astype(np.float64)
    com = kin.model.body_com.numpy().astype(np.float64)

    def energy(qq):
        poses = kin.body_poses(qq.reshape(1, -1))
        z = np.array([_rotate(p[3:], c)[2] + p[2] for p, c in zip(poses, com, strict=True)])
        return GRAVITY * float((mass * z).sum())

    h = 1e-4
    tau = np.zeros(len(joints))
    for j in range(len(joints)):
        dq = torch.zeros_like(q)
        dq[j] = h
        tau[j] = -(energy(q + dq) - energy(q - dq)) / (2 * h)
    return tau
