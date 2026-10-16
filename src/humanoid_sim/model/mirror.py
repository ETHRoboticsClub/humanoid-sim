"""Left/right mirror map of the robot: which joint and drive is whose mirror image, and with which sign.

Mirroring across the sagittal plane maps joint positions ``q`` to ``q'`` about the standing pose ``c``:
``q'[perm[i]] = c[perm[i]] + sign[i] · (q[i] - c[i])``; joints on the plane map to themselves (sign -1 for a yaw
or roll joint, +1 for a pitch joint). Velocities, torques and deviations map with the signs alone. Derived
from the description:

* pairs: joints named ``*_l`` / ``*_r``; drives follow their joints, the ankle cranks pair upper with upper;
* the sagittal plane: the plane between the first pair's joint frames, snapped to the nearest base axis;
* joint signs: a rotation axis is a pseudovector, so its mirror image is ``-M a`` (``M`` the reflection); the
  sign is +1 where the partner's axis equals that image, -1 where it is its opposite;
* drive signs: the joint sign through each drive's ``device = sign · q``; for the cranks, the sign that maps
  the left cranks of a pose onto the right cranks of its mirror image (checked over the ankle's range).

The description is not exactly mirror-symmetric: the standing pose puts the feet several centimetres from
each other's mirror image, and left and right leg axes differ by a few degrees. ``standing_asymmetry_m`` is the
largest mismatch of a joint frame at the standing pose.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from humanoid_sim.model.description import Description

AXIS_TOLERANCE = 0.98  # |cos| between an axis and its partner's mirror image


@dataclass(frozen=True)
class MirrorMap:
    joint_names: list[str]
    joint_perm: list[int]  # q'[joint_perm[i]] = joint_sign[i] * q[i]
    joint_sign: list[int]
    drive_names: list[str]
    drive_perm: list[int]
    drive_sign: list[int]
    lateral_axis: int  # base-frame axis normal to the sagittal plane (0 x, 1 y, 2 z)
    center: list[float]  # the standing pose (URDF joint coordinates, rad), mirrored about
    standing_asymmetry_m: float

    def mirror_joints(self, q):
        """Mirror ``[..., J]`` joint positions (numpy or torch)."""
        c = self.center if not hasattr(q, "clone") else q.new_tensor(self.center)
        c = np.asarray(c) if not hasattr(q, "clone") else c
        return _apply(q - c, self.joint_perm, self.joint_sign) + c

    def mirror_joint_rates(self, qd):
        """Mirror ``[..., J]`` joint velocities, torques or position deviations."""
        return _apply(qd, self.joint_perm, self.joint_sign)

    def mirror_drives(self, x):
        """Mirror ``[..., D]`` drive velocities, torques or position deviations about the center."""
        return _apply(x, self.drive_perm, self.drive_sign)


def _apply(x, perm, sign):
    out = x.clone() if hasattr(x, "clone") else np.array(x, copy=True)
    for i, (j, s) in enumerate(zip(perm, sign, strict=True)):
        out[..., j] = s * x[..., i]
    return out


def _partner(name: str) -> str | None:
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return None


def _joint_positions(urdf_path, q: dict[str, float]) -> dict[str, np.ndarray]:
    """Every joint frame's origin in the root link's frame at configuration ``q`` (other joints at zero)."""
    from humanoid_sim.model.urdf import _origin

    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    by_child = {j.find("child").get("link"): j for j in joints.values()}
    link_tf: dict[str, np.ndarray] = {}

    def frame(j) -> np.ndarray:
        return tf(j.find("parent").get("link")) @ _origin(j.find("origin"))

    def tf(link: str) -> np.ndarray:
        if link not in link_tf:
            j = by_child.get(link)
            if j is None:
                link_tf[link] = np.eye(4)
            else:
                motion = np.eye(4)
                if j.get("name") in q:
                    axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                    motion[:3, :3] = _rotation(axis, q[j.get("name")])
                link_tf[link] = frame(j) @ motion
        return link_tf[link]

    return {name: frame(j)[:3, 3] for name, j in joints.items()}


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def _joint_frames(urdf_path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every joint's origin position and rotation axis in the root link's frame, at the zero configuration."""
    from humanoid_sim.model.urdf import _origin

    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    by_child = {j.find("child").get("link"): j for j in joints.values()}
    link_tf: dict[str, np.ndarray] = {}

    def tf(link: str) -> np.ndarray:
        if link not in link_tf:
            j = by_child.get(link)
            link_tf[link] = np.eye(4) if j is None else tf(j.find("parent").get("link")) @ _origin(j.find("origin"))
        return link_tf[link]

    out = {}
    for name, j in joints.items():
        frame = tf(j.find("parent").get("link")) @ _origin(j.find("origin"))
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None else "1 0 0").split()])
        out[name] = (frame[:3, 3], frame[:3, :3] @ (axis / np.linalg.norm(axis)))
    return out


def mirror_map(desc: Description) -> MirrorMap:
    frames = _joint_frames(desc.urdf_path)
    joints = list(desc.joints)
    first = next(n for n in joints if _partner(n) in desc.joints and n.endswith("_l"))
    gap = frames[first][0] - frames[_partner(first)][0]
    lateral = int(np.argmax(np.abs(gap)))
    if abs(gap[lateral]) / np.linalg.norm(gap) < AXIS_TOLERANCE:
        raise ValueError(f"{first} and its partner are not mirrored across a base-axis plane")
    n = np.zeros(3)
    n[lateral] = 1.0
    reflect = np.eye(3) - 2 * np.outer(n, n)

    perm, sign = [], []
    for name in joints:
        partner = _partner(name) if _partner(name) in desc.joints else name
        image = -reflect @ frames[name][1]
        c = float(image @ frames[partner][1])
        if abs(c) < AXIS_TOLERANCE:
            raise ValueError(f"{name}: axis is not mirrored by {partner}'s (cos {c:.3f})")
        perm.append(joints.index(partner))
        sign.append(1 if c > 0 else -1)
    standing = [desc.joints[j].standing_offset_rad for j in joints]
    at_standing = _joint_positions(desc.urdf_path, dict(zip(joints, standing, strict=True)))
    asymmetry = max(float(np.abs(at_standing[joints[perm[i]]] - reflect @ at_standing[name]).max())
                    for i, name in enumerate(joints))

    drives = list(desc.drives)
    d_perm, d_sign = [], []
    cranks = {side: desc.ankle_cranks(side) for side in ("l", "r")}
    crank_sign = _crank_signs(desc, joints, perm, sign)
    for name in drives:
        d = desc.drives[name]
        if d.crank is None:
            j = joints.index(d.joints[0])
            partner = desc.joints[joints[perm[j]]]
            d_perm.append(drives.index(partner.device))
            d_sign.append(sign[j] * d.sign * partner.sign)
        else:
            side = "l" if name in cranks["l"] else "r"
            k = cranks[side].index(name)
            d_perm.append(drives.index(cranks["r" if side == "l" else "l"][k]))
            d_sign.append(crank_sign[k])
    return MirrorMap(joints, perm, sign, drives, d_perm, d_sign, lateral, standing, asymmetry)


def _crank_signs(desc: Description, joints: list[str], perm: list[int], sign: list[int]) -> list[int]:
    """Signs mapping the left cranks of an ankle pose onto the right cranks of its mirror image."""
    import torch

    left, right = desc.ankle_transmission("l", dtype=torch.float64), desc.ankle_transmission("r", dtype=torch.float64)
    names = ("foot_pitch_l", "foot_roll_l")
    s = torch.tensor([sign[joints.index(n)] for n in names], dtype=torch.float64)
    offsets_l = torch.tensor([desc.joints[n].standing_offset_rad for n in names], dtype=torch.float64)
    offsets_r = torch.tensor([desc.joints[joints[perm[joints.index(n)]]].standing_offset_rad for n in names],
                             dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    dq = (torch.rand(64, 2, generator=g, dtype=torch.float64) - 0.5) * math.radians(30)
    # Poses around each side's standing pose, mirrored about it.
    crank_l = left.cranks(offsets_l + dq) - left.cranks(offsets_l.unsqueeze(0))
    crank_r = right.cranks(offsets_r + s * dq) - right.cranks(offsets_r.unsqueeze(0))
    out = []
    for k in range(2):
        c = torch.corrcoef(torch.stack([crank_l[:, k], crank_r[:, k]]))[0, 1].item()
        if abs(c) < 0.99:
            raise ValueError(f"ankle crank {k}: left and mirrored right cranks do not correspond (r = {c:.3f})")
        out.append(1 if c > 0 else -1)
    return out
