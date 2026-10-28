"""World poses of the description's frames from a base pose and joint positions, on the host (NumPy, no simulator).

The ground truth of a sensor is where its frame was: the base pose and the joint positions of a recorded run
(:mod:`humanoid_sim.runtime.trajectory`) carried along the URDF's joint chain to the frame's link, fixed joints
included. Revolute, continuous and prismatic joints are followed; a joint that is on the path but not among the
given joint names is refused.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from humanoid_sim.sensors.geometry import quat_xyzw, rpy_matrix


def quat_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[N, 3, 3]`` of unit quaternions ``[N, 4]`` (x y z w)."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=-1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def axis_rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotations ``[N, 3, 3]`` about the unit ``axis`` by ``angle`` ``[N]`` (Rodrigues)."""
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


class FrameTree:
    def __init__(self, urdf: str | Path, joints: list[str]):
        root = ET.parse(urdf).getroot()
        self.joints = list(joints)
        self._by_child = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            axis = j.find("axis")
            self._by_child[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "xyz": np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]),
                "R": rpy_matrix(*[float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]),
                "axis": np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else
                np.array([1.0, 0.0, 0.0])}
        parents = set(self._by_child)
        roots = {j["parent"] for j in self._by_child.values()} - parents
        if len(roots) != 1:
            raise ValueError(f"{Path(urdf).name}: expected one root link, found {sorted(roots)}")
        self.root = roots.pop()

    def _chain(self, frame: str) -> list[dict]:
        chain = []
        while frame != self.root:
            if frame not in self._by_child:
                raise ValueError(f"no link {frame!r} in the description")
            chain.append(self._by_child[frame])
            frame = chain[-1]["parent"]
        return chain[::-1]

    def world_poses(self, frame: str, base_position_m: np.ndarray, base_quat_xyzw: np.ndarray,
                    joint_pos_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions ``[N, 3]`` and quaternions ``[N, 4]`` (x y z w) of ``frame`` in the world, for base poses
        ``[N, 3]``, ``[N, 4]`` (the root link's) and joint positions ``[N, J]`` (the names given at construction)."""
        R = quat_matrix(np.asarray(base_quat_xyzw, dtype=float))
        p = np.array(base_position_m, dtype=float)
        q = np.asarray(joint_pos_rad, dtype=float)
        for j in self._chain(frame):
            p = p + (R @ j["xyz"][:, None])[..., 0]
            R = R @ j["R"]
            if j["type"] in ("revolute", "continuous"):
                R = R @ axis_rotation(j["axis"] / np.linalg.norm(j["axis"]), q[:, self._index(j)])
            elif j["type"] == "prismatic":
                p = p + (R @ (j["axis"] / np.linalg.norm(j["axis"]))[:, None])[..., 0] * q[:, self._index(j)][:, None]
            elif j["type"] != "fixed":
                raise ValueError(f"joint {j['name']!r}: type {j['type']!r} is not followed")
        return p, np.array([quat_xyzw(r) for r in R])

    def _index(self, joint: dict) -> int:
        if joint["name"] not in self.joints:
            raise ValueError(f"joint {joint['name']!r} is on the path but not among the given joints")
        return self.joints.index(joint["name"])


def frame_poses(trajectory, urdf: str | Path, frame: str, t_ns=None) -> dict:
    """``frame``'s world poses along ``trajectory`` (:class:`~humanoid_sim.runtime.trajectory.Trajectory`): at its
    own record times, or at ``t_ns``."""
    truth = trajectory.at(trajectory.records["sim_time_ns"] if t_ns is None else t_ns)
    pos, quat = FrameTree(urdf, trajectory.joints).world_poses(
        frame, truth["base_position_m"], truth["base_quat_xyzw"], truth["joint_pos_rad"])
    return {"sim_time_ns": truth["sim_time_ns"], "position_m": pos, "quat_xyzw": quat}


def write_tum(poses: dict, path: str | Path) -> None:
    """TUM trajectory format (``t x y z qx qy qz qw``, t in s), what evo and most SLAM evaluators read."""
    t = poses["sim_time_ns"].astype(np.float64) * 1e-9
    rows = np.column_stack([t, poses["position_m"], poses["quat_xyzw"]])
    np.savetxt(path, rows, fmt=["%.9f"] + ["%.9f"] * 7)
