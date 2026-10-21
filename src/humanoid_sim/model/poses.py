"""Named poses the robot starts in, and the height of an upright base that puts each on the ground.

* ``"home"``: the pose the profile names (:func:`home_pose`). For the ETHRC V1 the pose its CAD is exported in, legs
  straight and both soles level at one height; for the G1 Unitree's deploy standing pose.
* ``"standing"``: every joint at the description's standing offset. The ETHRC V1's offsets are 0, so for it, as for
  the G1, the URDF's zero; on the V1 that is also the home pose.

Training scenes (``humanoid_sim.train.scene``) and the runtime host both start robots from these. A runtime session
starts in the profile's ``poses.start`` (:func:`start_pose`).
"""

from __future__ import annotations

from pathlib import Path

from humanoid_sim.model.description import Description

STANDING_CLEARANCE_M = 0.001  # spawn the soles just above the ground

POSES = ("home", "standing")


def home_pose(desc, profile: dict) -> dict[str, float]:
    """URDF joint positions of the profile's home pose (``poses.home``): ``zero``, ``{export: <table>}`` (the pose a
    CAD export was made in) or ``{preset: <preset_id>}`` (from the robot's ``position-presets.yaml``)."""
    spec = profile.get("poses", {}).get("home", "zero")
    if spec == "zero":
        return {n: 0.0 for n in desc.joints}
    if isinstance(spec, dict) and set(spec) == {"export"}:
        return desc.export_pose(spec["export"])
    if isinstance(spec, dict) and set(spec) == {"preset"}:
        return desc.preset_pose(spec["preset"])
    raise ValueError(f"poses.home {spec!r}: zero, {{export: <table>}} or {{preset: <preset_id>}}")


def start_pose(profile: dict) -> str:
    """The named pose a runtime session starts in: the profile's ``poses.start``."""
    name = profile.get("poses", {}).get("start", "home")
    if name not in POSES:
        raise ValueError(f"poses.start {name!r}: one of {POSES}")
    return name


def pose_joint_positions(desc, profile: dict, pose: str) -> dict[str, float]:
    """URDF joint positions (rad) of the named ``pose`` (see :data:`POSES`)."""
    if pose == "home":
        return home_pose(desc, profile)
    if pose == "standing":
        return {n: desc.joints[n].standing_offset_rad for n in desc.joints}
    raise ValueError(f"unknown pose {pose!r}; known: {POSES}")


def initial_pose(usd: Path, robot: str = "ethrc_v1", pose: str = "home"):
    """Joint positions ``[J]`` (description order) of ``pose`` (see :data:`POSES`) and the upright base height at
    which its lowest collision point touches the ground."""
    import torch

    from humanoid_sim.model.asset import load_profile
    from humanoid_sim.model.kinematics import standing_base_height

    desc = Description.load(robot)
    joints = pose_joint_positions(desc, load_profile(robot), pose)
    q = torch.tensor([joints[n] for n in desc.joints], dtype=torch.float64)
    return q, standing_base_height(usd, desc, q)
