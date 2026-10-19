"""Named poses the robot starts in, and the height of an upright base that puts each on the ground.

* ``"home"``: the description's home pose (:attr:`~humanoid_sim.model.description.Description.home_pose`), the
  pose the CAD is exported in: legs straight, both soles level and at the same height.
* ``"standing"``: the description's standing pose, captured on the robot. Under an upright base its soles are
  pitched by 10° and 17°.

Training scenes (``humanoid_sim.train.scene``) and the runtime host both start robots from these.
"""

from __future__ import annotations

from pathlib import Path

from humanoid_sim.model.description import Description

STANDING_CLEARANCE_M = 0.001  # spawn the soles just above the ground

POSES = ("home", "standing")


def initial_pose(usd: Path, robot: str = "ethrc_v1", pose: str = "home"):
    """Joint positions ``[J]`` (description order) of ``pose`` (see :data:`POSES`) and the upright base height at
    which its lowest sole corner touches the ground."""
    import torch

    from humanoid_sim.model.asset import asset_meta
    from humanoid_sim.model.kinematics import standing_base_height

    desc = Description.load(robot)
    soles = asset_meta(usd)["sole_boxes"]
    if pose == "home":
        q = torch.tensor([desc.home_pose[n] for n in desc.joints], dtype=torch.float64)
    elif pose == "standing":
        q = torch.tensor([desc.joints[n].standing_offset_rad for n in desc.joints], dtype=torch.float64)
    else:
        raise ValueError(f"unknown pose {pose!r}; known: {POSES}")
    return q, standing_base_height(usd, desc, soles, q)
