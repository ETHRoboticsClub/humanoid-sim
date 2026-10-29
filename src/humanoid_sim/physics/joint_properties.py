"""Per-joint solver properties, written to the articulation at spawn.

* ``stiffness``, ``damping``: the plant's constant solver PD (``k0``, ``d0``), on native crank shafts
  rather than the passive ankle hinges for a delivered USD mechanism.
* ``armature``: description rotor inertia, on native shafts without duplicate hinge armature.
* ``friction`` (Coulomb) and ``passive_damping`` (viscous): description, enabled-drive values.
* ``effort_limit``: the drive peak for 1:1 joints; for the ankle hinges, the joint torque the two cranks
  can reach at their peak at standing (``Σ |J| · peak`` per column), so the solver clamp never binds
  before the device layer's own envelope does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from humanoid_sim.core import PlantCore


@dataclass
class JointProperties:
    names: list[str]
    stiffness: list[float]
    damping: list[float]
    armature: list[float]
    friction: list[float]
    passive_damping: list[float]
    effort_limit: list[float]
    standing: list[float]


def joint_properties(plant: PlantCore, names: list[str]) -> JointProperties:
    desc = plant.desc
    properties = desc.physics_joints if any(n not in desc.joints for n in names) else desc.joints
    pj = plant.map.joint_names
    idx = [pj.index(n) for n in names]
    k0 = plant.k0[0, idx].tolist()
    d0 = plant.d0[0, idx].tolist()
    effort = {n: properties[n].effort_peak for n in names}
    if properties is not desc.joints:
        actuated = {pj[index] for index in plant.actuated.tolist()}
        effort.update({name: float("inf") for name in names if name not in actuated})
    standing = torch.tensor([properties[n].standing_offset_rad for n in pj], dtype=torch.float64)
    for _, d_idx, j_idx, tr in plant.map.ankles:
        jac = tr.to("cpu", torch.float64).jacobian(standing[j_idx.cpu()].unsqueeze(0))[0]
        peaks = torch.tensor([desc.drives[plant.map.drive_names[int(i)]].effort_peak for i in d_idx],
                             dtype=torch.float64)
        reach = (jac.abs() * peaks.unsqueeze(1)).sum(0)
        tr.to(plant.cfg.device, plant.cfg.dtype)
        for col, j in enumerate(j_idx.tolist()):
            effort[pj[j]] = float(reach[col])
    return JointProperties(
        names=list(names), stiffness=k0, damping=d0,
        armature=[properties[n].armature for n in names],
        friction=[properties[n].friction for n in names],
        passive_damping=[properties[n].damping for n in names],
        effort_limit=[effort[n] for n in names],
        standing=[properties[n].standing_offset_rad for n in names],
    )
