"""Per-joint solver properties, written to the articulation at spawn.

* ``stiffness``, ``damping``: the plant's constant solver PD (``k0``, ``d0``; ankles via J at standing).
* ``armature``: description (ankles: ``armature_joint``, the reflected crank inertia).
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
    if plant.solver_map is not plant.map:
        return native_joint_properties(plant, names)
    pj = plant.map.joint_names
    idx = [pj.index(n) for n in names]
    k0 = plant.k0[0, idx].tolist()
    d0 = plant.d0[0, idx].tolist()
    effort = {n: desc.joints[n].effort_peak for n in names}
    standing = torch.tensor([desc.joints[n].standing_offset_rad for n in pj], dtype=torch.float64)
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
        armature=[desc.joints[n].armature for n in names],
        friction=[desc.joints[n].friction for n in names],
        passive_damping=[desc.joints[n].damping for n in names],
        effort_limit=[effort[n] for n in names],
        standing=[desc.joints[n].standing_offset_rad for n in names],
    )


def native_joint_properties(plant: PlantCore, names: list[str]) -> JointProperties:
    """Native crank rotor inertia is on its shaft, not reflected onto both ankle hinges."""
    desc = plant.desc
    physical = plant.solver_map.joint_names
    indices = [physical.index(n) for n in names]
    cranks = {}
    hinges = {j for lk in desc.linkages for j in lk.joints}
    for spec in desc.ankle_model.payload["sides"].values():
        armatures = {float(desc.table["joints"][n]["armature"]) for n in spec["joint_names"]}
        if len(armatures) != 1:
            raise ValueError("native crank armature is ambiguous in the description")
        for role, drive in zip(("upper", "lower"), spec["motor_names"], strict=True):
            cranks[spec["usd_joints"][role]] = (desc.drives[drive], next(iter(armatures)))
    return JointProperties(
        names=list(names), stiffness=plant.k0[0, indices].tolist(), damping=plant.d0[0, indices].tolist(),
        armature=[cranks[n][1] if n in cranks else 0.0 if n in hinges else desc.joints[n].armature for n in names],
        friction=[0.0 if n in cranks else desc.joints[n].friction for n in names],
        passive_damping=[0.0 if n in cranks else desc.joints[n].damping for n in names],
        effort_limit=[cranks[n][0].effort_peak if n in cranks else desc.joints[n].effort_peak for n in names],
        standing=[0.0 if n in cranks else desc.joints[n].standing_offset_rad for n in names],
    )
