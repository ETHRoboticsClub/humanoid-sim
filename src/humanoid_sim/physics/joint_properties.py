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


def joint_dynamics(desc, *, native: bool | None = None) -> dict:
    """Nominal passive dynamics in solver coordinates, before task randomization.

    Armature is kg m²; static/sliding friction is Nm; viscous damping is Nm s/rad.
    Native ankles put rotor inertia on the crank shafts, never on both hinges too.
    """
    rows = {n: {"armature": j.armature, "friction": j.friction,
                "dynamic_friction": j.dynamic_friction, "viscous_damping": j.damping}
            for n, j in desc.joints.items()}
    if native if native is not None else desc.usd_path is not None:
        for spec in desc.ankle_model.payload["sides"].values():
            armatures = {float(desc.table["joints"][n]["armature"]) for n in spec["joint_names"]}
            if len(armatures) != 1:
                raise ValueError("native crank armature is ambiguous in the description")
            for n in spec["joint_names"]:
                rows[n]["armature"] = 0.0
            for role in ("upper", "lower"):
                rows[spec["usd_joints"][role]] = {
                    "armature": next(iter(armatures)), "friction": 0.0,
                    "dynamic_friction": 0.0, "viscous_damping": 0.0}
    return {"joint_names": list(rows), **{field: [r[field] for r in rows.values()]
            for field in ("armature", "friction", "dynamic_friction", "viscous_damping")}}


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
    passive = joint_dynamics(desc, native=False)
    pidx = [passive["joint_names"].index(n) for n in names]
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
        armature=[passive["armature"][i] for i in pidx],
        friction=[passive["friction"][i] for i in pidx],
        passive_damping=[passive["viscous_damping"][i] for i in pidx],
        effort_limit=[effort[n] for n in names],
        standing=[desc.joints[n].standing_offset_rad for n in names],
    )


def native_joint_properties(plant: PlantCore, names: list[str]) -> JointProperties:
    """Native crank rotor inertia is on its shaft, not reflected onto both ankle hinges."""
    desc = plant.desc
    physical = plant.solver_map.joint_names
    indices = [physical.index(n) for n in names]
    cranks = {}
    passive = joint_dynamics(desc)
    pidx = [passive["joint_names"].index(n) for n in names]
    for spec in desc.ankle_model.payload["sides"].values():
        for role, drive in zip(("upper", "lower"), spec["motor_names"], strict=True):
            cranks[spec["usd_joints"][role]] = desc.drives[drive]
    return JointProperties(
        names=list(names), stiffness=plant.k0[0, indices].tolist(), damping=plant.d0[0, indices].tolist(),
        armature=[passive["armature"][i] for i in pidx],
        friction=[passive["friction"][i] for i in pidx],
        passive_damping=[passive["viscous_damping"][i] for i in pidx],
        effort_limit=[cranks[n].effort_peak if n in cranks else desc.joints[n].effort_peak for n in names],
        standing=[0.0 if n in cranks else desc.joints[n].standing_offset_rad for n in names],
    )
