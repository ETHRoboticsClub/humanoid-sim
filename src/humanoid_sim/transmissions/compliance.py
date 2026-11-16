"""Ankle linkage compliance: a torsional spring and damper between each crank drive's output and its crank arm.

An asset built with ``ankle_compliance`` (:func:`humanoid_sim.model.usd.build_asset`) splits each native crank shaft
in two: the drive's joint turns a light output body (the drive's output flange), and a coaxial revolute joint, the
spring, joins that output to the crank arm. The spring's joint PD, target zero, is the compliance: stiffness
``k`` (N m/rad) and damping ``c`` (N m s/rad) per crank, written to the solver like any joint gain. The drive's
encoder reads the output, so the drive feedback (and :meth:`~humanoid_sim.train.plant.RobotPlant.measured`) stays
the crank reading a rigid linkage would give, while the foot (the ankle hinges, ``truth()``) deflects under load.

Statics, used by the self-checks and by fits. With ``J = dθ/dq`` the crank-per-hinge Jacobian
(:meth:`~humanoid_sim.transmissions.ankle.AnkleTransmission.jacobian`, drive frame) and ``K = diag(k)``:

* a static torque ``τ`` on the hinges (pitch, roll) moves the foot by ``δq = C τ`` from the pose the cranks' readings
  imply, with the foot compliance ``C = J⁻¹ K⁻¹ J⁻ᵀ``;
* the cranks hold that load with drive torques ``τ_d = -J⁻ᵀ τ``, so the deflection also follows from the drives'
  own torques: ``δq = -J⁻¹ K⁻¹ τ_d``. A crank's spring angle is ``-τ_d / k`` (drive frame).

The equivalent foot stiffness is ``Jᵀ K J``. On a symmetric two-crank ankle (the V1's, ``J ≈ [[-a, -b], [a, -b]]``
with ``a ≈ b ≈ 1``) its pitch and roll diagonal both follow the sum of the two crank stiffnesses, in a ratio the
geometry fixes; their difference sets only the pitch-roll coupling. A spring along each rod instead (axial stiffness ``k_a``) gives the same
statics with ``k = k_a r²``, ``r`` the crank's moment arm about the shaft across the rod; the two differ only in how
``r`` changes with the crank angle.

Nothing here is a robot value: the stiffness and damping are knobs (the profile's ``physics.ankle_compliance``, the
plant's ``ankle_spring_stiffness`` and ``ankle_spring_damping``), fitted in humanoid-learning.

Joint give is the same mechanism for any 1:1 drive joint (the V1's hips and knee first). An asset built with
``joint_give`` (:func:`humanoid_sim.model.usd.build_asset`) splits each named joint in two: the joint itself, still
named as the description names it and still the drive's, turns a light output body, and a coaxial revolute joint
``<joint>_give`` joins that output to the link. Its joint PD, target zero, is the give: stiffness ``k`` (N m/rad) and
damping ``c`` (N m s/rad) per joint, from the profile's ``physics.joint_give`` or the description (a joint's
``give_stiffness_nm_per_rad`` and ``give_damping_nm_s_per_rad``, else its family's), randomizable per env
(``joint_give_stiffness``, ``joint_give_damping``). The drive's encoder reads the output, so the measured joint position
(drive feedback, :meth:`~humanoid_sim.train.plant.RobotPlant.measured`, ``Truth.joint_pos``) stays the drive side's,
while the link sits ``θ`` further, ``Truth.joint_give_rad``, in the joint's own coordinates. Statics: a torque ``τ`` on
the link about the joint's axis, held by the drive, deflects the link by ``τ / k`` (:func:`give_deflection`).
"""

from __future__ import annotations

import re

import torch

SPRING_SUFFIX = "_spring"
GIVE_SUFFIX = "_give"
OUTPUT_SUFFIX = "_output"


def spring_joints(desc) -> dict[str, str]:
    """Each crank drive's spring joint name in an asset built with ankle compliance, in the description's drive
    order: ``{drive: <crank joint>_spring}``."""
    out = {}
    for linkage in desc.linkages:
        joints = desc.ankle_model.payload["sides"][linkage.side]["usd_joints"]
        for role, drive in (("upper", linkage.upper), ("lower", linkage.lower)):
            out[drive] = joints[role] + SPRING_SUFFIX
    return {d: out[d] for d in desc.drives if d in out}


def output_body(crank_joint: str) -> str:
    """The name of the drive output body the asset adds behind ``crank_joint``."""
    return crank_joint + OUTPUT_SUFFIX


def give_joint(joint: str) -> str:
    """The name of the give joint the asset adds behind drive joint ``joint``."""
    return joint + GIVE_SUFFIX


def give_candidates(desc) -> list[str]:
    """The joints that can carry give, in the description's order: those a drive moves one to one."""
    return [n for n in desc.joints if any(d.crank is None and d.joints == (n,) for d in desc.drives.values())]


def resolve_give(desc, patterns) -> tuple[str, ...]:
    """The joint names ``patterns`` select (each a joint name or a regex over them, matched in full), in the
    description's order. Raises for a pattern that selects nothing or a joint that cannot carry give (a joint no
    drive moves one to one: the ankle hinges, whose compliance is the cranks')."""
    if isinstance(patterns, str):
        patterns = [patterns]
    candidates = give_candidates(desc)
    chosen = set()
    for pattern in patterns:
        found = [n for n in candidates if re.fullmatch(pattern, n)]
        if not found:
            raise ValueError(f"joint give: {pattern!r} selects no joint of {candidates}")
        chosen.update(found)
    return tuple(n for n in candidates if n in chosen)


def give_deflection(load: torch.Tensor, stiffness: torch.Tensor) -> torch.Tensor:
    """Static link deflection ``τ / k`` (rad) under torque ``τ`` (N m) on the link about the joint's axis, held by
    the drive, for a give of stiffness ``k`` (N m/rad)."""
    return load / stiffness


def foot_compliance(jacobian: torch.Tensor, stiffness: torch.Tensor) -> torch.Tensor:
    """``C = J⁻¹ K⁻¹ J⁻ᵀ`` ``[..., 2, 2]`` (rad per N m on the hinges) from ``J`` ``[..., 2, 2]`` and the cranks'
    stiffness ``[..., 2]`` (N m/rad)."""
    inv = torch.linalg.inv(jacobian)
    return inv @ torch.diag_embed(1.0 / stiffness) @ inv.transpose(-1, -2)


def foot_stiffness(jacobian: torch.Tensor, stiffness: torch.Tensor) -> torch.Tensor:
    """``Jᵀ K J`` ``[..., 2, 2]`` (N m/rad on the hinges)."""
    return jacobian.transpose(-1, -2) @ torch.diag_embed(stiffness) @ jacobian


def deflection_under_load(jacobian: torch.Tensor, stiffness: torch.Tensor, hinge_torque: torch.Tensor) -> torch.Tensor:
    """Static foot deflection ``[..., 2]`` (pitch, roll) under hinge torque ``τ`` ``[..., 2]``: ``C τ``."""
    return (foot_compliance(jacobian, stiffness) @ hinge_torque.unsqueeze(-1)).squeeze(-1)


def deflection_from_drive_torque(jacobian: torch.Tensor, stiffness: torch.Tensor,
                                 drive_torque: torch.Tensor) -> torch.Tensor:
    """Static foot deflection ``[..., 2]`` from the cranks' drive torques ``[..., 2]`` (drive frame):
    ``-J⁻¹ K⁻¹ τ_d``."""
    spring = -drive_torque / stiffness
    return torch.linalg.solve(jacobian, spring.unsqueeze(-1)).squeeze(-1)


def equal_crank_stiffness(jacobian: torch.Tensor, pitch_nm_per_rad: torch.Tensor | float) -> torch.Tensor:
    """The equal stiffness ``[..., 2]`` of both cranks that gives the foot the pitch stiffness ``pitch_nm_per_rad``
    (the ``[0, 0]`` entry of ``Jᵀ K J``) at the pose of ``J``."""
    k = torch.as_tensor(pitch_nm_per_rad, dtype=jacobian.dtype, device=jacobian.device) / jacobian[..., :, 0].square().sum(-1)
    return k.unsqueeze(-1).expand(*k.shape, 2)
