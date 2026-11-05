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
"""

from __future__ import annotations

import torch

SPRING_SUFFIX = "_spring"
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
