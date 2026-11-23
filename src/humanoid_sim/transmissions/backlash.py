"""Ankle crank backlash: free play in each crank drive's gearbox, between the motor side and the output shaft.

An asset built with ``ankle_backlash`` (:func:`humanoid_sim.model.usd.build_asset`) splits each native crank shaft
at the gearbox: the drive's joint turns a light motor-side body (``<crank joint>_motor``, the gearbox input reflected
to the output), and a coaxial revolute joint, ``<crank joint>_backlash``, joins it to the output shaft (the crank
arm, or the drive's output flange when the asset also has ankle compliance). The backlash joint carries no gain; its
two stops at ``±w/2`` are the gear teeth, so the shaft turns freely within a band of width ``w`` and is driven
through the stop it rests on. The stops are joint limits, so both solvers treat them as constraints (Kamino's are
hard; MJWarp's take the profile's ``physics.mjwarp.ankle_backlash_solref``).

Encoders. The drive reports its output encoder, on the crank shaft past the gap, so the reported position is the
motor side plus the play (:meth:`~humanoid_sim.core.PlantCore.feedback`); the reported velocity stays the motor
side's. Which encoder its position loop reads is the profile's ``physics.ankle_backlash.loop_encoder``
(:func:`loop_encoder`): ``motor`` (the drive's joint) or ``output`` (the shaft past the gap, a dual-loop drive whose
velocity loop stays on the motor encoder). The drives' configuration has not confirmed either (modeled). With ankle compliance as well, the chain is motor, gap, output encoder,
spring, crank arm.

Output flange. With ankle compliance as well, the play frees the springs' output flange from the drive's joint and its
rotor armature: within the play the flange hangs on the spring alone, and a spring of stiffness ``k`` and damping
``b`` on an inertia ``I`` must satisfy :func:`check_output_stability` at the solver's step. The flange's inertia is a
build knob (the profile's ``physics.ankle_backlash.output_inertia_kg_m2``): a numerical floor, chosen for the stiffest
spring a caller wants, until the gearbox output's own inertia is measured.

Nothing here is a robot value: the width is a knob (the profile's ``physics.ankle_backlash``, the plant's
``ankle_backlash``), fitted in humanoid-learning.
"""

from __future__ import annotations

import torch

LOOP_ENCODERS = ("motor", "output")
BACKLASH_SUFFIX = "_backlash"
MOTOR_SUFFIX = "_motor"
# Half the band the asset authors the stops at, before the spawn opens them to the play: closed for any purpose, but
# not empty, since MuJoCo refuses a joint whose range has its lower end at its upper one when it compiles the model.
AUTHORED_HALF_WIDTH_RAD = 1e-6


def backlash_joints(desc) -> dict[str, str]:
    """Each crank drive's backlash joint name in an asset built with ankle backlash, in the description's drive
    order: ``{drive: <crank joint>_backlash}``."""
    out = {}
    for linkage in desc.linkages:
        joints = desc.ankle_model.payload["sides"][linkage.side]["usd_joints"]
        for role, drive in (("upper", linkage.upper), ("lower", linkage.lower)):
            out[drive] = joints[role] + BACKLASH_SUFFIX
    return {d: out[d] for d in desc.drives if d in out}


def loop_encoder(physics: dict) -> str:
    """The encoder the crank drives' position loop reads: the profile's ``physics.ankle_backlash.loop_encoder``."""
    v = (physics.get("ankle_backlash") or {}).get("loop_encoder", "motor")
    if v not in LOOP_ENCODERS:
        raise ValueError(f"physics.ankle_backlash.loop_encoder {v!r}: one of {', '.join(LOOP_ENCODERS)}")
    return v


def motor_body(crank_joint: str) -> str:
    """The name of the motor-side body the asset adds behind ``crank_joint``."""
    return crank_joint + MOTOR_SUFFIX


def limits(width_rad: torch.Tensor) -> torch.Tensor:
    """The backlash joints' stops ``[..., 2]`` (lower, upper; rad) for widths ``[...]``: ``±w/2``."""
    half = 0.5 * width_rad
    return torch.stack([-half, half], -1)


def dead_band(gap_rad: torch.Tensor, width_rad: torch.Tensor | float) -> torch.Tensor:
    """The part of ``gap_rad`` (output shaft past the motor side) beyond the band ``±w/2``: zero inside it. Statics of
    the stops for self-checks and fits: a stop holds load only once the shaft rests on it, by this much past it."""
    half = 0.5 * torch.as_tensor(width_rad, dtype=gap_rad.dtype, device=gap_rad.device)
    return gap_rad - gap_rad.clamp(-half, half)


def static_play(drive_torque: torch.Tensor, width_rad: torch.Tensor | float) -> torch.Tensor:
    """Where a rigid gearbox's shaft rests in the gap (shaft past motor, drive frame) under a static drive torque
    ``τ``: the motor side leads the shaft in the direction it pushes, so ``-sign(τ) · w/2``; the middle without load.
    The output encoder then reads ``motor + this``, and ``motor − output`` steps by ``w`` where ``τ`` changes sign."""
    half = 0.5 * torch.as_tensor(width_rad, dtype=drive_torque.dtype, device=drive_torque.device)
    return -torch.sign(drive_torque) * half


# A spring-damper (stiffness k, damping b) on an inertia I at step dt is stable in a symplectic Euler step when
# k dt²/I + 2 b dt/I < 4 (Jury's test; physics.crank_stops uses the same measure). MJWarp's RK4 is stable beyond it
# (an asset at a measure of 6 stays finite), Kamino's euler step jitters well before it diverges, so the bound is
# the limit for both, not a margin.
STABLE_MEASURE = 4.0


def stability_measure(stiffness_nm_per_rad, damping_nm_s_per_rad, inertia_kg_m2: float, period_s: float):
    """``k dt²/I + 2 b dt/I`` of a spring on the output flange: stable below :data:`STABLE_MEASURE`."""
    return (torch.as_tensor(stiffness_nm_per_rad) * period_s**2 + 2.0 * torch.as_tensor(damping_nm_s_per_rad)
            * period_s) / inertia_kg_m2


def check_output_stability(stiffness_nm_per_rad, damping_nm_s_per_rad, inertia_kg_m2: float, period_s: float) -> None:
    """Raise unless every crank spring (``[...]``, N m/rad and N m s/rad) keeps the freed output flange of inertia
    ``inertia_kg_m2`` stable at the step ``period_s``. Called where the gains are written (startup and level
    changes), so it may sync the host."""
    m = stability_measure(stiffness_nm_per_rad, damping_nm_s_per_rad, inertia_kg_m2, period_s)
    worst = float(m.max())
    if not worst < STABLE_MEASURE:
        raise ValueError(
            f"ankle springs on a backlash asset's output flange ({inertia_kg_m2:g} kg m²) are unstable at a "
            f"{period_s * 1e3:g} ms step: k dt²/I + 2 b dt/I reaches {worst:.3g}, needs < {STABLE_MEASURE:g}. Use "
            "softer springs or a heavier flange (physics.ankle_backlash.output_inertia_kg_m2, at build)")
