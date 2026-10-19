"""Target shaping: realize a modeled force with constant solver PD gains.

The solver integrates its PD implicitly (MJWarp ``implicitfast``: implicit in the damping), which is what
keeps stiff drives stable at 4 ms. Changing solver gains at run time costs a model update, so the gains
stay at constant per-episode values ``k0, d0`` and every other force law (the drive's own gains, the
torque-speed envelope and torque cap, arrival blending) is realized by moving the position target:

    k0 (target − q) + d0 (0 − q̇) = force   ⇒   target = q + (force + d0 q̇) / k0

With the drive's own gains equal to ``k0, d0`` this is exactly the implicit PD on the demanded target.
Otherwise the solver applies ``force − k0 Δq − d0 Δq̇`` over the step, stable while ``K ≤ 2 k0`` and
``D ≤ 2 d0`` (walk v2, rc_humanoid_rl_lab ``walk_v2/core.py:177-210``).
"""

from __future__ import annotations

import torch


def shaped_position_target(q: torch.Tensor, qd: torch.Tensor, force: torch.Tensor, k0: torch.Tensor,
                           d0: torch.Tensor) -> torch.Tensor:
    return q + (force + d0 * qd) / k0


def solver_force(target: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, k0: torch.Tensor,
                 d0: torch.Tensor) -> torch.Tensor:
    """Force the solver PD produces at state ``q, q̇`` (velocity target 0)."""
    return k0 * (target - q) - d0 * qd


def outside_stable_region(stiffness: torch.Tensor, damping: torch.Tensor, k0: torch.Tensor,
                          d0: torch.Tensor) -> torch.Tensor:
    """True where demanded gains exceed ``2 k0`` or ``2 d0`` (shaping is no longer guaranteed stable)."""
    return (stiffness > 2 * k0) | (damping > 2 * d0)
