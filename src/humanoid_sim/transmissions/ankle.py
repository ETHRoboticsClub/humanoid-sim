"""Batched torch transmission for the two-crank parallel ankle.

The ankle is simulated as two hinges (pitch about +Y, then roll about +X: the URDF tree). Two crank
drives move the foot through rods. This module maps between the hinge state and the crank state:

* ``cranks(q)``: hinge angles → crank angles (analytic circle-sphere IK on the branch connected to the
  linkage neutral, as in humanoid-urdf ``ankle_model``);
* ``jacobian(q)``: ``dθ/dq`` from the rod constraints;
* crank torque → hinge torque is ``Jᵀ τ``; hinge velocity → crank velocity is ``J q̇``.

Geometry, axes, frame signs and the linkage neutral come from the description (``actuators.yaml``
``ankle_model``); nothing is typed in here. ``q`` is in URDF (calibration) joint coordinates. The linkage
neutral, where both cranks read zero (the drives' calibration zero), sits at ``semantic_neutral_rad``; the
linkage is solved on the offset from it, which ``semantic_to_kinematic_sign`` maps to the linkage frame.
Crank angles are relative to the linkage neutral. With ``native_angles`` the hinges and crank shafts are the
delivered USD joints, and a crank's angle is its device angle, ``crank_sign * (shaft angle - crank_zero)``
(humanoid-urdf ``actuators.yaml`` ``drives.<crank>.shaft_sign`` and ``shaft_zero_rad``).

It is golden-tested against humanoid-urdf ``ankle_model``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CrankGeometry:
    p_anchor0: np.ndarray  # foot-side rod ball, neutral, pivot frame (m)
    center: np.ndarray  # crank circle centre
    u_hat: np.ndarray  # neutral direction centre → crank ball (θ_abs = 0)
    v_hat: np.ndarray  # n × u
    radius: float
    length: float
    theta0: float  # absolute angle of the neutral solution on the chosen branch
    branch: int  # ±1, sign of the acos term on the neutral-connected branch


def crank_geometry(spec: dict) -> CrankGeometry:
    """Derive the crank circle from the description's measured points (same algebra as ankle_model)."""
    p_shaft, n_shaft, p_crank0, p_anchor0 = (np.asarray(spec[k], dtype=np.float64)
                                             for k in ("P_shaft", "n_shaft", "P_crank_0", "P_anchor_0"))
    length = float(spec["L"])
    n = n_shaft / np.linalg.norm(n_shaft)
    center = p_shaft + np.dot(p_crank0 - p_shaft, n) * n
    arm = p_crank0 - center
    radius = float(np.linalg.norm(arm))
    if radius < 1e-9:
        raise ValueError("ankle crank radius is zero")
    u = arm / radius
    v = np.cross(n, u)
    d0 = center - p_anchor0
    a0, b0 = 2 * radius * np.dot(d0, u), 2 * radius * np.dot(d0, v)
    ratio0 = (length**2 - np.dot(d0, d0) - radius**2) / np.hypot(a0, b0)
    if not abs(ratio0) < 1:
        raise ValueError("neutral ankle linkage is singular or unreachable")
    candidates = np.arctan2(b0, a0) + np.array([-1.0, 1.0]) * np.arccos(ratio0)
    candidates = (candidates + np.pi) % (2 * np.pi) - np.pi
    index = int(np.argmin(np.abs(candidates)))
    return CrankGeometry(p_anchor0, center, u, v, radius, length, float(candidates[index]), (-1, 1)[index])


def mirror_right(right: dict) -> dict:
    """Left geometry: mirror the right side across the sagittal plane (axes are not mirrored)."""
    left = {k: v for k, v in right.items()}
    for crank in ("upper", "lower"):
        left[crank] = {k: (list(v) if isinstance(v, list) else v) for k, v in right[crank].items()}
        for key in ("P_shaft", "n_shaft", "P_crank_0", "P_anchor_0"):
            left[crank][key] = [left[crank][key][0], -left[crank][key][1], left[crank][key][2]]
    return left


class AnkleTransmission:
    """One ankle side, batched over leading dims. ``q``: ``[..., 2]`` (pitch, roll), URDF coordinates."""

    def __init__(self, geometry: dict, kinematic_sign, neutral=(0.0, 0.0), *, native_angles=False,
                 device="cpu", dtype=torch.float32, crank_sign=(1, 1), crank_zero=(0.0, 0.0)):
        for name, axis in (("pitch_axis", [0, 1, 0]), ("roll_axis", [1, 0, 0])):
            if not np.allclose(geometry[name], axis):
                raise ValueError(f"ankle transmission requires {name}={axis}")
        self.cranks_geom = (crank_geometry(geometry["upper"]), crank_geometry(geometry["lower"]))
        sign = np.asarray(kinematic_sign, dtype=np.float64)
        if sign.shape != (2,) or not np.isin(sign, (-1, 1)).all():
            raise ValueError("invalid semantic_to_kinematic_sign")
        neutral = np.asarray(neutral, dtype=np.float64)
        if neutral.shape != (2,) or not np.isfinite(neutral).all():
            raise ValueError("invalid semantic_neutral_rad")
        self._neutral_np = neutral
        crank_sign = np.asarray(crank_sign, dtype=np.float64)
        crank_zero = np.asarray(crank_zero, dtype=np.float64)
        if crank_sign.shape != (2,) or not np.isin(crank_sign, (-1, 1)).all():
            raise ValueError("invalid crank_sign")
        if crank_zero.shape != (2,) or not np.isfinite(crank_zero).all():
            raise ValueError("invalid crank_zero")
        self._crank_sign_np, self._crank_zero_np = crank_sign, crank_zero
        self.native_angles = native_angles
        self.to(device, dtype, sign=sign)

    def to(self, device, dtype, *, sign=None):
        sign = self._sign_np if sign is None else sign
        self._sign_np = sign

        def t(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float64), device=device, dtype=dtype)

        self.device, self.dtype = torch.device(device), dtype
        self.sign = t(sign)
        self.neutral = t(self._neutral_np)  # [2] joint angles of the linkage neutral
        g = self.cranks_geom
        self._p0 = t(np.stack([c.p_anchor0 for c in g]))  # [2, 3]
        self._c = t(np.stack([c.center for c in g]))
        self._u = t(np.stack([c.u_hat for c in g]))
        self._v = t(np.stack([c.v_hat for c in g]))
        self._r = t([c.radius for c in g])
        self._len = t([c.length for c in g])
        self._theta0 = t([0.0 if self.native_angles else c.theta0 for c in g])
        self._branch = t([float(c.branch) for c in g])
        self._crank_sign = t(self._crank_sign_np)
        self._crank_zero = t(self._crank_zero_np)
        return self

    def _foot(self, qk):
        """Rod foot balls for kinematic angles ``qk`` [..., 2] → ``P_f``, ``P_r`` [..., 2, 3]."""
        p, r = qk[..., 0:1], qk[..., 1:2]
        cp, sp, cr, sr = torch.cos(p), torch.sin(p), torch.cos(r), torch.sin(r)
        x, y, z = self._p0[:, 0], self._p0[:, 1], self._p0[:, 2]
        yr, zr = cr * y - sr * z, sr * y + cr * z
        p_r = torch.stack([x.expand_as(yr), yr, zr], dim=-1)
        p_f = torch.stack([cp * x + sp * zr, yr, -sp * x + cp * zr], dim=-1)
        return p_f, p_r, cp, sp

    def _theta_abs(self, p_f):
        d = self._c - p_f
        a = 2 * self._r * (d * self._u).sum(-1)
        b = 2 * self._r * (d * self._v).sum(-1)
        ratio = (self._len.square() - (d * d).sum(-1) - self._r.square()) / torch.hypot(a, b)
        theta = torch.atan2(b, a) + self._branch * torch.acos(ratio.clamp(-1.0, 1.0))
        return theta, ratio.abs()

    def cranks(self, q: torch.Tensor, *, return_reach: bool = False):
        """Crank angles relative to neutral, ``[..., 2]`` (upper, lower). ``reach`` = |K/R| (≥1: unreachable)."""
        p_f, _, _, _ = self._foot((q - self.neutral) * self.sign)
        theta, reach = self._theta_abs(p_f)
        rel = self._crank_sign * (theta - self._theta0 - self._crank_zero)
        rel = torch.remainder(rel + torch.pi, 2 * torch.pi) - torch.pi
        return (rel, reach) if return_reach else rel

    def shafts(self, q: torch.Tensor) -> torch.Tensor:
        """Crank shaft angles ``[..., 2]`` (upper, lower): the USD joint angles with ``native_angles``, else the
        crank angles relative to the neutral."""
        p_f, _, _, _ = self._foot((q - self.neutral) * self.sign)
        theta, _ = self._theta_abs(p_f)
        return torch.remainder(theta - self._theta0 + torch.pi, 2 * torch.pi) - torch.pi

    def jacobian(self, q: torch.Tensor) -> torch.Tensor:
        """``dθ/dq`` ``[..., 2 (crank), 2 (pitch, roll)]`` in URDF joint coordinates."""
        p_f, p_r, cp, sp = self._foot((q - self.neutral) * self.sign)
        theta, _ = self._theta_abs(p_f)
        c, s = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
        tip = self._c + self._r.unsqueeze(-1) * (c * self._u + s * self._v)
        e = tip - p_f
        tangent = self._r.unsqueeze(-1) * (-s * self._u + c * self._v)
        f_theta = 2 * (e * tangent).sum(-1)
        # d P_f / d pitch = a_p × P_f with a_p = +Y; d P_f / d roll = R_p (a_r × P_r) with a_r = +X.
        dp = torch.stack([p_f[..., 2], torch.zeros_like(p_f[..., 0]), -p_f[..., 0]], dim=-1)
        cp, sp = cp.unsqueeze(-1).squeeze(-1), sp
        dr = torch.stack([sp * p_r[..., 1], -p_r[..., 2], cp * p_r[..., 1]], dim=-1)
        f_q = torch.stack([-2 * (e * dp).sum(-1), -2 * (e * dr).sum(-1)], dim=-1)
        return -f_q / f_theta.unsqueeze(-1) * self.sign * self._crank_sign.unsqueeze(-1)

    def crank_state(self, q: torch.Tensor, qd: torch.Tensor):
        """Crank angles and velocities from hinge state."""
        jac = self.jacobian(q)
        return self.cranks(q), (jac @ qd.unsqueeze(-1)).squeeze(-1), jac

    def joints_from_cranks(self, theta: torch.Tensor, q0: torch.Tensor, iterations: int = 8) -> torch.Tensor:
        """Hinge angles whose cranks are ``theta`` (Newton on ``cranks(q) = θ`` from ``q0``; fixed iterations).

        Used for initial states and replay, never in the step loop. Starts must be on the same assembly
        branch (near the linkage neutral or the standing pose); the residual is the caller's to check.
        """
        q = q0.expand_as(theta).clone()
        for _ in range(iterations):
            q = q - torch.linalg.solve(self.jacobian(q), (self.cranks(q) - theta).unsqueeze(-1)).squeeze(-1)
        return q

    @staticmethod
    def joint_torque(jac: torch.Tensor, crank_torque: torch.Tensor) -> torch.Tensor:
        """Hinge torques from crank torques: ``Jᵀ τ``."""
        return (jac.transpose(-1, -2) @ crank_torque.unsqueeze(-1)).squeeze(-1)
