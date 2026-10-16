"""Drives ↔ URDF joints, batched.

* 1:1 drives: ``q_drive = sign · q_joint`` (URDF zero is the calibration pose, the drive's frame);
  ``τ_joint = sign · τ_drive``.
* Ankle cranks: crank angles from the hinge state through :class:`AnkleTransmission`;
  ``τ_hinges = Jᵀ τ_cranks``.

This is the plant side of Controls' ``Plant.physical_motor_feedback`` / ``Plant.step`` transmission.
"""

from __future__ import annotations

import torch

from humanoid_sim.model.description import Description


class DriveJointMap:
    def __init__(self, desc: Description, joint_names: list[str], drive_names: list[str] | None = None, *,
                 device="cpu", dtype=torch.float32):
        self.drive_names = list(drive_names or desc.drives)
        self.joint_names = list(joint_names)
        jidx = {n: i for i, n in enumerate(self.joint_names)}
        drives = desc.drives
        unknown = [d for d in self.drive_names if d not in drives]
        if unknown:
            raise ValueError(f"drives not in the description: {unknown}")
        lin_d, lin_j, lin_s = [], [], []
        for i, name in enumerate(self.drive_names):
            d = drives[name]
            if d.crank is None:
                lin_d.append(i)
                lin_j.append(jidx[d.joints[0]])
                lin_s.append(float(d.sign))
        self.device, self.dtype = torch.device(device), dtype
        self._lin_d = torch.tensor(lin_d, device=device, dtype=torch.long)
        self._lin_j = torch.tensor(lin_j, device=device, dtype=torch.long)
        self._lin_s = torch.tensor(lin_s, device=device, dtype=dtype)
        self.ankles = []
        for side in ("l", "r"):
            upper, lower = desc.ankle_cranks(side)
            if upper not in self.drive_names:
                continue
            self.ankles.append((
                side,
                torch.tensor([self.drive_names.index(upper), self.drive_names.index(lower)], device=device),
                torch.tensor([jidx[f"foot_pitch_{side}"], jidx[f"foot_roll_{side}"]], device=device),
                desc.ankle_transmission(side, device=device, dtype=dtype),
            ))
        self.actuated_joints = torch.tensor(
            sorted(lin_j + [int(j) for a in self.ankles for j in a[2]]), device=device, dtype=torch.long)

    @property
    def num_drives(self) -> int:
        return len(self.drive_names)

    def drive_state(self, q: torch.Tensor, qd: torch.Tensor):
        """Physical drive-frame position and velocity ``[E, D]`` plus the ankle Jacobians from joint state."""
        e = q.shape[0]
        qdrv = torch.zeros(e, self.num_drives, device=q.device, dtype=q.dtype)
        vdrv = torch.zeros_like(qdrv)
        qdrv[:, self._lin_d] = q[:, self._lin_j] * self._lin_s
        vdrv[:, self._lin_d] = qd[:, self._lin_j] * self._lin_s
        jacs = []
        for _, d_idx, j_idx, tr in self.ankles:
            theta, omega, jac = tr.crank_state(q[:, j_idx], qd[:, j_idx])
            qdrv[:, d_idx] = theta
            vdrv[:, d_idx] = omega
            jacs.append(jac)
        return qdrv, vdrv, jacs

    def joint_torque(self, tau_drive: torch.Tensor, jacs: list[torch.Tensor]) -> torch.Tensor:
        """Joint torques ``[E, J]`` (zero on joints without a drive)."""
        out = torch.zeros(tau_drive.shape[0], len(self.joint_names), device=tau_drive.device,
                          dtype=tau_drive.dtype)
        out[:, self._lin_j] = tau_drive[:, self._lin_d] * self._lin_s
        for (_, d_idx, j_idx, tr), jac in zip(self.ankles, jacs, strict=True):
            out[:, j_idx] = tr.joint_torque(jac, tau_drive[:, d_idx])
        return out

    def joint_velocity_from_drive(self, v_drive: torch.Tensor, jacs: list[torch.Tensor]) -> torch.Tensor:
        """Joint velocities that produce drive velocities ``v_drive`` (ankles: ``J⁻¹ ω``)."""
        out = torch.zeros(v_drive.shape[0], len(self.joint_names), device=v_drive.device, dtype=v_drive.dtype)
        out[:, self._lin_j] = v_drive[:, self._lin_d] * self._lin_s
        for (_, d_idx, j_idx, _), jac in zip(self.ankles, jacs, strict=True):
            out[:, j_idx] = torch.linalg.solve(jac, v_drive[:, d_idx].unsqueeze(-1)).squeeze(-1)
        return out

    def joint_space_gains(self, stiffness: torch.Tensor, damping: torch.Tensor, q_ref: torch.Tensor):
        """Diagonal joint-space PD equivalent of per-drive gains ``[D]`` at joint pose ``q_ref`` ``[J]``.

        1:1: unchanged. Ankle hinges: ``diag(Jᵀ K J)`` with ``J`` at ``q_ref`` (walk v2's ``diag(MᵀM)``).
        Returns ``(K0 [J], D0 [J])``; joints without a drive get 0.
        """
        k0 = torch.zeros(len(self.joint_names), device=self.device, dtype=self.dtype)
        d0 = torch.zeros_like(k0)
        k0[self._lin_j] = stiffness[self._lin_d]
        d0[self._lin_j] = damping[self._lin_d]
        for _, d_idx, j_idx, tr in self.ankles:
            jac = tr.jacobian(q_ref[j_idx].unsqueeze(0))[0]
            k0[j_idx] = torch.diagonal(jac.T @ torch.diag(stiffness[d_idx]) @ jac)
            d0[j_idx] = torch.diagonal(jac.T @ torch.diag(damping[d_idx]) @ jac)
        return k0, d0
