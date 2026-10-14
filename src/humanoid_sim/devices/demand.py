"""Drive demand: an exact, batched mirror of Controls' ``FakeDriveDemand``.

Source: ETHRC-Humanoid-Controls ``ros2_ws/src/sim/ethrc_fake_ethercat/include/ethrc_fake_ethercat/fake_bank.h``
(``FakeDriveDemand``) and ``plant_host.py`` ``references_from_demands``. Field meaning is what the
``Epos4Slave`` reports to the plant: positions in the physical drive-output frame (rad), velocity rad/s,
feed-forward Nm, raw JVPT gains (0x34C6:01..03), torque cap 0x34C6:04 in mNm (0 = unset).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

MODE_NONE = 0
MODE_HOMING = 6
MODE_JVPT = -64
MODE_FREEZE = -62

# Raw JVPT gain counts per SI gain (Nm/rad, Nm s/rad). Controls ethrc_ssot/drive_gains.py:29
# JVPT_PER_SI_GAIN; humanoid-urdf records it only in a comment. Modeled: to move into humanoid-urdf.
JVPT_PER_SI_GAIN = 1000.0


@dataclass
class DriveDemand:
    """One demand per env and drive, every field ``[E, D]``."""

    torque_enabled: torch.Tensor  # bool
    mode: torch.Tensor  # int8 (0x6061)
    target_position_rad: torch.Tensor
    target_velocity_rad_s: torch.Tensor
    feedforward_torque_nm: torch.Tensor
    p_gain_raw: torch.Tensor  # int64 holding uint32
    i_gain_raw: torch.Tensor
    d_gain_raw: torch.Tensor
    max_torque_mnm: torch.Tensor
    homing_hold: torch.Tensor  # bool; never read by Controls' plant_host, carried for the ABI

    @classmethod
    def disabled(cls, num_envs: int, num_drives: int, *, device="cpu", dtype=torch.float32) -> DriveDemand:
        def z(dt):
            return torch.zeros(num_envs, num_drives, device=device, dtype=dt)

        return cls(z(torch.bool), z(torch.int8), z(dtype), z(dtype), z(dtype), z(torch.int64), z(torch.int64),
                   z(torch.int64), z(torch.int64), z(torch.bool))

    @classmethod
    def jvpt(cls, position: torch.Tensor, stiffness: torch.Tensor, damping: torch.Tensor, *,
             velocity: torch.Tensor | None = None, feedforward: torch.Tensor | None = None,
             max_torque_nm: torch.Tensor | None = None) -> DriveDemand:
        """Enabled JVPT demand from SI quantities (gains rounded to raw counts as Controls does)."""
        zeros = torch.zeros_like(position)

        def raw(si):
            return torch.round(torch.as_tensor(si, device=position.device) * JVPT_PER_SI_GAIN).to(torch.int64) \
                .expand(position.shape).clone()

        return cls(
            torque_enabled=torch.ones_like(position, dtype=torch.bool),
            mode=torch.full_like(position, MODE_JVPT, dtype=torch.int8),
            target_position_rad=position.clone(),
            target_velocity_rad_s=zeros.clone() if velocity is None else velocity.clone(),
            feedforward_torque_nm=zeros.clone() if feedforward is None else feedforward.clone(),
            p_gain_raw=raw(stiffness), i_gain_raw=torch.zeros_like(position, dtype=torch.int64),
            d_gain_raw=raw(damping),
            max_torque_mnm=(torch.zeros_like(position, dtype=torch.int64) if max_torque_nm is None
                            else torch.round(max_torque_nm * 1000).to(torch.int64)),
            homing_hold=torch.zeros_like(position, dtype=torch.bool),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, torch.Tensor]) -> DriveDemand:
        return cls(**{f.name: d[f.name] for f in fields(cls)})
