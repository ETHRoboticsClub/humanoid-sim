"""Apply physical replay parameters, without owning a fitting space or search.

All supplied arrays are finite SI values [envs, named joints/drives]. Writes happen
between replays, before reset; omitted parameters retain their current values.
"""

from __future__ import annotations

import numpy as np
import torch


def apply_parameters(
    sim,
    joint_names,
    *,
    armature=None,
    damping=None,
    friction=None,
    friction_ratio=None,
    drive_names=None,
    position_offset=None,
) -> None:
    """Write named joint properties and drive reading offsets to every replay env.

    friction_ratio is dynamic/static Coulomb friction in [0, 1]; supplying it
    requires friction. position_offset is reading minus truth (rad), not a fit's
    bias convention. No bounds, fitted groups, optimization or description edits.
    Validate every supplied value before any backend write.
    """
    plant, robot = sim.plant, sim.backend.robot
    names = list(joint_names)
    if len(set(names)) != len(names) or set(names) - set(plant.map.joint_names):
        raise ValueError("joint_names must be unique plant joints")
    ids = torch.tensor([robot.joint_names.index(n) for n in names], device=robot.device)
    values = {}
    for key, raw in (
        ("armature", armature),
        ("damping", damping),
        ("friction", friction),
        ("friction_ratio", friction_ratio),
    ):
        if raw is None:
            continue
        a = np.asarray(raw)
        if a.shape != (plant.num_envs, len(names)) or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"{key}: finite nonnegative [envs, joints] expected")
        if key == "friction_ratio" and (a > 1).any():
            raise ValueError("friction_ratio must be in [0, 1]")
        values[key] = torch.as_tensor(a, dtype=torch.float32, device=robot.device)
    if friction_ratio is not None and friction is None:
        raise ValueError("friction_ratio requires friction")
    drives = list(plant.map.drive_names if drive_names is None else drive_names)
    offset = None
    if position_offset is not None:
        a = np.asarray(position_offset)
        if len(set(drives)) != len(drives) or set(drives) - set(plant.map.drive_names):
            raise ValueError("drive_names must be unique plant drives")
        if a.shape != (plant.num_envs, len(drives)) or not np.isfinite(a).all():
            raise ValueError("position_offset: finite [envs, drives] expected")
        offset = torch.as_tensor(a, dtype=plant.cfg.dtype, device=plant.cfg.device)
    for key, method, argument in (
        ("armature", "write_joint_armature_to_sim_index", "armature"),
        ("damping", "write_joint_viscous_friction_coefficient_to_sim_index", "joint_viscous_friction_coeff"),
        ("friction", "write_joint_friction_coefficient_to_sim_index", "joint_friction_coeff"),
    ):
        if key in values:
            getattr(robot, method)(**{argument: values[key], "joint_ids": ids})
    if friction_ratio is not None:
        # Compute in the caller's precision, as the replay parameter model does,
        # then cast once to the plant. This preserves the existing fit's writes.
        relief = np.asarray(friction) * (1.0 - np.asarray(friction_ratio))
        cols = [plant.map.joint_names.index(n) for n in names]
        plant.friction_relief[:, cols] = torch.as_tensor(
            relief, dtype=plant.cfg.dtype, device=plant.cfg.device
        )
    if offset is not None:
        cols = [plant.map.drive_names.index(n) for n in drives]
        plant.bank.position_offset[:, cols] = offset
