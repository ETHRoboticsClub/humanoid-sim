"""Drive the delivered crank joints while retaining the semantic joint API."""

from __future__ import annotations

import torch

from humanoid_sim.core import PlantCore


class NativeDrivePlant:
    def __init__(self, plant: PlantCore):
        self.plant = plant
        self.names = list(plant.desc.physics_joints)
        self.core = PlantCore(plant.desc, self.names, plant.num_envs, plant.cfg)
        self.core.bank = plant.bank
        self.effort = torch.zeros_like(self.core.external_torque)
        self.semantic_ids = torch.tensor([self.names.index(name) for name in plant.map.joint_names],
                                         device=plant.cfg.device)
        from humanoid_sim.model.native_state import NativeState

        self.state_model = NativeState(plant.desc, plant.map.joint_names,
                                       device=plant.cfg.device, dtype=plant.cfg.dtype)

    def pre_physics(self, t_ns, positions, velocities):
        self.core.external_torque.zero_()
        self.core.external_torque[:, self.semantic_ids] = self.plant.external_torque
        self.core.friction_relief[:, self.semantic_ids] = self.plant.friction_relief
        targets = self.core.pre_physics(t_ns, positions, velocities)
        self.effort.copy_(self.core.external_torque + self.core._relief(velocities, self.core.last))
        self.effort.index_fill_(1, self.core.actuated, 0)
        self.plant.last = self.core.last
        return targets

    def feedback(self, positions, velocities):
        return self.core.feedback(positions, velocities)

    def complete_state(self, positions, velocities):
        """Initialize native crank coordinates from a semantic pose and velocity."""
        full = torch.zeros(*positions.shape[:-1], len(self.names), dtype=positions.dtype, device=positions.device)
        rates = torch.zeros_like(full)
        full[..., self.semantic_ids] = positions
        rates[..., self.semantic_ids] = velocities
        for linkage in self.plant.desc.linkages:
            spec = self.plant.desc.ankle_model.payload["sides"][linkage.side]
            semantic = [self.plant.map.joint_names.index(name) for name in linkage.joints]
            native = [self.names.index(spec["usd_joints"][name]) for name in ("upper", "lower")]
            transmission = self.plant.desc.ankle_transmission(linkage.side, device=positions.device,
                                                             dtype=positions.dtype)
            angles, speeds, _ = transmission.crank_state(positions[..., semantic], velocities[..., semantic])
            full[..., native] = angles
            rates[..., native] = speeds
        return full, rates

    def complete_articulation_state(self, positions, velocities, full_positions, full_velocities, names):
        full_positions, full_velocities = full_positions.clone(), full_velocities.clone()
        ids = [names.index(name) for name in self.plant.map.joint_names]
        full_positions[..., ids] = positions.to(full_positions.dtype)
        full_velocities[..., ids] = velocities.to(full_velocities.dtype)
        for name, (position, velocity) in self.state_model.state(positions, velocities).items():
            index = names.index(name)
            full_positions[..., index] = position.to(full_positions.dtype)
            full_velocities[..., index] = velocity.to(full_velocities.dtype)
        return full_positions, full_velocities


def check_native_cranks(desc):
    import numpy as np
    from isaaclab_newton.physics import NewtonManager
    from pxr import Usd, UsdPhysics

    from humanoid_sim.model.usd import crank_joints

    model = NewtonManager.get_model()
    labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
    starts = model.joint_qd_start.numpy()
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    counts = set()
    for drive, name in crank_joints(desc).items():
        indices = [index for index, label in enumerate(labels) if label == name]
        if not indices:
            raise RuntimeError(f"native USD articulation lacks crank {name}")
        counts.add(len(indices))
        for index in indices:
            actual = [lower[starts[index]], upper[starts[index]]]
            if not np.allclose(actual, desc.drives[drive].position_limit_device, atol=1e-6, rtol=0):
                raise RuntimeError(f"native crank {name} has incorrect physical stops: {actual}")
    stage = Usd.Stage.Open(str(desc.usd_path))
    enabled = model.joint_enabled.numpy()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or not UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
            continue
        indices = [index for index, label in enumerate(labels) if label == prim.GetName()]
        if {len(indices)} != counts or not np.all(enabled[indices]):
            raise RuntimeError(f"native USD articulation lacks enabled loop closures for {prim.GetName()}")
