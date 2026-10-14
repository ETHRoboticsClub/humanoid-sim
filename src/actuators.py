"""Vectorized output-shaft torque model, intentionally uncalibrated.

The optional torque-speed law is a conservative symmetric linear envelope,
not a measured HEJ four-quadrant map. Armature belongs in the physical backend
inertia; it is never approximated as a torque here.
"""

import torch


class ActuatorModel:
    def __init__(self, configs, device, dtype, firmware_period_ns):
        def row(name):
            return torch.tensor([getattr(c, name) for c in configs], device=device, dtype=dtype)[None, :]

        self.stiffness = row("stiffness")
        self.damping = row("damping")
        self.torque_limit = row("torque_limit")
        self.speed_limit = torch.tensor(
            [c.speed_limit or float("inf") for c in configs], device=device, dtype=dtype
        )[None, :]
        self.coulomb = row("coulomb_friction")
        self.viscous = row("viscous_friction")
        self.armature = row("armature")
        tau = row("torque_filter_tau_s")
        dt = firmware_period_ns * 1e-9
        self.alpha = torch.where(tau > 0, 1 - torch.exp(-dt / tau), torch.ones_like(tau))

    def evaluate(
        self, position, velocity, demand_position, demand_velocity, demand_effort, enabled, previous
    ):
        requested = self.stiffness * (demand_position - position)
        requested = requested + self.damping * (demand_velocity - velocity) + demand_effort
        cap = self.torque_limit * torch.clamp(1 - velocity.abs() / self.speed_limit, min=0, max=1)
        clipped = torch.maximum(torch.minimum(requested, cap), -cap)
        filtered = previous + self.alpha * (clipped - previous)
        # Reapply the current envelope after filtering; decreasing speed capacity
        # cannot leave a previously filtered torque outside the current cap.
        filtered = torch.maximum(torch.minimum(filtered, cap), -cap)
        motor = torch.where(enabled, filtered, torch.zeros_like(filtered))
        friction = self.coulomb * torch.sign(velocity) + self.viscous * velocity
        return requested, motor, friction, motor - friction
