"""Backend-owned physical feedback; no synthetic current/torque sensor."""

import torch


class SensorBank:
    def __init__(self, shape, device, dtype):
        self.shape, self.device, self.dtype = shape, device, dtype
        self.position = torch.zeros(shape, device=device, dtype=dtype)
        self.velocity = torch.zeros_like(self.position)
        # NaN explicitly means the backend has not supplied a measured effort.
        self.measured_effort = torch.full_like(self.position, float("nan"))

    def validate(self, value, name, *, check_finite=True):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != self.shape:
            raise ValueError(f"{name} must be a Tensor of shape {self.shape}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if not value.is_floating_point():
            raise ValueError(f"{name} must contain floating point values")
        if check_finite and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite floating point values")

    def update(self, position, velocity, effort=None, *, check_finite=True):
        self.validate(position, "position", check_finite=check_finite)
        self.validate(velocity, "velocity", check_finite=check_finite)
        if effort is not None:
            self.validate(effort, "effort", check_finite=check_finite)
        self.position.copy_(position)
        self.velocity.copy_(velocity)
        if effort is not None:
            self.measured_effort.copy_(effort)
        else:
            self.measured_effort.fill_(float("nan"))
