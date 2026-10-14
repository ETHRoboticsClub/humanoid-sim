"""Integer time and explicit ordering for firmware/physics co-simulation."""

from dataclasses import dataclass


def validate_tick_ns(value: int, name: str = "tick_ns") -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be a nonnegative Python int64 integer")
    return value


@dataclass(frozen=True)
class TickEvent:
    now_ns: int
    firmware: bool
    physics: bool = True


class TickScheduler:
    """At each boundary: receive frames, firmware, apply held torque, physics.

    Physics must resolve every firmware boundary. A backend needing larger
    steps must explicitly substep; stale feedback cannot stand in for substeps.
    The event at zero initializes firmware before the first physics step.
    """

    def __init__(self, firmware_period_ns: int = 400_000, physics_period_ns: int = 400_000):
        validate_tick_ns(firmware_period_ns, "firmware_period_ns")
        validate_tick_ns(physics_period_ns, "physics_period_ns")
        if not firmware_period_ns or not physics_period_ns:
            raise ValueError("periods must be positive")
        if firmware_period_ns % physics_period_ns:
            raise ValueError("physics_period_ns must divide firmware_period_ns")
        self.firmware_period_ns = firmware_period_ns
        self.physics_period_ns = physics_period_ns
        self.now_ns = 0

    def current(self) -> TickEvent:
        return TickEvent(self.now_ns, self.now_ns % self.firmware_period_ns == 0)

    def step(self) -> TickEvent:
        self.now_ns += self.physics_period_ns
        return self.current()

    def reset(self, now_ns: int = 0) -> None:
        validate_tick_ns(now_ns)
        if now_ns % self.firmware_period_ns:
            raise ValueError("reset time must be on a firmware boundary")
        self.now_ns = now_ns
