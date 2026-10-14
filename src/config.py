"""Small explicit configuration boundary; no framework-specific registry."""

from dataclasses import dataclass
from pathlib import Path

import yaml


def read_mapping(path: str | Path) -> dict:
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


@dataclass(frozen=True)
class TimingConfig:
    physics_period_ns: int = 400_000
    firmware_period_ns: int = 400_000
    bus_period_ns: int = 2_000_000

    def __post_init__(self):
        for value in (self.physics_period_ns, self.firmware_period_ns, self.bus_period_ns):
            if type(value) is not int or value <= 0:
                raise ValueError("periods must be positive integer nanoseconds")
        if self.firmware_period_ns % self.physics_period_ns or self.bus_period_ns % self.physics_period_ns:
            raise ValueError("physics ticks must resolve firmware and bus boundaries exactly")
