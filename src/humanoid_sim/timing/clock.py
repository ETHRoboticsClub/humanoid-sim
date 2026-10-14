"""Integer-nanosecond conversions."""

NS_PER_US = 1_000
NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000


def ns_from_s(seconds: float) -> int:
    """Round seconds to the nearest integer nanosecond."""
    return int(round(seconds * NS_PER_S))


def s_from_ns(ns: int) -> float:
    return ns / NS_PER_S


def check_period(name: str, period_ns: int, *, divides: int | None = None) -> int:
    """Validate a period; optionally require it to divide another period exactly."""
    if not isinstance(period_ns, int) or period_ns <= 0:
        raise ValueError(f"{name} must be a positive integer number of nanoseconds, got {period_ns!r}")
    if divides is not None and divides % period_ns:
        raise ValueError(f"{name}={period_ns} ns does not divide {divides} ns")
    return period_ns
