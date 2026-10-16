"""Named signals into Rerun, on the same ``time`` timeline as the 3D viewer.

Naming: ``/{source}/{env}/{group}/{name}`` for simulated signals and ``/{source}/{group}/{name}`` for one
robot's recorded data, e.g. ``/sim/0/drives/maxonHEJ90L4/position``, ``/real/drives/maxonHEJ90L4/position``,
``/residual/0/drives/maxonHEJ90L4/position``. Units are SI and implied by the name.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

TIMELINE = "time"  # elapsed seconds since the start of the recording


def signal_path(source: str, group: str, name: str, env: int | None = None) -> str:
    if env is None:
        return f"/{source}/{group}/{name}"
    return f"/{source}/{env}/{group}/{name}"


class SignalLog:
    """Thin wrapper over the global Rerun stream. ``init=False`` joins a recording another component opened."""

    def __init__(self, app_id: str = "humanoid-sim", *, rrd: str | None = None, spawn: bool = False,
                 init: bool = True):
        import rerun as rr

        self.rr = rr
        if init:
            rr.init(app_id)
            if rrd:
                rr.save(rrd)
            elif spawn:
                rr.spawn()

    def at(self, t_ns: int) -> None:
        self.rr.set_time(TIMELINE, duration=t_ns / 1e9)

    def log(self, name: str, value, *, source: str = "sim", group: str = "signals",
            envs: Iterable[int] | None = (0,)) -> None:
        """Log one value per env (``value`` indexed by env) at the current time."""
        v = np.asarray(value)
        if envs is None:
            self.rr.log(signal_path(source, group, name), self.rr.Scalars(v))
            return
        for e in envs:
            self.rr.log(signal_path(source, group, name, e), self.rr.Scalars(v[e] if v.ndim else v))

    def series(self, path: str, t_ns: np.ndarray, values: np.ndarray) -> None:
        """A whole time series in one call (columnar; the efficient path for recorded data)."""
        rr = self.rr
        rr.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=np.asarray(t_ns) / 1e9)],
                        columns=rr.Scalars.columns(scalars=np.asarray(values, dtype=np.float64)))
