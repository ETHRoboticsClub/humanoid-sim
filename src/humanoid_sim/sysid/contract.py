"""What a capture must hold for a PACE fit, checked without physics.

:func:`check_capture` applies the contract in ``docs/replay-and-sysid.md`` ("The capture contract") to an
``exc-*`` capture or a rosbag, and ``check-bag`` and ``fit-pace`` both run it. Errors make a fit wrong and refuse
it; warnings are worth a look. That the robot hung on the gantry, and which excitation band a capture holds, are
not in the data and stay the caller's to know.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_OVERLAP_NS = 5_000_000_000  # commanded feedback a fit needs at least (chosen: a few periods of a low band)
MAX_GAP_NS = 50_000_000  # a feedback gap longer than this is reported (ten 200 Hz samples)
SHIFT_MARGIN_NS = 10_000_000  # the rebuild check scans this far beyond the fit's delay range
SHIFT_STEP_NS = 1_000_000


@dataclass
class CaptureCheck:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    rebuild: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> str:
        lines = [f"{self.name}: {'ok' if self.ok else 'REFUSED'}"]
        lines += [f"  {k}: {v}" for k, v in self.info.items()]
        lines += [f"  error: {e}" for e in self.errors] + [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def check_capture(capture, plant, *, bias_bound_rad: float, delay_bounds_ns: tuple[int, int],
                  scored: list[str] | None = None, need_imu: bool = True) -> CaptureCheck:
    """``capture``: an ``ExcitationCapture`` or ``BagCapture``; ``plant``: anything with ``desc`` and ``map``
    (a ``PlantCore``). ``scored``: the drives the fit scores (default all); the others need only stream."""
    from humanoid_sim.replay.replay import command_demand_rms, drive_targets

    c = CaptureCheck(getattr(capture, "name", str(capture)))
    drives = list(plant.map.drive_names)
    scored = drives if scored is None else scored
    try:
        pdo = capture.pdo()
    except ValueError as e:
        c.errors.append(str(e))
        return c
    missing = sorted(set(drives) - set(pdo))
    if missing:
        c.errors.append(f"no feedback from drives {missing}")
        return c
    t0, t1 = capture.time_span_ns()
    c.info["feedback"] = f"{len(drives)} drives, {(t1 - t0) / 1e9:.1f} s"

    no_demand = [d for d in scored if not np.isfinite(pdo[d]["position_demand_rad"]).any()]
    if no_demand:
        which = "any scored drive" if len(no_demand) == len(scored) else f"scored drives {no_demand}"
        c.errors.append(f"no 0x6062 position demand from {which}: the fit cannot check the rebuilt targets against "
                        "what the drives acted on")
    bad = [d for d in drives if not np.isfinite(pdo[d]["native_position_rad"]).all()]
    if bad:
        c.errors.append(f"non-finite measured positions on {bad}")
    gaps = {d: int(np.diff(pdo[d].t_ns).max()) for d in drives if len(pdo[d].t_ns) > 1}
    long = sorted(d for d, g in gaps.items() if g > MAX_GAP_NS)
    if long:
        c.warnings.append(f"feedback gaps up to {max(gaps[d] for d in long) / 1e6:.0f} ms on "
                          + ("every drive" if len(long) == len(drives) else str(long)))

    for what, read in (("gains", capture.gains), ("IMU", capture.imu if need_imu else None)):
        if read is None:
            continue
        try:
            read()
        except ValueError as e:
            c.errors.append(f"{what}: {e}")

    try:
        _, t_cmd = drive_targets(plant, capture.commands())
    except ValueError as e:
        c.errors.append(f"commands: {e}")
        return c
    overlap = min(t1, int(t_cmd[-1])) - max(t0, int(t_cmd[0]))
    c.info["commands"] = (f"{len(t_cmd)} frames over {(t_cmd[-1] - t_cmd[0]) / 1e9:.1f} s, "
                          f"{max(overlap, 0) / 1e9:.1f} s of it with feedback")
    if overlap < MIN_OVERLAP_NS:
        c.errors.append(f"the command frames and the feedback overlap for {max(overlap, 0) / 1e9:.1f} s (at least "
                        f"{MIN_OVERLAP_NS / 1e9:g} s needed); stamps on different clocks would do this")
        return c

    lo, hi = delay_bounds_ns
    shifts = np.arange(lo - SHIFT_MARGIN_NS, hi + SHIFT_MARGIN_NS + 1, SHIFT_STEP_NS)
    r = c.rebuild = command_demand_rms(capture, plant, shifts)
    best = r["best_shift_ns"]
    rms, mean = r["rms_at_best_rad"], r["mean_at_best_rad"]
    worst = max((d for d in scored), key=lambda d: rms[d])
    against = "0x6062" if not no_demand else "0x6062 or, where missing, the measured position"
    c.info["rebuild"] = f"vs {against}: best shift {best / 1e6:g} ms, worst {worst} {rms[worst] * 1e3:.2f} mrad RMS"
    # Against the measured position the shift includes the drive's tracking lag, so it says nothing of the delay.
    if not no_demand and not lo <= best <= hi:
        c.errors.append(f"the rebuilt targets best explain 0x6062 {best / 1e6:g} ms later, outside the fit's delay "
                        f"range {lo / 1e6:g} to {hi / 1e6:g} ms (--delay-ms)")
    off = {d: mean[d] for d in scored if abs(mean[d]) > bias_bound_rad}
    if off:
        c.errors.append("rebuilt targets sit beyond the bias bound on "
                        + ", ".join(f"{d} ({m * 1e3:+.0f} mrad)" for d, m in sorted(off.items()))
                        + ": the capture's zeros or signs differ from the description's")
    return c
