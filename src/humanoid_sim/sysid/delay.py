"""The delay between a capture's recorded drive demand and the drive acting on it.

The captures record each drive's position demand (0x6062) as the host received it (about 200 Hz); the drive
acted on it earlier or later than that stamp, by its PDO, bus and interpolation timing. A batched replay gives
every env its own delay (the demand recorded at ``t`` applied at ``t + delay``; negative: earlier) and scores the
position RMS of every drive: the delay that minimizes it is what the replay, and the plant's latency model, should
use. Reported per drive and per region (the description's ``region`` of the drive's joint; ``ankle`` for the
cranks).
"""

from __future__ import annotations

import numpy as np


def drive_regions(desc, drives: list[str]) -> dict[str, list[int]]:
    regions: dict[str, list[int]] = {}
    for i, name in enumerate(drives):
        d = desc.drives[name]
        region = "ankle" if d.crank else desc.joints[d.joints[0]].region
        regions.setdefault(region, []).append(i)
    return regions


def scan_delay(sim, captures, delays_ns: np.ndarray, *, base_motion: str | None, duration_s: float | None,
               skip_s: float = 1.0) -> dict:
    """Position RMS for every delay (env) and drive, over the captures; the best delay per drive and region."""
    from humanoid_sim.replay.replay import replay_position_rms

    delays_ns = np.asarray(delays_ns, dtype=np.int64)
    if len(delays_ns) != sim.plant.num_envs:
        raise ValueError(f"{len(delays_ns)} delays for {sim.plant.num_envs} envs")
    per = np.stack([replay_position_rms(c, sim, base_motion=base_motion, duration_s=duration_s, skip_s=skip_s,
                                        delays_ns=delays_ns).cpu().numpy() for c in captures])  # [C, E, D]
    rms = per.mean(axis=0)  # [E, D]
    drives = sim.plant.map.drive_names
    zero = int(np.argmin(np.abs(delays_ns)))
    regions = {}
    for region, idx in drive_regions(sim.desc, drives).items():
        curve = rms[:, idx].mean(axis=1)
        k = int(np.argmin(curve))
        regions[region] = {"best_delay_ns": int(delays_ns[k]), "rms_best_rad": float(curve[k]),
                           "rms_at_zero_rad": float(curve[zero])}
    best = np.argmin(rms, axis=0)
    return {
        "delays_ns": delays_ns.tolist(), "drives": drives, "rms_rad": rms.tolist(),
        "best_delay_ns": {d: int(delays_ns[best[i]]) for i, d in enumerate(drives)},
        "regions": regions, "per_capture_rms_rad": per.tolist(),
    }
