"""Recorded motor gains versus fixed numerical solver references; no fitting."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from humanoid_sim.devices.shaping import SolverGainReference


def capture_solver_reference(captures, desc):
    """One fixed per-drive maximum envelope for explicitly selected captures.

    Supports train/holdout captures with different P/D without changing references
    during stepping or substituting their actual hardware demands.
    """
    drives = tuple(desc.drives)
    if not captures or len({c.dataset_id for c in captures}) != len(captures):
        raise ValueError("solver references need nonempty unique capture identities")
    p, d, signatures = [], [], []
    for cap in captures:
        gains = cap.gains()
        if set(gains) != set(drives):
            raise ValueError("capture solver reference gain inventory differs from description")
        pairs = [(gains[n].stiffness, gains[n].damping) for n in drives]
        signatures.append({"capture": cap.dataset_id, "gains": pairs})
        p.append([v[0] for v in pairs])
        d.append([v[1] for v in pairs])
    digest = hashlib.sha256(json.dumps(signatures, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return SolverGainReference(drives, tuple(np.max(p, axis=0).tolist()), tuple(np.max(d, axis=0).tolist()),
                               "capture-envelope:" + digest)


def gain_support(capture, plant, *, gains=None):
    """CPU preflight, once outside stepping. Refuse unsupported recorded gains.

    Explicit references must dominate recorded hardware P/D; this makes the
    runtime coupled-ankle reference bound conservative for the captured demands.
    Nominal mode retains the existing twice-reference documented region.
    """
    gains = capture.gains() if gains is None else gains
    names = list(plant.map.drive_names)
    if set(gains) != set(names):
        raise ValueError("replay gain inventory differs from plant drives")
    recorded = np.array([(gains[n].stiffness, gains[n].damping) for n in names])
    if not np.isfinite(recorded).all() or (recorded[:, 0] <= 0).any() or (recorded[:, 1] < 0).any():
        raise ValueError("replay requires finite positive P and nonnegative D")
    reference = np.stack([plant.solver_drive_stiffness.detach().cpu().numpy(),
                          plant.solver_drive_damping.detach().cpu().numpy()], axis=1)
    explicit = plant.solver_gains is not None
    limit = 1.0 if explicit else 2.0
    rounding = 4 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(reference))
    scales = np.stack([plant.bank.p_scale.detach().cpu().numpy(), plant.bank.d_scale.detach().cpu().numpy()], axis=-1)
    realized = recorded[None] * scales
    if not np.isfinite(realized).all() or (realized < 0).any():
        raise ValueError("invalid realized replay gains")
    unsupported = (realized > limit * reference[None] + rounding[None]).any(axis=(0, 2))
    report = {"mode": "capture" if explicit else "nominal", "drives": names,
              "recorded_p_d": recorded.tolist(), "reference_p_d": reference.tolist(),
              "reference_source": "description nominal" if not explicit else plant.solver_gains.source,
              "realized_p_d_max": realized.max(axis=0).tolist(),
              "joint_names": list(plant.map.joint_names),
              "solver_k0": plant.k0[0].detach().cpu().tolist(),
              "solver_d0": plant.d0[0].detach().cpu().tolist(),
              "criterion": "explicit envelope dominates hardware P/D; coupled bound checked during replay"
                           if explicit else "recorded P/D <= twice nominal references",
              "unsupported_drives": [n for n, bad in zip(names, unsupported, strict=True) if bad]}
    if unsupported.any():
        raise ValueError("recorded gains exceed solver reference support: " + json.dumps(report))
    return report


def check_run_gain_support(plant):
    """One end-of-run sync; no steady-state CPU synchronization."""
    warned = plant.gain_warnings.any().item()
    ratio = float(plant.ankle_reference_ratio_peak.max())
    if warned or not np.isfinite(ratio) or ratio > 2.0 + 1e-5:
        raise RuntimeError(f"solver gain support exceeded during replay: drive_warning={warned}, "
                           f"coupled_ankle_reference_ratio_peak={ratio}")
    return {"drive_warning": bool(warned), "coupled_ankle_reference_ratio_peak": ratio,
            "coupled_bound": "local J^T K J eigenvalue / diagonal solver reference; not a nonlinear stability proof"}
