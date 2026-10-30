"""Offline current diagnostics; no current-loop model or actuator input path.

Controls standalone.cpp @ 9930f8e2 publishes Reading::getActualJointCurrent /
getDemandedJointCurrent, not getActualCurrent (a different SDK signal). Pinned
maxon SDK e0753006 Reading.cpp converts both selected signals from mA to A.
The recorded torque estimate is current-derived, not an independent load cell.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.replay.inertial import source_identity


@dataclass(frozen=True)
class EffortCfg:
    mode: str = "raw"
    calibration: Path | None = None
    max_gap_ns: int = 50_000_000
    min_valid_fraction: float = 0.0

    def __post_init__(self):
        if self.mode not in ("raw", "current"):
            raise ValueError("effort mode must be raw or current")
        if (self.mode == "current") != (self.calibration is not None):
            raise ValueError("current comparison requires calibration; raw mode does not accept it")
        if type(self.max_gap_ns) is not int or self.max_gap_ns <= 0:
            raise ValueError("effort maximum gap must be positive integer ns")
        if not np.isfinite(self.min_valid_fraction) or not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("effort minimum valid fraction must be in [0, 1]")


@dataclass(frozen=True)
class CurrentConversion:
    """Fixed sourced current-to-output-equivalent torque mapping, not a fit.

    gain includes gearing, sign and stated loss assumptions. Its physical validity
    is the calibration producer's responsibility; source metadata is mandatory.
    No description-family constant alone is silently promoted to output torque.
    """

    drives: tuple[str, ...]
    gain_nm_per_a: np.ndarray
    zero_a: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path, desc, drives):
        data = Path(path).read_bytes()
        doc = json.loads(data)
        if doc.get("schema") != "ethrc-effort-calibration/v1":
            raise ValueError("unsupported effort calibration schema")
        if doc.get("description_digest") != desc.digest:
            raise ValueError("effort calibration description mismatch")
        if doc.get("current_signal") != "current_a" or doc.get("torque_reference") != "drive-output-equivalent":
            raise ValueError("calibration must map actual current_a to native drive-output-equivalent torque")
        for key in ("source", "assumptions"):
            if not isinstance(doc.get(key), str) or not doc[key].strip():
                raise ValueError(f"effort calibration requires {key}")
        if doc.get("status") not in ("modeled", "measured", "synthetic"):
            raise ValueError("calibration status must be modeled, measured or synthetic")
        try:
            gain = np.array([doc["drives"][d]["gain_nm_per_a"] for d in drives], dtype=float)
            zero = np.array([doc["drives"][d]["zero_a"] for d in drives], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("calibration requires gain_nm_per_a and zero_a for each selected drive") from error
        if gain.shape != (len(drives),) or zero.shape != gain.shape or not np.isfinite([gain, zero]).all() or (gain == 0).any():
            raise ValueError("finite nonzero scalar current gains and finite zero offsets required")
        return cls(tuple(drives), gain, zero, doc | {"sha256": hashlib.sha256(data).hexdigest()})

    def torque(self, current_a):
        return (current_a - self.zero_a) * self.gain_nm_per_a

    def equivalent_current(self, torque_nm):
        return torque_nm / self.gain_nm_per_a + self.zero_a


def sample_current(t_ns, values, query_ns, max_gap_ns):
    """Gap-aware linear telemetry sampling, no extrapolation or NaN bridging."""
    t, v, q = np.asarray(t_ns), np.asarray(values), np.asarray(query_ns)
    if (t.dtype != np.int64 or q.dtype != np.int64 or t.ndim != 1 or q.ndim != 1
            or not len(t) or v.shape != t.shape or np.any(np.diff(t) <= 0)):
        raise ValueError("current sampling requires ordered int64 source/query times and scalar samples")
    right = np.searchsorted(t, q, side="left")
    safe = np.minimum(right, len(t) - 1)
    exact = (right < len(t)) & (t[safe] == q)
    left = np.maximum(right - 1, 0)
    h = t[safe] - t[left]
    between = (~exact) & (right > 0) & (right < len(t)) & (h <= max_gap_ns)
    valid = (exact & np.isfinite(v[safe])) | (between & np.isfinite(v[left]) & np.isfinite(v[safe]))
    out = np.full(q.shape, np.nan)
    out[exact] = v[safe[exact]]
    k = between & valid
    out[k] = v[left[k]] + (v[safe[k]] - v[left[k]]) * ((q[k] - t[left[k]]) / h[k])
    out[~valid] = np.nan
    return out, valid


def coverage(t_ns, mask, max_gap_ns):
    """Supported span sums only adjacent valid samples within the gap limit."""
    gaps = np.diff(t_ns)
    supported = mask[:-1] & mask[1:] & (gaps <= max_gap_ns)
    valid_times = t_ns[mask]
    return {"samples": len(mask), "valid_samples": int(mask.sum()),
            "valid_fraction": float(mask.mean()) if len(mask) else 0.0,
            "covered_duration_ns": int(gaps[supported].sum()),
            "longest_valid_sample_gap_ns": int(np.diff(valid_times).max()) if len(valid_times) > 1 else None}


@dataclass(frozen=True)
class PreparedEffort:
    capture_id: str
    drives: tuple[str, ...]
    times: tuple[np.ndarray, ...]
    actual: tuple[np.ndarray, ...]
    demand: tuple[np.ndarray, ...]
    cfg: EffortCfg
    conversion: CurrentConversion | None
    metadata: dict

    def compare(self, t_ns, sim_applied_nm, *, skip_ns=1_000_000_000):
        if type(skip_ns) is not int or skip_ns < 0:
            raise ValueError("effort warmup must be nonnegative integer ns")
        t = np.asarray(t_ns)
        if t.dtype != np.int64 or t.ndim != 1 or not len(t) or np.any(np.diff(t) <= 0):
            raise ValueError("effort comparison requires ordered int64 replay times")
        applied = np.asarray(sim_applied_nm)
        if applied.shape != (len(t), len(self.drives)) or not np.isfinite(applied).all():
            raise ValueError("finite simulated applied torque [samples, selected drives] required")
        arrays = {"t_ns": t, "sim_applied_torque_nm": applied}
        report = {"metadata": self.metadata, "warmup_ns": skip_ns, "drives": {}}
        keep = t >= t[0] + skip_ns
        if not keep.any():
            raise ValueError("no effort samples after score warmup")
        actual, demand = [], []
        for i, d in enumerate(self.drives):
            a, av = sample_current(self.times[i], self.actual[i], t, self.cfg.max_gap_ns)
            b, bv = sample_current(self.times[i], self.demand[i], t, self.cfg.max_gap_ns)
            ca, cd = coverage(t[keep], av[keep], self.cfg.max_gap_ns), coverage(t[keep], bv[keep], self.cfg.max_gap_ns)
            if not ca["valid_samples"] or ca["valid_fraction"] < self.cfg.min_valid_fraction:
                raise ValueError(f"insufficient actual current coverage for {d}: {ca}")
            report["drives"][d] = {"actual": ca, "demand": cd}
            arrays[f"source_t_ns_{i}"] = self.times[i]
            arrays[f"source_current_a_{i}"] = self.actual[i]
            arrays[f"source_current_demand_a_{i}"] = self.demand[i]
            actual.append(a)
            demand.append(b)
        arrays["recorded_current_a"] = np.stack(actual, axis=1)
        arrays["recorded_current_demand_a"] = np.stack(demand, axis=1)
        arrays["actual_valid"] = np.isfinite(arrays["recorded_current_a"])
        arrays["demand_valid"] = np.isfinite(arrays["recorded_current_demand_a"])
        if self.conversion is not None:
            reference = self.conversion.torque(arrays["recorded_current_a"])
            residual = applied - reference
            arrays.update(current_derived_torque_nm=reference, applied_torque_residual_nm=residual,
                          model_equivalent_current_a=self.conversion.equivalent_current(applied))
            for i, d in enumerate(self.drives):
                err = residual[keep & arrays["actual_valid"][:, i], i]
                report["drives"][d]["rms_torque_nm"] = float(np.sqrt(np.mean(err ** 2)))
                report["drives"][d]["mean_torque_error_nm"] = float(err.mean())
        return arrays, report


def prepare_effort(capture, desc, drives, cfg):
    """Read once, retain actual and demand separately, and never alter the plant."""
    drives = tuple(drives)
    if not drives or len(set(drives)) != len(drives) or set(drives) - set(desc.drives):
        raise ValueError("effort requires nonempty unique description drive names")
    conversion = CurrentConversion.load(cfg.calibration, desc, drives) if cfg.mode == "current" else None
    pdo = capture.pdo()
    times, actual, demand = [], [], []
    digest = hashlib.sha256()
    info = {}
    for d in drives:
        if d not in pdo:
            raise ValueError(f"no current feedback from {d}")
        trace = pdo[d]
        t = np.array(trace.t_ns, copy=True)
        a = np.array(trace.fields.get("current_a", np.full(len(t), np.nan)), dtype=float, copy=True)
        b = np.array(trace.fields.get("current_demand_a", np.full(len(t), np.nan)), dtype=float, copy=True)
        sample_current(t, a, t, cfg.max_gap_ns)
        sample_current(t, b, t, cfg.max_gap_ns)
        if not np.isfinite(a).any():
            raise ValueError(f"no finite actual current for {d}")
        digest.update(d.encode())
        for v in (t, a, b):
            digest.update(np.ascontiguousarray(v).tobytes())
        info[d] = {"actual": coverage(t, np.isfinite(a), cfg.max_gap_ns),
                   "demand": coverage(t, np.isfinite(b), cfg.max_gap_ns)}
        times.append(t)
        actual.append(a)
        demand.append(b)
    metadata = {"mode": cfg.mode, "capture_id": capture.dataset_id, "drives": list(drives),
                "input_sha256": digest.hexdigest(), "description_digest": desc.digest,
                "mechanism_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sim_source": source_identity(__file__), "max_gap_ns": cfg.max_gap_ns,
                "min_valid_fraction": cfg.min_valid_fraction, "source_coverage": info,
                "sampling": "linear, exact source samples preserved; no gap bridging/extrapolation",
                "frame": "native drive output; ankle cranks", "time_reference": "per-drive source ROS ns",
                "telemetry_shift_ns": 0, "filter": "none", "current_units": "A", "torque_units": "N m",
                "demand_comparison": "raw only; hardware limiting/current-loop semantics not validated",
                "calibration": None if conversion is None else conversion.metadata,
                "quantity": "modeled output-equivalent estimate, not verified delivered torque"}
    return PreparedEffort(capture.dataset_id, drives, tuple(times), tuple(actual), tuple(demand), cfg, conversion, metadata)


def add_effort_args(parser):
    parser.add_argument("--effort-comparison", choices=("none", "raw", "current"), default="none")
    parser.add_argument("--effort-calibration", type=Path)
    parser.add_argument("--effort-max-gap-ms", type=float, default=50.0)
    parser.add_argument("--effort-min-valid-fraction", type=float, default=0.0)


def effort_config(args):
    if args.effort_comparison == "none":
        if args.effort_calibration is not None:
            raise ValueError("--effort-calibration requires --effort-comparison current")
        return None
    if not np.isfinite(args.effort_max_gap_ms):
        raise ValueError("finite effort maximum gap required")
    return EffortCfg(args.effort_comparison, args.effort_calibration,
                     int(round(args.effort_max_gap_ms * 1e6)), args.effort_min_valid_fraction)
