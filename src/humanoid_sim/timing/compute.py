"""The robot computer's policy compute time: from a control tick to the moment its demand leaves for the drives.

A policy on the robot reads its inputs at a tick, computes, and sends its demand when it is done, so the demand
leaves its compute time after the tick. A workload's measured compute time is kept as an empirical distribution,
its quantiles: ``probs`` from 0 to 1 and ``ns`` non-decreasing. A draw inverts it by linear interpolation between
the quantile points, so draws stay inside the measured range and keep its tail. Draws are independent per call,
as the benchmark measured them per call.

A robot profile lists its measured workloads under ``platform.compute`` (:meth:`ComputeDelay.from_profile`).
:func:`profile_block` writes that block from a Controls policy-compute benchmark report: the paced runs, timed
from each tick to the step's answer (``response_ns``: the wake-up lateness plus the compute).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The quantiles kept per workload and load: the median, the bulk, and the tail down to the 1-in-1000 calls a
# 1000-tick paced run resolves (its 0.999 quantile lies between its two slowest calls).
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)

BENCH_SCHEMA = "ethrc-policy-compute-bench/v1"
BENCH_FILES = ("policy-compute.json", "policy-compute.npz")
PACED_SERIES = "response_ns"  # from the tick to the answer: what delays the demand on the robot


@dataclass(frozen=True)
class ComputeDelay:
    """An empirical compute-time distribution: ``ns[i]`` is the ``probs[i]`` quantile."""

    probs: tuple[float, ...]
    ns: tuple[int, ...]

    def __post_init__(self):
        p, v = self.probs, self.ns
        if len(p) < 2 or len(p) != len(v):
            raise ValueError("a compute delay needs at least two quantiles, as many values as probabilities")
        if p[0] != 0.0 or p[-1] != 1.0:
            raise ValueError(f"quantile probabilities run from 0 to 1, got {p[0]} to {p[-1]}")
        if any(b <= a for a, b in zip(p, p[1:], strict=False)):
            raise ValueError("quantile probabilities must increase")
        if v[0] < 0 or any(b < a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError("quantile values must be non-negative and non-decreasing")

    @classmethod
    def constant(cls, ns: int) -> ComputeDelay:
        return cls((0.0, 1.0), (int(ns), int(ns)))

    @classmethod
    def from_samples(cls, samples_ns, probs: tuple[float, ...] = QUANTILES) -> ComputeDelay:
        """Quantiles of measured per-call times (``numpy.quantile``, linear interpolation)."""
        x = np.asarray(samples_ns, dtype=np.float64)
        if x.size == 0:
            raise ValueError("no samples")
        q = np.quantile(x, probs, method="linear")
        return cls(tuple(float(p) for p in probs), tuple(int(round(v)) for v in q))

    @classmethod
    def from_profile(cls, profile: dict, workload: str, load: str) -> ComputeDelay:
        """The profile's measured ``workload`` under ``load`` (``platform.compute``)."""
        block = profile.get("platform", {}).get("compute")
        if not block:
            raise KeyError(f"profile of {profile.get('robot')!r} has no platform.compute block")
        workloads = block["workloads"]
        if workload not in workloads:
            raise KeyError(f"no compute workload {workload!r}; measured: {sorted(workloads)}")
        loads = workloads[workload]["loads"]
        if load not in loads:
            raise KeyError(f"{workload}: no load {load!r}; measured: {sorted(loads)}")
        return cls(tuple(float(p) for p in block["quantiles"]), tuple(int(v) for v in loads[load]["ns"]))

    def at(self, u: torch.Tensor) -> torch.Tensor:
        """The delay at probability ``u`` (any shape, in [0, 1]): the inverse of the piecewise-linear CDF, int64
        ns."""
        p = torch.tensor(self.probs, dtype=torch.float64, device=u.device)
        v = torch.tensor(self.ns, dtype=torch.float64, device=u.device)
        u = u.to(torch.float64).clamp(0.0, 1.0)
        i = (torch.searchsorted(p, u, right=True) - 1).clamp(0, p.numel() - 2)
        p0, p1, v0, v1 = p[i], p[i + 1], v[i], v[i + 1]
        return torch.round(v0 + (u - p0) / (p1 - p0) * (v1 - v0)).to(torch.int64)

    def sample(self, n: int, *, device="cpu", generator: torch.Generator | None = None) -> torch.Tensor:
        """``n`` independent draws, int64 ns."""
        return self.at(torch.rand(n, dtype=torch.float64, device=device, generator=generator))


# -- Controls' policy-compute benchmark ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bench(report_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """The benchmark's report and its raw series (``<run>/<workload>/<series>``, ns)."""
    report_dir = Path(report_dir)
    report = json.loads((report_dir / BENCH_FILES[0]).read_text())
    if report.get("schema") != BENCH_SCHEMA:
        raise ValueError(f"{report_dir}: schema {report.get('schema')!r}, expected {BENCH_SCHEMA!r}")
    with np.load(report_dir / BENCH_FILES[1]) as z:
        series = {k: z[k] for k in z.files}
    return report, series


def paced_samples(report: dict, series: dict[str, np.ndarray], workload: str) -> dict[str, tuple[str, np.ndarray]]:
    """Per load, the paced run timing ``workload`` (the bench's name, e.g. ``"isaac-step b [cpu]"``): its run name
    and its ``response_ns``."""
    out: dict[str, tuple[str, np.ndarray]] = {}
    for i, run in enumerate(report["runs"]):
        if run["mode"] != "paced":
            continue
        for j, w in enumerate(run["workloads"]):
            if w["name"] != workload or w.get("status") != "ok":
                continue
            load = run["load"]["name"]
            if load in out:
                raise ValueError(f"{workload}: two paced runs under load {load!r}")
            out[load] = (run["name"], series[f"{i}/{j}/{PACED_SERIES}"])
    if not out:
        raise ValueError(f"no paced run timed {workload!r}")
    return out


def profile_block(report_dir: Path, workloads: dict[str, dict], *, unit: str, controls: str, tool: str) -> dict:
    """The ``platform.compute`` block of a robot profile from a benchmark report: ``workloads`` maps a name to
    ``{"bench": <bench workload name>, "doc": <what it is>}``. ``unit``, ``controls`` (Controls' commit on the
    robot) and ``tool`` (the benchmark's own commit) are provenance the report does not carry."""
    report_dir = Path(report_dir)
    report, series = read_bench(report_dir)
    plan, probe = report["core_plan"], report["probe"]
    block = {
        "source": {
            "tool": tool,
            "unit": unit,
            "controls": controls,
            "report": {"schema": report["schema"], "started": report["started"], "finished": report["finished"],
                       "files": {name: _sha256(report_dir / name) for name in BENCH_FILES}},
            "machine": {"gpu": probe["gpu"], "kernel": report["host_before"]["uname"]["release"],
                        "description": probe["description"], "versions": probe["versions"]},
            "placement": {"control_cpu": plan["control_cpu"], "isolated": plan["isolated"],
                          "housekeeping": plan["housekeeping"], "fifo_priority": plan["rt_priority"]},
            "series": f"{PACED_SERIES} of the paced runs: from the control tick to the step's answer",
        },
        "policy_period_ns": int(round(probe["policy_period_s"] * 1e9)),
        "quantiles": list(QUANTILES),
        "workloads": {},
    }
    for name, spec in workloads.items():
        loads = {}
        for load, (run, samples) in sorted(paced_samples(report, series, spec["bench"]).items()):
            d = ComputeDelay.from_samples(samples)
            loads[load] = {"run": run, "samples": int(samples.size), "ns": list(d.ns)}
        block["workloads"][name] = {"bench": spec["bench"], "doc": spec["doc"], "loads": loads}
    return block
