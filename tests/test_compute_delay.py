"""The robot computer's measured policy compute time: the quantile distribution, its draws, the benchmark reader and
the G1 profile's block."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from humanoid_sim.model.asset import load_profile
from humanoid_sim.timing.compute import QUANTILES, ComputeDelay, profile_block

G1 = "g1_29dof_with_hand_rev_1_0"
G1_COMPUTE = Path(os.environ.get("HUMANOID_SIM_G1_COMPUTE", Path(__file__).resolve().parents[2] /
                                 "datasets/g1-lab-1/compute-2026-10-20"))


def test_a_draw_inverts_the_piecewise_linear_distribution():
    # probability 0.5 at 2 ms, the range 1 to 5 ms: linear in between
    d = ComputeDelay((0.0, 0.5, 1.0), (1_000_000, 2_000_000, 5_000_000))
    u = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert d.at(u).tolist() == [1_000_000, 1_500_000, 2_000_000, 3_500_000, 5_000_000]
    assert d.at(u).dtype == torch.int64
    assert ComputeDelay.constant(7).at(torch.rand(5)).tolist() == [7] * 5


def test_draws_follow_the_distribution():
    d = ComputeDelay((0.0, 0.5, 0.9, 1.0), (1_000, 2_000, 3_000, 10_000))
    n = 200_000
    x = d.sample(n, generator=torch.Generator().manual_seed(0)).numpy()
    assert x.min() >= d.ns[0] and x.max() <= d.ns[-1]
    # Dvoretzky-Kiefer-Wolfowitz: the empirical CDF lies within eps of the true one, except with probability alpha
    alpha = 1e-6
    eps = math.sqrt(math.log(2 / alpha) / (2 * n))
    for p, v in zip(d.probs, d.ns, strict=True):
        assert abs((x <= v).mean() - p) <= eps


def test_quantiles_of_samples():
    x = np.arange(1, 1001)  # numpy's linear quantile of 1..N at p is 1 + p (N - 1)
    d = ComputeDelay.from_samples(x)
    assert d.probs == QUANTILES
    assert d.ns == tuple(round(1 + p * 999) for p in QUANTILES)


@pytest.mark.parametrize("probs, ns", [((0.1, 1.0), (1, 2)), ((0.0, 0.9), (1, 2)), ((0.0, 0.5, 0.5, 1.0), (1, 2, 3, 4)),
                                       ((0.0, 1.0), (2, 1)), ((0.0, 1.0), (-1, 1)), ((0.0, 1.0), (1,))])
def test_a_malformed_distribution_is_refused(probs, ns):
    with pytest.raises(ValueError):
        ComputeDelay(probs, ns)


def _report(tmp_path: Path, runs: list[dict], series: dict[str, np.ndarray]) -> Path:
    """A policy-compute report with the fields profile_block reads."""
    report = {
        "schema": "ethrc-policy-compute-bench/v1", "started": "s", "finished": "f",
        "core_plan": {"control_cpu": 5, "isolated": [5, 6, 7], "housekeeping": [0, 1, 2, 3, 4], "rt_priority": 20},
        "probe": {"gpu": "g", "description": "d", "versions": {"torch": "t"}, "policy_period_s": 0.02},
        "host_before": {"uname": {"release": "k"}},
        "runs": runs,
    }
    (tmp_path / "policy-compute.json").write_text(json.dumps(report))
    np.savez(tmp_path / "policy-compute.npz", **series)
    return tmp_path


def _run(mode, load, names):
    return {"name": f"{mode} control-cpu {load}", "mode": mode, "load": {"name": load},
            "workloads": [{"name": n, "status": "ok"} for n in names]}


def test_a_profile_block_from_a_benchmark_report(tmp_path):
    w = "isaac-step b [cpu]"
    idle, busy = np.arange(1, 1001), np.arange(1001, 3001)
    runs = [_run("burst", "idle", [w]), _run("paced", "idle", ["other", w]), _run("paced", "housekeeping", [w])]
    series = {"0/0/call_ns": np.full(10, 99), "1/0/response_ns": np.full(10, 5), "1/1/response_ns": idle,
              "2/0/response_ns": busy}
    path = _report(tmp_path, runs, series)
    block = profile_block(path, {"core": {"bench": w, "doc": "x"}}, unit="u", controls="c", tool="t")
    assert block["policy_period_ns"] == 20_000_000
    assert block["quantiles"] == list(QUANTILES)
    loads = block["workloads"]["core"]["loads"]
    assert sorted(loads) == ["housekeeping", "idle"]  # the burst run is not a paced one
    assert loads["idle"]["ns"] == [round(1 + p * 999) for p in QUANTILES]
    assert loads["housekeeping"]["ns"] == [round(1001 + p * 1999) for p in QUANTILES]
    assert loads["housekeeping"]["samples"] == busy.size
    files = block["source"]["report"]["files"]
    for name in ("policy-compute.json", "policy-compute.npz"):
        assert files[name] == hashlib.sha256((path / name).read_bytes()).hexdigest()
    assert block["source"]["placement"]["control_cpu"] == 5


def test_a_report_with_two_paced_runs_under_one_load_is_refused(tmp_path):
    w = "isaac-step b [cpu]"
    runs = [_run("paced", "idle", [w]), _run("paced", "idle", [w])]
    path = _report(tmp_path, runs, {"0/0/response_ns": np.ones(3), "1/0/response_ns": np.ones(3)})
    with pytest.raises(ValueError, match="two paced runs"):
        profile_block(path, {"core": {"bench": w, "doc": "x"}}, unit="u", controls="c", tool="t")


def test_the_g1_profile_names_its_measured_workloads():
    profile = load_profile(G1)
    block = profile["platform"]["compute"]
    for workload, spec in block["workloads"].items():
        for load in spec["loads"]:
            d = ComputeDelay.from_profile(profile, workload, load)
            assert d.probs == tuple(block["quantiles"])
            # every compute time fits inside the policy period it was measured at
            assert d.ns[-1] < block["policy_period_ns"]
    with pytest.raises(KeyError, match="measured"):
        ComputeDelay.from_profile(profile, "no-such-workload", "idle")
    with pytest.raises(KeyError, match="no platform.compute"):
        ComputeDelay.from_profile(load_profile("ethrc_v1"), "policy_core", "idle")


@pytest.mark.data
@pytest.mark.skipif(not (G1_COMPUTE / "policy-compute.json").exists(), reason=f"no G1 compute report at {G1_COMPUTE} "
                    "(HUMANOID_SIM_G1_COMPUTE)")
def test_the_g1_profile_block_is_what_the_report_gives():
    block = load_profile(G1)["platform"]["compute"]
    src = block["source"]
    workloads = {n: {"bench": w["bench"], "doc": w["doc"]} for n, w in block["workloads"].items()}
    again = profile_block(G1_COMPUTE, workloads, unit=src["unit"], controls=src["controls"], tool=src["tool"])
    assert again == block
