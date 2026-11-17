"""Does `serve` keep up at the native V1's 1 ms period? Runs on a CUDA device; the timing assertions only on the
reference node class (an L40S): other GPUs, and the CPU, run the protocol checks and skip the numbers."""

import statistics
import time

import numpy as np
import pytest
import torch
from conftest import built_usd

from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi

pytestmark = pytest.mark.physics

# What one host step (the client's write and the host's whole step, unpaced) may cost on an L40S, in ms, with the
# measured mean and p99 of the latest benchmark run in docs/benchmarks.md. A regression fails here before it shows
# up as late steps in a Controls session.
STEP_BUDGET_MS = {"mean": 2.6, "p99": 3.2}


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""


def _host(tmp_path):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from humanoid_sim.runtime.host import RuntimeHost

    host = RuntimeHost(tmp_path / "plant", built_usd("ethrc_v1"), device="cuda:0", support="suspended")
    return host, abi.Segment.open(tmp_path / "plant", wait_s=5)


def _drive(host, client, steps: int, *, start: int = 0) -> np.ndarray:
    """Step the host ``steps`` times, the client writing a demand on every bus cycle (1.5 ms, so one step in three
    sees no new record), and return each step's wall cost in ms."""
    desc = Description.load("ethrc_v1")
    q, _ = host._named_pose(host.start_pose)
    dtype = host.sim.plant.cfg.dtype
    hold = host.sim.plant.map.drive_state(q.to(dtype), torch.zeros_like(q).to(dtype))[0][0].cpu().numpy()
    demand = abi.pd_demand(hold, [desc.drives[d].baseline_stiffness for d in host.drives],
                           [desc.drives[d].baseline_damping for d in host.drives])
    bus = host._segment_args["bus_period_ns"]
    cost, cycle = np.empty(steps), start
    for k in range(steps):
        t = time.perf_counter()
        if (k + 1) * host.period_ns // bus > k * host.period_ns // bus:
            cycle += 1
            client.write_demands(demand, cycle=cycle)
        host.step()
        cost[k] = (time.perf_counter() - t) * 1e3
    return cost


def test_a_step_costs_no_more_than_its_budget_on_the_reference_gpu(tmp_path):
    host, client = _host(tmp_path)
    try:
        _drive(host, client, 100)  # the first steps compile and settle
        torch.cuda.synchronize()
        cost = _drive(host, client, 1500, start=100)
        mean, p99 = float(cost.mean()), float(np.percentile(cost, 99))
        print(f"step cost on {_gpu_name()}: mean {mean:.3f} ms, p99 {p99:.3f} ms, max {cost.max():.3f} ms, "
              f"period {host.period_ns / 1e6:.3f} ms, real-time factor {host.period_ns / 1e6 / mean:.2f}")
        if "L40S" not in _gpu_name():
            pytest.skip(f"step timing is asserted on an L40S, this is a {_gpu_name()}")
        assert mean < STEP_BUDGET_MS["mean"] and p99 < STEP_BUDGET_MS["p99"], (mean, p99)
    finally:
        client.close()
        host.close()


@pytest.mark.xfail(reason="MJWarp's RK4 step alone costs 1.35 ms on an L40S and the host's own work about 0.9 ms "
                   "(docs/benchmarks.md): the native V1 at 1 ms runs below real time until the integrator and the host's "
                   "per-step work both shrink", strict=True)
def test_a_step_costs_less_than_the_period_on_the_reference_gpu(tmp_path):
    host, client = _host(tmp_path)
    try:
        _drive(host, client, 100)
        torch.cuda.synchronize()
        cost = _drive(host, client, 1000, start=100)
        if "L40S" not in _gpu_name():
            pytest.skip(f"step timing is asserted on an L40S, this is a {_gpu_name()}")
        assert statistics.fmean(cost) < host.period_ns / 1e6
    finally:
        client.close()
        host.close()


def test_the_descriptor_and_status_report_lateness(tmp_path):
    host, client = _host(tmp_path)
    try:
        timing = client.descriptor()["timing"]
        assert timing["late_after_ns"] == host.period_ns == 1_000_000
        assert timing["keeping_up_window_ns"] > 0
        host.run(max_steps=60)  # paced: the first steps compile nothing, but a step may cost over the period
        status = host.status()
        assert set(status) >= {"late_steps", "overrun_steps", "step_ms_mean", "step_ms_max"}
        assert status["step_ms_mean"] > 0 and status["step_ms_max"] >= status["step_ms_mean"]
        assert status["overrun_steps"] <= 60
    finally:
        client.close()
        host.close()


def test_the_graphed_step_writes_the_solver_targets_the_eager_step_would(tmp_path):
    """The plant's joint targets and efforts go into the solver's buffers inside the captured graph. Replayed on the
    current state, they are bit-for-bit what the eager assembly writes, and the buffers have not moved."""
    host, client = _host(tmp_path)
    try:
        sim = host.sim
        for _ in range(5):
            host.step()
        graphs = sim._graphs
        assert graphs.stable()
        position, velocity, effort = graphs.command
        graphs.t.fill_(sim.t_ns)
        graphs.g_pre.replay()
        graphed = [t.clone() for t in (position, velocity, effort)]
        sim._write_targets(graphs.target)  # the eager assembly, on the same state
        for got, want in zip(graphed, (position, velocity, effort), strict=True):
            assert torch.equal(got, want)
        assert graphs.stable()
    finally:
        client.close()
        host.close()
