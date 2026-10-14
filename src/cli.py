"""Small behavioral-device and real Newton smoke runners, with lazy imports."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time


def parser():
    result = argparse.ArgumentParser(prog="humanoid-sim")
    sub = result.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="single drive process-data → firmware → physics → sensor smoke")
    smoke.add_argument("--backend", choices=("cpu", "isaac"), default="cpu")
    smoke.add_argument("--steps", type=int, default=120)
    smoke.add_argument("--num-envs", type=int, default=4)
    smoke.add_argument("--device", default=None, help="cpu or cuda:N; Isaac defaults to cuda:0")
    smoke.add_argument("--physics-period-ns", type=int, default=400_000)
    smoke.add_argument("--firmware-period-ns", type=int, default=400_000)
    smoke.add_argument("--headless", action="store_true", help="omit a native window")
    smoke.add_argument("--visualizer", choices=("none", "viser", "newton_gl"), default="none")
    smoke.add_argument("--viewer-host", default="127.0.0.1")
    smoke.add_argument("--viewer-port", type=int, default=8080)
    smoke.add_argument("--realtime", action="store_true", help="pace smoke to simulated time for a viewer")
    smoke.add_argument("--output-dir", type=Path, help="write trace.jsonl and summary.json")
    return result


def _devices(args, device):
    from .contracts import DriveConfig
    from .firmware import DeviceBank

    return DeviceBank(
        [DriveConfig("hinge", stiffness=20.0, damping=1.0, torque_limit=5.0)],
        args.num_envs,
        device=device,
        firmware_period_ns=args.firmware_period_ns,
        check_finite=args.backend == "cpu",
    )


def _submit(devices, now_ns):
    import torch
    from .contracts import CommandBatch, MODE_JVPT

    tick = now_ns // devices.firmware_period_ns
    controlword = (6, 7, 15)[min(tick, 2)]

    def real(value):
        return torch.full(devices.shape, value, device=devices.device, dtype=devices.dtype)

    def integer(value):
        return torch.full(devices.shape, value, device=devices.device, dtype=torch.int64)

    devices.submit(
        CommandBatch(integer(controlword), integer(MODE_JVPT), real(0.2), real(0.0), real(0.0)), now_ns
    )


def _run_steps(args, adapter, physics_step, *, backend):
    import torch

    initial = adapter.devices.feedback.position.clone()
    peak_effort = 0.0
    start = time.monotonic()
    with ExitStack() as outputs:
        trace_handle = None
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            trace_handle = outputs.enter_context((args.output_dir / "trace.jsonl").open("w"))
        for index in range(args.steps):
            if adapter.clock.current().firmware:
                _submit(adapter.devices, adapter.clock.now_ns)
            effort = adapter.pre()
            # Exactly one physical step; the adapter never owns or advances physics.
            physics_step()
            feedback = adapter.post()
            peak_effort = max(peak_effort, float(effort.abs().max()))
            finite = bool(torch.isfinite(feedback.position).all() & torch.isfinite(feedback.velocity).all())
            if not finite:
                raise RuntimeError("smoke physical state became nonfinite")
            if trace_handle is not None:
                row = {
                    "step": index + 1,
                    "physics_time_ns": feedback.timestamp_ns,
                    "firmware_time_ns": feedback.firmware_timestamp_ns,
                    "position_rad": feedback.position.cpu().tolist(),
                    "velocity_rad_s": feedback.velocity.cpu().tolist(),
                    "applied_device_effort_nm": feedback.applied_effort.cpu().tolist(),
                    "statusword": feedback.statusword.cpu().tolist(),
                }
                trace_handle.write(json.dumps(row) + "\n")
            if args.realtime:
                remaining = start + adapter.clock.now_ns * 1e-9 - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
    feedback = adapter.devices.feedback
    moved = float((feedback.position - initial).abs().max())
    if peak_effort <= 0.0 or moved <= 1e-8:
        raise RuntimeError("smoke failed to apply nonzero firmware effort and move the joint")
    summary = {
        "backend": backend,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "device": str(adapter.devices.device),
        "physics_time_ns": adapter.clock.now_ns,
        "firmware_period_ns": args.firmware_period_ns,
        "physics_period_ns": args.physics_period_ns,
        "peak_applied_effort_nm": peak_effort,
        "max_displacement_rad": moved,
        "final_position_rad": feedback.position.cpu().tolist(),
        "finite": True,
        "physical_effort_sensor": False,
    }
    if args.output_dir:
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def cpu_smoke(args):
    """Analytic one-DOF inertia stub validates lifecycle; it is not Isaac evidence."""
    from types import SimpleNamespace
    import torch
    from .isaac import IsaacEffortAdapter

    devices = _devices(args, args.device or "cpu")
    q = torch.zeros(devices.shape, device=devices.device, dtype=devices.dtype)
    dq = torch.zeros_like(q)
    commanded = torch.zeros_like(q)

    class EffortTarget:
        def set_effort_index(self, *, value, joint_ids=None, env_ids=None):
            commanded.copy_(value)

    class InertiaStub:
        data = SimpleNamespace(
            joint_pos=q, joint_vel=dq, joint_stiffness=torch.zeros_like(q), joint_damping=torch.zeros_like(q)
        )
        cfg = SimpleNamespace(
            actuators={"effort": SimpleNamespace(class_type=SimpleNamespace(is_implicit_model=True))}
        )
        actuators = SimpleNamespace(target_command=EffortTarget())

        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

        def step(self):
            dt = args.physics_period_ns * 1e-9
            dq.add_(commanded * (dt / (1.0 / 12.0)))
            q.add_(dq * dt)

    articulation = InertiaStub()
    adapter = IsaacEffortAdapter(articulation, devices, physics_period_ns=args.physics_period_ns)
    return _run_steps(args, adapter, articulation.step, backend="cpu-inertia-stub")


def isaac_smoke(args):
    # This import path only runs when explicitly selected. Core/help need no Kit,
    # USD, Warp, Newton, CUDA, scene assets, or RL installation.
    try:
        from isaaclab.app import launch_simulation
        from isaaclab.sim import SimulationContext
        from .isaac import (
            IsaacEffortAdapter,
            author_single_joint,
            make_newton_simulation_config,
            make_single_joint_articulation,
        )

        cfg = make_newton_simulation_config(
            device=args.device or "cuda:0",
            physics_period_ns=args.physics_period_ns,
            visualizer=args.visualizer,
            viewer_host=args.viewer_host,
            viewer_port=args.viewer_port,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Isaac smoke requires the pinned Isaac Lab + Newton runtime; "
            "build the repository container. CPU smoke remains available. "
            f"Dependency error: {exc}"
        ) from exc
    launcher = {
        "device": cfg.device,
        "headless": args.headless,
        "visualizer": [] if args.visualizer == "none" else [args.visualizer],
        "livestream": 0,
    }
    with launch_simulation(cfg, launcher):
        sim = SimulationContext(cfg)
        manager = sim.physics_manager
        if not manager.__module__.startswith("isaaclab_newton.") or "Featherstone" not in manager.__name__:
            raise RuntimeError(f"requested Newton Featherstone but initialized {manager}")
        author_single_joint(sim.stage, num_envs=args.num_envs)
        articulation = make_single_joint_articulation()
        sim.reset()
        devices = _devices(args, sim.device)
        adapter = IsaacEffortAdapter(articulation, devices, physics_period_ns=args.physics_period_ns)
        return _run_steps(
            args,
            adapter,
            lambda: sim.step(render=args.visualizer != "none"),
            backend="isaaclab-newton-featherstone",
        )


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.steps < 4 or args.num_envs < 1:
        cli.error("smoke requires --steps >= 4 and --num-envs >= 1")
    if not 1 <= args.viewer_port <= 65535:
        cli.error("--viewer-port must be between 1 and 65535")
    if args.backend == "cpu" and args.visualizer != "none":
        cli.error("visualizers require --backend isaac")
    try:
        from .clock import TickScheduler

        TickScheduler(args.firmware_period_ns, args.physics_period_ns)
        result = cpu_smoke(args) if args.backend == "cpu" else isaac_smoke(args)
    except (RuntimeError, ValueError, ImportError) as exc:
        print(f"humanoid-sim: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
