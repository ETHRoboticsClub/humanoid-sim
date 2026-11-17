"""The ``replay`` of a rosbag2 recording and the ``regress`` command (see :mod:`humanoid_sim.replay.regression`)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

RECORDINGS_ENV = "HUMANOID_SIM_RECORDINGS"
DEFAULT_RECORDINGS = "../../data/recordings"
DEFAULT_CASES = "regression/cases.yaml"


def recordings_dir(args) -> Path:
    return Path(args.recordings or os.environ.get(RECORDINGS_ENV) or DEFAULT_RECORDINGS)


def is_bag(path) -> bool:
    return (Path(path) / "metadata.yaml").is_file()


def build_sim(args, usd: Path):
    """The sim for a bag replay: the asset with base joints (the base follows the IMU), the profile's plant."""
    from humanoid_sim.sim import SimCore

    return SimCore(usd, device=args.device, robot=args.robot,
                   period_ns=None if args.period_ms is None else int(round(args.period_ms * 1e6)),
                   fixed_base=False, base_joints=True, solver=args.solver, integrator=args.integrator,
                   self_collision=False if getattr(args, "no_self_collision", False) else None)


def _stats_json(results) -> dict:
    return {m: r.as_dict() for m, r in results.items()}


def _write_trace(out: Path, run) -> None:
    res = run.replay
    np.savez_compressed(out / "trace.npz", t_ns=res.t_ns, demand=res.demand,
                        **{f"sim_{k}": v for k, v in res.sim.items()},
                        **{f"real_{k}": v for k, v in res.real.items()})


def _json_default(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def replay_bag(args, usd: Path) -> int:
    """``replay <bag>``: the recording's window on the sim, the base following its IMU, each COMMAND frame's targets
    arriving with the profile's measured command path; summary.json holds the metrics, robot and sim."""
    from humanoid_sim.model.asset import plant_identity
    from humanoid_sim.replay import regression as R
    from humanoid_sim.replay.bag import BagCapture

    sim = build_sim(args, usd)
    drives = list(sim.plant.map.drive_names)
    bag = BagCapture(args.capture, drives=drives)
    if args.window:
        window = (float(args.window[0]), float(args.window[1]))
    else:  # from the first frame naming every joint (and the warm-up after it) to the end of the drives' feedback
        commands = R.drive_commands(bag, sim.desc, drives)
        first = max(bag.time_span_ns()[0], int(commands.t_ns[0]))
        window = ((first - bag.start_ns) / 1e9 + args.warmup, (bag.time_span_ns()[1] - bag.start_ns) / 1e9 - 0.1)
    case = R.Case(Path(args.capture).name, "", {"s3": "", "sha256": bag.sha256, "path": ""}, window, args.warmup, {}, {},
                  support="unknown", scoring="open_loop")
    t = time.time()
    run = R.run_case(case, Path(args.capture), sim, base=args.base)
    wall = time.time() - t
    tag = f"{bag.dataset_id}_{sim.solver}_{sim.integrator}_{sim.period_ns // 1000}us_base-{args.base}"
    out = Path(args.out) / f"{tag}_{window[0]:g}-{window[1]:g}s"
    out.mkdir(parents=True, exist_ok=True)
    _write_trace(out, run)
    res = run.replay
    summary = {"dataset_id": bag.dataset_id, "name": bag.name, "window_s": list(window), "wall_s": wall,
               "sim_s": len(res.t_ns) * sim.period_ns / 1e9, "drives": res.drives, "device": args.device,
               **plant_identity(sim, usd), **run.meta, "metrics": _stats_json(run.results),
               "rms_position_rad": res.rms("position"), "rms_velocity_rad_s": res.rms("velocity"),
               "rms_torque_nm": res.rms("torque"), "tracking_rms_rad": res.tracking_rms()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    print(json.dumps({k: summary[k] for k in ("dataset_id", "period_ns", "wall_s", "sim_s", "window_s")}))
    for d in res.drives:
        key = f"tracking_rms.{d}"
        r = run.results.get(key)
        sim_err = "" if r is None or r.sim is None else f"  tracking sim {r.sim.mean * 1e3:6.2f} robot {r.robot.mean * 1e3:6.2f} mrad"
        print(f"{d:22s} {summary['rms_position_rad'][d] * 1e3:7.2f} mrad sim-real{sim_err}")
    return 0


def regress(args, usd: Path | None) -> int:
    """``regress``: every case on the sim, a verdict per metric; exit 1 when any case fails. ``--propose`` rewrites
    the cases file's bands from the robot's values instead (no sim)."""
    from humanoid_sim.model.description import Description
    from humanoid_sim.replay import regression as R

    cases = R.load_cases(Path(args.cases))
    chosen = [c for c in cases.cases if not args.case or c.id in args.case]
    if args.case and len(chosen) != len(set(args.case)):
        raise SystemExit(f"unknown case in {args.case}; the file has {[c.id for c in cases.cases]}")
    recordings = recordings_dir(args)
    if args.propose:
        desc = Description.load(args.robot)
        proposed = R.propose_file(R.Cases(cases.version, cases.tolerances, chosen), recordings, desc, list(desc.drives))
        keep = {c.id: c for c in proposed.cases}
        merged = R.Cases(cases.version, cases.tolerances, [keep.get(c.id, c) for c in cases.cases])
        Path(args.cases).write_text(R.dump_cases(merged))
        for c in proposed.cases:
            print(f"{c.id}: {len(c.bands)} bands over {c.robot_windows} windows -> {args.cases}")
        return 0

    sim = build_sim(args, usd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, failed = {}, False
    for case in chosen:
        t = time.time()
        root = R.resolve_bag(case, recordings)
        run = R.run_case(case, root, sim)
        verdicts = R.judge(run)
        ok = R.case_passed(verdicts)
        failed |= not ok
        state = R.case_state(case, verdicts)
        print("\n".join(R.report_lines(case, verdicts)), flush=True)
        d = out / case.id
        d.mkdir(parents=True, exist_ok=True)
        _write_trace(d, run)
        entry = {"case": case.id, "state": state, "passed": ok, "support": case.support, "scoring": case.scoring, "counts": R.counts(verdicts), "window_s": list(case.window_s),
                 "wall_s": time.time() - t, "device": args.device, **run.meta,
                 "metrics": {v.metric: v.as_dict() for v in verdicts}}
        (d / "summary.json").write_text(json.dumps(entry, indent=1, default=_json_default))
        table[case.id] = {"state": state, "counts": entry["counts"]}
    (out / "regress.json").write_text(json.dumps(table, indent=1))
    print("\n" + "\n".join(f"{k}: {v['state']}  {v['counts']}" for k, v in table.items()))
    return 1 if failed else 0
