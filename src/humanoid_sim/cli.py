"""humanoid-sim command line.

    humanoid-sim build-asset [--base-joints]       robot USD from the description (Isaac Lab image)
    humanoid-sim replay <exc-dir> [--period-ms 4] [--base yaw|orientation|fixed]
                                                   drive-level replay on MJWarp, scored against the capture;
                                                   base following the capture's IMU yaw (default), or fixed
    humanoid-sim view-replay <replay-dir> --save x.rrd   sim and real robot + every drive signal in Rerun
    humanoid-sim bench --num-envs N [--train]      real-time factor / throughput (Isaac Lab / MJWarp); --train runs
                                                   the Train-Time API scene (sensors, randomization); --runtime
                                                   the real-time host's step
    humanoid-sim fit-passive <exc-dir>... --num-envs N --generations G
                                                   fit upper-body passive joint parameters by batched replay
    humanoid-sim serve --segment PATH [--support suspended|free]
                                                   one robot in real time behind the runtime segment, for
                                                   Controls' plant host
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_ASSETS = Path("outputs/assets")


def _usd(args, *, base_joints: bool = False) -> Path:
    if args.usd:
        return Path(args.usd)
    from humanoid_sim.model.asset import asset_dir, load_profile
    from humanoid_sim.model.description import Description

    out = asset_dir(Description.load(args.robot), load_profile(args.robot), Path(args.assets),
                    base_joints=base_joints)
    meta = out / "asset.json"
    if not meta.exists():
        flag = " --base-joints" if base_joints else ""
        raise SystemExit(f"asset {out} is not built for this description and profile; "
                         f"run `humanoid-sim build-asset{flag}` in the Isaac Lab image")
    return out / json.loads(meta.read_text())["usd"]


def cmd_build_asset(args) -> int:
    from humanoid_sim.model.asset import build_asset, load_profile
    from humanoid_sim.model.description import Description

    usd = build_asset(Description.load(args.robot), load_profile(args.robot), Path(args.assets), force=args.force,
                      base_joints=args.base_joints)
    print(usd)
    return 0


def cmd_replay(args) -> int:
    import numpy as np

    from humanoid_sim.replay.exc import ExcitationCapture
    from humanoid_sim.replay.replay import replay
    from humanoid_sim.sim import SimCore

    cap = ExcitationCapture(Path(args.capture))
    period_ns = int(round(args.period_ms * 1e6))
    moving = args.base != "fixed"
    sim = SimCore(_usd(args, base_joints=moving), device=args.device, robot=args.robot, period_ns=period_ns,
                  fixed_base=not moving, base_joints=moving, integrator=args.integrator)
    t = time.time()
    res = replay(cap, sim, duration_s=args.duration, base_motion=args.base if moving else None)
    wall = time.time() - t
    integrator = args.integrator or sim.profile["physics"]["integrator"]
    out = Path(args.out) / f"{cap.dataset_id}_mjwarp_{integrator}_{period_ns // 1000}us_base-{args.base}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trace.npz", t_ns=res.t_ns, demand=res.demand,
                        **{f"sim_{k}": v for k, v in res.sim.items()}, **{f"real_{k}": v for k, v in res.real.items()})
    summary = {"dataset_id": cap.dataset_id, "name": cap.name, "period_ns": period_ns,
               "physics": "isaaclab/mjwarp", "device": args.device, "integrator": integrator, "base": args.base,
               "wall_s": wall,
               "sim_s": len(res.t_ns) * period_ns / 1e9, "drives": res.drives,
               "rms_position_rad": res.rms("position"), "rms_velocity_rad_s": res.rms("velocity"),
               "rms_torque_nm": res.rms("torque"), "tracking_rms_rad": res.tracking_rms(), **res.meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("dataset_id", "period_ns", "wall_s", "sim_s")}))
    for d in res.drives:
        print(f"{d:22s} {summary['rms_position_rad'][d] * 1e3:7.2f} mrad  (real-demand "
              f"{summary['tracking_rms_rad'][d] * 1e3:6.2f})")
    return 0


def cmd_view_replay(args) -> int:
    from humanoid_sim.replay.view import view_replay

    view_replay(Path(args.trace), _usd(args), args.save, fps=args.fps, robot=args.robot, connect=args.connect)
    if args.save:
        print(args.save)
    return 0


def cmd_bench(args) -> int:
    if args.runtime:
        from humanoid_sim.bench.runner import run_runtime_bench

        print(json.dumps(run_runtime_bench(args, _usd(args)), indent=1))
        return 0
    if args.train:
        from humanoid_sim.bench.train import run_train_bench

        print(json.dumps(run_train_bench(args, _usd(args)), indent=1))
        return 0
    from humanoid_sim.bench.runner import run_bench

    print(json.dumps(run_bench(args, _usd(args)), indent=1))
    return 0


# Joints tied left/right for the passive fit: the upper body, the worst-replayed part of the robot.
UPPER_BODY_GROUPS = {
    "waist": ("waist_upperbody_yaw",),
    "shoulder_flexion": ("shoulder_flexion_l", "shoulder_flexion_r"),
    "shoulder_abduction": ("shoulder_abduction_l", "shoulder_abduction_r"),
    "shoulder_rotation": ("shoulder_rotation_l", "shoulder_rotation_r"),
    "elbow": ("elbow_flexion_l", "elbow_flexion_r"),
}


def cmd_fit_passive(args) -> int:
    import math

    import numpy as np

    from humanoid_sim.replay.exc import ExcitationCapture
    from humanoid_sim.sim import SimCore
    from humanoid_sim.sysid.passive import FitSpace, Group, evaluate, fit

    moving = args.base != "fixed"
    sim = SimCore(_usd(args, base_joints=moving), device=args.device, robot=args.robot, num_envs=args.num_envs,
                  fixed_base=not moving, base_joints=moving)
    space = FitSpace([Group(name, joints) for name, joints in UPPER_BODY_GROUPS.items()],
                     log_bound=math.log(args.bound))
    drives = sim.plant.map.drive_names
    scored = sorted({drives.index(sim.desc.joints[j].device) for g in space.groups for j in g.joints})
    captures = [ExcitationCapture(Path(c)) for c in args.captures]
    base = args.base if moving else None
    t = time.time()
    result = fit(sim, captures, space, scored, generations=args.generations, base_motion=base,
                 duration_s=args.duration, seed=args.seed)
    result |= {"captures": [c.dataset_id for c in captures], "base": args.base, "num_envs": args.num_envs,
               "bound": args.bound,
               "wall_s": time.time() - t, "drives": drives, "scored": [drives[i] for i in scored],
               **sim.desc.identity()}
    m = dict(zip(space.names, result["multipliers"], strict=True))
    result["values"] = {j: {p: getattr(sim.desc.joints[j], p) * m[f"{g.name}.{p}"] for p in space.params}
                        for g in space.groups for j in g.joints}
    if args.holdout:
        x = np.zeros((args.num_envs, space.dim))
        x[1:] = result["x"]
        held = [ExcitationCapture(Path(c)) for c in args.holdout]
        f, per = evaluate(sim, held, space, x, scored, base_motion=base, duration_s=None)
        result["holdout"] = {"captures": [c.dataset_id for c in held], "objective_description": float(f[0]),
                             "objective_fit": float(f[1]), "rms_description": per[:, 0].tolist(),
                             "rms_fit": per[:, 1].tolist()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit-passive.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("objective_description", "objective", "multipliers", "names")}))
    if args.holdout:
        print(json.dumps(result["holdout"]["objective_description"]), json.dumps(result["holdout"]["objective_fit"]))
    return 0


def _revision() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_serve(args) -> int:
    from humanoid_sim.runtime.host import RuntimeHost

    host = RuntimeHost(args.segment, _usd(args), device=args.device, robot=args.robot, support=args.support,
                       bus_period_ns=int(args.bus_period_us * 1000),
                       client_timeout_ns=int(args.client_timeout_ms * 1_000_000), seed=args.seed,
                       sim_revision=_revision())
    print(json.dumps({"segment": str(args.segment), "session_id": host.session_id, "ready": True}), flush=True)
    try:
        code = host.run(max_steps=args.max_steps)
    finally:
        status = host.status()
        host.close()
    print(json.dumps(status | {"exit": code}), flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="humanoid-sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="ethrc_v1")
    p.add_argument("--assets", default=str(DEFAULT_ASSETS))
    p.add_argument("--usd", default=None, help="robot USD (default: the asset built for this description and profile)")
    sub = p.add_subparsers(dest="command")

    b = sub.add_parser("build-asset")
    b.add_argument("--force", action="store_true")
    b.add_argument("--base-joints", action="store_true",
                   help="the variant with yaw/pitch/roll joints above the base, for replays with a moving base")
    b.set_defaults(func=cmd_build_asset)

    r = sub.add_parser("replay")
    r.add_argument("capture")
    r.add_argument("--period-ms", type=float, default=4.0)
    r.add_argument("--integrator", default=None)
    r.add_argument("--device", default="cuda:0", help="cuda:N, or cpu (MJWarp on Warp's CPU device: slow)")
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--base", choices=("yaw", "orientation", "fixed"), default="yaw",
                   help="base yaw or full orientation from the capture's IMU (asset built with --base-joints), "
                        "or a fixed base")
    r.add_argument("--out", default="outputs/replay")
    r.set_defaults(func=cmd_replay)

    v = sub.add_parser("view-replay", help="Rerun recording of a replay: sim and real robot, all drive signals")
    v.add_argument("trace", help="replay output directory (trace.npz + summary.json)")
    v.add_argument("--save", default=None, help="write an .rrd file instead of streaming")
    v.add_argument("--connect", default=None, help="stream to a running viewer, e.g. rerun+http://127.0.0.1:9876/proxy")
    v.add_argument("--fps", type=float, default=50.0)
    v.set_defaults(func=cmd_view_replay)

    be = sub.add_parser("bench")
    be.add_argument("--num-envs", type=int, default=1)
    be.add_argument("--steps", type=int, default=1000)
    be.add_argument("--device", default="cuda:0")
    be.add_argument("--integrator", default=None)
    be.add_argument("--no-self-collision", action="store_true")
    be.add_argument("--train", action="store_true", help="Train-Time API scene: RobotPlant, IMU, contacts")
    be.add_argument("--runtime", action="store_true", help="the real-time host's whole step, unpaced")
    be.set_defaults(func=cmd_bench)

    fp = sub.add_parser("fit-passive", help="fit upper-body passive joint parameters to captures")
    fp.add_argument("captures", nargs="+")
    fp.add_argument("--holdout", nargs="*", default=[],
                    help="captures to score the fit on; use a different excitation profile than the fitted ones")
    fp.add_argument("--num-envs", type=int, default=256, help="CMA-ES population + 1 (env 0 keeps the best)")
    fp.add_argument("--generations", type=int, default=15)
    fp.add_argument("--duration", type=float, default=None, help="replay only the first seconds of each capture")
    fp.add_argument("--base", choices=("yaw", "orientation", "fixed"), default="yaw")
    fp.add_argument("--device", default="cuda:0")
    fp.add_argument("--seed", type=int, default=0)
    fp.add_argument("--bound", type=float, default=10.0, help="multipliers stay within [1/bound, bound]")
    fp.add_argument("--out", default="outputs/sysid")
    fp.set_defaults(func=cmd_fit_passive)

    sv = sub.add_parser("serve", help="one robot in real time behind the runtime segment")
    sv.add_argument("--segment", required=True, help="segment file (on a volume shared with the plant host)")
    sv.add_argument("--support", choices=("suspended", "free"), default="suspended",
                    help="suspended welds the base to the world; free stands it on the ground (gantry by command)")
    sv.add_argument("--device", default="cuda:0")
    sv.add_argument("--bus-period-us", type=float, default=1500.0, help="the client's bus cycle (informational)")
    sv.add_argument("--client-timeout-ms", type=float, default=500.0,
                    help="end the session when an attached client's demands are this old")
    sv.add_argument("--max-steps", type=int, default=None)
    sv.add_argument("--seed", type=int, default=0)
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
