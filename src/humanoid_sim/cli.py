"""humanoid-sim command line.

    humanoid-sim [--solver mjwarp|kamino] <command>  Newton solver for the commands that step physics
                                                   (default: the profile's)
    humanoid-sim build-asset [--base-joints]       robot USD from the description (Isaac Lab image)
    humanoid-sim replay <exc-dir> [--period-ms 4] [--base yaw|orientation|fixed]
                                                   drive-level replay, scored against the capture;
                                                   base following the capture's IMU yaw (default), or fixed
    humanoid-sim view-replay <replay-dir> --save x.rrd   sim and real robot + every drive signal in Rerun
    humanoid-sim view-runtime <segment> [--save x.rrd]   watch a running `serve` in Rerun (read-only)
    humanoid-sim bench --num-envs N [--train]      real-time factor / throughput (Isaac Lab / Newton); --train runs
                                                   the Train-Time API scene (sensors, randomization); --runtime
                                                   the real-time host's step
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

    from humanoid_sim.core import PlantCore
    from humanoid_sim.model.description import Description
    from humanoid_sim.replay.coordinates import CaptureCoordinates, validate_capture_coordinates
    from humanoid_sim.replay.effort import effort_config, prepare_effort
    from humanoid_sim.replay.exc import ExcitationCapture
    from humanoid_sim.replay.gains import capture_solver_reference, gain_support
    from humanoid_sim.replay.inertial import acceleration_config
    from humanoid_sim.replay.replay import replay
    from humanoid_sim.sim import SimCore

    cfg = acceleration_config(args)
    effort_cfg = effort_config(args)
    desc = Description.load(args.robot)
    coordinate_path = getattr(args, "capture_coordinates", None)
    coordinates = CaptureCoordinates.load(coordinate_path, desc) if coordinate_path else None
    cap = ExcitationCapture(Path(args.capture), coordinates=coordinates)
    validate_capture_coordinates(cap, desc)
    solver_reference = capture_solver_reference([cap], desc) if args.solver_gain_reference == "capture" else None
    gain_support(cap, PlantCore(desc, list(desc.joints), 1, solver_gains=solver_reference))
    effort = None
    if effort_cfg is not None:
        effort = prepare_effort(cap, desc, list(desc.drives), effort_cfg)
    period_ns = int(round(args.period_ms * 1e6))
    moving = args.base != "fixed"
    sim = SimCore(_usd(args, base_joints=moving), device=args.device, robot=args.robot, period_ns=period_ns,
                  fixed_base=not moving, base_joints=moving, contacts=cfg is not None, solver=args.solver, integrator=args.integrator,
                  solver_gains=solver_reference)
    t = time.time()
    res = replay(cap, sim, duration_s=args.duration, base_motion=args.base if moving else None,
                 demand_delay_ns=int(round(args.demand_delay_ms * 1e6)), base_acceleration=cfg, effort=effort)
    wall = time.time() - t
    integrator = sim.integrator
    delay = f"_demand{args.demand_delay_ms:+g}ms" if args.demand_delay_ms else ""
    out = Path(args.out) / f"{cap.dataset_id}_{sim.solver}_{integrator}_{period_ns // 1000}us_base-{args.base}{delay}"
    if cfg is not None:
        out = out / f"accel-imu-w{cfg.window_ns}-g{cfg.max_gap_ns}"
    if effort_cfg is not None:
        out = out / f"effort-{effort_cfg.mode}"
    if coordinates is not None:
        out = out / f"coordinates-{coordinates.sha256[:12]}"
    if solver_reference is not None:
        out = out / "solver-capture"
    if cfg is not None or effort_cfg is not None or solver_reference is not None or coordinates is not None:
        if out.exists():
            raise ValueError(f"replay output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trace.npz", t_ns=res.t_ns, demand=res.demand,
                        **{f"sim_{k}": v for k, v in res.sim.items()}, **{f"real_{k}": v for k, v in res.real.items()},
                        **{f"motion_{k}": v for k, v in res.motion.items()},
                        **{f"effort_{k}": v for k, v in res.effort.items()})
    summary = {"dataset_id": cap.dataset_id, "name": cap.name, "period_ns": period_ns,
               "physics": f"isaaclab/{sim.solver}", "device": args.device, "integrator": integrator, "base": args.base,
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


def cmd_view_runtime(args) -> int:
    from humanoid_sim.runtime.view import view_runtime

    frames = view_runtime(args.segment, _usd(args), rrd=args.save, connect=args.connect, fps=args.fps,
                          seconds=args.seconds, wait_s=args.wait, robot=args.robot, draw=args.draw,
                          detail=args.detail)
    print(f"{frames} frames" + (f" -> {args.save}" if args.save else ""))
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

        print(json.dumps(run_runtime_bench(args, _usd(args), scene=_scene(args)), indent=1))
        return 0
    if args.train:
        from humanoid_sim.bench.train import run_train_bench

        print(json.dumps(run_train_bench(args, _usd(args), scene=_scene(args)), indent=1))
        return 0
    from humanoid_sim.bench.runner import run_bench

    print(json.dumps(run_bench(args, _usd(args), scene=_scene(args)), indent=1))
    return 0


# Joints tied left/right for the passive fit: the upper body, the worst-replayed part of the robot.


def _revision() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _scene(args):
    """The static boxes of ``--scene`` with ``--spawn`` at the world origin, or None."""
    if args.scene is None:
        if args.spawn is not None or getattr(args, "scan_mesh", None) is not None:
            raise SystemExit("--spawn and --scan-* go with a --scene")
        return None
    from humanoid_sim.world.boxes import StaticBoxes
    from humanoid_sim.world.scan import ScanFiles

    scan = getattr(args, "scan_mesh", None)
    if (scan is None) != (getattr(args, "scan_texture", None) is None):
        raise SystemExit("--scan-mesh and --scan-texture go together")
    if scan is None and getattr(args, "scan_floor_texture", None) is not None:
        raise SystemExit("--scan-floor-texture comes with --scan-mesh and --scan-texture")
    files = None if scan is None else ScanFiles.of(scan, args.scan_texture, args.scan_floor_texture)
    return StaticBoxes.load(args.scene, args.spawn or (0.0, 0.0), scan=files)


def _add_scene_args(parser) -> None:
    parser.add_argument("--scene", default=None, help="static boxes (walls, furniture) around the robot: an .npz as "
                        "world/boxes.py describes, such as Controls' hangar_proxy.npz")
    parser.add_argument("--spawn", nargs=2, type=float, default=None, metavar=("X", "Y"),
                        help="the scene point (m) the robot starts at, placed at the world origin (default 0 0)")


def _gantry(args):
    """The crane of ``--gantry``, placed with ``--spawn`` like the scene, or None."""
    if args.gantry is None:
        return None
    from humanoid_sim.world.trolley import GantryRig

    return GantryRig.load(args.gantry).placed(tuple(args.spawn or (0.0, 0.0)))


def cmd_serve(args) -> int:
    from humanoid_sim.runtime.host import EXIT_CLIENT_LOST, EXIT_DETACHED, RuntimeHost

    host = RuntimeHost(args.segment, _usd(args), device=args.device, robot=args.robot, solver=args.solver,
                       support=args.support, pose=args.pose, scene=_scene(args), gantry=_gantry(args),
                       bus_period_ns=int(args.bus_period_us * 1000),
                       client_timeout_ns=int(args.client_timeout_ms * 1_000_000), seed=args.seed,
                       sim_revision=_revision(), truth_every=args.truth_every)
    try:
        while True:
            print(json.dumps({"segment": str(args.segment), "session_id": host.session_id, "ready": True}),
                  flush=True)
            code = host.run(max_steps=args.max_steps)
            print(json.dumps(host.status() | {"exit": code}), flush=True)
            if not (args.persist and code in (EXIT_CLIENT_LOST, EXIT_DETACHED)):
                break
            host.new_session()
    finally:
        host.close()
    return 0 if code == EXIT_DETACHED else code


def cmd_render_runtime(args) -> int:
    from humanoid_sim.runtime.render import render_runtime
    from humanoid_sim.sensors.session import SessionCfg

    cfg = SessionCfg(robot=args.robot, sensors=args.sensors, splat=args.splat, splat_intensity=args.splat_intensity,
                     scan_mesh=args.scan_mesh, cameras=not args.no_cameras, lidar=not args.no_lidar,
                     feeds=args.feeds, lidar_domain=args.lidar_domain, rrd=args.rrd, images=args.images,
                     device=args.device, assets=args.assets)
    render_runtime(args.segment, cfg, seconds=args.seconds, wait_s=args.wait, summary=args.summary)
    return 0


def cmd_splat_usd(args) -> int:
    from humanoid_sim.world.splat import convert

    spawn = (0.0, 0.0) if args.spawn is None else tuple(args.spawn)
    print(json.dumps(convert(args.ply, args.out, spawn_m=spawn, sh_degree=args.sh_degree, half=args.half,
                             parts=args.parts), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="humanoid-sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="ethrc_v1")
    p.add_argument("--solver", default=None, help="Newton solver: mjwarp or kamino (default: the profile's)")
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
    r.add_argument("--device", default="cuda:0", help="cuda:N, or cpu (Warp's CPU device: slow)")
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--base", choices=("yaw", "orientation", "fixed"), default="yaw",
                   help="base yaw or full orientation from the capture's IMU (asset built with --base-joints), "
                        "or a fixed base")
    r.add_argument("--demand-delay-ms", type=float, default=0.0,
                   help="apply each recorded demand this much after its stamp (negative: before); fit delays with humanoid_learning.sysid")
    r.add_argument("--out", default="outputs/replay")
    from humanoid_sim.replay.inertial import add_acceleration_args

    add_acceleration_args(r)
    from humanoid_sim.replay.effort import add_effort_args

    add_effort_args(r)
    r.add_argument("--solver-gain-reference", choices=("nominal", "capture"), default="nominal")
    r.add_argument("--capture-coordinates", default=None, help="explicit sourced capture-to-description coordinate artifact")
    r.set_defaults(func=cmd_replay)

    v = sub.add_parser("view-replay", help="Rerun recording of a replay: sim and real robot, all drive signals")
    v.add_argument("trace", help="replay output directory (trace.npz + summary.json)")
    v.add_argument("--save", default=None, help="write an .rrd file instead of spawning a viewer")
    v.add_argument("--connect", default=None, help="stream to a running viewer, e.g. rerun+http://127.0.0.1:9876/proxy")
    v.add_argument("--fps", type=float, default=50.0)
    v.set_defaults(func=cmd_view_replay)

    vr = sub.add_parser("view-runtime", help="watch a running `serve` in Rerun: the robot and its signals (read-only)")
    vr.add_argument("segment", help="the segment file `serve --segment` publishes")
    vr.add_argument("--save", default=None, help="write an .rrd file instead of spawning a viewer")
    vr.add_argument("--connect", default=None, help="stream to a running viewer, e.g. rerun+http://127.0.0.1:9876/proxy")
    vr.add_argument("--fps", type=float, default=30.0, help="frames read from the segment per second")
    vr.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until the server goes)")
    vr.add_argument("--wait", type=float, default=60.0,
                    help="seconds to wait for the segment to appear (the next session of a --persist host is "
                         "followed if it starts within 10 s)")
    vr.add_argument("--draw", choices=("scan", "boxes"), default=None,
                    help="the host's scene as its textured scan or its boxes (default: the scan when it has one)")
    vr.add_argument("--detail", choices=("full", "live"), default=None,
                    help="the scan's detail (default: full with --save, live otherwise)")
    vr.set_defaults(func=cmd_view_runtime)

    rr_ = sub.add_parser("render-runtime", help="the rig's RTX cameras and lidars on a running `serve`, published as "
                         "Controls reads them (Isaac Sim image, RTX GPU; read-only)")
    rr_.add_argument("segment", help="the segment file `serve --segment` publishes")
    rr_.add_argument("--sensors", default=None, help="an extra rig whose sensors join the robot's (e.g. g1-cameras)")
    rr_.add_argument("--splat", default=None, help="the room's Gaussian splat as USD (`splat-usd`), for the cameras")
    rr_.add_argument("--splat-intensity", type=float, default=1.0, help="scale the splat's emission")
    rr_.add_argument("--scan-mesh", default=None,
                     help="the room's scan (OBJ) for the lidar (default: the host's, when readable here)")
    rr_.add_argument("--no-cameras", action="store_true")
    rr_.add_argument("--no-lidar", action="store_true")
    rr_.add_argument("--feeds", default=None, help="camera feed root (ethrc-camera-sim-feed/v1)")
    rr_.add_argument("--lidar-domain", type=int, default=None, help="publish the Mid-360 on Unitree's DDS domain")
    rr_.add_argument("--rrd", default=None)
    rr_.add_argument("--images", default=None, help="save every stream's first and last frame here (PNG)")
    rr_.add_argument("--summary", default=None, help="write the summary here (JSON)")
    rr_.add_argument("--device", default="cuda:0")
    rr_.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until the server goes)")
    rr_.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the segment to appear")
    rr_.set_defaults(func=cmd_render_runtime)

    be = sub.add_parser("bench")
    be.add_argument("--num-envs", type=int, default=1)
    be.add_argument("--steps", type=int, default=1000)
    be.add_argument("--device", default="cuda:0")
    be.add_argument("--integrator", default=None)
    be.add_argument("--no-self-collision", action="store_true")
    be.add_argument("--train", action="store_true", help="Train-Time API scene: RobotPlant, IMU, contacts")
    be.add_argument("--runtime", action="store_true", help="the real-time host's whole step, unpaced")
    _add_scene_args(be)
    be.set_defaults(func=cmd_bench)

    sv = sub.add_parser("serve", help="one robot in real time behind the runtime segment")
    sv.add_argument("--segment", required=True, help="segment file (on a volume shared with the plant host)")
    sv.add_argument("--support", choices=("suspended", "free"), default="suspended",
                    help="suspended hangs it from the lab gantry's harness cable; free stands it on the ground (gantry by command)")
    sv.add_argument("--pose", choices=("home", "standing"), default=None,
                    help="the pose every session starts in (default: the profile's poses.start)")
    sv.add_argument("--device", default="cuda:0")
    sv.add_argument("--bus-period-us", type=float, default=1500.0, help="the client's bus cycle (informational)")
    sv.add_argument("--client-timeout-ms", type=float, default=500.0,
                    help="end the session when an attached client's demands are this old")
    sv.add_argument("--truth-every", type=int, default=5, help="publish the truth every this many steps (and after "
                    "every command); 1 for test oracles that track maxima")
    sv.add_argument("--persist", action="store_true", help="when a client detaches or is lost, reset the robot "
                    "and serve the next client on a new session, until stopped")
    sv.add_argument("--max-steps", type=int, default=None, help="periods per session")
    sv.add_argument("--seed", type=int, default=0)
    _add_scene_args(sv)
    sv.add_argument("--scan-mesh", default=None, help="with --scene: the room's textured scan (an OBJ), for viewers "
                    "only; the descriptor names it and its digest")
    sv.add_argument("--scan-texture", default=None, help="the scan's texture (PNG)")
    sv.add_argument("--scan-floor-texture", default=None, help="the scan's floor seen from above (PNG), drawn over "
                    "the scene file's floor_corners")
    sv.add_argument("--gantry", default=None, help="a gantry crane over the scene (JSON, world/trolley.py's "
                    "GantryRig, in the scene file's frame): MOVE_GANTRY moves it, a hoist rope hangs from it")
    sv.set_defaults(func=cmd_serve)

    sp = sub.add_parser("splat-usd", help="convert a 3D Gaussian Splatting PLY (a scanned room) to a USD splat for "
                        "cameras to render, placed like the room's boxes")
    sp.add_argument("ply")
    sp.add_argument("out", help="the USD file to write (.usdc for a large splat)")
    sp.add_argument("--spawn", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="the scene point (m) placed at the world origin, as for --scene (default 0 0)")
    sp.add_argument("--parts", default=None, help="JSON: part names to boxes in the PLY's frame ({center, half, "
                    "yaw}); each part is written as a prim of its own that a renderer moves, 'drop' boxes are cut")
    sp.add_argument("--sh-degree", type=int, default=None, help="keep spherical harmonics up to this degree "
                    "(default: all the PLY has; 0 keeps each Gaussian's base colour only)")
    sp.add_argument("--half", action="store_true", help="store the coefficients as half floats")
    sp.set_defaults(func=cmd_splat_usd)

    args = p.parse_args(argv)
    if args.command is None:
        p.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
