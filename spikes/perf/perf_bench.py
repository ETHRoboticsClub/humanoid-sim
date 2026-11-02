"""Training-scene throughput with per-phase timing and an optional kernel profile.

Same workload as ``humanoid-sim bench --train`` (humanoid_sim.bench.train.run_train_bench), plus:
  --no-cuda-graph       Isaac Lab's Newton step and the plant run eagerly
  --phases              per-phase GPU time (CUDA events) and host time for pre_physics / write / step / update
  --profile N           torch.profiler over N physics steps after timing: kernel table and a chrome trace
  --overrides JSON      extra profile physics overrides merged after --period-ms / --integrator
  --contact-gap M       every shape's contact gap after the reset; --broad-phase explicit|sap|nxn
  --record NPZ          instead of timing: joint and base state every policy step (with --excite AMP: every drive
                        demand on a sine of AMP rad, seeded), for traj_compare.py; also the peak contact and
                        constraint-row occupancy
Writes one JSON object to --out (and stdout), with collision statistics (pairs, candidates, contacts and their
distances, self and ground).

    sky launch -y -d -c sim-perf -i 10 spikes/perf/perf.sky.yaml --env MATRIX=spikes/perf/baseline.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def collision_stats(manager) -> dict:
    """Shapes, explicit pairs, broad-phase candidates and contacts, split into self and ground pairs."""
    import numpy as np
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    pipe = NewtonManager._collision_pipeline
    contacts = NewtonManager.get_contacts()
    world = model.shape_world.numpy()
    body = model.shape_body.numpy()
    nworld = int(model.world_count) if hasattr(model, "world_count") else int(world.max() + 1)
    out = {"shapes_total": int(len(world)), "shapes_world0": int((world == 0).sum()),
           "broad_phase": getattr(pipe, "broad_phase_mode", None)}

    def split(pairs):
        pairs = np.asarray(pairs).reshape(-1, 2)
        ground = (world[pairs[:, 0]] < 0) | (world[pairs[:, 1]] < 0)
        return {"total": int(len(pairs)), "per_world_self": float((~ground).sum() / nworld),
                "per_world_ground": float(ground.sum() / nworld)}

    if getattr(pipe, "shape_pairs_filtered", None) is not None:
        out["explicit_pairs"] = split(pipe.shape_pairs_filtered.numpy())
    if hasattr(pipe, "broad_phase_pair_count"):
        n = int(pipe.broad_phase_pair_count.numpy()[0])
        out["candidate_pairs"] = split(pipe.broad_phase_shape_pairs.numpy()[:n])
    if contacts is not None:
        n = int(contacts.rigid_contact_count.numpy()[0])
        out["contacts"] = split(np.stack([contacts.rigid_contact_shape0.numpy()[:n],
                                          contacts.rigid_contact_shape1.numpy()[:n]], -1))
        out["rigid_contact_max"] = int(contacts.rigid_contact_max)
        s0, s1 = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
        q = NewtonManager.get_state_0().body_q.numpy()

        def world_point(shape, p):
            b = body[shape]
            out_p = p.copy()
            ok = b >= 0
            t, r = q[b[ok], :3], q[b[ok], 3:]  # xyzw
            v = p[ok]
            u = r[:, :3]
            w = r[:, 3:4]
            out_p[ok] = t + v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))
            return out_p

        a = world_point(s0, contacts.rigid_contact_point0.numpy()[:n])
        b = world_point(s1, contacts.rigid_contact_point1.numpy()[:n])
        nrm = contacts.rigid_contact_normal.numpy()[:n]
        dist = ((b - a) * nrm).sum(-1) - contacts.rigid_contact_margin0.numpy()[:n] - contacts.rigid_contact_margin1.numpy()[:n]
        ground = (world[s0] < 0) | (world[s1] < 0)
        for name, sel in (("self", ~ground), ("ground", ground)):
            d = dist[sel]
            if len(d):
                out[f"contact_dist_{name}_m"] = {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max()),
                                                "fraction_speculative": float((d > 0).mean())}
    types = model.shape_type.numpy()[world == 0]
    out["shape_types_world0"] = {str(t): int((types == t).sum()) for t in np.unique(types)}
    out["bodies_world0"] = int(len(np.unique(body[world == 0])))
    for name in ("shape_gap", "shape_margin"):
        values = getattr(model, name).numpy()
        out[name] = {"world0": np.unique(np.round(values[world == 0], 5)).tolist()[:10],
                     "static": np.unique(np.round(values[world < 0], 5)).tolist()[:10]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="ethrc_v1")
    p.add_argument("--solver", default=None)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--integrator", default=None)
    p.add_argument("--period-ms", type=float, default=None)
    p.add_argument("--no-self-collision", action="store_true")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--no-plant-graph", action="store_true")
    p.add_argument("--phases", action="store_true")
    p.add_argument("--profile", type=int, default=0)
    p.add_argument("--overrides", default=None)
    p.add_argument("--contact-gap", type=float, default=None, help="set every shape's contact gap [m]")
    p.add_argument("--record", default=None, help="npz of joint and base state every policy step")
    p.add_argument("--excite", type=float, default=0.0, help="sine amplitude [rad] on every drive demand")
    p.add_argument("--broad-phase", default=None, help="explicit, sap or nxn")
    p.add_argument("--assets", default="outputs/assets")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    import humanoid_sim.physics.isaaclab as hpi
    from humanoid_sim.bench.train import POLICY_PERIOD_NS, physics_overrides, randomization
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.model.asset import _merged, asset_dir, load_profile
    from humanoid_sim.model.description import Description
    from humanoid_sim.timing.clock import NS_PER_S
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg
    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg

    if args.no_cuda_graph or args.broad_phase:
        orig = hpi.simulation_cfg

        def no_graph(*a, **k):
            cfg = orig(*a, **k)
            if args.no_cuda_graph:
                cfg.physics.use_cuda_graph = False
            if args.broad_phase:
                from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
                cfg.physics.collision_cfg = NewtonCollisionPipelineCfg(broad_phase=args.broad_phase)
            return cfg
        hpi.simulation_cfg = no_graph

    overrides = physics_overrides(args)
    if args.overrides:
        overrides = _merged(overrides, json.loads(args.overrides))
    profile = load_profile(args.robot, physics_overrides=overrides)
    adir = asset_dir(Description.load(args.robot), load_profile(args.robot), Path(args.assets))
    usd = adir / json.loads((adir / "asset.json").read_text())["usd"]
    n = args.num_envs
    sim = sim_utils.SimulationContext(robot_sim_cfg(args.robot, device=args.device, solver=args.solver,
                                                    physics_overrides=overrides))
    with robot_build(args.robot, solver=args.solver, physics_overrides=overrides):
        scene = instantiate(robot_scene_cfg(n, usd, robot=args.robot, ground_friction=0.0, pose="home",
                                            self_collision=False if args.no_self_collision else None,
                                            physics_overrides=overrides))
        sim.reset()
    robot = scene["robot"]
    if args.contact_gap is not None:
        from isaaclab_newton.physics import NewtonManager
        model = NewtonManager.get_model()
        model.shape_gap.fill_(args.contact_gap)
    graph = not args.no_plant_graph
    heaviest = robot.body_names[int(robot.data.body_mass.torch[0].argmax())]
    plant = RobotPlant(robot, TrainPlantCfg(robot=args.robot, randomization=randomization(Description.load(args.robot), heaviest),
                                            physics_overrides=overrides, cuda_graph=graph, compile=graph),
                       imu=scene["imu"], contacts=scene["contacts"], physics_manager=sim.physics_manager)
    q, qd = plant.joint_state()
    q = q.clone()
    stand, _, _ = plant.core.map.drive_state(q, qd)
    stiff, damp = plant.core.drive_stiffness.expand(n, -1), plant.core.drive_damping.expand(n, -1)
    every = POLICY_PERIOD_NS // plant.period_ns
    period_s = plant.period_ns / NS_PER_S
    gen = torch.Generator(device=q.device).manual_seed(0)
    resets = max(1, n // 50)
    sink = torch.zeros((), device=q.device)
    peak_spin = torch.zeros((), device=q.device)
    names = ("policy", "pre_physics", "write", "step", "update")
    events = []
    host = {k: 0.0 for k in names}

    phase0 = 2 * torch.pi * torch.rand(stand.shape, generator=torch.Generator(device=q.device).manual_seed(1),
                                       device=q.device)
    freq = 0.5 + 2.0 * torch.rand(stand.shape, generator=torch.Generator(device=q.device).manual_seed(2), device=q.device)
    record = []
    occupancy = {}

    def track_occupancy():
        import warp as wp
        from isaaclab_newton.physics import NewtonManager
        c = NewtonManager.get_contacts()
        if c is not None:
            v = wp.to_torch(c.rigid_contact_count)[0].float()
            occupancy["newton_contacts_total"] = torch.maximum(occupancy.get("newton_contacts_total", v), v)
        mjd = getattr(NewtonManager._solver, "mjw_data", None)
        if mjd is not None:
            v = wp.to_torch(mjd.nacon)[0].float()
            occupancy["mjwarp_nacon_total"] = torch.maximum(occupancy.get("mjwarp_nacon_total", v), v)
            v = wp.to_torch(mjd.nefc).max().float()
            occupancy["mjwarp_nefc_max_per_world"] = torch.maximum(occupancy.get("mjwarp_nefc_max_per_world", v), v)

    def policy_step():
        nonlocal sink
        if args.record:
            data = robot.data
            record.append(torch.cat([plant.joint_state()[0], data.root_link_pose_w.torch[:, :3] - scene.env_origins,
                                     data.root_link_pose_w.torch[:, 3:]], -1).cpu())
        m, t = plant.measured(), plant.truth()
        sink = sink + m.drive.position_rad.sum() + m.imu.gyro[:, 0].sum() + t.contact_forces_w[..., 2].sum()
        ids = torch.randint(0, n, (resets,), generator=gen, device=q.device)
        root = robot.data.default_root_pose.torch[ids].clone()
        root[:, :3] += scene.env_origins[ids]
        plant.reset(ids, joint_pos=q[ids], joint_vel=torch.zeros_like(q[ids]), root_pose=root,
                    root_vel=torch.zeros(resets, 6, device=q.device))
        sent = torch.full((n,), plant.t_ns, dtype=torch.int64, device=q.device)
        target = stand
        if args.excite:
            target = stand + args.excite * torch.sin(2 * torch.pi * freq * (plant.t_ns / 1e9) + phase0)
        plant.submit(DriveDemand.pd(target, stiff, damp), sent)

    def phase(name, fn, timed):
        if not timed:
            fn()
            return
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        h = time.perf_counter()
        e0.record()
        fn()
        e1.record()
        host[name] += time.perf_counter() - h
        events.append((name, e0, e1))

    def run(steps, timed=False):
        nonlocal peak_spin
        for k in range(steps):
            if k % every == 0:
                phase("policy", policy_step, timed)
            phase("pre_physics", plant.pre_physics, timed)
            phase("write", scene.write_data_to_sim, timed)
            phase("step", sim.step, timed)
            phase("update", lambda: scene.update(period_s), timed)
            if args.record:
                track_occupancy()
            peak_spin = torch.maximum(peak_spin, robot.data.body_link_vel_w.torch[..., 3:].norm(dim=-1).max())

    if args.record:
        run(args.steps)
        np_out = {"state": torch.stack(record).numpy(), "period_ns": plant.period_ns,
                  "joints": list(plant.joints)}
        import numpy as np
        np.savez_compressed(args.record, **np_out)
        occ = {k: v.item() for k, v in occupancy.items()}
        from isaaclab_newton.physics import NewtonManager
        mjd = getattr(NewtonManager._solver, "mjw_data", None)
        if mjd is not None:
            occ |= {"mjwarp_naconmax": int(mjd.naconmax), "mjwarp_njmax": int(mjd.njmax)}
        height = robot.data.root_link_pose_w.torch[:, 2] - scene.env_origins[:, 2]
        occ["fraction_envs_up_end"] = height.gt(0.75 * robot.data.default_root_pose.torch[:, 2]).float().mean().item()
        Path(args.out).write_text(json.dumps({"recorded": args.record, "policy_steps": len(record), "occupancy": occ,
                                             "peak_body_ang_vel_rad_s": peak_spin.item(), "num_envs": n}))
        print("recorded", args.record, len(record))
        return
    run(60)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    t = time.perf_counter()
    sync_error = None
    try:
        run(args.steps)
    except RuntimeError as e:
        sync_error = str(e).splitlines()[0]
    torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    out = {"robot": args.robot, "num_envs": n, "steps": args.steps, "solver": args.solver or profile["physics"]["solver"],
           "period_ns": plant.period_ns, "self_collision": not args.no_self_collision,
           "cuda_graph": not args.no_cuda_graph, "contact_gap": args.contact_gap, "broad_phase": args.broad_phase, "plant_graph": graph, "overrides": overrides,
           "gpu": torch.cuda.get_device_name(), "wall_s": wall, "sync_error": sync_error,
           "ms_per_physics_step": 1e3 * wall / args.steps, "env_steps_per_s": n * args.steps / wall,
           "sim_seconds_per_wall_second": n * args.steps * period_s / wall}
    solver = out["solver"]
    out["integrator"] = profile["physics"][solver].get("integrator")
    if args.phases:
        run(args.steps, timed=True)
        torch.cuda.synchronize()
        gpu = {k: 0.0 for k in names}
        for name, e0, e1 in events:
            gpu[name] += e0.elapsed_time(e1)
        out["phase_gpu_ms_per_step"] = {k: v / args.steps for k, v in gpu.items()}
        out["phase_host_ms_per_step"] = {k: 1e3 * v / args.steps for k, v in host.items()}
    if args.profile:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as tprof

        with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(args.profile)
            torch.cuda.synchronize()
        base = Path(args.out).with_suffix("")
        prof.export_chrome_trace(str(base) + ".trace.json")
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=80, max_name_column_width=90)
        Path(str(base) + ".kernels.txt").write_text(table)
        rows = []
        for ev in prof.key_averages():
            dt = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            if dt and ev.device_type.name == "CUDA":
                rows.append((ev.key, dt / 1e3 / args.profile, ev.count / args.profile))
        rows.sort(key=lambda r: -r[1])
        out["top_kernels_ms_per_step"] = [[k[:100], round(ms, 4), round(c, 2)] for k, ms, c in rows[:40]]
        out["kernel_ms_per_step_total"] = sum(r[1] for r in rows)
    try:
        out["collision"] = collision_stats(sim.physics_manager)
    except Exception as e:  # diagnostics only
        out["collision"] = {"error": repr(e)}
    height = robot.data.root_link_pose_w.torch[:, 2] - scene.env_origins[:, 2]
    out["fraction_envs_up"] = height.gt(0.75 * robot.data.default_root_pose.torch[:, 2]).float().mean().item()
    out["checksum_finite"] = bool(torch.isfinite(sink).item())
    out["peak_body_ang_vel_rad_s"] = peak_spin.item()
    text = json.dumps(out, indent=1)
    Path(args.out).write_text(text)
    print(text)
    sim_utils.SimulationContext.clear_instance()


if __name__ == "__main__":
    main()
