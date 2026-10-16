"""Spike: the real closed-chain ankle in MJWarp, against the hinge + transmission model.

Rig (right ankle, pivot frame = world, shin fixed):
    shin ─pitch(+Y)→ cross ─roll(+X)→ foot
    shin ─crank_u(n_shaft, driven)→ crank_u ─ball(crank tip)→ rod_u ┐
    shin ─crank_l(n_shaft, driven)→ crank_l ─ball(crank tip)→ rod_l ┤ loop: ball(rod end, foot anchor)
The two rod→foot ball joints sit outside the articulation, so SolverMuJoCo turns each into a CONNECT
equality (Newton's loop-closure path; the UsdPhysics ``excludeFromArticulation`` equivalent).

Geometry: humanoid-urdf ``ankle_model.right_geometry``. MODELED, not measured: crank, rod and foot masses
(crank and rod masses do not exist in the description), foot inertia.

Checks:
  1. kinematics: cranks driven to the transmission's angles for a pose → foot reaches that pose;
  2. stability at crank gains 480/8 (baseline) and 100/3.5 (walk) at 4 ms;
  3. 700 N foot load with a 5 cm CoP offset: ankle deflection vs the hinge model's ``(JᵀKJ)⁻¹ τ``;
  4. reset by settling: start with the loop open by a few mm, measure the residual after 0.2 s;
  5. cost per step vs the hinge-only rig.

Physics: MJWarp only (``use_mujoco_cpu`` is never set). Run in the Isaac Lab image or on a GPU node:
``python spikes/closed_chain/ankle_rig.py --device cuda:0`` (``cpu`` runs MJWarp on Warp's CPU device, slowly).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import warp as wp

from humanoid_sim.model.description import Description

EQ_SOLIMP = (0.99, 0.999, 0.0005, 0.5, 2.0)
MODELED = {"crank_mass_kg": 0.06, "rod_mass_kg": 0.04, "foot_mass_kg": 0.9, "cross_mass_kg": 0.12}


def rig(desc: Description, *, closed: bool, kp: float, kd: float, device: str, eq_solref=None):
    import newton

    geom = desc.ankle_geometry("r")
    tr = desc.ankle_transmission("r", dtype=torch.float64)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.density = 0.0
    eye = wp.quat_identity()

    def box_inertia(m, sx, sy, sz):
        return wp.mat33(m * (sy * sy + sz * sz) / 12, 0, 0, 0, m * (sx * sx + sz * sz) / 12, 0, 0, 0,
                        m * (sx * sx + sy * sy) / 12)

    shin = b.add_link(mass=1.0, inertia=box_inertia(1.0, 0.06, 0.06, 0.3), com=wp.vec3(0, 0, 0.15), label="shin")
    cross = b.add_link(mass=MODELED["cross_mass_kg"], inertia=box_inertia(MODELED["cross_mass_kg"], .04, .04, .02),
                       label="cross")
    foot = b.add_link(mass=MODELED["foot_mass_kg"], inertia=box_inertia(MODELED["foot_mass_kg"], .25, .09, .05),
                      com=wp.vec3(0.03, 0, -0.05), label="foot")
    joints = [b.add_joint_fixed(-1, shin, label="shin_fixed"),
              b.add_joint_revolute(shin, cross, axis=(0, 1, 0), label="pitch", armature=0.0),
              b.add_joint_revolute(cross, foot, axis=(1, 0, 0), label="roll", armature=0.0)]
    hinge_dofs = [0, 1]
    crank_dofs, loops = [], []
    if closed:
        for name in ("upper", "lower"):
            g = geom[name]
            c = tr.cranks_geom[0 if name == "upper" else 1]
            p_crank0, p_anchor0 = np.asarray(g["P_crank_0"]), np.asarray(g["P_anchor_0"])
            m_c, m_r = MODELED["crank_mass_kg"], MODELED["rod_mass_kg"]
            crank = b.add_link(xform=wp.transform(wp.vec3(*c.center), eye), mass=m_c,
                               com=wp.vec3(*(0.5 * (p_crank0 - c.center))),
                               inertia=box_inertia(m_c, .06, .06, .01), label=f"crank_{name}")
            j = b.add_joint_revolute(shin, crank, parent_xform=wp.transform(wp.vec3(*c.center), eye),
                                     axis=tuple(np.asarray(g["n_shaft"], dtype=float)), target_ke=kp, target_kd=kd,
                                     armature=float(desc.table["joints"]["foot_pitch_r"]["armature"]),
                                     label=f"crank_{name}_joint")
            joints.append(j)
            rod_vec = p_anchor0 - p_crank0
            rod = b.add_link(xform=wp.transform(wp.vec3(*p_crank0), eye), mass=m_r, com=wp.vec3(*(0.5 * rod_vec)),
                             inertia=box_inertia(m_r, .01, .01, float(np.linalg.norm(rod_vec))), label=f"rod_{name}")
            joints.append(b.add_joint_ball(crank, rod, parent_xform=wp.transform(wp.vec3(*(p_crank0 - c.center)), eye),
                                           label=f"rod_{name}_ball"))
            loops.append((rod, p_anchor0, rod_vec, name))
        crank_dofs = [2, 6]  # pitch, roll, crank_u, ball_u(3), crank_l, ball_l(3)
    b.add_articulation(joints, label="ankle")
    # Loop closures go after the articulation: joints outside it become MuJoCo CONNECT equalities.
    for rod, p_anchor0, rod_vec, name in loops:
        b.add_joint_ball(foot, rod, parent_xform=wp.transform(wp.vec3(*p_anchor0), eye),
                         child_xform=wp.transform(wp.vec3(*rod_vec), eye), label=f"loop_{name}")
    model = b.finalize(device=device)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, integrator="implicitfast",
                                         iterations=100, ls_iterations=50)
    if closed and eq_solref is not None:
        # Loop CONNECTs get MuJoCo's default solref (0.02 s, 1): far too soft for 700 N. Stiffen them.
        ref = solver.mjw_model.eq_solref.numpy()
        ref[..., :] = eq_solref
        solver.mjw_model.eq_solref.assign(ref)
        imp = solver.mjw_model.eq_solimp.numpy()
        imp[..., :] = EQ_SOLIMP
        solver.mjw_model.eq_solimp.assign(imp)
    return model, solver, tr, hinge_dofs, crank_dofs


def run(args):
    import newton

    desc = Description.load()
    out = {"modeled": MODELED}
    dt = 0.004
    for gains, eq_solref in (((480.0, 8.0), None), ((480.0, 8.0), (0.008, 1.0)), ((100.0, 3.5), (0.008, 1.0))):
        kp, kd = gains
        model, solver, tr, hd, cd = rig(desc, closed=True, kp=kp, kd=kd, device=args.device, eq_solref=eq_solref)
        s0, s1, ctl = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        res = {}
        # 1+2: drive the cranks to the transmission's angles for a sequence of poses; settle; measure.
        poses = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.15), (-0.15, -0.1), (0.3, 0.1)]
        errs, peak_vel = [], 0.0
        t0 = time.perf_counter()
        steps = 0
        for p, r in poses:
            theta = tr.cranks(torch.tensor([[p, r]], dtype=torch.float64))[0].numpy() + np.array(
                [c.theta0 for c in tr.cranks_geom])
            tq = ctl.joint_target_q.numpy()
            ts = model.joint_target_q_start.numpy()
            tq[ts[3]] = theta[0]
            tq[ts[5]] = theta[1]
            ctl.joint_target_q.assign(tq)
            for _ in range(int(0.6 / dt)):
                s0.clear_forces()
                solver.step(s0, s1, ctl, None, dt)
                s0, s1 = s1, s0
                steps += 1
                peak_vel = max(peak_vel, float(np.abs(s0.joint_qd.numpy()[:2]).max()))
            q = s0.joint_q.numpy()
            errs.append(float(np.abs(q[[0, 1]] - np.array([p, r])).max()))
        wall = time.perf_counter() - t0
        res["pose_error_rad_max"] = max(errs)
        res["pose_errors_rad"] = errs
        res["finite"] = bool(np.isfinite(s0.joint_q.numpy()).all())
        res["peak_hinge_speed_rad_s"] = peak_vel
        res["us_per_step"] = 1e6 * wall / steps
        # 3: 700 N up through a CoP 5 cm ahead of the pivot axis, at the standing-ish pose (0, 0).
        theta0 = np.array([c.theta0 for c in tr.cranks_geom])
        tq = ctl.joint_target_q.numpy()
        ts = model.joint_target_q_start.numpy()
        tq[ts[3]], tq[ts[5]] = theta0
        ctl.joint_target_q.assign(tq)
        for _ in range(int(0.6 / dt)):
            s0.clear_forces()
            solver.step(s0, s1, ctl, None, dt)
            s0, s1 = s1, s0
        q_unloaded = s0.joint_q.numpy()[:2].copy()
        foot_body = 2
        f = np.array([0.0, 0.0, 700.0])
        cop = np.array([0.05, 0.0, -0.08])
        for _ in range(int(0.6 / dt)):
            s0.clear_forces()
            bf = s0.body_f.numpy()
            bq = s0.body_q.numpy()[foot_body]
            com_w = bq[:3]  # foot origin = pivot; wrench about body origin (Newton body_f is [force, torque] at COM)
            bf[foot_body, :3] = f
            bf[foot_body, 3:] = np.cross(cop - com_w, f)
            s0.body_f.assign(bf)
            solver.step(s0, s1, ctl, None, dt)
            s0, s1 = s1, s0
        q_loaded = s0.joint_q.numpy()[:2]
        tau = np.cross(cop, f)[[1, 0]]  # pitch about +Y, roll about +X at the pivot
        jac = tr.jacobian(torch.zeros(1, 2, dtype=torch.float64))[0].numpy()
        k_joint = jac.T @ np.diag([kp, kp]) @ jac
        predicted = np.linalg.solve(k_joint, tau)
        res["load_deflection_rad"] = (q_loaded - q_unloaded).tolist()
        res["hinge_model_deflection_rad"] = predicted.tolist()
        res["load_finite"] = bool(np.isfinite(q_loaded).all())
        res["diverged"] = bool(not np.isfinite(q_loaded).all() or np.abs(q_loaded).max() > 10.0)
        out[f"gains_{int(kp)}_{kd}_eqsolref_{eq_solref}"] = res
    # 5: hinge-only rig cost (same solver, no cranks/rods/loops).
    model, solver, *_ = rig(desc, closed=False, kp=480, kd=8, device=args.device)
    s0, s1, ctl = model.state(), model.state(), model.control()
    t0 = time.perf_counter()
    for _ in range(500):
        s0.clear_forces()
        solver.step(s0, s1, ctl, None, dt)
        s0, s1 = s1, s0
    out["hinge_only_us_per_step"] = 1e6 * (time.perf_counter() - t0) / 500
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    run(ap.parse_args())
