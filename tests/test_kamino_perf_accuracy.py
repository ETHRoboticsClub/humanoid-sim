"""Accuracy of Kamino's speed levers on the two minimal reproductions (one PD joint, a planar four-bar), as numbers per
PADMM variant: the joint's frequency response against the exact ODE, the four-bar's crank-to-rocker response against MJWarp
and its loop drift. The mechanisms are those of tests/test_kamino_mjwarp_parity.py (humanoid-sim feat/kamino-parity); the
solver settings are the variable here: library defaults, the training's (Isaac Lab's tolerances 1e-4, rho_0 0.05, 250
iterations) and looser tolerances and lower caps. Each variant must stay inside the parity tests' bounds; the table is
printed (pytest -s) and written to $KAMINO_PERF_ACCURACY_JSON when set.

Marked ``physics``: CUDA, MJWarp on the GPU."""

import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.physics

DT = 1e-3
FREQS_HZ = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0])
SETTLE_S, RUN_S = 1.5, 4.0
HIP_KP, HIP_KD = 197.4, 12.6
CRANK_KP, CRANK_KD = 22.1, 1.41
LINK_M, LINK_L, LINK_I, LINK_ARM, LINK_B = 2.0, 0.1, 0.02, 0.02, 0.5
LOOP_SOLREF = (0.002, 1.0)

TRAINING = {"primal_tolerance": 1e-4, "dual_tolerance": 1e-4, "compl_tolerance": 1e-4, "rho_0": 0.05, "max_iterations": 250}
VARIANTS = {
    "library-default": {},
    "training": TRAINING,
    "tol-3e-4": TRAINING | {"primal_tolerance": 3e-4, "dual_tolerance": 3e-4, "compl_tolerance": 3e-4},
    "tol-1e-3": TRAINING | {"primal_tolerance": 1e-3, "dual_tolerance": 1e-3, "compl_tolerance": 1e-3},
    "iters-100": TRAINING | {"max_iterations": 100},
    "iters-50": TRAINING | {"max_iterations": 50},
    "no-warmstart": TRAINING | {"warmstart_mode": "none"},
}


def cuda():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("Kamino accuracy runs on CUDA")
    return "cuda:0"


def multisine(t, amp):
    return amp * np.sin(2 * np.pi * FREQS_HZ[:, None] * t[None] + np.arange(len(FREQS_HZ))[:, None]).sum(0)


def response(t, u, y):
    w = t > SETTLE_S
    e = np.exp(-2j * np.pi * FREQS_HZ[:, None] * t[None, w])
    return (e @ y[w]) / (e @ u[w])


PROFILE_KAMINO = {  # profiles/ethrc_v1.yaml physics.kamino as the training builds it
    "integrator": "euler", "iterations": 250, "max_contacts_per_world": 96, "use_fk_solver": False,
    "sparse_jacobian": True, "sparse_dynamics": False, "dynamics_solver": "padmm", "linear_solver": "LLTB",
    "linear_solver_kwargs": {"solve_block_dim": 256, "solve_max_registers": 48}, "matrix_preconditioner_stride": 1024,
    "tracked_active_clear": True, "use_graph_conditionals": True,
    "padmm": {"rho_0": 0.05, "algorithm": "bilateral_reduced", "reduced_capacity": 96, "reduced_ordering": "rcm",
              "reduced_dynamic_pattern": True, "reduced_sparse_diagnostics": True, "reduced_lazy_preconditioning": True},
}
# The same mechanisms through the production reduced path (every world of a mechanism without limits or contacts is
# rejected by the reduced classifier, so these run the full-PADMM fallback inside it), with a speed lever's switch.
for _name, _env in (("reduced", {}), ("reduced+per-world-fallback", {"KAMINO_PER_WORLD_FALLBACK": "1"}),
                    ("reduced+skip-diagnostics", {"KAMINO_SKIP_DIAGNOSTICS": "1"})):
    VARIANTS[_name] = {"profile": True, "env": _env}


def solver(kind, model, padmm=None):
    import newton

    if kind == "kamino":
        if padmm and padmm.get("profile"):
            from humanoid_sim.physics.isaaclab import solver_cfg

            os.environ.pop("KAMINO_PER_WORLD_FALLBACK", None)
            os.environ.pop("KAMINO_SKIP_DIAGNOSTICS", None)
            os.environ.update(padmm["env"])
            return newton.solvers.SolverKamino(model, config=solver_cfg("kamino", PROFILE_KAMINO).to_solver_config())
        if not padmm:
            return newton.solvers.SolverKamino(model)
        from newton._src.solvers.kamino.config import PADMMSolverConfig

        config = newton.solvers.SolverKamino.Config()
        config.padmm = PADMMSolverConfig(**padmm)
        return newton.solvers.SolverKamino(model, config=config)
    return newton.solvers.SolverMuJoCo(model, integrator="implicitfast", use_mujoco_cpu=False)


def step_all(model, s, targets, dof, read):
    import newton

    state, nxt, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    target = np.zeros(model.joint_dof_count, dtype=np.float32)
    out = []
    for u in targets:
        target[dof] = u
        control.joint_target_q.assign(target)
        s.step(state, nxt, control, None, DT)
        state, nxt = nxt, state
        out.append(read(state))
    return np.array(out)


def pd_joint(device):
    import newton
    import warp as wp

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(b)
    i = LINK_I
    link = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), com=wp.vec3(LINK_L, 0.0, 0.0),
                      inertia=wp.mat33(i, 0, 0, 0, i, 0, 0, 0, i), mass=LINK_M)
    j = b.add_joint_revolute(-1, link, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                             axis=newton.Axis.Y, target_ke=HIP_KP, target_kd=HIP_KD, damping=LINK_B, armature=LINK_ARM,
                             limit_lower=-3.0, limit_upper=3.0)
    b.add_articulation([j])
    return b.finalize(device=device)


def exact_pd(targets):
    from scipy.linalg import expm

    inertia = LINK_I + LINK_M * LINK_L**2 + LINK_ARM
    a = np.array([[0.0, 1.0], [-HIP_KP / inertia, -(HIP_KD + LINK_B) / inertia]])
    bvec = np.array([0.0, HIP_KP / inertia])
    big = expm(np.block([[a, bvec[:, None]], [np.zeros((1, 3))]]) * DT)
    ad, bd = big[:2, :2], big[:2, 2]
    x, out = np.zeros(2), []
    for u in targets:
        x = ad @ x + bd * u
        out.append(x[0])
    return np.array(out)


def four_bar(device):
    import newton
    import warp as wp

    def x(px, pz):
        return wp.transform(wp.vec3(px, 0.0, pz), wp.quat_identity())

    def eye(v):
        return wp.mat33(v, 0, 0, 0, v, 0, 0, 0, v)

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(b)
    crank = b.add_link(xform=x(0.0, 0.0), com=wp.vec3(0.0, 0.0, 0.015), inertia=eye(2e-5), mass=0.1)
    rod = b.add_link(xform=x(0.0, 0.03), com=wp.vec3(0.04, 0.0, 0.005), inertia=eye(5e-5), mass=0.05)
    rocker = b.add_link(xform=x(0.08, 0.0), com=wp.vec3(0.03, 0.0, 0.02), inertia=eye(0.01), mass=1.0)
    jc = b.add_joint_revolute(-1, crank, parent_xform=x(0.0, 0.0), axis=newton.Axis.Y, target_ke=CRANK_KP,
                              target_kd=CRANK_KD, armature=5.6e-3, damping=0.05)
    jb = b.add_joint_revolute(crank, rod, parent_xform=x(0.0, 0.03), axis=newton.Axis.Y)
    jr = b.add_joint_revolute(-1, rocker, parent_xform=x(0.08, 0.0), axis=newton.Axis.Y, damping=0.02)
    b.add_joint_revolute(rocker, rod, parent_xform=x(0.0, 0.04), child_xform=x(0.08, 0.01), axis=newton.Axis.Y)
    b.add_articulation([jc, jb])
    b.add_articulation([jr])
    return b.finalize(device=device)


def loop_gap_m(body_q):
    from scipy.spatial.transform import Rotation

    def point(row, local):
        return row[:3] + Rotation.from_quat(row[3:7]).apply(local)

    return np.linalg.norm(point(body_q[2], [0.0, 0.0, 0.04]) - point(body_q[1], [0.08, 0.0, 0.01]))


def measure(device, padmm) -> dict:
    t = np.arange(round(RUN_S / DT)) * DT
    u = multisine(t, 0.01)
    exact = response(t, u, exact_pd(u))
    model = pd_joint(device)
    q = step_all(model, solver("kamino", model, padmm), u, 0, lambda s: s.joint_q.numpy()[0])
    got = response(t, u, q - q[t > SETTLE_S].mean())
    pd_gain = float(np.max(np.abs(np.abs(got) / np.abs(exact) - 1.0)))
    pd_phase = float(np.max(np.abs(np.degrees(np.angle(got / exact)))))
    u2 = multisine(t, 0.02)
    resp, gap = {}, {}
    for kind in ("kamino", "mjwarp"):
        model = four_bar(device)
        s = solver(kind, model, padmm if kind == "kamino" else None)
        if kind == "mjwarp":
            solref = s.mjw_model.eq_solref.numpy()
            solref[...] = LOOP_SOLREF
            s.mjw_model.eq_solref.assign(solref)
        out = step_all(model, s, u2, 0, lambda st: np.concatenate([st.joint_q.numpy()[2:3], [loop_gap_m(st.body_q.numpy())]]))
        y = out[:, 0] - out[t > SETTLE_S, 0].mean()
        resp[kind], gap[kind] = response(t, u2, y), out[t > SETTLE_S, 1]
    ratio = resp["kamino"] / resp["mjwarp"]
    return {"pd_gain_err_max": pd_gain, "pd_phase_err_deg_max": pd_phase,
            "fourbar_gain_ratio_err_max": float(np.max(np.abs(np.abs(ratio) - 1.0))),
            "fourbar_phase_err_deg_max": float(np.max(np.abs(np.degrees(np.angle(ratio))))),
            "loop_drift_um_mean": float(gap["kamino"].mean() * 1e6), "loop_drift_um_max": float(gap["kamino"].max() * 1e6)}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_stays_inside_the_parity_bounds(name):
    device = cuda()
    m = measure(device, VARIANTS[name])
    print(f"[accuracy] {name}: {json.dumps(m)}")
    path = os.environ.get("KAMINO_PERF_ACCURACY_JSON")
    if path:
        rows = json.load(open(path)) if os.path.exists(path) else {}
        rows[name] = m
        json.dump(rows, open(path, "w"), indent=1)
    assert m["pd_gain_err_max"] < 0.03 and m["pd_phase_err_deg_max"] < 1.5, m
    assert m["fourbar_gain_ratio_err_max"] < 0.05 and m["fourbar_phase_err_deg_max"] < 3.0, m
    assert m["loop_drift_um_max"] < 5.0, m
