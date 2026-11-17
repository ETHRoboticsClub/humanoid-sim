"""Kamino against MJWarp (and the exact ODE) on the same small mechanisms, inputs and step: the solver numerics a
training plant inherits, checked where they can be told apart.

- One PD-driven revolute joint (the V1's HEJ90 baseline gains, armature, passive damping): the frequency response from
  the position target to the joint angle, 1 to 15 Hz, against the exact linear ODE under the same zero-order hold.
  Implicit PD on Kamino (``m_j = a + dt b + dt kd + dt^2 kp``) and implicitfast on MJWarp add no lag or damping a
  closed loop at 3 to 6 Hz could notice.
- A planar four-bar closed by a loop joint outside the articulation (a crank at the V1 crank gains, a coupler rod, a
  loaded rocker): the response from the crank target to the rocker angle, and the loop's constraint drift. Kamino holds
  the loop as a hard bilateral constraint; MJWarp as an equality at humanoid-sim's loop solref (2 ms), so it drifts
  more. The closed-chain response is the V1 ankle's mechanism and the V2's many loops.

Marked ``physics``: CUDA, MJWarp on the GPU (``use_mujoco_cpu`` is never set in this repo)."""

import numpy as np
import pytest

pytestmark = pytest.mark.physics

DT = 1e-3
FREQS_HZ = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0])
SETTLE_S, RUN_S = 1.0, 4.0  # a 3 s window: whole cycles of every frequency, no leakage
# V1 baseline gains (humanoid-urdf baseline_position_gains): HEJ90 197.4 / 12.6, ankle cranks 22.1 / 1.41.
HIP_KP, HIP_KD = 197.4, 12.6
CRANK_KP, CRANK_KD = 22.1, 1.41
# One link on a hip-sized drive: mass, centre-of-mass offset, inertia about the centre of mass, armature, damping.
LINK_M, LINK_L, LINK_I, LINK_ARM, LINK_B = 2.0, 0.1, 0.02, 0.02, 0.5
# humanoid-sim's MJWarp loop closure response (profiles/ethrc_v1.yaml physics.mjwarp.loop_closure_solref)
LOOP_SOLREF = (0.002, 1.0)


def cuda():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("Kamino and MJWarp parity runs on CUDA")
    return "cuda:0"


def multisine(t, amp):
    return amp * np.sin(2 * np.pi * FREQS_HZ[:, None] * t[None] + np.arange(len(FREQS_HZ))[:, None]).sum(0)


def response(t, u, y):
    """Complex response y/u at each of FREQS_HZ over the settled part."""
    w = t > SETTLE_S
    e = np.exp(-2j * np.pi * FREQS_HZ[:, None] * t[None, w])
    return (e @ y[w]) / (e @ u[w])


def solver(kind, model):
    import newton

    if kind == "kamino":
        return newton.solvers.SolverKamino(model)
    s = newton.solvers.SolverMuJoCo(model, integrator="implicitfast", use_mujoco_cpu=False)
    return s


def step_all(model, s, targets, dof, read):
    """Run ``targets`` (one per step) into DoF ``dof`` and return ``read(state)`` after every step."""
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
    """The joint's exact response to the zero-order-held targets (no gravity: the response is about a pose, and the
    linear part is what the drive loop sees; the gravity load only shifts the pose)."""
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


def test_pd_joint_response_matches_mjwarp_and_the_exact_ode():
    device = cuda()
    t = np.arange(round(RUN_S / DT)) * DT
    u = multisine(t, 0.01)
    exact = response(t, u, exact_pd(u))
    got_by = {}
    for kind in ("kamino", "mjwarp"):
        model = pd_joint(device)
        q = step_all(model, solver(kind, model), u, 0, lambda s: s.joint_q.numpy()[0])
        q = q - q[t > SETTLE_S].mean()  # the gravity offset
        got = response(t, u, q)
        gain = np.abs(got) / np.abs(exact)
        phase = np.degrees(np.angle(got / exact))
        got_by[kind] = got
        assert np.all(np.abs(gain - 1.0) < 0.03), f"{kind}: gain ratio to the exact ODE {np.round(gain, 4)}"
        assert np.all(np.abs(phase) < 1.5), f"{kind}: phase off the exact ODE {np.round(phase, 2)} deg"
    ratio = got_by["kamino"] / got_by["mjwarp"]
    assert np.all(np.abs(np.abs(ratio) - 1.0) < 0.02), f"Kamino/MJWarp gain {np.round(np.abs(ratio), 4)}"


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
    """Distance between the loop joint's two anchors: the rocker's (0, 0, 0.04) and the rod's (0.08, 0, 0.01)."""
    from scipy.spatial.transform import Rotation

    def point(row, local):
        return row[:3] + Rotation.from_quat(row[3:7]).apply(local)

    return np.linalg.norm(point(body_q[2], [0.0, 0.0, 0.04]) - point(body_q[1], [0.08, 0.0, 0.01]))


def test_four_bar_closed_chain_response_and_drift_match_mjwarp():
    device = cuda()
    t = np.arange(round(RUN_S / DT)) * DT
    u = multisine(t, 0.02)
    got, gap = {}, {}
    for kind in ("kamino", "mjwarp"):
        model = four_bar(device)
        s = solver(kind, model)
        if kind == "mjwarp":  # the loop is the model's only equality
            solref = s.mjw_model.eq_solref.numpy()
            solref[...] = LOOP_SOLREF
            s.mjw_model.eq_solref.assign(solref)
        out = step_all(model, s, u, 0, lambda st: np.concatenate([st.joint_q.numpy()[2:3], [loop_gap_m(st.body_q.numpy())]]))
        y = out[:, 0] - out[t > SETTLE_S, 0].mean()
        got[kind], gap[kind] = response(t, u, y), out[t > SETTLE_S, 1].max()
    ratio = got["kamino"] / got["mjwarp"]
    assert np.all(np.abs(np.abs(ratio) - 1.0) < 0.05), f"gain ratio Kamino/MJWarp {np.round(np.abs(ratio), 4)}"
    assert np.all(np.abs(np.degrees(np.angle(ratio))) < 3.0), f"phase {np.round(np.degrees(np.angle(ratio)), 2)} deg"
    assert gap["kamino"] < 5e-6, f"Kamino loop drift {gap['kamino'] * 1e6:.2f} um"
    assert gap["kamino"] <= gap["mjwarp"], f"Kamino {gap['kamino']:.2e} m against MJWarp {gap['mjwarp']:.2e} m"


# A crank spring as humanoid-sim's compliance asset builds it: the drive's PD (the contract's crank gains, 160 / 4.2,
# with the shaft's armature) on a 10 g output body, a spring joint (PD to zero) to a loaded link.
SPRING_KP, SPRING_KD, SPRING_ARM = 160.0, 4.2, 5.6e-3
OUTPUT_M, OUTPUT_I, LOAD_I = 0.01, 1e-6, 0.02
SPRING_K, SPRING_C = 120.0, 0.12  # the shank-phone fit's crank stiffness, the range's mean damping


def series_spring(device):
    import newton
    import warp as wp

    def eye(v):
        return wp.mat33(v, 0, 0, 0, v, 0, 0, 0, v)

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(b)
    pose = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
    out = b.add_link(xform=pose, com=wp.vec3(0.0, 0.0, 0.0), inertia=eye(OUTPUT_I), mass=OUTPUT_M)
    load = b.add_link(xform=pose, com=wp.vec3(0.0, 0.0, 0.0), inertia=eye(LOAD_I), mass=1.0)
    drive = b.add_joint_revolute(-1, out, parent_xform=pose, axis=newton.Axis.Y, target_ke=SPRING_KP,
                                 target_kd=SPRING_KD, armature=SPRING_ARM)
    spring = b.add_joint_revolute(out, load, axis=newton.Axis.Y, target_ke=SPRING_K, target_kd=SPRING_C)
    b.add_articulation([drive, spring])
    return b.finalize(device=device)


def exact_series_spring(targets):
    from scipy.linalg import expm

    j1, j2, k, c = OUTPUT_I + SPRING_ARM, LOAD_I, SPRING_K, SPRING_C
    a = np.array([[0, 1, 0, 0], [-(SPRING_KP + k) / j1, -(SPRING_KD + c) / j1, k / j1, c / j1], [0, 0, 0, 1],
                  [k / j2, c / j2, -k / j2, -c / j2]])
    bvec = np.array([0.0, SPRING_KP / j1, 0.0, 0.0])
    big = expm(np.block([[a, bvec[:, None]], [np.zeros((1, 5))]]) * DT)
    x, out = np.zeros(4), []
    for u in targets:
        x = big[:4, :4] @ x + big[:4, 4] * u
        out.append(x[2])
    return np.array(out)


def test_series_spring_response_matches_the_exact_ode_where_the_loop_lives():
    """Target to load angle through the spring. MJWarp's rk4 (the profile's) is exact to 0.5%. Kamino's first-order
    implicit step damps fast modes by about omega dt / 2 in damping ratio: within 2% and 1.5 deg up to 6 Hz (the 3 to 6
    Hz band of the stand chatter), 5 to 16% low at 8 to 15 Hz around this spring's 12 Hz resonance (2026-11-17, L4). A
    regression past that is a change in Kamino's integration."""
    import newton

    device = cuda()
    t = np.arange(round(RUN_S / DT)) * DT
    u = multisine(t, 0.01)
    exact = response(t, u, exact_series_spring(u))
    low = FREQS_HZ <= 6.0
    for kind in ("kamino", "rk4"):
        model = series_spring(device)
        s = newton.solvers.SolverKamino(model) if kind == "kamino" else newton.solvers.SolverMuJoCo(
            model, integrator="rk4", use_mujoco_cpu=False)
        q = step_all(model, s, u, 0, lambda st: float(st.joint_q.numpy()[0] + st.joint_q.numpy()[1]))
        r = response(t, u, q - q[t > SETTLE_S].mean()) / exact
        gain, phase = np.abs(r), np.degrees(np.angle(r))
        if kind == "rk4":
            assert np.all(np.abs(gain - 1.0) < 0.005) and np.all(np.abs(phase) < 0.5), (np.round(gain, 4), phase)
            continue
        assert np.all(np.abs(gain[low] - 1.0) < 0.02), f"Kamino gain/exact to 6 Hz {np.round(gain[low], 4)}"
        assert np.all(np.abs(phase[low]) < 1.5), f"Kamino phase to 6 Hz {np.round(phase[low], 2)} deg"
        assert np.all(gain[~low] > 0.8) and np.all(gain[~low] < 1.0), f"Kamino above 6 Hz {np.round(gain[~low], 4)}"
