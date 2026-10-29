"""Physics on the built robot assets through Isaac Lab / Newton (Warp CPU device), for every robot in
``PHYSICS_ROBOTS``. Runs in the Isaac Lab image."""

import numpy as np
import pytest
import torch
from conftest import (
    GRAVITY,
    PARTS,
    PHYSICS_ROBOTS,
    built_usd,
    gravity_torque,
    imu_samples_per_s,
    urdf_mass_kg,
)

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_kinematic_model_mass_and_bodies(robot):
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    from humanoid_sim.model.kinematics import RobotKinematics

    usd = built_usd(robot)
    desc = Description.load(robot)
    kin = RobotKinematics(usd, list(desc.joints))
    assert float(kin.model.body_mass.numpy().sum()) == pytest.approx(urdf_mass_kg(robot), abs=0.01)
    if desc.usd_model is not None:
        assert {n.rsplit("/", 1)[-1] for n in kin.model.body_label} == set(desc.usd_model.bodies)
    else:
        assert kin.model.body_count == len(desc.joints) + 1  # a tree: one body per movable joint, plus the base


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_home_pose_stands_its_lowest_point_on_the_ground(robot):
    """The profile's home pose under an upright base at the height that puts its lowest collision point on the
    ground."""
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    from humanoid_sim.model.kinematics import RobotKinematics, lowest_collision_z
    from humanoid_sim.model.poses import home_pose, initial_pose

    usd = built_usd(robot)
    desc = Description.load(robot)
    joints = list(desc.joints)
    kin = RobotKinematics(usd, joints, base_height_m=0.0)

    q, height = initial_pose(usd, robot, pose="home")
    home = home_pose(desc, load_profile(robot))
    assert q.tolist() == pytest.approx([home[n] for n in joints])
    assert height == pytest.approx(-lowest_collision_z(kin, kin.body_poses(q.reshape(1, -1))))


def _body_shapes(model) -> dict[str, list[int]]:
    """Body name → the indices of its colliding shapes, for a Newton model holding one robot."""
    import newton

    owner, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    out = {}
    for s in range(model.shape_count):
        if owner[s] >= 0 and flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            out.setdefault(model.body_label[owner[s]].rsplit("/", 1)[-1], []).append(s)
    return out


@pytest.mark.parametrize("robot", [r for r in PHYSICS_ROBOTS if load_profile(r)["physics"]["self_collision"]])
def test_home_and_start_poses_have_no_self_contact(robot):
    """In the home pose and the runtime start pose no two of the robot's shapes that the asset lets collide
    interpenetrate, so a reset into either starts without contact forces between its own bodies."""
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    import newton
    import warp as wp

    from humanoid_sim.model.kinematics import RobotKinematics
    from humanoid_sim.model.poses import pose_joint_positions, start_pose

    desc, profile = Description.load(robot), load_profile(robot)
    joints = list(desc.joints)
    kin = RobotKinematics(built_usd(robot), joints, self_collision=True)
    body = {s: name for name, shapes in _body_shapes(kin.model).items() for s in shapes}
    pipeline = newton.CollisionPipeline(kin.model)
    contacts = pipeline.contacts()
    distance = wp.zeros(contacts.rigid_contact_max, dtype=float, device=kin.model.device)
    for pose in {"home", start_pose(profile)}:
        q = pose_joint_positions(desc, profile, pose)
        kin.body_poses(torch.tensor([[q[n] for n in joints]], dtype=torch.float64))
        pipeline.collide(kin.state, contacts)
        newton.eval_rigid_contact_kinematics(kin.model, kin.state, contacts, out_distance=distance)
        n = int(contacts.rigid_contact_count.numpy()[0])
        pairs = zip(contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n],
                    distance.numpy()[:n], strict=True)
        touching = {(body[a], body[b]) for a, b, d in pairs if d < 0}
        assert not touching, (pose, touching)


@pytest.mark.physics
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_newton_model_filters_what_the_profile_excludes(robot):
    """The Newton model of the scene filters, with the profile's ``physics.self_collision`` on, exactly the asset's
    filtered pairs (``model.contact_excludes`` and the pairs the self-collision sets exclude) and each joint's parent
    and child; with it off, every pair of the robot's bodies. Bodies of two sets the profile pairs collide."""
    pytest.importorskip("isaaclab_newton")
    from isaaclab_newton.physics import NewtonManager

    from humanoid_sim.model.urdf import PreparedUrdf
    from humanoid_sim.sim import SimCore

    desc, profile = Description.load(robot), load_profile(robot)
    sim = SimCore(built_usd(robot), device="cpu", robot=robot)
    try:
        model = NewtonManager.get_model()
        shapes = _body_shapes(model)
        filters = model.shape_collision_filter_pairs
        filtered = {frozenset((a, b)): {(x, y) in filters or (y, x) in filters for x in shapes[a] for y in shapes[b]}
                    for i, a in enumerate(shapes) for b in list(shapes)[i + 1:]}
    finally:
        sim.close()

    prepared = PreparedUrdf(desc, profile).build()
    contact_excludes = {frozenset(pair) for pair in profile["model"].get("contact_excludes", [])}
    if profile["physics"]["self_collision"]:
        excluded = contact_excludes | {frozenset(pair) for pair in prepared.self_collision_excludes()}
        excluded |= {frozenset((b, parent)) for b, (_, parent) in prepared.body_tree().items() if parent}
    else:
        excluded = set(filtered)
    for pair, flags in filtered.items():
        assert flags == {pair in excluded}, sorted(pair)

    cfg = profile["model"].get("self_collision")
    if profile["physics"]["self_collision"] and cfg is not None:
        sets = prepared.body_sets()
        paired = None if cfg["pairs"] == "all" else {frozenset(p) for p in cfg["pairs"]}
        crossing = [pair for pair in filtered if pair not in contact_excludes
                    and len(named := {sets[b] for b in pair}) == 2 and named <= set(cfg["sets"])
                    and (paired is None or named in paired)]
        assert crossing
        assert all(filtered[pair] == {False} for pair in crossing)


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_holds_its_pose_on_a_fixed_base_against_gravity(request, robot, solver):
    """Hanging on a welded base, each 1:1-driven joint held at the baseline gains sags by its gravity torque over
    the drive's stiffness, less what the joint's friction holds: the sag, the gravity torque and the gain are each
    taken independently of the solver."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.sim import SimCore

    usd = built_usd(robot)
    sim = SimCore(usd, device="cpu", robot=robot, fixed_base=True, solver=solver)
    request.addfinalizer(sim.close)
    q, qd = sim.joint_state()
    p = sim.plant
    qdrv, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.pd(qdrv, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    for _ in range(100):  # 0.4 s: settled at the baseline damping
        fb = sim.step()
    q1, _ = sim.joint_state()
    assert torch.isfinite(fb.applied_torque_nm).all()
    sag = (q1 - q)[0].numpy()
    desc = sim.desc
    joints = list(desc.joints)
    tau = gravity_torque(usd, joints, q[0].double(), sim.backend.cfg.base_height_m)
    linked = [joints.index(n) for lk in desc.linkages for n in lk.joints]
    if linked:  # the crank-driven ankles: their stiffness in joint space is the cranks' through the linkage
        assert np.abs(sag[linked]).max() < 0.02
    for d in desc.drives.values():
        if d.crank is not None:
            continue
        j = joints.index(d.joints[0])
        k, friction = d.baseline_stiffness, desc.joints[d.joints[0]].friction
        most = abs(tau[j]) / k
        least = max(abs(tau[j]) - friction, 0.0) / k
        assert least * 0.8 - 1e-3 <= abs(sag[j]) <= most * 1.2 + 1e-3, (d.joints[0], sag[j], tau[j], k, friction)
        if abs(sag[j]) > 2e-3:
            assert np.sign(sag[j]) == np.sign(tau[j]), d.joints[0]  # it sags the way gravity pulls


@pytest.mark.physics
def test_base_joints_follow_prescribed_yaw(request):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.model.urdf import BASE_JOINTS
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd("ethrc_v1", base_joints=True), device="cpu", base_joints=True)
    request.addfinalizer(sim.close)
    q, qd = sim.joint_state()
    p = sim.plant
    qdrv, _, _ = p.map.drive_state(q, qd)
    p.submit(DriveDemand.pd(qdrv, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1)),
             torch.zeros(1, dtype=torch.long))
    names = sim.backend.joint_names
    yaw = names.index(BASE_JOINTS[0][0])
    rate = 0.5  # rad/s
    sim.set_base(torch.zeros(1, 3), torch.tensor([[rate, 0.0, 0.0]]), write_state=True)
    for k in range(100):
        # Targets at the step start: with the velocity target the base lands on the step-end angle.
        sim.set_base(torch.tensor([[rate * k * 0.004, 0.0, 0.0]]), torch.tensor([[rate, 0.0, 0.0]]))
        sim.step()
    bq, _ = sim.backend.joint_state()
    assert bq[0, yaw].item() == pytest.approx(rate * 0.4, abs=2e-4)
    q1, _ = sim.joint_state()
    assert (q1 - q).abs().max().item() < 0.02  # the robot holds its pose while the base turns


@pytest.mark.physics
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_imu_delivers_raw_samples_on_a_fixed_base(request, robot):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd(robot), device="cpu", robot=robot, fixed_base=True, imu=True, contacts=True)
    request.addfinalizer(sim.close)
    for _ in range(50):  # 0.2 s
        sim.step()
    s = sim.imu.samples
    n = int(s.count[0])
    expected = 0.2 * imu_samples_per_s(robot)
    assert 0.7 * expected <= n <= 1.25 * expected  # a burst's worth may still be in flight
    assert s.specific_force[0, 0].norm().item() == pytest.approx(GRAVITY, abs=0.1)  # at rest: gravity only
    model_bias = sim.imu.turn_on[0, :3] + sim.imu.drift[0, :3]  # the IMU model's turn-on bias and drift
    assert (s.gyro[0, 0] - model_bias).norm().item() < 0.01
    assert (s.stamp_ns[0, :8] >= s.sample_ns[0, :8]).all()
    assert sim.backend.contacts is not None


@pytest.mark.physics
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_cuda_graph_step_matches_eager(robot):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.sim import SimCore

    usd = built_usd(robot)

    def run(cuda_graph, imu=False, compile=False):
        sim = SimCore(usd, device="cuda:0", robot=robot, fixed_base=True, cuda_graph=cuda_graph, imu=imu,
                      compile=compile)
        try:
            p = sim.plant
            q, qd = sim.joint_state()
            stand, _, _ = p.map.drive_state(q, qd)
            out = []
            for k in range(60):
                if k % 5 == 0:  # a new demand every 20 ms, mid-step arrival
                    p.submit(DriveDemand.pd(stand + 0.02 * (k % 3), p.drive_stiffness.expand(1, -1),
                                              p.drive_damping.expand(1, -1)),
                             torch.full((1,), sim.t_ns + 1_500_000, dtype=torch.long, device=q.device))
                fb = sim.step()
                out.append(torch.stack([fb.position_rad, fb.applied_torque_nm]).clone())
            return torch.stack(out), (sim.imu.samples.seq[0].clone(), sim.imu.samples.gyro[0].clone()) if imu else None
        finally:
            sim.close()

    eager, _ = run(False)
    graphed, _ = run(True)
    assert torch.allclose(eager, graphed, atol=1e-5)
    compiled, _ = run(True, compile=True)
    # Fused kernels reorder float operations; the difference feeds back through the physics.
    assert torch.allclose(eager, compiled, rtol=1e-3, atol=1e-3)
    from humanoid_sim.sensors import rigs

    noisy = rigs.rig(robot).imu.gyro_noise_density_rad_s_sqrt_hz > 0
    for compile in (False, True):
        _, (seq, gyro) = run(True, imu=True, compile=compile)
        assert (seq[:8].diff() == -1).all()  # samples delivered, newest first
        if noisy:
            assert gyro[:8].std() > 0  # noise drawn per replay


# MuJoCo's default joint-limit constraint, which MJWarp applies when Newton's force-space limit gains are zero
# (profile joint_limits: mujoco; MuJoCo XML reference, solref and solimp defaults).
LIMIT_TIMECONST_S, LIMIT_DAMPRATIO = 0.02, 1.0
LIMIT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)  # d0, dwidth, width, midpoint, power


def _impedance(r):
    d0, dwidth, width, mid, power = LIMIT_SOLIMP
    x = min(abs(r) / width, 1.0)
    y = x**power / mid ** (power - 1) if x <= mid else 1 - (1 - x) ** power / (1 - mid) ** (power - 1)
    return d0 + y * (dwidth - d0)


def _soft_limit_sink(inverse_inertia, force):
    """How far MuJoCo's soft limit lets a joint sink past its stop under a static ``force`` (Nm): at rest its
    constraint law reads ``d k r = (1 - d) a0`` with ``a0 = force * inverse_inertia`` (the joint's diagonal of the
    inverse mass matrix) and ``k = d / (dwidth^2 timeconst^2 dampratio^2)``; ``d`` depends on ``r``, so iterate."""
    dwidth = LIMIT_SOLIMP[1]
    r = 0.0
    for _ in range(50):
        d = _impedance(r)
        r = (1 - d) * force * inverse_inertia * (dwidth * LIMIT_TIMECONST_S * LIMIT_DAMPRATIO) ** 2 / d**2
    return r


def _inverse_inertia(usd, desc, q, joint, base_height_m):
    """The joint's diagonal of the inverse joint-space mass matrix (fixed base, armature included), from Newton's
    mass matrix on the host."""
    import newton

    from humanoid_sim.model.kinematics import RobotKinematics

    joints = list(desc.joints)
    kin = RobotKinematics(usd, joints, base_height_m=base_height_m)
    labels = [lbl.rsplit("/", 1)[-1] for lbl in kin.model.joint_label]
    qd_start = kin.model.joint_qd_start.numpy()
    dof = {n: int(qd_start[labels.index(n)]) for n in joints}
    kin.body_poses(q.reshape(1, -1))
    mass = newton.eval_mass_matrix(kin.model, kin.state).numpy()[0].astype(np.float64)
    for n in joints:
        mass[dof[n], dof[n]] += desc.joints[n].armature
    return float(np.linalg.inv(mass)[dof[joint], dof[joint]])


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_a_joint_pushed_into_its_stop_stays_at_it(robot, solver):
    """40 Nm into a waist yaw stop, every other joint held at the baseline gains. On MJWarp the joint sinks past
    the stop as far as MuJoCo's own soft limit lets it (profile joint_limits: mujoco, as Controls' MJCF has it),
    which grows with the joint's inverse inertia: under 1 mrad for the V1, about 9 mrad for the G1's lighter upper
    body. On Kamino the stop is a hard constraint. The waist turns about a vertical axis, so gravity adds no torque
    of its own; self-collision is off so that only the stop resists."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.devices.demand import LAW_DISABLED, DriveDemand
    from humanoid_sim.sim import SimCore

    joint, push = PARTS[robot]["vertical_joint"], 40.0
    usd = built_usd(robot)
    sim = SimCore(usd, device="cpu", robot=robot, fixed_base=True, solver=solver, self_collision=False)
    try:
        p = sim.plant
        j = list(sim.desc.joints).index(joint)
        lo, hi = sim.desc.calibration_travel(joint)
        q, qd = sim.joint_state()
        q = q.clone()
        q[0, j] = hi - 0.01
        sim.write_joint_state(q, torch.zeros_like(qd))
        held, _, _ = p.map.drive_state(q, torch.zeros_like(q))
        demand = DriveDemand.pd(held, p.drive_stiffness.expand(1, -1), p.drive_damping.expand(1, -1))
        demand.law[0, p.map.drive_names.index(next(n for n, d in sim.desc.drives.items() if d.joints == (joint,)))] \
            = LAW_DISABLED  # the waist yaw takes the push alone
        p.submit(demand, torch.zeros(1, dtype=torch.long))
        p.external_torque[0, j] = push
        for _ in range(150):
            sim.step()
        q1 = sim.joint_state()[0][0]
        past = float(q1[j]) - hi
        if solver == "kamino":
            assert -0.001 < past < 0.002, past
        else:
            # MuJoCo weights the limit by an inverse inertia it takes at a reference pose, not the current one:
            # the zero pose and the held pose bracket it.
            h = sim.backend.cfg.base_height_m
            weights = [_inverse_inertia(usd, sim.desc, pose, joint, h)
                       for pose in (torch.zeros_like(q1, dtype=torch.float64), q1.double())]
            least, most = (_soft_limit_sink(w, push) for w in (min(weights), max(weights)))
            assert 0.85 * least < past < 1.15 * most, (past, least, most)
    finally:
        sim.close()


@pytest.mark.physics
def test_pace_objective_is_zero_at_planted_parameters(request):
    """A chirp recorded with planted PACE parameters replays exactly at them, and measurably worse at the start
    values: the fit's plumbing (bias, delay, friction relief, solver writes) round-trips."""
    pytest.importorskip("isaaclab_newton")
    import numpy as np

    from humanoid_sim.sim import SimCore
    from humanoid_sim.sysid.pace import PaceSpace, chirp_commands, evaluate, start_values, synthetic_capture

    sim = SimCore(built_usd("ethrc_v1"), device="cpu", robot="ethrc_v1", num_envs=2, fixed_base=True)
    request.addfinalizer(sim.close)
    joints, drives = list(sim.plant.map.joint_names), list(sim.plant.map.drive_names)
    space = PaceSpace(joints, drives, start_values(sim.desc, joints))
    x_true = np.random.default_rng(0).uniform(-0.5, 0.5, space.dim)
    amp = {j: 0.05 for j in joints}
    commands = chirp_commands(joints, duration_s=3.0, f0_hz=0.5, f1_hz=3.0, amplitude_rad=amp)
    capture = synthetic_capture(sim, space, space.decode(np.tile(x_true, (2, 1))), commands)
    f, _ = evaluate(sim, [capture], space, np.stack([x_true, space.x0()]), list(range(len(drives))),
                    base_motion=None, duration_s=None)
    assert f[0] < 1e-4 and f[1] > 10 * f[0]
