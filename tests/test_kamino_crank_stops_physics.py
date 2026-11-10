"""Kamino's soft crank stops on the solver (physics/crank_stops.py, KaminoCrankStops): a static push past a stop rests
at torque over stiffness, no push gives no overshoot, a stiff stop gives the hard stop's crank angle and torques,
the per-env knobs reach the solver, and the stops stay stable under a shaking load across the knobs' range, on the
native V1 and on the ankle-compliance asset.

Every check hangs the V1 by its base, holds the cranks with their drives' baseline PD and drives one crank per env
past one of its stops. With no load on the foot the rods carry no force at rest (the foot's equilibrium is
``Jᵀ τ = 0`` for the cranks' net torques ``τ``, and ``J`` is invertible), so each crank's drive torque is held by its
stop alone: the overshoot is the drive's applied torque over the stop's stiffness.
"""
import math

import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand

PUSH_RAD = 1.0  # how far past a stop the pushed crank's target lies: 22 N m with the baseline spring
STEPS = 2500  # 2.5 s at the V1's 1 ms: the cranks' soft hold settles in half of it
SIDES = 8  # one env per stop: four cranks, lower and upper
HINGE_FRICTION = 1e-6  # next to nothing (Kamino keeps a joint's friction structure from its build)


@pytest.fixture(autouse=True)
def clean_solver_output(capfd):
    yield
    output = "".join(capfd.readouterr())
    assert "possibly invalid inertia tensor" not in output
    assert "exceeded MJWarp limit" not in output
    assert "Contact buffer overflowed" not in output


def _cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("physics checks run on CUDA")


def _run(*, compliance=False, soft=None, scales=None, specs=(), push=PUSH_RAD, steps=STEPS, gravity=(0.0, 0.0, 0.0),
         shake=None, envs=SIDES):
    """Hang the robot, hold its cranks, push crank ``env // 2`` of each env past its lower (even env) or upper (odd
    env) stop and step. ``soft``: the profile's ``crank_stops.soft`` (``None``: left to the profile); ``scales``:
    ``(stiffness, damping)`` ratios per env for the plant's knobs; ``shake``: ``(hinge torque N m, Hz)``. Returns the
    settled state; the simulation is closed (one per process)."""
    from humanoid_sim.model.usd import crank_joints
    from humanoid_sim.randomization import ParamSpec
    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    overrides = {}
    if compliance:
        overrides["ankle_compliance"] = {"stiffness_nm_per_rad": 600.0, "damping_nm_s_per_rad": 0.3}
    if soft is not None:
        overrides["kamino"] = {"crank_stops": {"soft": soft}}
    sim = SimCore(built_usd("ethrc_v1", ankle_compliance=compliance), device="cuda:0", num_envs=envs, fixed_base=True,
                  solver="kamino", gravity_m_s2=gravity, physics_overrides=overrides or None)
    try:
        core, desc, robot = sim.plant, sim.desc, sim.backend.robot
        specs = list(specs)
        if scales is not None:  # the knobs held at their nominal until set below
            specs += [ParamSpec(name, "uniform", (1.0, 1.0), "scale")
                      for name in ("crank_stop_stiffness", "crank_stop_damping")]
        if specs:
            plant = RobotPlant(robot, TrainPlantCfg(reporting=False, feedback_sampling=False, randomization=specs,
                                                    physics_overrides=overrides or None))
            if scales is not None:
                for name, value in zip(("crank_stop_stiffness", "crank_stop_damping"), scales, strict=True):
                    plant.randomizer.values[name] = torch.as_tensor(value, dtype=torch.float32, device="cuda:0").reshape(-1, 1)
                plant._write_physics()
        sim.write_joint_state(*sim.joint_state())
        hinges = [j for lk in desc.linkages for j in lk.joints]
        cols = [core.solver_map.joint_names.index(j) for j in hinges]
        core.friction_relief[:, cols] = 0.0
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.full((envs, len(cols)), HINGE_FRICTION, device="cuda:0"),
            joint_ids=sim._to_solver[cols])
        cranks = list(crank_joints(desc))
        drive_ids = [core.map.drive_names.index(d) for d in cranks]
        limits = torch.tensor([desc.drives[d].position_limit_device for d in cranks], device="cuda:0")
        feedback = core.feedback(*sim.solver_state())
        target = feedback.position_rad.clone()
        pushed = torch.arange(envs, device="cuda:0") // 2 % len(cranks)
        upper = torch.arange(envs, device="cuda:0") % 2 == 1
        for e in range(envs):
            target[e, drive_ids[int(pushed[e])]] = (limits[pushed[e], 1] + push if upper[e] else limits[pushed[e], 0] - push)
        core.submit(DriveDemand.pd(target, core.drive_stiffness, core.drive_damping),
                    torch.zeros(envs, dtype=torch.int64, device="cuda:0"))
        hinge_ids = torch.tensor([core.map.joint_names.index(j) for j in hinges], device="cuda:0")
        stops = getattr(robot, "_kamino_crank_stops", None)
        peak = torch.zeros(envs, len(cranks), device="cuda:0")
        row = torch.arange(envs, device="cuda:0")
        drive_cols = torch.tensor(drive_ids, device="cuda:0")
        deepest = torch.full((envs,), -1.0, device="cuda:0")
        worst_rate = torch.zeros((), device="cuda:0")
        for step in range(steps):
            if shake is not None:
                load, hz = shake
                phase = 2 * math.pi * hz * step * sim.period_ns * 1e-9
                core.external_torque[:, hinge_ids] = load * math.sin(phase)
            feedback = sim.step()
            if stops is not None:
                peak = torch.maximum(peak, stops.torque.abs())
            at = feedback.position_rad[row, drive_cols[pushed]]
            deepest = torch.maximum(deepest, torch.where(upper, at - limits[pushed, 1], limits[pushed, 0] - at))
            worst_rate = torch.maximum(worst_rate, sim.solver_state()[1].abs().max())
        q, qd = (t.clone() for t in sim.joint_state())
        position = feedback.position_rad[:, drive_ids].clone()
        out = {"position": position, "torque": feedback.applied_torque_nm[:, drive_ids].clone(), "q": q, "qd": qd, "cranks": cranks,
               "pushed": pushed, "upper": upper, "limits": limits, "peak": peak, "worst_rate": worst_rate,
               "overshoot": torch.where(upper, position[row, pushed] - limits[pushed, 1],
                                        limits[pushed, 0] - position[row, pushed]),
               "drive_torque": feedback.applied_torque_nm[row, drive_cols[pushed]].abs().clone(), "deepest": deepest,
               "stiffness": None if stops is None else stops.stiffness.clone(),
               "damping": None if stops is None else stops.damping.clone(),
               "stop_torque": None if stops is None else stops.torque.clone(), "hard_limits": _hard_limits(desc)}
        return out
    finally:
        sim.close()


def _hard_limits(desc):
    """``{crank joint: [E, 2]}`` the solver's own (hard) limits of the cranks, from the Newton model."""
    from isaaclab_newton.physics import NewtonManager

    from humanoid_sim.model.usd import crank_joints

    model = NewtonManager.get_model()
    labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
    starts, lower, upper = model.joint_qd_start.numpy(), model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    return {name: torch.tensor([[lower[starts[i]], upper[starts[i]]] for i, label in enumerate(labels) if label == name])
            for name in crank_joints(desc).values()}


def _nominal(physics_key):
    from humanoid_sim.model.asset import load_profile

    return load_profile("ethrc_v1")["physics"]["kamino"]["crank_stops"][physics_key]


@pytest.mark.physics
@pytest.mark.parametrize("compliance", [False, True])
def test_a_static_push_rests_past_the_stop_by_torque_over_stiffness(compliance):
    _cuda()
    # Per env stiffness ratios (and the nominal damping): each stop pair at a different stiffness through the plant.
    ratios = torch.tensor([1.0, 1.0, 0.25, 0.25, 3.0, 3.0, 0.1, 0.5])
    state = _run(compliance=compliance, scales=(ratios, torch.ones(SIDES)))
    k = state["stiffness"][:, 0]
    torch.testing.assert_close(k, ratios.to("cuda:0") * _nominal("stiffness_nm_per_rad"))
    assert state["qd"].abs().max() < (0.03 if compliance else 2e-3), "not settled"
    torque = state["drive_torque"]
    assert torque.min() > 0.8 * 22.1 * (PUSH_RAD - 0.15) and torque.max() < 30.01  # about the baseline spring's push
    predicted = torque / k
    print(f"[static compliance={compliance}] k {k.tolist()}; drive torque {torque.tolist()}; overshoot {state['overshoot'].tolist()}; "
          f"predicted {predicted.tolist()}")
    # The overshoot is the drive torque over the stiffness (within 5 % and 0.3 mrad), at every stop and every stiffness.
    assert predicted.max() > 0.01 and predicted.min() > 1e-3
    torch.testing.assert_close(state["overshoot"], predicted, rtol=0.05, atol=3e-4)
    # The stop torque the solver was given balances the drive's.
    torch.testing.assert_close(state["stop_torque"].abs().max(dim=1).values, torque, rtol=0.03, atol=0.1)
    # The solver's hard limit moved out to the backstop, past the deepest overshoot.
    from humanoid_sim.model.description import Description

    desc = Description.load("ethrc_v1")
    pad = _nominal("backstop_rad")
    for drive, joint in zip(state["cranks"], state["hard_limits"], strict=True):
        lo, hi = desc.drives[drive].shaft_limits
        torch.testing.assert_close(state["hard_limits"][joint], torch.tensor([[lo - pad, hi + pad]]).expand(SIDES, 2),
                                   rtol=0, atol=1e-5)
    assert state["overshoot"].max() < pad


@pytest.mark.physics
@pytest.mark.parametrize("compliance", [False, True])
def test_without_a_push_there_is_no_overshoot_and_no_stop_torque(compliance):
    _cuda()
    # A crank held at its stop (target on the bound) and one held inside the window: neither presses on a stop.
    state = _run(compliance=compliance, soft=True, push=0.0, scales=(torch.ones(SIDES), torch.ones(SIDES)))
    assert state["qd"].abs().max() < (0.03 if compliance else 2e-3), "not settled"
    print(f"[no push compliance={compliance}] overshoot {state['overshoot'].tolist()}; drive torque {state['drive_torque'].tolist()}; "
          f"peak stop torque {float(state['peak'].max()):.4f} N m")
    assert state["overshoot"].abs().max() < 3e-4
    assert state["drive_torque"].max() < 0.5
    assert state["peak"].max() < 0.6  # the stop touches at most the settling's last 0.3 mrad (k 2000 N m/rad)


@pytest.mark.physics
@pytest.mark.parametrize("compliance", [False, True])
def test_a_stiff_stop_reproduces_the_hard_stop(compliance):
    _cuda()
    hard = _run(compliance=compliance, soft=False)
    assert hard["stiffness"] is None  # no soft stops built: today's hard limits
    stiff_ratio, stiff_damping = 3.0, 0.5  # 6000 N m/rad and 1 N m s/rad: the stiffest the 1 ms step carries
    stiff = _run(compliance=compliance, scales=(torch.full((SIDES,), stiff_ratio), torch.full((SIDES,), stiff_damping)))
    nominal = _run(compliance=compliance, soft=True)  # the profile's nominal through SimCore alone
    settled = 0.03 if compliance else 2e-3  # the compliance springs ring down slowly
    assert max(float(r["qd"].abs().max()) for r in (hard, stiff, nominal)) < settled, "not settled"
    # The hard stop holds the crank at its bound; the stiff one lets it past by torque over stiffness.
    assert hard["overshoot"].abs().max() < (4e-3 if compliance else 1e-3)  # the springs keep ringing down slowly
    k_stiff = stiff_ratio * _nominal("stiffness_nm_per_rad")
    torch.testing.assert_close(stiff["overshoot"], hard["overshoot"], rtol=0, atol=5e-3)
    assert (stiff["overshoot"] - hard["overshoot"]).abs().max() <= 1.2 * float(hard["drive_torque"].max()) / k_stiff + 3e-4
    # The same crank angles (every crank, drive frame) and the same torques.
    torch.testing.assert_close(stiff["position"], hard["position"], rtol=0, atol=6e-3)
    torch.testing.assert_close(stiff["torque"], hard["torque"], rtol=0.03, atol=0.3)
    torch.testing.assert_close(stiff["q"], hard["q"], rtol=0, atol=6e-3)
    print(f"[stiff-vs-hard compliance={compliance}] hard overshoot max {float(hard['overshoot'].abs().max()) * 1e3:.3f} mrad; "
          f"stiff (6000 N m/rad) {stiff['overshoot'].tolist()} rad vs predicted {(stiff['drive_torque'] / 6000).tolist()}; "
          f"max |crank angle diff| stiff-hard {float((stiff['position'] - hard['position']).abs().max()) * 1e3:.3f} mrad, "
          f"nominal-hard {float((nominal['position'] - hard['position']).abs().max()) * 1e3:.3f} mrad; "
          f"max |torque diff| stiff-hard {float((stiff['torque'] - hard['torque']).abs().max()):.3f} N m, "
          f"nominal-hard {float((nominal['torque'] - hard['torque']).abs().max()):.3f} N m; "
          f"hard torque {hard['drive_torque'].tolist()}")
    # The nominal is a visibly softer stop that still holds within a degree.
    assert nominal["overshoot"].max() > stiff["overshoot"].max()
    torch.testing.assert_close(nominal["overshoot"], hard["overshoot"], rtol=0, atol=0.017)
    torch.testing.assert_close(nominal["torque"], hard["torque"], rtol=0.05, atol=0.5)


@pytest.mark.physics
def test_damping_reaches_the_solver_and_takes_the_energy_out_of_an_impact():
    _cuda()
    ones = torch.ones(SIDES)
    low, high = torch.full((SIDES,), 0.1), torch.full((SIDES,), 1.25)
    # Hit the stops with the drives' full push (2 rad, 30 N m cap) with little and with much damping.
    first = _run(scales=(ones, low), push=2.0)
    second = _run(scales=(ones, high), push=2.0)
    torch.testing.assert_close(first["damping"][:, 0], low.to("cuda:0") * _nominal("damping_nm_s_per_rad"))
    torch.testing.assert_close(second["damping"][:, 0], high.to("cuda:0") * _nominal("damping_nm_s_per_rad"))
    assert first["qd"].abs().max() < 2e-3 and second["qd"].abs().max() < 2e-3, "not settled"
    # The settled overshoot is the same (it is torque over stiffness); the impact's deepest reach is not.
    torch.testing.assert_close(first["overshoot"], second["overshoot"], rtol=0.03, atol=3e-4)
    print(f"[damping] impact reach beyond the settled overshoot: damping x0.1 {(first['deepest'] - first['overshoot']).mean() * 1e3:.2f} mrad, "
          f"x1.25 {(second['deepest'] - second['overshoot']).mean() * 1e3:.2f} mrad")
    assert (first["deepest"] - first["overshoot"]).mean() > (second["deepest"] - second["overshoot"]).mean() + 1e-3


@pytest.mark.physics
def test_the_training_path_applies_the_stops_too():
    """RobotPlant.pre_physics (without SimCore.step) adds the torques."""
    _cuda()
    from humanoid_sim.model.usd import crank_joints
    from humanoid_sim.randomization import ParamSpec
    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    sim = SimCore(built_usd("ethrc_v1"), device="cuda:0", num_envs=2, fixed_base=True, solver="kamino",
                  gravity_m_s2=(0.0, 0.0, 0.0), cuda_graph=False)
    try:
        plant = RobotPlant(sim.backend.robot, TrainPlantCfg(reporting=False, randomization=[
            ParamSpec("crank_stop_stiffness", "uniform", (1.0, 1.0), "scale")]))
        drives = list(crank_joints(sim.desc))
        ids = [plant.drives.index(d) for d in drives]
        upper = torch.tensor([sim.desc.drives[d].position_limit_device[1] for d in drives], device="cuda:0")
        target = plant.truth().drive.position_rad.clone()
        for e in range(2):  # one crank per env: pushing both of an ankle's cranks drives the hinges into their box
            target[e, ids[e]] = upper[e] + PUSH_RAD
        plant.submit(DriveDemand.pd(target, plant.core.drive_stiffness, plant.core.drive_damping),
                     torch.zeros(2, dtype=torch.int64, device="cuda:0"))
        for _ in range(STEPS):
            plant.pre_physics()
            sim.backend.step()
        over = (plant.truth().drive.position_rad[:, ids] - upper).clamp(min=0)[torch.arange(2), torch.arange(2)]
        stops = plant._kamino_stops
        assert stops is sim.backend.robot._kamino_crank_stops and stops.torque[torch.arange(2), torch.arange(2)].min() > 5.0
        print(f"[training path] overshoot {over.tolist()}, stop torque {stops.torque[torch.arange(2), torch.arange(2)].tolist()}")
        assert over.min() > 1e-3 and over.max() < 0.05
    finally:
        sim.close()


@pytest.mark.physics
@pytest.mark.parametrize("compliance", [False, True])
def test_the_stops_stay_stable_across_the_stiffness_and_damping_range_under_a_shaking_load(compliance):
    """Gravity on, a 10 N m hinge torque shaking at 8 Hz on top of the drives' push past the stops (the stops load and
    unload with the shaking), the softest and stiffest stops and the least and most damping the plant allows."""
    _cuda()
    from humanoid_sim.randomization import ParamSpec

    specs = [ParamSpec("crank_stop_stiffness", "discrete_uniform", (0.05, 3.0), "scale", quantum=2.95),
             ParamSpec("crank_stop_damping", "discrete_uniform", (0.25, 1.25), "scale", quantum=1.0)]
    state = _run(compliance=compliance, specs=specs, gravity=(0.0, 0.0, -9.81), shake=(10.0, 8.0), steps=2000,
                 envs=16, push=0.3)
    k = [round(v, 2) for v in (state["stiffness"][:, 0] / _nominal("stiffness_nm_per_rad")).tolist()]
    b = [round(v, 2) for v in (state["damping"][:, 0] / _nominal("damping_nm_s_per_rad")).tolist()]
    assert {(0.05, 0.25), (0.05, 1.25), (3.0, 0.25), (3.0, 1.25)} <= set(zip(k, b, strict=True))  # all four corners
    assert torch.isfinite(state["q"]).all() and torch.isfinite(state["qd"]).all()
    assert state["worst_rate"] < 60.0
    # Bounded: no crank past the backstop's range, and the stops were in work in every env.
    assert state["overshoot"].abs().max() < _nominal("backstop_rad")
    assert state["peak"].max(dim=1).values.min() > 1.0
    # Stable means the late crank state is no larger than the early one's bound: nothing ringing at the step.
    assert state["qd"].abs().max() < 20.0
    print(f"[shaking compliance={compliance}] peak joint speed {float(state['worst_rate']):.2f} rad/s; deepest overshoot "
          f"{float(state['deepest'].max()):.4f} rad; peak stop torque per env {state['peak'].max(dim=1).values.tolist()}")
