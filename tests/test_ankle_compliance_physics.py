"""Ankle linkage compliance on the solvers: the crank springs deflect the foot as their statics say, not at all
without a load, and like the rigid linkage when stiff; their per-env knobs reach the solver; the drive feedback stays
the crank reading.

Each check hangs the V1 by its base without gravity, holds the cranks with their drives' baseline PD and puts a
static torque on the ankle hinges, with the hinges' Coulomb friction removed (it would hold part of a static load).
The foot's deflection is its hinge angles less the pose the cranks' readings imply through the rigid linkage. Every
spring deflection is taken against the rigid asset under the same load, so the solver's own loop compliance (MJWarp
closes the rods with a soft constraint) is not counted as the spring's.
"""
import math

import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand

STEPS = 2000  # 2 s at the V1's 1 ms: the held cranks settle in well under 1 s
NOMINAL = {"stiffness_nm_per_rad": 600.0, "damping_nm_s_per_rad": 0.3}
# Hinge torques (pitch, roll), N m, the same on both ankles: none, toes up, toes down with roll, roll alone.
LOADS = ((0.0, 0.0), (5.0, 0.0), (-5.0, 2.0), (0.0, 3.0))
# A spring stiff enough to stand for the rigid linkage on each solver (MJWarp's rk4 is explicit: see the profile).
STIFF = {"mjwarp": 4000.0, "kamino": 1e5}


@pytest.fixture(autouse=True)
def clean_solver_output(capfd):
    yield
    output = "".join(capfd.readouterr())
    assert "possibly invalid inertia tensor" not in output
    assert "exceeded MJWarp limit" not in output


def _cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("physics checks run on CUDA")


def _settle(request, solver, *, compliance: bool, stiffness: dict | None = None, specs=(), gravity=(0.0, 0.0, 0.0),
            loads=LOADS, steps=STEPS, shake_hz: float | None = None):
    """Hang the robot, hold its cranks, load its hinges and step. Returns the settled state per env."""
    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    gains = {**NOMINAL, **(stiffness or {})}
    overrides = {"ankle_compliance": gains}
    sim = SimCore(built_usd("ethrc_v1", ankle_compliance=compliance), device="cuda:0", num_envs=len(loads),
                  fixed_base=True, solver=solver, gravity_m_s2=gravity, physics_overrides=overrides)
    request.addfinalizer(sim.close)
    plant = RobotPlant(sim.backend.robot, TrainPlantCfg(reporting=False, feedback_sampling=False,
                                                        randomization=list(specs), physics_overrides=overrides))
    sim.write_joint_state(*sim.joint_state())
    desc, core = sim.desc, sim.plant
    hinges = [j for lk in desc.linkages for j in lk.joints]
    ids = torch.tensor([core.map.joint_names.index(j) for j in hinges], device="cuda:0")
    cols = [core.solver_map.joint_names.index(j) for j in hinges]
    core.enabled_friction[:, cols] = 0.0
    core.friction_relief[:, cols] = 0.0
    core.friction_changed = True
    load = torch.tensor(loads, device="cuda:0", dtype=core.cfg.dtype).repeat(1, len(desc.linkages))
    feedback = core.feedback(*sim.solver_state())
    core.submit(DriveDemand.pd(feedback.position_rad.clone(), core.drive_stiffness, core.drive_damping),
                torch.zeros(len(loads), dtype=torch.int64, device="cuda:0"))
    for step in range(steps):
        phase = 2 * math.pi * (shake_hz or 0.0) * step * sim.period_ns * 1e-9
        core.external_torque[:, ids] = load * (1.0 if shake_hz is None else math.sin(phase))
        feedback = sim.step()
    q, qd = sim.joint_state()
    stiffness = None
    if compliance:
        stiffness = plant.params().get("ankle_spring_stiffness")
        if stiffness is None:
            stiffness = torch.full((len(loads), len(plant.cranks)), gains["stiffness_nm_per_rad"], device="cuda:0")
    return {"q": q, "qd": qd, "feedback": feedback, "desc": desc, "plant": plant, "core": core, "load": load,
            "truth": plant.truth() if compliance else None, "stiffness": stiffness}


def _per_side(state):
    """Per ankle: hinge angles, crank readings and drive torques, the readings' rigid pose, J there, and each
    crank's spring stiffness (None for the rigid asset)."""
    desc, core, fb = state["desc"], state["core"], state["feedback"]
    sides = []
    for k, lk in enumerate(desc.linkages):
        tr = desc.ankle_transmission(lk.side, device="cuda:0", dtype=torch.float64)
        q = state["q"][:, [core.map.joint_names.index(j) for j in lk.joints]].double()
        drives = [core.map.drive_names.index(d) for d in (lk.upper, lk.lower)]
        theta = fb.position_rad[:, drives].double()
        rigid = tr.joints_from_cranks(theta, q, iterations=20)
        torch.testing.assert_close(tr.cranks(rigid), theta, atol=1e-9, rtol=0)
        stiffness = None
        if state["stiffness"] is not None:
            cranks = state["plant"].cranks
            stiffness = state["stiffness"][:, [cranks.index(lk.upper), cranks.index(lk.lower)]].double()
        sides.append({"q": q, "deflection": q - rigid, "jac": tr.jacobian(q), "torque": fb.applied_torque_nm[:, drives].double(),
                      "load": state["load"][:, 2 * k:2 * k + 2].double(), "stiffness": stiffness,
                      "spring": None if state["truth"] is None else state["truth"].ankle_spring_rad[
                          :, [state["plant"].cranks.index(lk.upper), state["plant"].cranks.index(lk.lower)]].double()})
    return sides


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_a_static_load_deflects_the_foot_by_the_springs_compliance(request, solver):
    _cuda()
    from humanoid_sim.randomization import ParamSpec
    from humanoid_sim.transmissions.compliance import deflection_from_drive_torque, deflection_under_load

    rigid = _per_side(_settle(request, solver, compliance=False))
    for side in rigid:
        # The solver's own loop compliance is small next to the springs' (MJWarp's soft closures; none on Kamino).
        assert side["deflection"].abs().max() < 1e-3
    # Per env and crank stiffness through the plant's knob (a solver write), damping at the nominal.
    specs = [ParamSpec("ankle_spring_stiffness", "log_uniform", (300.0, 1500.0), "set")]
    state = _settle(request, solver, compliance=True, specs=specs)
    k = state["plant"].params()["ankle_spring_stiffness"]
    torch.testing.assert_close(state["plant"].robot.data.joint_stiffness.torch[:, state["plant"].spring_ids], k.float())
    assert k[:, 0].unique().numel() == len(LOADS)
    for side, base in zip(_per_side(state), rigid, strict=True):
        assert state["qd"].abs().max() < 1e-3, "not settled"
        spring = side["deflection"] - base["deflection"]
        expected = deflection_under_load(side["jac"], side["stiffness"], side["load"])
        # Zero load, zero deflection; loaded, the statics within 3 % (plus 0.1 mrad).
        assert spring[0].abs().max() < 1e-4 and side["spring"][0].abs().max() < 1e-4
        assert expected[1:].abs().max() > 5e-3
        torch.testing.assert_close(spring, expected, rtol=0.03, atol=1e-4)
        # The same deflection from the drives' own torques (what a fit sees), and each spring's angle from its torque.
        torch.testing.assert_close(deflection_from_drive_torque(side["jac"], side["stiffness"], side["torque"]),
                                   expected, rtol=0.03, atol=1e-4)
        torch.testing.assert_close(side["spring"], -side["torque"] / side["stiffness"], rtol=0.03, atol=2e-5)


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_stiff_springs_reproduce_the_rigid_linkage(request, solver):
    _cuda()
    rigid_state = _settle(request, solver, compliance=False)
    rigid = _per_side(rigid_state)
    stiff_state = _settle(request, solver, compliance=True, stiffness={"stiffness_nm_per_rad": STIFF[solver]})
    stiff = _per_side(stiff_state)
    from humanoid_sim.transmissions.compliance import deflection_under_load

    for side, base in zip(stiff, rigid, strict=True):
        left = deflection_under_load(side["jac"], side["stiffness"], side["load"])
        assert left.abs().max() < 2e-3  # what the stiff springs still allow
        torch.testing.assert_close(side["q"], base["q"], rtol=0, atol=float(left.abs().max()) + 3e-4)
        torch.testing.assert_close(side["deflection"] - base["deflection"], left, rtol=0.1, atol=1e-4)
    # The cranks' feedback is the drives' own: the same readings and torques as the rigid linkage's.
    torch.testing.assert_close(stiff_state["feedback"].applied_torque_nm, rigid_state["feedback"].applied_torque_nm,
                               rtol=0.02, atol=0.02)


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_springs_stay_stable_at_the_ends_of_their_range_under_a_shaking_load(request, solver):
    """Gravity on, a 10 N m hinge torque shaking at 8 Hz, the softest and stiffest springs a fit could ask for."""
    _cuda()
    from humanoid_sim.randomization import ParamSpec

    specs = [ParamSpec("ankle_spring_stiffness", "discrete_uniform", (100.0, STIFF[solver]),
                       "set", quantum=STIFF[solver] - 100.0),
             ParamSpec("ankle_spring_damping", "discrete_uniform", (0.05, 1.0), "set", quantum=0.95)]
    loads = ((10.0, 4.0),) * 8
    state = _settle(request, solver, compliance=True, specs=specs, gravity=(0.0, 0.0, -9.81), loads=loads,
                    steps=1000, shake_hz=8.0)
    k = state["plant"].params()["ankle_spring_stiffness"]
    assert {100.0, STIFF[solver]} <= set(k.flatten().tolist())
    assert torch.isfinite(state["q"]).all() and torch.isfinite(state["qd"]).all()
    spring = state["truth"].ankle_spring_rad
    # Bounded by the static deflection a 10 N m hinge torque gives the softest spring, with dynamic headroom.
    assert spring.abs().max() < 0.2 and state["qd"].abs().max() < 50.0
