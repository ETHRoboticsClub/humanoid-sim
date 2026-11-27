"""Ankle crank backlash on the solvers: a static load rests each crank's output shaft on the stop it pushes against;
the per-env widths reach the solver; the reported position is the output shaft while the drive holds its motor side;
motor minus output steps by the play where the drive torque changes sign, as on the robot's captures; and the play
stays stable with the linkage springs under a shaking load; springs too stiff for the output flange the play frees are
refused.

Each check hangs the V1 by its base, holds the cranks with their drives' baseline PD and puts a torque on the ankle
hinges. Kamino's stops are hard constraints; MJWarp's are soft (the profile's ``ankle_backlash_solref``), so on MJWarp
the shaft also sinks past the stop it rests on and the checks bound the play from below only.
"""
import math

import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand

WIDTH = 0.008  # rad, the nominal play of every crank in these checks
STEPS = 3000  # 3 s at the V1's 1 ms
# Hinge torques (pitch, roll), N m, the same on both ankles: none, toes up, toes down, roll either way.
LOADS = ((0.0, 0.0), (8.0, 0.0), (-8.0, 0.0), (0.0, 5.0), (0.0, -5.0))
LOADED_NM = 0.3  # a crank torque past which the shaft rests on a stop (the captures' plateau threshold)
SPRINGS = {"stiffness_nm_per_rad": 600.0, "damping_nm_s_per_rad": 0.3}


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


def _run(solver, *, compliance: bool = False, specs=(), gravity=(0.0, 0.0, 0.0), loads=LOADS, steps=STEPS,
         load_hz: float | None = None, record: bool = False):
    """Hang the robot, hold its cranks, load its hinges (steadily, or as a sine at ``load_hz``) and step. Returns the
    last state, and with ``record`` every step's motor-minus-output and crank torque; the simulation is closed (one
    Isaac Lab simulation per process at a time)."""
    import isaaclab.sim as sim_utils

    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    overrides = {"ankle_backlash": {"width_rad": WIDTH}, "ankle_compliance": SPRINGS}
    usd = built_usd("ethrc_v1", ankle_backlash=True, ankle_compliance=compliance)
    try:
        sim = SimCore(usd, device="cuda:0", num_envs=len(loads), fixed_base=True, solver=solver,
                      gravity_m_s2=gravity, physics_overrides=overrides)
        plant = RobotPlant(sim.backend.robot, TrainPlantCfg(reporting=False, feedback_sampling=False,
                                                            randomization=list(specs), physics_overrides=overrides))
        sim.write_joint_state(*sim.joint_state())
        desc, core = sim.desc, sim.plant
        hinges = [j for lk in desc.linkages for j in lk.joints]
        ids = torch.tensor([core.map.joint_names.index(j) for j in hinges], device="cuda:0")
        cols = [core.solver_map.joint_names.index(j) for j in hinges]
        # Next to nothing rather than zero (Kamino keeps a joint's friction structure from its build): the hinges'
        # friction would hold part of the load.
        core.friction_relief[:, cols] = 0.0
        sim.backend.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.full((len(loads), len(cols)), 1e-6, device="cuda:0"),
            joint_ids=sim._to_solver[cols])
        load = torch.tensor(loads, device="cuda:0", dtype=core.cfg.dtype).repeat(1, len(desc.linkages))
        cranks = core.crank_drives
        feedback = core.feedback(*sim.solver_state())
        core.submit(DriveDemand.pd(feedback.position_rad.clone(), core.drive_stiffness, core.drive_damping),
                    torch.zeros(len(loads), dtype=torch.int64, device="cuda:0"))
        trace = []
        travel = torch.zeros((), device="cuda:0")  # the play's largest change over one step
        play = core.backlash_rad.clone()
        for step in range(steps):
            scale = 1.0 if load_hz is None else math.sin(2 * math.pi * load_hz * step * sim.period_ns * 1e-9)
            core.external_torque[:, ids] = load * scale
            feedback = sim.step()
            travel = torch.maximum(travel, (core.backlash_rad - play).abs().max())
            play = core.backlash_rad.clone()
            if record:
                motor = core.post_state[0][:, cranks]
                trace.append(torch.stack([motor - feedback.position_rad[:, cranks],
                                          feedback.applied_torque_nm[:, cranks]], -1))
        truth = plant.truth()
        width = plant.params().get("ankle_backlash")
        return {"q": truth.joint_pos.clone(), "qd": truth.joint_vel.clone(), "motor": core.post_state[0][:, cranks].clone(),
                "position": feedback.position_rad[:, cranks].clone(),
                "torque": feedback.applied_torque_nm[:, cranks].clone(),
                "play": truth.ankle_backlash_rad.clone(),
                "width": (torch.full_like(truth.ankle_backlash_rad, WIDTH) if width is None else width.clone()),
                "limits": plant.robot.data.joint_pos_limits.torch[:, plant.backlash_ids].clone(),
                "trace": torch.stack(trace, 1) if record else None, "travel": float(travel)}
    finally:
        # What SimCore.close does, here also after a build that failed half way (no SimCore to close): a simulation
        # left behind would fail every later test of the process.
        sim_utils.SimulationContext.clear_instance()


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_a_static_load_rests_each_shaft_on_the_stop_it_pushes_against(solver):
    _cuda()
    from humanoid_sim.randomization import ParamSpec
    from humanoid_sim.transmissions.backlash import limits, static_play

    specs = [ParamSpec("ankle_backlash", "uniform", (0.005, 0.012), "set")]
    state = _run(solver, specs=specs)
    width = state["width"]
    assert width[:, 0].unique().numel() == len(LOADS)
    torch.testing.assert_close(state["limits"], limits(width).to(state["limits"].dtype))  # per env, in the solver
    assert state["qd"].abs().max() < 2e-3, "not settled"
    # The reported position is the output shaft: the motor side the drive holds plus the play.
    torch.testing.assert_close(state["position"] - state["motor"], state["play"], rtol=0, atol=1e-6)
    loaded = state["torque"].abs() > LOADED_NM
    assert loaded[1:].any(-1).all() and not loaded[0].any()
    expected = static_play(state["torque"], width)
    if solver == "kamino":  # hard stops: on the stop, within the band
        torch.testing.assert_close(state["play"][loaded], expected[loaded], rtol=0, atol=2e-4)
        assert (state["play"].abs() <= width / 2 + 2e-4).all()
    else:  # soft stops: on the stop's side, at or past it
        assert (torch.sign(state["play"][loaded]) == torch.sign(expected[loaded])).all()
        assert (state["play"][loaded].abs() >= 0.9 * width[loaded] / 2).all()


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_motor_minus_output_steps_by_the_play_where_the_torque_changes_sign(solver):
    """The captures' measurement: a slow sine load on the hinges, motor minus output encoder against the crank torque,
    the plateaus' medians for torques beyond ±0.3 N m."""
    _cuda()
    state = _run(solver, loads=((8.0, 4.0),), steps=4000, load_hz=0.5, record=True)
    gap, torque = state["trace"][0, 1000:].unbind(-1)  # past the first quarter period
    for c in range(gap.shape[1]):
        push, pull = torque[:, c] > LOADED_NM, torque[:, c] < -LOADED_NM
        assert push.sum() > 200 and pull.sum() > 200
        step = float(gap[push, c].median() - gap[pull, c].median())
        if solver == "kamino":
            assert step == pytest.approx(WIDTH, rel=0.05)
        else:
            assert step >= 0.95 * WIDTH


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
@pytest.mark.parametrize("compliance", [False, True])
def test_the_play_stays_stable_under_a_shaking_load(solver, compliance):
    """Gravity on, a 10 N m hinge torque shaking at 8 Hz, the narrowest and widest play a fit could ask for."""
    _cuda()
    from humanoid_sim.randomization import ParamSpec

    specs = [ParamSpec("ankle_backlash", "discrete_uniform", (0.002, 0.02), "set", quantum=0.018)]
    state = _run(solver, compliance=compliance, specs=specs, gravity=(0.0, 0.0, -9.81), loads=((10.0, 4.0),) * 8,
                 steps=1000, load_hz=8.0)
    assert {0.002, 0.02} <= {round(w, 6) for w in state["width"].flatten().tolist()}
    assert torch.isfinite(state["q"]).all() and torch.isfinite(state["qd"]).all()
    assert state["qd"].abs().max() < 50.0
    # Within the band, plus what a stop lets through: on Kamino one step of the play's travel (its hard stop acts
    # once reached), on MJWarp what a soft stop gives under the load.
    slack = state["travel"] + 2e-4 if solver == "kamino" else 0.1
    assert (state["play"].abs() <= state["width"] / 2 + slack).all()


@pytest.mark.physics
def test_springs_too_stiff_for_the_freed_flange_are_refused_before_the_simulation():
    _cuda()
    from humanoid_sim.model.asset import backlash_output_inertia, load_profile
    from humanoid_sim.sim import SimCore
    from humanoid_sim.transmissions.backlash import STABLE_MEASURE

    usd = built_usd("ethrc_v1", ankle_backlash=True, ankle_compliance=True)
    dt = load_profile("ethrc_v1")["physics"]["period_ns"] * 1e-9
    stiff = 2 * STABLE_MEASURE * backlash_output_inertia(usd) / dt**2
    with pytest.raises(ValueError, match="output_inertia_kg_m2"):
        SimCore(usd, device="cuda:0", num_envs=1, fixed_base=True, physics_overrides={
            "ankle_backlash": {"width_rad": WIDTH},
            "ankle_compliance": {"stiffness_nm_per_rad": stiff, "damping_nm_s_per_rad": SPRINGS["damping_nm_s_per_rad"]}})
