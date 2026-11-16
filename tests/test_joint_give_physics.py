"""Joint give on the solvers: a torque on a link deflects it from its drive's reading by the give's statics, none
gives none, a stiff give is the rigid joint, the per-env knobs reach the solver, the measured position stays the
drive's, a rigid asset is untouched by the give's settings, and the give stays stable under a shaking load.

Each check hangs the V1 by its base without gravity (with it, for the shaking one), holds every drive with its baseline
PD at the pose it starts in, and puts a known torque on the links of the given joints: ``+τ`` on the link and ``-τ`` on
its parent, about the joint's axis, as one internal load. The drive holds the output body, so the link sits ``τ / k``
past the drive's reading, whatever the drive's own softness lets the output move. The link's angle is also read from
the bodies' poses, independently of the joint states.
"""
import math

import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand

STEPS = 2000  # 2 s at the V1's 1 ms
NOMINAL = {"stiffness_nm_per_rad": 600.0, "damping_nm_s_per_rad": 0.5}
# Test inputs: the joints given give, by pattern, and the torques (N m) put on their links, one value per env: none,
# one way, the other, a larger one.
GIVE = ("hip_flexion_.*", "hip_abduction_.*", "knee_flexion_.*")
LOADS = (0.0, 6.0, -6.0, 12.0)
# The stiffest give each solver takes as the rigid joint and the softest and stiffest of the shaking check's range
# (MJWarp's rk4 is explicit: see the profile).
STIFF = {"mjwarp": 5000.0, "kamino": 1e5}
SOFT = 100.0
# The most damping a give may have in the shaking check.
DAMPED = {"mjwarp": 2.0, "kamino": 5.0}


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


def _names():
    from humanoid_sim.model.description import Description
    from humanoid_sim.transmissions.compliance import resolve_give

    return resolve_give(Description.load("ethrc_v1"), GIVE)


def _axes(desc, robot, names):
    """Each joint's parent and link body ids and its axis in the parent body's frame."""
    bodies = list(robot.body_names)
    parents, links, axes = [], [], []
    for name in names:
        parent, link = desc.usd_model.joint_bodies(name)
        frame0, _ = desc.usd_model.joint_frames(name)
        parents.append(bodies.index(parent))
        links.append(bodies.index(link))
        axes.append(frame0[:3, :3] @ desc.usd_model.joint_axis(name))
    return (torch.tensor(parents, device="cuda:0"), torch.tensor(links, device="cuda:0"),
            torch.tensor(axes, device="cuda:0", dtype=torch.float32))


def _link_angle(desc, robot, names, parents, links):
    """Each given joint's link angle ``[E, G]`` from the bodies' poses: the parent's rotation turned by the angle about
    the joint's axis is the link's."""
    from humanoid_sim.physics.crank_stops import _matrix

    pose = robot.data.body_link_pose_w.torch
    out = []
    for c, name in enumerate(names):
        frame0, frame1 = (torch.tensor(f[:3, :3], device="cuda:0", dtype=torch.float32)
                          for f in desc.usd_model.joint_frames(name))
        axis = torch.tensor(desc.usd_model.joint_axis(name), device="cuda:0", dtype=torch.float32)
        m = frame0.T @ _matrix(pose[:, parents[c], 3:]).transpose(-1, -2) @ _matrix(pose[:, links[c], 3:]) @ frame1
        sin = axis @ torch.stack([m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]]) / 2
        out.append(torch.atan2(sin, (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2))
    return torch.stack(out, -1)


def _settle(solver, *, give: bool, overrides=None, specs=(), gravity=(0.0, 0.0, 0.0), loads=LOADS, steps=STEPS,
            shake_hz: float | None = None):
    """Hang the robot, hold its drives, load its given joints and step. Returns the settled state per env; the
    simulation is closed (one Isaac Lab simulation per process at a time). ``give``: the asset with give (else the
    rigid one, loaded the same way)."""
    from humanoid_sim.physics.crank_stops import _rotate
    from humanoid_sim.sim import SimCore
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    names = _names()
    overrides = overrides if overrides is not None else {"joint_give": NOMINAL}
    envs = len(loads)
    sim = SimCore(built_usd("ethrc_v1", joint_give=names if give else ()), device="cuda:0", num_envs=envs,
                  fixed_base=True, solver=solver, gravity_m_s2=gravity, physics_overrides=overrides)
    try:
        plant = RobotPlant(sim.backend.robot, TrainPlantCfg(reporting=False, feedback_sampling=False,
                                                            randomization=list(specs), physics_overrides=overrides))
        sim.write_joint_state(*sim.joint_state())
        desc, core, robot = sim.desc, sim.plant, sim.backend.robot
        # The given joints' Coulomb friction would hold part of a static load: next to nothing (Kamino keeps a joint's
        # friction structure from its build, so not zero). Written here: SimCore writes only the back-drive joints'.
        cols = [core.solver_map.joint_names.index(n) for n in names]
        core.friction_relief[:, cols] = 0.0
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.full((envs, len(cols)), 1e-6, device="cuda:0"), joint_ids=sim._to_solver[cols])
        parents, links, axes = _axes(desc, robot, names)
        feedback = core.feedback(*sim.solver_state())
        held = feedback.position_rad.clone()
        core.submit(DriveDemand.pd(held, core.drive_stiffness, core.drive_damping),
                    torch.zeros(envs, dtype=torch.int64, device="cuda:0"))
        load = torch.tensor(loads, device="cuda:0", dtype=torch.float32).unsqueeze(1).expand(envs, len(names))
        everyone = torch.arange(envs, device="cuda:0")
        both = torch.cat([parents, links]).unique()
        share = torch.zeros(len(names), len(both), device="cuda:0")  # the link's share +1, its parent's -1
        for c in range(len(names)):
            share[c, (both == links[c]).nonzero()] = 1.0
            share[c, (both == parents[c]).nonzero()] = -1.0
        for step in range(steps):
            phase = 2 * math.pi * (shake_hz or 0.0) * step * sim.period_ns * 1e-9
            tau = load * (1.0 if shake_hz is None else math.sin(phase))
            axis_w = _rotate(robot.data.body_link_pose_w.torch[:, parents, 3:], axes.expand(envs, -1, -1))
            wrench = torch.einsum("egx,gb->ebx", tau.unsqueeze(-1) * axis_w, share).contiguous()
            robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(wrench), torques=wrench, body_ids=both, env_ids=everyone, is_global=True)
            feedback = sim.step()
        q, qd = (t.clone() for t in sim.joint_state())
        truth = plant.truth()
        out = {"q": q, "qd": qd, "position": feedback.position_rad.clone(), "torque": feedback.applied_torque_nm.clone(),
               "held": held, "desc": desc, "map": core.map, "names": names, "load": load.clone(), "give": None,
               "stiffness": None, "damping": None, "solver_stiffness": None, "solver_damping": None,
               "link": _link_angle(desc, robot, names, parents, links), "state_qd": truth.joint_vel.clone()}
        if give:
            out["give"] = truth.joint_give_rad.clone()
            out["solver_stiffness"] = plant.robot.data.joint_stiffness.torch[:, plant.give_ids].clone()
            out["solver_damping"] = plant.robot.data.joint_damping.torch[:, plant.give_ids].clone()
            k, c = (plant.params().get(f"joint_give_{n}") for n in ("stiffness", "damping"))
            for key, value, nominal in (("stiffness", k, NOMINAL["stiffness_nm_per_rad"]),
                                        ("damping", c, NOMINAL["damping_nm_s_per_rad"])):
                out[key] = torch.full((envs, len(names)), nominal, device="cuda:0") if value is None else value.clone()
        return out
    finally:
        sim.close()


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_a_load_deflects_the_link_by_the_gives_stiffness_and_the_reading_stays_the_drives(solver):
    _cuda()
    from humanoid_sim.randomization import ParamSpec
    from humanoid_sim.transmissions.compliance import give_deflection

    # Per env and joint stiffness through the plant's knob (a solver write), damping at the nominal.
    specs = [ParamSpec("joint_give_stiffness", "log_uniform", (300.0, 1500.0), "set")]
    state = _settle(solver, give=True, specs=specs)
    k = state["stiffness"]
    torch.testing.assert_close(state["solver_stiffness"], k.float())
    torch.testing.assert_close(state["solver_damping"], state["damping"].float())
    assert k[:, 0].unique().numel() == len(LOADS) and (k[:, 0] != k[:, 1]).any()
    assert state["state_qd"].abs().max() < 2e-3, "not settled"
    expected = give_deflection(state["load"], k)
    give = state["give"]
    # Zero load, zero give; loaded, the statics within 3 % (plus 0.1 mrad).
    assert give[0].abs().max() < 1e-4
    assert expected[1:].abs().min() > 4e-3
    torch.testing.assert_close(give, expected, rtol=0.03, atol=1e-4)
    # The link is where the bodies' poses say: the drive's reading plus the give (the output body turns with the
    # drive, the link past it), and the drive's reading is the joint's own, not the link's.
    q = state["q"][:, [list(state["desc"].joints).index(n) for n in state["names"]]]
    torch.testing.assert_close(state["link"], q + give, rtol=0, atol=2e-4)
    assert (state["link"] - q).abs().max() > 4e-3
    # The drive holds the whole load, so its torque is the load's.
    drives = [state["map"].drive_names.index(state["desc"].joints[n].device) for n in state["names"]]
    torque = state["torque"][:, drives]
    torch.testing.assert_close(torque.abs(), state["load"].abs().expand_as(torque), rtol=0.05, atol=0.1)


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_a_stiff_give_is_the_rigid_joint(solver):
    _cuda()
    rigid = _settle(solver, give=False)
    stiff = _settle(solver, give=True, overrides={"joint_give": {"stiffness_nm_per_rad": STIFF[solver],
                                                                 "damping_nm_s_per_rad": NOMINAL["damping_nm_s_per_rad"]}})
    left = stiff["load"].abs().max() / STIFF[solver]  # what the stiff give still allows
    assert float(left) < 3e-3
    torch.testing.assert_close(stiff["q"], rigid["q"], rtol=0, atol=float(left) + 5e-4)
    torch.testing.assert_close(stiff["link"], rigid["link"], rtol=0, atol=float(left) + 5e-4)
    # The drives hold the same load as the rigid joints' drives do.
    drives = [rigid["map"].drive_names.index(rigid["desc"].joints[n].device) for n in rigid["names"]]
    torch.testing.assert_close(stiff["torque"][:, drives], rigid["torque"][:, drives], rtol=0.02, atol=0.1)


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_a_rigid_asset_ignores_the_gives_settings_and_refuses_its_knobs(solver):
    _cuda()
    from humanoid_sim.randomization import ParamSpec

    off = _settle(solver, give=False, overrides={})
    on = _settle(solver, give=False, overrides={"joint_give": NOMINAL})
    for key in ("q", "qd", "position", "torque", "link"):
        torch.testing.assert_close(on[key], off[key], rtol=0, atol=1e-6)
    assert off["give"] is None
    with pytest.raises(ValueError, match="joint give"):
        _settle(solver, give=False, specs=[ParamSpec("joint_give_stiffness", "uniform", (300.0, 600.0), "set")])


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["kamino", "mjwarp"])
def test_give_stays_stable_at_the_ends_of_its_range_under_a_shaking_load(solver):
    """Gravity on, a 10 N m load shaking at 8 Hz on every given link, the softest and stiffest give a fit could ask
    for, the least and most damping the solver takes."""
    _cuda()
    from humanoid_sim.randomization import ParamSpec

    specs = [ParamSpec("joint_give_stiffness", "discrete_uniform", (SOFT, STIFF[solver]), "set",
                       quantum=STIFF[solver] - SOFT),
             ParamSpec("joint_give_damping", "discrete_uniform", (0.05, DAMPED[solver]), "set",
                       quantum=DAMPED[solver] - 0.05)]
    state = _settle(solver, give=True, specs=specs, gravity=(0.0, 0.0, -9.81), loads=(10.0,) * 8, steps=1000,
                    shake_hz=8.0)
    assert {SOFT, STIFF[solver]} <= set(state["stiffness"].flatten().tolist())
    assert {0.05, DAMPED[solver]} <= set(state["damping"].flatten().tolist())
    assert torch.isfinite(state["q"]).all() and torch.isfinite(state["qd"]).all()
    # Bounded by the static deflection a 10 N m load gives the softest give, with dynamic headroom.
    assert state["give"].abs().max() < 0.3 and state["qd"].abs().max() < 50.0
