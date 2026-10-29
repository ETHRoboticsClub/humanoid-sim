"""Solver selection from the profile's physics block (no Isaac Lab needed)."""

import pytest

from humanoid_sim.model.asset import load_profile
from humanoid_sim.physics.isaaclab import SOLVERS, solver_settings


def test_profile_selects_its_solver_and_has_a_block_per_solver():
    physics = load_profile("ethrc_v1")["physics"]
    assert solver_settings(physics)[0] == physics["solver"]
    for name in SOLVERS:
        solver, settings = solver_settings(physics, name)
        assert solver == name and "integrator" in settings and "iterations" in settings


def test_unknown_or_unconfigured_solver_raises():
    physics = load_profile("ethrc_v1")["physics"]
    with pytest.raises(ValueError, match="physics solver"):
        solver_settings(physics, "physx")
    with pytest.raises(ValueError, match="no 'kamino' block"):
        solver_settings({k: v for k, v in physics.items() if k != "kamino"}, "kamino")


def test_explicit_physics_overrides_merge_without_mutating_inputs_or_files(tmp_path):
    defaults = "physics:\n  period_ns: 2000000\n  kamino:\n    iterations: 7\n    padmm: {rho_0: 0.1}\n"
    (tmp_path / "defaults.yaml").write_text(defaults)
    (tmp_path / "fixture.yaml").write_text("robot: fixture\nphysics:\n  kamino:\n    iterations: 11\n")
    overrides = {"kamino": {"iterations": 19, "dvi": {"tolerance": 0.002}}}
    result = load_profile("fixture", tmp_path, physics_overrides=overrides)
    assert result["physics"] == {"period_ns": 2000000, "kamino": {
        "iterations": 19, "padmm": {"rho_0": 0.1}, "dvi": {"tolerance": 0.002}}}
    result["physics"]["kamino"]["dvi"]["tolerance"] = 0.5
    assert overrides["kamino"]["dvi"]["tolerance"] == 0.002
    assert load_profile("fixture", tmp_path)["physics"]["kamino"]["iterations"] == 11
    assert (tmp_path / "defaults.yaml").read_text() == defaults


def test_model_and_simulation_use_the_same_physics_overrides(monkeypatch):
    from humanoid_sim.physics import isaaclab, model_build
    from humanoid_sim.train.scene import robot_build, robot_sim_cfg

    baseline = load_profile("ethrc_v1")
    overrides = {"period_ns": 2_000_000, "kamino": {"iterations": 17, "collapse_fixed_joints": False}}
    calls = {}

    def simulation(physics, **kwargs):
        calls["simulation"] = physics, kwargs
        return "simulation"

    def build(robot, physics, solver):
        calls["build"] = robot, physics, solver
        return "build"

    monkeypatch.setattr(isaaclab, "simulation_cfg", simulation)
    monkeypatch.setattr(model_build, "model_build", build)
    assert robot_sim_cfg(solver="kamino", physics_overrides=overrides) == "simulation"
    assert robot_build(solver="kamino", physics_overrides=overrides) == "build"
    physics, options = calls["simulation"]
    assert calls["build"] == ("ethrc_v1", physics, "kamino")
    assert options["period_s"] == .002 and options["solver"] == "kamino"
    assert physics["kamino"]["iterations"] == 17
    assert physics["kamino"]["collapse_fixed_joints"] is False
    assert load_profile("ethrc_v1") == baseline


@pytest.mark.physics
def test_native_integrator_preserves_torque_free_slender_body_spin():
    """An axisymmetric fixture has constant angular-speed magnitude without torque.

    Isolate the GPU integrator from contacts, resets and controls: slender native
    rods can acquire axial spin through contact, even with closed endpoints.
    Fixture inertias and velocities are independent of the delivered robot.
    """
    import newton
    import numpy as np
    import warp as wp
    from newton.solvers import SolverMuJoCo

    if not wp.is_cuda_available():
        pytest.skip("rotational integration regression requires CUDA")
    builder = newton.ModelBuilder()
    builder.gravity = wp.vec3(0., 0., 0.)
    body = builder.add_link(mass=.1, inertia=wp.mat33(1e-6, 0., 0., 0., 1e-4, 0., 0., 0., 1e-4))
    builder.add_articulation([builder.add_joint_free(body)])
    builder.joint_qd[:] = [0., 0., 0., 200., 20., 10.]
    model = builder.finalize(device="cuda:0")
    integrator = load_profile("ethrc_v1")["physics"]["mjwarp"]["integrator"]
    solver = SolverMuJoCo(model, integrator=integrator, use_mujoco_cpu=False)
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    initial_speed = np.linalg.norm(state.body_qd.numpy()[0, 3:])
    period_ns, duration_ns, check_period_ns = 1_000_000, 1_000_000_000, 10_000_000
    for t_ns in range(0, duration_ns, period_ns):
        solver.step(state, next_state, control, None, period_ns / 1e9)
        state, next_state = next_state, state
        if (t_ns + period_ns) % check_period_ns == 0:
            speed = np.linalg.norm(state.body_qd.numpy()[0, 3:])
            np.testing.assert_allclose(speed, initial_speed, rtol=.01, atol=0.)
