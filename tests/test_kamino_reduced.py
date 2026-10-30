"""Explicit reduced-solver plumbing and incompatible-mode rejection in the physics image."""
import dataclasses

import pytest

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("options", [
    {"algorithm": "full"},
    {"algorithm": "bilateral_reduced"},
    {"algorithm": "bilateral_reduced", "reduced_capacity": 64},
    {"reduced_capacity": 32},
    {"algorithm": "bilateral_reduced", "reduced_ordering": "rcm"},
    {"algorithm": "bilateral_reduced", "reduced_ordering": "rcm", "reduced_dynamic_pattern": True},
    {"reduced_ordering": "natural", "reduced_dynamic_pattern": False},
    {"reduced_sparse_diagnostics": False},
    {"algorithm": "bilateral_reduced", "reduced_sparse_diagnostics": True},
    {"reduced_lazy_preconditioning": False},
    {"algorithm": "bilateral_reduced", "reduced_ordering": "rcm", "reduced_dynamic_pattern": True, "reduced_lazy_preconditioning": True},
])
def test_algorithm_bridge_preserves_other_solver_parameters(options):
    from humanoid_sim.physics.isaaclab import solver_cfg

    settings = {"integrator": "euler", "iterations": 19, "max_contacts_per_world": 3,
                "tracked_active_clear": True, "matrix_preconditioner_stride": 128,
                "use_graph_conditionals": True, "padmm": {"rho_0": 0.08}}
    baseline = solver_cfg("kamino", settings).to_solver_config()
    configured = solver_cfg("kamino", settings | {"padmm": settings["padmm"] | options}).to_solver_config()
    assert configured.padmm.algorithm == options.get("algorithm", "full")
    assert configured.padmm.reduced_capacity == options.get("reduced_capacity", 96)
    assert configured.padmm.reduced_ordering == options.get("reduced_ordering", "natural")
    assert configured.padmm.reduced_dynamic_pattern == options.get("reduced_dynamic_pattern", False)
    assert configured.padmm.reduced_sparse_diagnostics == options.get("reduced_sparse_diagnostics", False)
    assert configured.padmm.reduced_lazy_preconditioning == options.get("reduced_lazy_preconditioning", False)
    configured.padmm.reduced_lazy_preconditioning = baseline.padmm.reduced_lazy_preconditioning
    configured.padmm.reduced_sparse_diagnostics = baseline.padmm.reduced_sparse_diagnostics
    configured.padmm.reduced_ordering = baseline.padmm.reduced_ordering
    configured.padmm.reduced_dynamic_pattern = baseline.padmm.reduced_dynamic_pattern
    configured.padmm.algorithm = baseline.padmm.algorithm
    configured.padmm.reduced_capacity = baseline.padmm.reduced_capacity
    assert dataclasses.asdict(configured) == dataclasses.asdict(baseline)


@pytest.mark.parametrize("changed, constructor, message", [
    ({"algorithm": "unknown"}, {}, "explicit PADMM algorithm"),
    ({"reduced_capacity": 33}, {}, "reduced_capacity"),
    ({"reduced_ordering": "unknown"}, {}, "reduced_ordering"),
    ({"reduced_dynamic_pattern": True, "reduced_ordering": "natural"}, {}, "requires reduced_ordering=rcm"),
    ({}, {"use_graph_conditionals": False}, "graph conditionals"),
    ({}, {"use_acceleration": False}, "acceleration"),
    ({}, {"collect_info": True}, "collect_info"),
    ({"penalty_update_method": "balanced"}, {}, "fixed-penalty"),
])
def test_reduced_rejects_incompatible_modes_before_solve(changed, constructor, message):
    # A small real fixture supplies model metadata; these are mechanism constraints,
    # not robot-specific joint counts or gains.
    from kamino_reduced_fixture import make
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    model, _ = make()
    default = PADMMSolver(model=model)
    assert default._reduced_runtime is None
    if changed.get("penalty_update_method") == "balanced":
        # This is a supported full-PADMM mode, rejected specifically by reduced
        # execution. A nonexistent method would fail earlier and miss this gate.
        adaptive = PADMMSolver(model=model, config=PADMMSolverConfig(penalty_update_method="balanced"))
        assert adaptive._reduced_runtime is None
    cfg = PADMMSolverConfig(algorithm="bilateral_reduced")
    for key, value in changed.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match=message):
        PADMMSolver(model=model, config=cfg, **constructor)


@pytest.mark.parametrize("eta", [1e-5, 1e-4])
def test_reduced_sigma_matches_initialized_source_and_factor(eta):
    """The operator uses the source GPU config rounding, including the 1-ULP case."""
    import numpy as np
    import warp as wp
    from kamino_reduced_fixture import make
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_direct_factor import make_runtime_direct_factor
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    model, problem = make()
    cfg = PADMMSolverConfig(algorithm="bilateral_reduced", eta=eta, rho_0=0.05,
                           primal_tolerance=1e-4, dual_tolerance=1e-4,
                           compl_tolerance=1e-4, max_iterations=100)
    solver = PADMMSolver(model=model, config=cfg, use_graph_conditionals=True)
    runtime = solver._reduced_runtime
    n = int(problem.data.dim.numpy()[0])
    free = int(problem.data.njc.numpy()[0])
    raw = np.zeros(problem.data.D.shape[0], dtype=np.float32)
    raw[:n*n] = (np.eye(n, dtype=np.float32) * 2.0).ravel()
    pristine = wp.array(raw, device=model.device)
    q = np.zeros(problem.data.v_f.shape[0], dtype=np.float32)
    q[:n] = np.linspace(-0.2, 0.2, n, dtype=np.float32)
    problem.data.v_f.assign(q)

    def solve():
        wp.copy(problem.data.D, pristine)
        solver.coldstart()
        solver.solve(problem)

    solve()
    wp.synchronize_device()
    initialized_sigma = solver.data.state.sigma.numpy()[0, 0]
    expected = np.float32(np.float32(eta) + np.float32(cfg.rho_0))
    assert initialized_sigma.view(np.uint32) == expected.view(np.uint32)
    assert np.float32(runtime.sigma).view(np.uint32) == initialized_sigma.view(np.uint32)
    if eta == 1e-4:
        assert np.float32(eta + cfg.rho_0).view(np.uint32) != initialized_sigma.view(np.uint32)
    assert not runtime.bad.numpy().any()

    # A separate factor launch consumes sigma produced by the original GPU
    # initializer, not the corrected runtime scalar. Compare every lower entry.
    reference = wp.zeros_like(runtime.L)
    wp.launch(make_runtime_direct_factor(16), dim=(1, 128),
              inputs=[runtime.dim, runtime.mio, pristine, problem.data.dim, problem.data.mio,
                      runtime.bad, free, float(initialized_sigma), reference],
              device=model.device, block_dim=128)
    wp.synchronize_device()
    lower = np.tril_indices(n)
    actual = runtime.L.numpy()[:n*n].reshape(n, n)[lower]
    expected_factor = reference.numpy()[:n*n].reshape(n, n)[lower]
    np.testing.assert_array_equal(actual.view(np.uint32), expected_factor.view(np.uint32))
    # Same well-posed closed-form fixture used by the reduced runtime smoke.
    expected_solution = np.clip(-q[:n] / 2.0, -0.2, 0.2)
    result = solver.data.solution.lambdas.numpy()[:n].copy()
    np.testing.assert_allclose(result, expected_solution, atol=2e-4, rtol=2e-4)
    with wp.ScopedCapture(device=model.device) as capture:
        solve()
    wp.capture_launch(capture.graph)
    wp.synchronize_device()
    np.testing.assert_array_equal(solver.data.solution.lambdas.numpy()[:n], result)
