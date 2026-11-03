"""Check active-entry preconditioning against the existing CUDA kernel."""

import pytest

pytestmark = pytest.mark.physics


def test_preconditioner_execution_config_reaches_newton_and_rejects_invalid_values():
    from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
    from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig

    from humanoid_sim.physics.isaaclab import solver_cfg

    settings = {'integrator': 'euler', 'iterations': 9, 'max_contacts_per_world': 2}
    baseline = solver_cfg('kamino', settings).to_solver_config()
    tuned = solver_cfg('kamino', settings | {'matrix_preconditioner_stride': 1024}).to_solver_config()
    assert baseline.dynamics.matrix_preconditioner_stride == 0
    assert tuned.dynamics.matrix_preconditioner_stride == 1024
    tuned.dynamics.matrix_preconditioner_stride = 0
    assert tuned == baseline
    for invalid in (-1, 2.5, True):
        with pytest.raises(ValueError, match='non-negative integer'):
            ConstrainedDynamicsConfig(matrix_preconditioner_stride=invalid)
    with pytest.raises(ValueError, match='uniform across worlds'):
        DualProblem._check_config([
            DualProblem.Config(),
            DualProblem.Config(dynamics=ConstrainedDynamicsConfig(matrix_preconditioner_stride=128)),
        ], 2)


def test_active_preconditioner_preserves_padding_and_graph_dimension_updates():
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.dual import (
        _apply_dual_preconditioner_to_active_matrix,
        _apply_dual_preconditioner_to_matrix,
    )

    device = 'cuda:0'
    capacity = np.array([2, 3, 20, 32, 70], dtype=np.int32)
    dims = np.array([0, 1, 17, 32, 65], dtype=np.int32)
    mio = np.array([0, *np.cumsum(capacity[:-1] ** 2)], dtype=np.int32)
    vio = np.array([0, *np.cumsum(capacity[:-1])], dtype=np.int32)
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=int((capacity ** 2).sum())).astype(np.float32)
    weights = rng.uniform(-2., 2., size=int(capacity.sum())).astype(np.float32)
    dim = wp.array(dims, dtype=wp.int32, device=device)
    offsets = wp.array(mio, dtype=wp.int32, device=device)
    vectors = wp.array(vio, dtype=wp.int32, device=device)
    P = wp.array(weights, device=device)
    original = wp.array(matrix, device=device)
    baseline, candidate = wp.empty_like(original), wp.empty_like(original)
    stride = 37  # Several iterations per thread and nonuniform row boundaries.

    def run_baseline():
        wp.launch(_apply_dual_preconditioner_to_matrix, dim=(len(dims), int(capacity.max()) ** 2),
                  inputs=[dim, offsets, vectors, P, baseline], device=device)

    def run_candidate():
        wp.launch(_apply_dual_preconditioner_to_active_matrix, dim=(len(dims), stride),
                  inputs=[dim, offsets, vectors, P, candidate, stride], device=device)

    wp.copy(baseline, original)
    wp.copy(candidate, original)
    run_baseline()
    run_candidate()
    wp.synchronize_device()
    np.testing.assert_array_equal(candidate.numpy(), baseline.numpy())
    for world, n in enumerate(dims):
        padding = slice(mio[world] + n * n, mio[world] + capacity[world] ** 2)
        np.testing.assert_array_equal(candidate.numpy()[padding], matrix[padding])

    with wp.ScopedCapture() as captured:
        run_candidate()
    for current in (np.array([2, 0, 20, 1, 70], np.int32), np.zeros_like(dims)):
        dim.assign(current)
        wp.copy(baseline, original)
        wp.copy(candidate, original)
        run_baseline()
        wp.capture_launch(captured.graph)
        wp.synchronize_device()
        np.testing.assert_array_equal(candidate.numpy(), baseline.numpy())
