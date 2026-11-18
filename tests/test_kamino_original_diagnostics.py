"""The reduced solve's original-equation diagnostics are a switch that tests turn on; the solution does not depend on it."""

import numpy as np
import pytest

pytestmark = pytest.mark.physics


def _solve(original):
    import warp as wp
    from kamino_reduced_fixture import make
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    model, problem = make()
    cfg = PADMMSolverConfig(algorithm="bilateral_reduced", max_iterations=100, reduced_original_diagnostics=original)
    solver = PADMMSolver(model=model, config=cfg, use_graph_conditionals=True)
    p = problem.data
    n = int(p.dim.numpy()[0])
    raw = np.zeros(p.D.shape[0], np.float32)
    matrix = np.eye(n, dtype=np.float32) * 2
    matrix[0, 1] = matrix[1, 0] = np.float32(0.125)
    raw[: n * n] = matrix.ravel()
    p.D.assign(raw)
    q = np.zeros(p.v_f.shape[0], np.float32)
    q[:n] = np.linspace(-0.1, 0.2, n, dtype=np.float32)
    p.v_f.assign(q)
    solver.coldstart()
    solver.solve(problem)
    wp.synchronize_device()
    return solver, n


def test_original_diagnostics_are_off_by_default_and_change_no_solution():
    import warp as wp
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    assert PADMMSolverConfig().reduced_original_diagnostics is False
    with pytest.raises(ValueError, match="reduced_original_diagnostics must be bool"):
        PADMMSolverConfig(reduced_original_diagnostics=1)

    off, n = _solve(False)
    on, _ = _solve(True)
    assert not off._reduced_runtime.original_residuals.numpy().any()
    assert np.isfinite(on._reduced_runtime.original_residuals.numpy()).all()
    assert on._reduced_runtime.original_residuals.numpy().any()
    for name in ("x", "y", "z", "x_p", "y_p", "z_p", "y_hat", "z_hat"):
        left = getattr(off.data.state, name).numpy()[:n]
        right = getattr(on.data.state, name).numpy()[:n]
        assert left.tobytes() == right.tobytes(), name
    for name, value in vars(off.data.solution).items():
        if isinstance(value, wp.array):
            assert value.numpy()[:n].tobytes() == getattr(on.data.solution, name).numpy()[:n].tobytes(), name
    left, right = off.data.status.numpy(), on.data.status.numpy()
    for name in left.dtype.names:
        assert left[name].tobytes() == right[name].tobytes(), name
