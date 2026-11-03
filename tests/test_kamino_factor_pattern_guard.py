"""An untrusted Jacobian pattern must not discard entries in the supplied operator."""

import numpy as np
import pytest

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("detached", [False, True])
def test_public_solve_preserves_off_pattern_matrix_entries(monkeypatch, dynamic, detached):
    import warp as wp
    from kamino_reduced_fixture import model_fixture
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    model, problem = model_fixture(wp, 9, "cpu")
    solver = PADMMSolver(model=model, config=PADMMSolverConfig(
        algorithm="bilateral_reduced", reduced_ordering="rcm", reduced_dynamic_pattern=dynamic,
    ), use_graph_conditionals=True)
    runtime = solver._reduced_runtime
    p = problem.data
    n = int(p.dim.numpy()[0])
    permutation = runtime.permutation.numpy()[:n]
    # Two separated factor tiles, including a coupling absent from model/J topology.
    left, right = int(permutation[0]), int(permutation[32])
    matrix = 2 * np.eye(n, dtype=np.float32)
    matrix[left, right] = matrix[right, left] = .5
    values = np.zeros(p.D.shape, dtype=np.float32)
    values[:n * n] = matrix.ravel()
    if detached:
        p.D = wp.array(values, device="cpu")
    else:
        p.D.assign(values)
    # The dynamic pattern is deliberately empty, as before its first build.
    problem.jacobians._J_cts.bsm.num_nzb.zero_()
    launch = runtime.launch

    class FactorReached(Exception):
        pass

    def intercept(kernel, dim, args, block=128):
        if kernel is runtime.factor:
            raise FactorReached
        return launch(kernel, dim, args, block)

    monkeypatch.setattr(runtime, "launch", intercept)
    with pytest.raises(FactorReached):
        solver.solve(problem)
    occupied = runtime.occupied.numpy()
    included = bool(int(occupied[0, 2]) & 1) if dynamic else bool(occupied[2, 0])
    assert included, "factor schedule dropped a nonzero entry from the public matrix"
    # Show why this is a numerical error, not merely a conservative schedule choice.
    rhs = np.zeros(n)
    rhs[left] = 1
    candidate = matrix.copy()
    if not included:
        candidate[left, right] = candidate[right, left] = 0
    result = np.linalg.solve(candidate, rhs)
    np.testing.assert_allclose(matrix @ result, rhs, atol=1e-14, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_factor_pattern_switches_between_certified_and_public_graphs(device, dynamic):
    import warp as wp
    from kamino_reduced_fixture import model_fixture
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    model, problem = model_fixture(wp, 9, device)
    solver = PADMMSolver(model=model, config=PADMMSolverConfig(
        algorithm="bilateral_reduced", reduced_ordering="rcm", reduced_dynamic_pattern=dynamic,
    ), use_graph_conditionals=True)
    runtime = solver._reduced_runtime
    # Exercise real current sparse topology, then change its body connectivity.
    jac = problem.jacobians._J_cts.bsm
    p = problem.data
    n = int(p.dim.numpy()[0])
    coords = np.zeros(jac.nzb_coords.shape, np.int32)
    coords[:n, 0] = np.arange(n)
    coords[:n, 1] = (np.argsort(runtime.permutation.numpy())[:n] // 16) * 6
    jac.nzb_coords.assign(coords)
    jac.num_nzb.assign(np.array([n], np.int32))

    def prepare(fresh):
        runtime._prepare_factor_pattern(problem, fresh_sparse_build=fresh)

    prepare(True)
    sparse = runtime.occupied.numpy().copy()
    assert not np.all(sparse == (0xffffffff if dynamic else 1))
    prepare(False)
    dense = runtime.occupied.numpy().copy()
    assert np.all(dense == (0xffffffff if dynamic else 1))
    # Cached graphs must write their chosen schedule on every replay.
    with wp.ScopedCapture(device=device) as trusted:
        prepare(True)
    with wp.ScopedCapture(device=device) as public:
        prepare(False)
    for _ in range(2):
        wp.capture_launch(trusted.graph)
        np.testing.assert_array_equal(runtime.occupied.numpy(), sparse)
        wp.capture_launch(public.graph)
        np.testing.assert_array_equal(runtime.occupied.numpy(), dense)
    # Even the internal route must reject detached source/layout arrays.
    for field in ("D", "dim", "mio", "vio"):
        original = getattr(p, field)
        setattr(p, field, wp.clone(original))
        prepare(True)
        np.testing.assert_array_equal(runtime.occupied.numpy(), dense)
        setattr(p, field, original)
        prepare(True)
        np.testing.assert_array_equal(runtime.occupied.numpy(), sparse)


@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_public_rcm_solve_matches_dense_original_equations(dynamic):
    import warp as wp
    from kamino_reduced_fixture import model_fixture
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from newton._src.solvers.kamino.config import PADMMSolverConfig

    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    model, problem = model_fixture(wp, 9, "cuda:0")
    solver = PADMMSolver(model=model, config=PADMMSolverConfig(
        algorithm="bilateral_reduced", reduced_ordering="rcm", reduced_dynamic_pattern=dynamic,
    ), use_graph_conditionals=True)
    rt, p = solver._reduced_runtime, problem.data
    n = int(p.dim.numpy()[0])
    left, right = rt.permutation.numpy()[[0, 32]]
    matrix = 2 * np.eye(n, dtype=np.float32)
    matrix[left, right] = matrix[right, left] = .5
    q = np.zeros(p.v_f.shape, dtype=np.float32)
    q[left] = -.1
    p.v_f.assign(q)
    raw = np.zeros(p.D.shape, dtype=np.float32)
    raw[:n * n] = matrix.ravel()
    pristine = wp.array(raw, device=model.device)
    problem.jacobians._J_cts.bsm.num_nzb.zero_()

    def solve():
        wp.copy(p.D, pristine)
        solver.coldstart()
        solver.solve(problem)

    solve()
    with wp.ScopedCapture(device=model.device) as capture:
        solve()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        actual = solver.data.solution.lambdas.numpy()[:n]
        # The only nonzero solution coordinates are bilateral: no projection
        # boundary or iteration-cap ambiguity in this independently solved SPD case.
        expected = np.linalg.solve(matrix.astype(np.float64), -q[:n])
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(matrix @ actual + q[:n], 0, atol=1e-6, rtol=0)
        assert not rt.bad.numpy().any()
