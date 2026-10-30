"""Real public-solve alias behavior and generic sparse-diagnostic fallback."""
import numpy as np
import pytest

pytestmark = pytest.mark.physics

@pytest.mark.parametrize('mode', ['public_mutated_D', 'private_negative_P', 'private_body_overflow', 'private_detached_D'])
def test_sparse_diagnostics_preserve_generic_dense_contract(mode):
    import warp as wp
    from newton._src.solvers.kamino.config import PADMMSolverConfig
    from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver
    from kamino_reduced_fixture import make
    def fixture(candidate):
        model, problem = make()
        cfg = PADMMSolverConfig(
            algorithm='bilateral_reduced', reduced_sparse_diagnostics=candidate,
            max_iterations=100,
        )
        solver = PADMMSolver(model=model, config=cfg, use_graph_conditionals=True)
        rt, p = solver._reduced_runtime, problem.data
        n = int(p.dim.numpy()[0])
        raw = np.zeros(p.D.shape[0], np.float32)
        matrix = np.eye(n, dtype=np.float32)*2
        # Deliberately unsupported by the fixture's empty J topology. Direct
        # public solve supports manual D, irrespective of the sparse opt-in.
        matrix[0, 1] = matrix[1, 0] = np.float32(.125)
        raw[:n*n] = matrix.ravel()
        if mode == 'private_detached_D':
            p.D = wp.array(raw, device=model.device)
        else:
            p.D.assign(raw)
        q = np.zeros(p.v_f.shape[0], np.float32)
        q[:n] = np.linspace(-.1,.2,n,dtype=np.float32)
        p.v_f.assign(q)
        if mode == 'private_negative_P':
            p.P.fill_(-1.)
        if candidate:
            if mode == 'private_body_overflow':
                rt.diagnostic_bodies.fill_(33)
            rt.diagnostic_masks.fill_(0x12345678)
        solver.coldstart()
        return model, problem, solver, rt, n

    # Separate identical mathematical problems: the reference executes the
    # public dense diagnostic path with the option disabled. Both solvers
    # record original_residuals at the same pre-publication state lifecycle.
    # Do not recompute diagnostics on state.x/y/z after solve(): the source
    # _update_solution has already scaled those arrays by P or 1/P in place.
    reference_model, reference_problem, reference, reference_rt, reference_n = fixture(False)
    model, problem, solver, rt, n = fixture(True)
    assert n == reference_n
    reference.solve(reference_problem)
    if mode == 'public_mutated_D':
        solver.solve(problem)
    else:
        solver._solve_from_fresh_build(problem)
    wp.synchronize_device()
    if mode in ('public_mutated_D','private_detached_D'):
        assert np.all(rt.diagnostic_masks.numpy()==0x12345678), 'Uncertified alias entered sparse branch'
    else:
        assert np.all(rt.diagnostic_dense.numpy()!=0), 'Generic guard failed to select dense per-world path'
    expected = reference_rt.original_residuals.numpy()
    actual = rt.original_residuals.numpy()
    assert expected.shape == actual.shape == (1,4)
    assert expected.dtype == actual.dtype == np.float32
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    assert expected.tobytes() == actual.tobytes(), 'All four pre-publication diagnostics must be bitwise equal'
    # Also compare published active iterates, active solutions and each status
    # field. Capacity padding of solution arrays is not written by the source.
    for name in ('x','y','z','x_p','y_p','z_p','y_hat','z_hat'):
        left = getattr(reference.data.state,name).numpy()[:n]
        right = getattr(solver.data.state,name).numpy()[:n]
        assert left.tobytes() == right.tobytes(), ('published state',name)
    for name, value in vars(reference.data.solution).items():
        if isinstance(value,wp.array):
            left = value.numpy()[:n]
            right = getattr(solver.data.solution,name).numpy()[:n]
            assert left.tobytes() == right.tobytes(), ('active solution',name)
    left, right = reference.data.status.numpy(), solver.data.status.numpy()
    for name in left.dtype.names:
        assert left[name].tobytes() == right[name].tobytes(), ('status',name)
