"""Validate masked LLTB solves on the patched GPU image."""

import pytest

pytestmark = pytest.mark.physics


def test_masked_solves_preserve_inactive_outputs_and_graph_updates():
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.linalg.factorize.llt_blocked import (
        llt_blocked_solve,
        make_llt_blocked_solve_kernel,
    )

    rng = np.random.default_rng(11)
    dims = np.array([0, 1, 17, 32, 65], np.int32)
    matrix_offsets = np.array([0, *np.cumsum(dims[:-1] ** 2)], np.int32)
    vector_offsets = np.array([0, *np.cumsum(dims[:-1])], np.int32)
    references, factors, rhs = [], [], []
    for n in dims:
        matrix = rng.normal(size=(n, n))
        matrix = (matrix @ matrix.T + np.eye(n) * 3).astype(np.float32)
        vector = rng.normal(size=n).astype(np.float32)
        factors.append(np.linalg.cholesky(matrix).flatten())
        rhs.append(vector)
        references.append(np.linalg.solve(matrix, vector))

    device = "cuda:0"
    dim = wp.array(dims, dtype=wp.int32, device=device)
    mio = wp.array(matrix_offsets, dtype=wp.int32, device=device)
    vio = wp.array(vector_offsets, dtype=wp.int32, device=device)
    factor = wp.array(np.concatenate(factors), device=device)
    b = wp.array(np.concatenate(rhs), device=device)
    y = wp.full(int(dims.sum()), -99.0, device=device)
    x = wp.full(int(dims.sum()), -77.0, device=device)
    mask = wp.array(np.array([False, True, False, True, False]), dtype=wp.bool, device=device)
    kernel = make_llt_blocked_solve_kernel(32)

    def solve(active):
        llt_blocked_solve(kernel, dim, mio, vio, factor, b, y, x, num_blocks=len(dims), active_mask=active)

    solve(None)
    wp.synchronize()
    baseline = x.numpy().copy()
    np.testing.assert_allclose(baseline, np.concatenate(references), rtol=2e-5, atol=2e-6)

    x.fill_(-77.0)
    y.fill_(-99.0)
    solve(mask)
    wp.synchronize()
    actual_x, actual_y = x.numpy(), y.numpy()
    for world, n in enumerate(dims):
        rows = slice(vector_offsets[world], vector_offsets[world] + n)
        if world in (1, 3):
            np.testing.assert_array_equal(actual_x[rows], baseline[rows])
        else:
            np.testing.assert_array_equal(actual_x[rows], np.full(n, -77.0, np.float32))
            np.testing.assert_array_equal(actual_y[rows], np.full(n, -99.0, np.float32))

    # The captured graph must read the current mask, not the capture-time values.
    with wp.ScopedCapture() as capture:
        solve(mask)
    mask.fill_(True)
    wp.capture_launch(capture.graph)
    wp.synchronize()
    np.testing.assert_array_equal(x.numpy(), baseline)

    mask.fill_(False)
    x.fill_(4.0)
    y.fill_(5.0)
    wp.capture_launch(capture.graph)
    wp.synchronize()
    np.testing.assert_array_equal(x.numpy(), np.full(int(dims.sum()), 4.0, np.float32))
    np.testing.assert_array_equal(y.numpy(), np.full(int(dims.sum()), 5.0, np.float32))
