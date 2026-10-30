"""Regression for partial free-column tiles reading unwritten factor storage."""

import numpy as np
import warp as wp
from kamino_reduced_q_reference import make_reduced_q as make_reference_q
from newton._src.solvers.kamino._src.solvers.padmm.reduced_direct_factor import make_runtime_direct_factor
from newton._src.solvers.kamino._src.solvers.padmm.reduced_setup import (
    make_free_solve,
    make_reduced_inverse,
    make_reduced_q,
)


@wp.kernel(enable_backward=False)
def clear_reference(values: wp.array[wp.float32]):
    values[wp.tid()] = wp.float32(0.0)


@wp.kernel(enable_backward=False)
def bounded_cross_factor(
    L: wp.array[wp.float32],
    dims: wp.array[wp.int32],
    mio: wp.array[wp.int32],
    free: int,
    result: wp.array[wp.float32],
):
    world, lane = wp.tid()
    n = dims[world]
    for index in range(lane, n * n, 128):
        value = wp.float32(0.0)
        if index % n < free:
            value = L[mio[world] + index]
        result[mio[world] + index] = value


def bits_equal(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def run_partial_free_q_poison_and_changing_dimension_graph(free):
    """NaN-filled upper tiles never enter q; current free180 is a control."""
    wp.init()
    device = "cuda:0"
    stream = wp.Stream(device)
    count = 3
    capacity = 96
    width = free + capacity
    sigma = float(np.float32(np.float32(1e-5) + np.float32(0.05)))
    rng = np.random.default_rng(332 + free)
    matrices = []
    vectors = []
    for _ in range(count):
        M = rng.normal(size=(width, width)) / np.sqrt(width)
        matrices.append((np.eye(width) + 0.1 * (M @ M.T)).astype(np.float32))
        vectors.append(rng.normal(size=width).astype(np.float32))
    mio_np = np.arange(count, dtype=np.int32) * width * width
    mio = wp.array(mio_np, device=device)
    dims = wp.empty(count, dtype=wp.int32, device=device)
    bad = wp.zeros(count, dtype=wp.bool, device=device)
    D = wp.zeros(count * width * width, device=device)
    q = wp.array(np.stack(vectors).ravel(), device=device)
    qF = wp.array(np.stack([v[:free] for v in vectors]).ravel(), device=device)
    L = wp.full(count * width * width, float("nan"), device=device)
    clean = wp.zeros_like(L)
    cross = wp.zeros_like(L)
    v = wp.zeros(count * free, device=device)
    vref = wp.zeros_like(v)
    result = wp.full(count * capacity, 73.0, device=device)
    reference = wp.full(count * capacity, 73.0, device=device)
    old_result = wp.full(count * capacity, 73.0, device=device)
    Y = wp.full(count * capacity * capacity, 73.0, device=device)
    X = wp.full(count * capacity * capacity, 73.0, device=device)
    Yref = wp.full(count * capacity * capacity, 73.0, device=device)
    Xref = wp.full(count * capacity * capacity, 73.0, device=device)
    factor = make_runtime_direct_factor(16)
    forward = make_free_solve(False)
    corrected_q = make_reduced_q()
    old_q = make_reference_q()
    inverse = make_reduced_inverse()

    def launch(kernel, args, shape=(count, 128)):
        wp.launch(kernel, shape, inputs=args, block_dim=128, device=device, stream=stream)

    def pipeline():
        # Only the reference is sanitized each replay. The candidate retains
        # previously written L, including stale packing after dimension changes.
        wp.launch(clear_reference, dim=clean.size, inputs=[clean], device=device, stream=stream)
        launch(factor, [dims, mio, D, dims, mio, bad, free, sigma, L])
        launch(factor, [dims, mio, D, dims, mio, bad, free, sigma, clean])
        launch(forward, [L, dims, mio, qF, free, v])
        launch(forward, [clean, dims, mio, qF, free, vref])
        launch(bounded_cross_factor, [clean, dims, mio, free, cross])
        launch(corrected_q, [L, dims, mio, v, q, free, width, capacity, result])
        launch(old_q, [cross, dims, mio, vref, q, free, width, capacity, reference])
        launch(old_q, [L, dims, mio, v, q, free, width, capacity, old_result])
        launch(inverse, [L, dims, mio, free, capacity, capacity // 32, Y, X], (count * (capacity // 32), 128))
        launch(
            inverse,
            [clean, dims, mio, free, capacity, capacity // 32, Yref, Xref],
            (count * (capacity // 32), 128),
        )

    def set_state(reduced):
        dims_np = np.asarray(reduced, dtype=np.int32) + free
        raw = np.zeros(count * width * width, dtype=np.float32)
        for w, n in enumerate(dims_np):
            raw[mio_np[w] : mio_np[w] + n * n] = matrices[w][:n, :n].ravel()
        wp.synchronize_stream(stream)
        dims.assign(dims_np)
        D.assign(raw)
        wp.synchronize_device(device)
        return dims_np, raw

    def verify(dims_np, raw, label):
        wp.synchronize_stream(stream)
        actualL, referenceL = L.numpy(), clean.numpy()
        actualQ, referenceQ = result.numpy(), reference.numpy()
        bits_equal(actualQ, referenceQ)
        bits_equal(v.numpy(), vref.numpy())
        bits_equal(X.numpy(), Xref.numpy())
        bits_equal(Y.numpy(), Yref.numpy())
        bits_equal(D.numpy(), raw)
        for w, n in enumerate(dims_np):
            reduced = n - free
            lo = np.tril_indices(n)
            a = actualL[mio_np[w] : mio_np[w] + n * n].reshape(n, n)
            b = referenceL[mio_np[w] : mio_np[w] + n * n].reshape(n, n)
            bits_equal(a[lo], b[lo])
            assert np.isfinite(a[lo]).all()
            K = matrices[w][:n, :n].astype(np.float64).copy()
            K[np.arange(free, n), np.arange(free, n)] += sigma
            goldL = np.linalg.cholesky(K)
            np.testing.assert_allclose(a[lo], goldL[lo], atol=2e-6, rtol=2e-5, err_msg=label)
            F = K[:free, :free]
            B = K[:free, free:]
            goldq = vectors[w][free:n].astype(np.float64) - B.T @ np.linalg.solve(
                F, vectors[w][:free].astype(np.float64)
            )
            goldS = K[free:, free:] - B.T @ np.linalg.solve(F, B)
            actualq = actualQ[w * capacity : w * capacity + reduced]
            actualinv = X.numpy().reshape(count, capacity, capacity)[w, :reduced, :reduced]
            assert np.isfinite(actualq).all() and np.isfinite(actualinv).all()
            np.testing.assert_allclose(actualq, goldq, atol=2e-6, rtol=3e-5, err_msg=label)
            np.testing.assert_allclose(actualinv, np.linalg.inv(goldS), atol=3e-6, rtol=3e-5, err_msg=label)
        return actualQ.copy()

    initial, raw = set_state([17, 44, 65])
    pipeline()
    verify(initial, raw, "initial poison")
    old = old_result.numpy().reshape(count, capacity)
    if free == 161:
        assert any(not np.isfinite(old[w, :r]).all() for w, r in enumerate([17, 44, 65])), (
            "Fixture did not expose the original invalid read"
        )
    else:
        assert all(np.isfinite(old[w, :r]).all() for w, r in enumerate([17, 44, 65]))
    wp.synchronize_device(device)
    with wp.ScopedCapture(stream=stream) as capture:
        pipeline()
    for step, reduced in enumerate(([65, 33, 1], [1, 65, 44], [17, 44, 65])):
        dims_np, raw = set_state(reduced)
        wp.capture_launch(capture.graph, stream=stream)
        first = verify(dims_np, raw, f"graph transition {step}")
        wp.capture_launch(capture.graph, stream=stream)
        second = verify(dims_np, raw, f"graph replay {step}")
        bits_equal(first, second)
