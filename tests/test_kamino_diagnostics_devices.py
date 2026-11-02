"""Reduced PADMM stopping diagnostics agree with the one-thread-per-row source kernel on every device."""
import numpy as np
import pytest

pytestmark = pytest.mark.physics

# Per world: box, limit and contact rows. One world exceeds the 128 threads a world launches with,
# and none is a multiple of the 32 lanes a CUDA warp reduces over.
WORLDS = ((2, 2, 1), (10, 5, 10), (60, 30, 20))


def _problem(rng):
    dims = [box + limit + 3 * contact for box, limit, contact in WORLDS]
    vio = np.r_[0, np.cumsum(dims)[:-1]]
    mio = np.r_[0, np.cumsum(np.square(dims))[:-1]]
    boxes, limits, contacts = (np.array(column) for column in zip(*WORLDS))
    rows, entries = int(sum(dims)), int(np.square(dims).sum())
    D = rng.normal(size=entries).astype(np.float32)
    lower = -rng.uniform(0.1, 1.0, int(boxes.sum())).astype(np.float32)
    return {
        "D": D,
        "q": rng.normal(size=rows).astype(np.float32),
        "P": rng.uniform(0.5, 2.0, rows).astype(np.float32),
        "dim": np.array(dims, np.int32), "mio": mio.astype(np.int32), "vio": vio.astype(np.int32),
        "nbc": boxes.astype(np.int32), "nl": limits.astype(np.int32),
        "bcgo": np.zeros(len(dims), np.int32), "lcgo": boxes.astype(np.int32),
        "ccgo": (boxes + limits).astype(np.int32),
        "bcio": np.r_[0, np.cumsum(boxes)[:-1]].astype(np.int32),
        "cio": np.r_[0, np.cumsum(contacts)[:-1]].astype(np.int32),
        "lower": lower, "upper": -lower[::-1].copy(),
        "mu": rng.uniform(0.2, 1.0, int(contacts.sum())).astype(np.float32),
        "x": np.zeros(rows, np.float32),
        "y": rng.normal(size=rows).astype(np.float32),
        "z": rng.normal(size=rows).astype(np.float32),
    }


def _launch(wp, kernel, device, inputs, threads, block_dim):
    result = wp.zeros((len(WORLDS), 4), dtype=wp.float32, device=device)
    wp.launch(kernel, dim=(len(WORLDS), threads), inputs=[*inputs, result], device=device, block_dim=block_dim)
    return result.numpy()


def _reference(wp, device, problem, D, sigma):
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_runtime_kernels import original_diagnostics

    inputs = [wp.array(problem[key], device=device) for key in problem]
    inputs[0] = wp.array(D, device=device)
    return _launch(wp, original_diagnostics, device, [*inputs, sigma], max(problem["dim"]), 256)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reduced_diagnostics_cover_every_row_and_lane(device):
    import warp as wp
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_diagnostics import make_bounded
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_lazy_diagnostics import (
        make_bounded as make_lazy,
    )
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_lazy_sparse_diagnostics import (
        make_sparse as make_lazy_sparse,
    )
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_sparse_diagnostics import make_sparse

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(7)
    problem = _problem(rng)
    sigma = 0.0625
    inputs = [wp.array(value, device=device) for value in problem.values()] + [sigma]
    dense = _reference(wp, device, problem, problem["D"], sigma)
    assert np.all(dense > 0)

    def check(actual, expected, label):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, err_msg=label)

    for threads in (128, 256):
        for reduction in ("block", "warp"):
            check(_launch(wp, make_bounded(threads, reduction), device, inputs, threads, threads), dense,
                  f"dense {reduction} {threads}")

    # Sparse loads keep the diagonal and entries whose rows share a body; world 0 falls back to dense.
    dims, mio = problem["dim"], problem["mio"]
    capacity = int(dims.max())
    masks = np.zeros((len(WORLDS), capacity), np.uint32)
    flags = np.array([1, 0, 0], np.int32)
    sparse_D = problem["D"].copy()
    for w, n in enumerate(dims):
        masks[w, :n] = np.uint32(1) << rng.integers(0, 8, n).astype(np.uint32)
        if not flags[w]:
            shared = (masks[w, :n, None] & masks[w, None, :n]) != 0
            block = sparse_D[mio[w]:mio[w] + n * n].reshape(n, n)
            block[~(shared | np.eye(n, dtype=bool))] = 0.0
    sparse = _reference(wp, device, problem, sparse_D, sigma)
    tail = [wp.array(masks, device=device), wp.array(flags, device=device)]
    check(_launch(wp, make_sparse(capacity), device, inputs + tail, 128, 128), sparse, "sparse")

    # Lazy loads: raw (phase 0), preconditioned (1) and preconditioned plus a pending diagonal shift (2).
    P_epoch = rng.uniform(0.5, 2.0, len(problem["P"])).astype(np.float32)
    phase = np.array([0, 1, 2], np.int32)
    delta = np.array([0.0, 0.0, 0.25], np.float32)
    apply = np.array([0, 0, 1], np.int32)
    lazy_D, lazy_sparse_D = problem["D"].copy(), sparse_D.copy()
    for w, n in enumerate(dims):
        if phase[w]:
            p = P_epoch[problem["vio"][w]:problem["vio"][w] + n]
            for matrix in (lazy_D, lazy_sparse_D):
                block = matrix[mio[w]:mio[w] + n * n].reshape(n, n)
                block[:] = p[:, None] * (p[None, :] * block)
                if phase[w] == 2 and apply[w]:
                    block[np.diag_indices(n)] += delta[w]
    epoch = [wp.array(value, device=device) for value in
             (P_epoch, phase, problem["dim"], problem["mio"], problem["vio"], delta, apply)]
    lazy_inputs = inputs[:1] + epoch + inputs[1:]
    lazy = _reference(wp, device, problem, lazy_D, sigma)
    for reduction in ("block", "warp"):
        check(_launch(wp, make_lazy(128, reduction), device, lazy_inputs, 128, 128), lazy, f"lazy {reduction}")
    check(_launch(wp, make_lazy_sparse(capacity), device, lazy_inputs + tail, 128, 128),
          _reference(wp, device, problem, lazy_sparse_D, sigma), "lazy sparse")
