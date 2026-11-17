"""The tracked clear with sparse Jacobians assembles each entry once; it matches the bucketed assembly and an FP64 product."""

import numpy as np
import pytest
from test_kamino_body_assembly import _dense_delassus
from test_kamino_tracked_clear import _fixture

pytestmark = pytest.mark.physics


def _random_blocks(rng, model, data, jacobians, active, body_counts):
    """Rows with one to three blocks on distinct bodies, shuffled, inactive capacity filled with junk."""

    blocks = jacobians._J_cts.bsm
    starts = blocks.nzb_start.numpy()
    capacity = np.diff(np.append(starts, blocks.sum_of_num_nzb))
    coords = np.full((blocks.sum_of_num_nzb, 2), -99, np.int32)
    values = np.full((blocks.sum_of_num_nzb, 6), 7.0, np.float32)
    counts = np.zeros(len(active), np.int32)
    references = []
    inv_m = model.bodies.inv_m_i.numpy()
    inv_I = data.bodies.inv_I_i.numpy()
    offsets = np.concatenate([[0], np.cumsum(body_counts)])
    for world, rows in enumerate(active):
        pairs = []
        for row in range(rows):
            for body in rng.choice(body_counts[world], size=min(body_counts[world], rng.integers(1, 4)), replace=False):
                pairs.append((row, 6 * int(body)))
        pairs = pairs[: capacity[world]]
        rng.shuffle(pairs)
        pairs = np.asarray(pairs, np.int32).reshape(-1, 2)
        vals = rng.normal(size=(len(pairs), 6)).astype(np.float32)
        counts[world] = len(pairs)
        coords[starts[world] : starts[world] + len(pairs)] = pairs
        values[starts[world] : starts[world] + len(pairs)] = vals
        bodies = slice(offsets[world], offsets[world + 1])
        references.append(_dense_delassus(pairs, vals, inv_m[bodies], inv_I[bodies], rows))
    blocks.num_nzb.assign(counts)
    blocks.nzb_coords.assign(coords)
    blocks.nzb_values.assign(values)
    blocks.dims.assign(np.column_stack([active, np.asarray(body_counts) * 6]).astype(np.int32))
    data.info.num_total_cts.assign(np.asarray(active, np.int32))
    return references


def test_gathered_assembly_matches_bucketed_assembly_and_fp64_over_changing_dimensions():
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.delassus import DelassusOperator

    device = "cuda:0"
    body_counts = [3, 4, 5]
    model, data, jacobians, set_dims, maxdims = _fixture(wp, body_counts, device)
    rng = np.random.default_rng(1234)
    model.bodies.inv_m_i.assign(rng.uniform(0.2, 3, size=model.bodies.inv_m_i.shape[0]).astype(np.float32))
    factors = rng.normal(size=(model.bodies.inv_m_i.shape[0], 3, 3)).astype(np.float32)
    data.bodies.inv_I_i.assign(factors @ factors.transpose(0, 2, 1) + 0.1 * np.eye(3, dtype=np.float32))
    bucketed = DelassusOperator(model, data)
    tracked = DelassusOperator(model, data, tracked_active_clear=True)
    for active in ([12, 18, 24], [5, 4, 9], [0, 0, 0], [20, 3, 11], [2, 5, 1], [12, 18, 24]):
        active = np.minimum(active, maxdims)
        references = _random_blocks(rng, model, data, jacobians, active, body_counts)
        bucketed.build(model, data, jacobians)
        tracked.build(model, data, jacobians)
        wp.synchronize_device()
        assert tracked._gathered is not None and bucketed._gathered is None
        expected = np.zeros(int(tracked.num_maxsize), np.float64)
        mio = tracked._operator.info.mio.numpy()
        for world, rows in enumerate(active):
            expected[mio[world] : mio[world] + rows * rows] = references[world].ravel()
        actual = tracked.D.numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(actual, bucketed.D.numpy(), rtol=3e-5, atol=3e-5)
        # The tail a larger earlier prefix left is cleared, exactly.
        outside = np.ones(actual.shape, bool)
        for world, rows in enumerate(active):
            outside[mio[world] : mio[world] + rows * rows] = False
        assert not actual[outside].any()
        for world, rows in enumerate(active):
            matrix = actual[mio[world] : mio[world] + rows * rows].reshape(rows, rows)
            np.testing.assert_array_equal(matrix, matrix.T)
