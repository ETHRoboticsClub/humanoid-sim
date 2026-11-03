"""Current sparse topology, including changing contacts, drives conservative factor fill."""
import numpy as np
import pytest

pytestmark = pytest.mark.physics


def _reference(entries, dimension, bodies, inverse, block, capacity):
    if dimension <= 0 or (dimension + block - 1) // block > capacity:
        return np.full(capacity, 0xffffffff, dtype=np.uint32)
    n = (dimension + block - 1) // block
    graph = np.eye(n, dtype=bool)
    incident = [set() for _ in range(bodies)]
    for row, column in entries:
        if row < 0 or row >= dimension or row >= len(inverse) or column < 0 or column % 6 or column // 6 >= bodies:
            return np.full(capacity, 0xffffffff, dtype=np.uint32)
        tile = inverse[row] // block
        if tile >= capacity:
            return np.full(capacity, 0xffffffff, dtype=np.uint32)
        incident[column // 6].add(tile)
    for tiles in incident:
        for i in tiles:
            for j in tiles:
                graph[i, j] = True
    # Dense Boolean elimination is independent of the device bitset operations.
    for pivot in range(n):
        neighbors = np.flatnonzero(graph[pivot, pivot + 1:]) + pivot + 1
        graph[np.ix_(neighbors, neighbors)] = True
    result = np.zeros(capacity, dtype=np.uint32)
    for i in range(n):
        for j in range(n):
            if graph[i, j]:
                result[i] |= np.uint32(1 << j)
    return result


def test_dynamic_pattern_rebuilds_changed_coordinates_in_one_cuda_graph():
    import warp as wp
    from newton._src.solvers.kamino._src.solvers.padmm.reduced_pattern import (
        collect_body_tile_bits,
        make_symbolic_fill,
    )

    device = 'cuda:0'
    block, capacity, worlds, bodies, slots = 4, 3, 3, 3, 8
    inverse = np.array([2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
    coords = wp.zeros((worlds * slots, 2), dtype=wp.int32, device=device)
    starts = wp.array(np.arange(worlds, dtype=np.int32) * slots, device=device)
    counts = wp.zeros(worlds, dtype=wp.int32, device=device)
    dims = wp.zeros(worlds, dtype=wp.int32, device=device)
    body_counts = wp.array(np.full(worlds, bodies, dtype=np.int32), device=device)
    perm = wp.array(inverse, dtype=wp.int32, device=device)
    body_tiles = wp.zeros((worlds, bodies), dtype=wp.uint32, device=device)
    invalid = wp.zeros(worlds, dtype=wp.int32, device=device)
    occupied = wp.zeros((worlds, capacity), dtype=wp.uint32, device=device)
    fill = make_symbolic_fill(capacity)

    def launch():
        body_tiles.zero_()
        invalid.zero_()
        wp.launch(collect_body_tile_bits, dim=(worlds, slots), inputs=[coords, starts, counts, dims, body_counts, perm, block, capacity, body_tiles, invalid], device=device)
        wp.launch(fill, dim=worlds, inputs=[body_tiles, body_counts, dims, block, invalid, occupied], device=device)

    launch()  # Compile before CUDA capture.
    with wp.ScopedCapture(device=device) as capture:
        launch()
    cases = [
        ([11, 8, 0], [[(0, 0), (8, 0), (4, 6), (10, 6)], [(1, 0), (7, 0), (3, 12)], []]),
        ([7, 11, 0], [[(1, 12), (6, 12)], [(0, 0), (10, 0), (6, 7)], []]),
    ]
    for dimensions, rows in cases:
        values = np.zeros((worlds * slots, 2), dtype=np.int32)
        for world, entries in enumerate(rows):
            values[world * slots:world * slots + len(entries)] = np.asarray(entries, dtype=np.int32).reshape(-1, 2)
        coords.assign(values)
        counts.assign(np.array([len(entries) for entries in rows], dtype=np.int32))
        dims.assign(np.array(dimensions, dtype=np.int32))
        wp.capture_launch(capture.graph)
        expected = np.stack([_reference(entries, n, bodies, inverse, block, capacity) for entries, n in zip(rows, dimensions, strict=True)])
        np.testing.assert_array_equal(occupied.numpy(), expected)
