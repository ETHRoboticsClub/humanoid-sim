"""Check sparse Delassus assembly against independent dense FP64 products."""

import pytest

pytestmark = pytest.mark.physics


def _dense_delassus(coords, values, inverse_mass, inverse_inertia, rows):
    """Assemble J M^-1 J^T without sparse block-pair enumeration."""
    import numpy as np

    bodies = len(inverse_mass)
    jacobian = np.zeros((rows, 6 * bodies), dtype=np.float64)
    mass_inverse = np.zeros((6 * bodies, 6 * bodies), dtype=np.float64)
    for (row, column), value in zip(coords, values, strict=True):
        jacobian[row, column : column + 6] += value.astype(np.float64)
    for body in range(bodies):
        offset = 6 * body
        mass_inverse[offset : offset + 3, offset : offset + 3] = np.eye(3) * inverse_mass[body]
        mass_inverse[offset + 3 : offset + 6, offset + 3 : offset + 6] = inverse_inertia[body]
    return jacobian @ mass_inverse @ jacobian.T


def test_body_assembly_tracks_changing_contacts_padding_and_graph_inputs():
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.delassus import _BodyGroupedDelassus
    from newton._src.solvers.kamino._src.linalg.sparse_matrix import BlockDType, BlockSparseMatrices

    device = "cuda:0"
    rng = np.random.default_rng(8546)
    body_counts = np.array([2, 5, 3], dtype=np.int32)
    body_offsets = np.array([0, 2, 7, 10], dtype=np.int32)
    block_capacity = [75, 170, 125]
    matrix_offsets = np.array([0, 900, 1900], dtype=np.int32)
    matrix_size = 2600
    jacobian = BlockSparseMatrices()
    jacobian.finalize(
        max_dims=[(29, 12), (31, 30), (25, 18)],
        capacities=block_capacity,
        nzb_dtype=BlockDType(wp.float32, 6),
        index_dtype=wp.int32,
        device=device,
    )
    block_starts = jacobian.nzb_start.numpy()
    block_count = jacobian.sum_of_num_nzb
    coordinates = np.full((block_count, 2), -99, dtype=np.int32)
    values = np.zeros((block_count, 6), dtype=np.float32)
    inverse_mass = rng.uniform(0.2, 3, size=int(body_counts.sum())).astype(np.float32)
    factors = rng.normal(size=(int(body_counts.sum()), 3, 3)).astype(np.float32)
    inverse_inertia = factors @ factors.transpose(0, 2, 1)
    offsets = wp.array(body_offsets, device=device)
    mass = wp.array(inverse_mass, device=device)
    inertia = wp.array(inverse_inertia, dtype=wp.mat33f, device=device)
    dims = wp.zeros(3, dtype=wp.int32, device=device)
    matrix_starts = wp.array(matrix_offsets, device=device)
    output = wp.zeros(matrix_size, device=device)
    workspace = _BodyGroupedDelassus(jacobian, int(body_counts.sum()), device)

    def assemble():
        workspace.assemble(offsets, mass, inertia, dims, matrix_starts, output)

    graph = None
    for active in ([0, 27, 9], [28, 1, 0], [3, 0, 24], [0, 0, 0], [7, 23, 11]):
        coordinates.fill(-99)  # Inactive capacity must never be interpreted as a block.
        values.fill(0)
        counts = np.zeros(3, dtype=np.int32)
        initial = np.full(matrix_size, -7, dtype=np.float32)
        expected = initial.astype(np.float64)
        for world, rows in enumerate(active):
            pairs = np.asarray(
                [
                    (row, 6 * body)
                    for row in range(rows)
                    for body in range(body_counts[world])
                    if body == 0 or rng.random() < 0.35
                ],
                dtype=np.int32,
            ).reshape(-1, 2)
            rng.shuffle(pairs)
            count = len(pairs)
            assert count <= block_capacity[world]
            blocks = rng.normal(size=(count, 6)).astype(np.float32)
            start = int(block_starts[world])
            counts[world] = count
            coordinates[start : start + count] = pairs
            values[start : start + count] = blocks
            bodies = slice(body_offsets[world], body_offsets[world + 1])
            reference = _dense_delassus(pairs, blocks, inverse_mass[bodies], inverse_inertia[bodies], rows)
            matrix = slice(matrix_offsets[world], matrix_offsets[world] + rows * rows)
            initial[matrix] = 0
            expected[matrix] = reference.ravel()
        jacobian.num_nzb.assign(counts)
        jacobian.nzb_coords.assign(coordinates)
        jacobian.nzb_values.assign(values)
        jacobian.dims.assign(np.column_stack((active, body_counts * 6)).astype(np.int32))
        dims.assign(np.asarray(active, dtype=np.int32))
        output.assign(initial)
        if graph is None:
            assemble()  # Warm every kernel before capture.
            wp.synchronize_device()
            np.testing.assert_allclose(output.numpy(), expected, rtol=3e-5, atol=3e-5)
            with wp.ScopedCapture(device=device) as captured:
                assemble()
            graph = captured.graph
            output.assign(initial)
        wp.capture_launch(graph)
        wp.synchronize_device()
        actual = output.numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        padding = np.ones(matrix_size, dtype=bool)
        for world, rows in enumerate(active):
            start = matrix_offsets[world]
            padding[start : start + rows * rows] = False
            matrix = actual[start : start + rows * rows].reshape(rows, rows)
            np.testing.assert_allclose(matrix, matrix.T, rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(actual[padding], initial[padding])
        # The workspace adds contributions; clearing belongs to DelassusOperator.
        assemble()
        wp.synchronize_device()
        expected_additive = expected.copy()
        expected_additive[~padding] *= 2
        np.testing.assert_allclose(output.numpy(), expected_additive, rtol=3e-5, atol=5e-5)
        np.testing.assert_array_equal(output.numpy()[padding], initial[padding])


def _fixed_chain_fixture(body_count, device):
    """Construct real Kamino containers; populate a synthetic valid sparse J."""
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
    from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDoFType
    from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
    from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians

    builder = ModelBuilderKamino(default_world=False)
    world = builder.add_world(name="fixture")
    identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    bodies = [
        builder.add_rigid_body(
            m_i=1.0,
            i_I_i=identity,
            q_i_0=wp.transformf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            name=f"body_{index}",
            world_index=world,
        )
        for index in range(body_count)
    ]
    for index in range(body_count - 1):
        builder.add_joint(
            act_type=JointActuationType.PASSIVE,
            dof_type=JointDoFType.FIXED,
            bid_B=bodies[index],
            bid_F=bodies[index + 1],
            B_r_Bj=wp.vec3f(0.0),
            F_r_Fj=wp.vec3f(0.0),
            X_Bj=identity,
            name=f"joint_{index}",
            world_index=world,
        )
    model = builder.finalize(device)
    data = model.data()
    # Match SolverKamino's setup order: this also allocates bilateral row offsets.
    make_unilateral_constraints_info(model=model, data=data, limits=None, contacts=None)
    jacobians = SparseSystemJacobians(model)
    blocks = jacobians._J_cts.bsm
    rows = int(data.info.num_total_cts.numpy()[0])
    assert rows > 0
    coordinates = np.zeros((blocks.sum_of_num_nzb, 2), dtype=np.int32)
    values = np.zeros((blocks.sum_of_num_nzb, 6), dtype=np.float32)
    assert 2 * rows <= len(values)
    for row in range(rows):
        coordinates[2 * row] = (row, 0)
        coordinates[2 * row + 1] = (row, 6)
        values[2 * row, row % 6] = 1.0
        values[2 * row + 1, row % 6] = -1.0
    blocks.num_nzb.assign(np.array([2 * rows], dtype=np.int32))
    blocks.dims.assign(np.array([[rows, body_count * 6]], dtype=np.int32))
    blocks.nzb_coords.assign(coordinates)
    blocks.nzb_values.assign(values)
    return model, data, jacobians


def test_body_assembly_workspace_follows_operator_finalize_and_jacobian_replacement():
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.delassus import DelassusOperator
    from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians

    device = "cuda:0"
    model, data, jacobians = _fixed_chain_fixture(2, device)
    operator = DelassusOperator(model, data, solver=None)
    assert operator._body_grouped is None
    operator.build(model, data, jacobians)
    wp.synchronize_device()
    first_workspace = operator._body_grouped
    baseline = operator.D.numpy().copy()
    assert np.any(baseline != 0)
    operator.build(model, data, jacobians)
    assert operator._body_grouped is first_workspace
    np.testing.assert_allclose(operator.D.numpy(), baseline, rtol=1e-6, atol=1e-7)
    operator.build(model, data, jacobians, reset_to_zero=False)
    np.testing.assert_allclose(operator.D.numpy(), 2 * baseline, rtol=1e-6, atol=1e-7)

    replacement = SparseSystemJacobians(model)
    original_blocks, new_blocks = jacobians._J_cts.bsm, replacement._J_cts.bsm
    for name in ("num_nzb", "dims", "nzb_coords", "nzb_values"):
        wp.copy(getattr(new_blocks, name), getattr(original_blocks, name))
    operator.build(model, data, replacement)
    second_workspace = operator._body_grouped
    assert second_workspace is not first_workspace
    assert second_workspace.jacobian is new_blocks
    np.testing.assert_allclose(operator.D.numpy(), baseline, rtol=1e-6, atol=1e-7)

    # Re-finalizing the same owner replaces its BSM without finalizing the operator.
    replacement.finalize(model=model)
    refinalized_blocks = replacement._J_cts.bsm
    assert refinalized_blocks is not new_blocks
    for name in ("num_nzb", "dims", "nzb_coords", "nzb_values"):
        wp.copy(getattr(refinalized_blocks, name), getattr(new_blocks, name))
    operator.build(model, data, replacement)
    third_workspace = operator._body_grouped
    assert third_workspace is not second_workspace
    assert third_workspace.jacobian is refinalized_blocks
    np.testing.assert_allclose(operator.D.numpy(), baseline, rtol=1e-6, atol=1e-7)
    operator.build(model, data, replacement)
    assert operator._body_grouped is third_workspace
    np.testing.assert_allclose(operator.D.numpy(), baseline, rtol=1e-6, atol=1e-7)

    larger_model, larger_data, larger_jacobians = _fixed_chain_fixture(3, device)
    operator.finalize(larger_model, larger_data, solver=None)
    assert operator._body_grouped is None
    operator.build(larger_model, larger_data, larger_jacobians)
    assert operator._body_grouped is not third_workspace
    assert operator._body_grouped.jacobian is larger_jacobians._J_cts.bsm
    assert operator._body_grouped.counts.shape[0] == larger_model.size.sum_of_num_bodies
    expected = DelassusOperator(larger_model, larger_data, solver=None)
    expected.build(larger_model, larger_data, larger_jacobians)
    np.testing.assert_allclose(operator.D.numpy(), expected.D.numpy(), rtol=1e-6, atol=1e-7)
