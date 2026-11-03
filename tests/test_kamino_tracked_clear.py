"""Tracked clearing must preserve every matrix entry across the public lifecycle."""

import pytest

pytestmark = pytest.mark.physics


def _fixture(wp, body_counts, device):
    import numpy as np
    from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
    from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDoFType
    from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
    from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians

    builder = ModelBuilderKamino(default_world=False)
    identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for wi, count in enumerate(body_counts):
        world = builder.add_world(name=f"w{wi}")
        bodies = [
            builder.add_rigid_body(
                m_i=1.0,
                i_I_i=identity,
                q_i_0=wp.transformf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                name=f"b{wi}_{i}",
                world_index=world,
            )
            for i in range(count)
        ]
        for i in range(count - 1):
            builder.add_joint(
                act_type=JointActuationType.PASSIVE,
                dof_type=JointDoFType.FIXED,
                bid_B=bodies[i],
                bid_F=bodies[i + 1],
                B_r_Bj=wp.vec3f(0.0),
                F_r_Fj=wp.vec3f(0.0),
                X_Bj=identity,
                name=f"j{wi}_{i}",
                world_index=world,
            )
    model = builder.finalize(device)
    data = model.data()
    make_unilateral_constraints_info(model=model, data=data, limits=None, contacts=None)
    jac = SparseSystemJacobians(model)
    blocks = jac._J_cts.bsm
    starts = blocks.nzb_start.numpy()
    maxdims = data.info.num_total_cts.numpy().copy()
    coords = np.full((blocks.sum_of_num_nzb, 2), -99, np.int32)
    values = np.zeros((blocks.sum_of_num_nzb, 6), np.float32)

    def set_dims(active):
        active = np.asarray(active, np.int32)
        assert np.all(active <= maxdims) and np.all(active >= 0)
        coords.fill(-99)
        values.fill(0)
        for world, n in enumerate(active):
            start = starts[world]
            for row in range(n):
                coords[start + 2 * row] = (row, 0)
                coords[start + 2 * row + 1] = (row, 6)
                values[start + 2 * row, row % 6] = 1.0
                values[start + 2 * row + 1, row % 6] = -1.0
        blocks.num_nzb.assign(2 * active)
        blocks.dims.assign(np.column_stack([active, np.asarray(body_counts) * 6]).astype(np.int32))
        blocks.nzb_coords.assign(coords)
        blocks.nzb_values.assign(values)
        data.info.num_total_cts.assign(active)

    set_dims(maxdims)
    return model, data, jac, set_dims, maxdims


def test_tracked_clear_matches_full_allocation_with_changing_graph_inputs():
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.delassus import DelassusOperator

    device = "cuda:0"
    model, data, jacobians, set_dims, _ = _fixture(wp, [3, 4, 5], device)
    baseline = DelassusOperator(model, data)
    tracked = DelassusOperator(model, data, tracked_active_clear=True)
    assert not baseline.tracked_active_clear
    assert baseline._tracked_clear_highwater is None
    eta = wp.full(int(baseline.num_maxdims), 0.125, dtype=wp.float32, device=device)

    def build_reset():
        baseline.build(model, data, jacobians)
        tracked.build(model, data, jacobians)

    def build_add():
        baseline.build(model, data, jacobians, reset_to_zero=False)
        tracked.build(model, data, jacobians, reset_to_zero=False)

    def regularize():
        baseline.regularize(eta)
        tracked.regularize(eta)

    actions = {"build_reset": build_reset, "build_add": build_add, "regularize": regularize}
    for action in actions.values():
        action()
    graphs = {}
    for name, action in actions.items():
        with wp.ScopedCapture(device=device) as capture:
            action()
        graphs[name] = capture.graph
    baseline.zero()
    tracked.zero()
    # The large regularization-only write followed by a small reset catches a
    # tracker that records only the previous build dimension. Additive shrink
    # must retain the largest earlier dirty prefix until an actual clear.
    sequence = [
        ("build_reset", [12, 18, 24]),
        ("build_reset", [5, 4, 9]),
        ("build_add", [8, 12, 17]),
        ("build_add", [3, 2, 4]),
        ("regularize", [10, 15, 20]),
        ("build_reset", [2, 5, 1]),
        ("build_reset", [0, 0, 0]),
        ("regularize", [11, 17, 23]),
        ("build_add", [1, 1, 1]),
        ("build_reset", [3, 2, 4]),
    ]
    for _ in range(3):
        for name, dimensions in sequence:
            set_dims(dimensions)
            wp.capture_launch(graphs[name])
            wp.synchronize_device()
            np.testing.assert_array_equal(tracked.D.numpy(), baseline.D.numpy())


def test_public_zero_mode_changes_external_writes_and_refinalization():
    import numpy as np
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.delassus import DelassusOperator
    from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians

    device = "cuda:0"
    model, data, jacobians, set_dims, _ = _fixture(wp, [3, 4, 5], device)
    baseline = DelassusOperator(model, data)
    tracked = DelassusOperator(model, data, tracked_active_clear=True)

    def equal():
        wp.synchronize_device()
        np.testing.assert_array_equal(tracked.D.numpy(), baseline.D.numpy())

    def build():
        baseline.build(model, data, jacobians)
        tracked.build(model, data, jacobians)
        equal()

    # Public zero clears arbitrary writes, including inactive allocation tails.
    baseline.D.fill_(7.0)
    tracked.D.fill_(7.0)
    baseline.zero()
    tracked.zero()
    equal()
    assert not np.any(tracked.D.numpy())
    assert not np.any(tracked._tracked_clear_highwater.numpy())

    # Enabling preserves current data and conservatively invalidates capacity.
    tracked.set_tracked_active_clear(False)
    baseline.D.fill_(11.0)
    tracked.D.fill_(11.0)
    before = tracked.D.numpy().copy()
    tracked.set_tracked_active_clear(True)
    np.testing.assert_array_equal(tracked.D.numpy(), before)
    set_dims([2, 3, 1])
    build()

    baseline.D.fill_(-3.0)
    tracked.D.fill_(-3.0)
    tracked.invalidate_tracked_clear()
    set_dims([1, 2, 3])
    build()

    tracked.set_tracked_active_clear(False)
    baseline.D.fill_(19.0)
    tracked.D.fill_(19.0)
    set_dims([3, 1, 2])
    build()
    tracked.set_tracked_active_clear(True)

    dense = DenseSystemJacobians(model)
    dense.data.J_cts_data.zero_()
    set_dims([4, 6, 8])
    baseline.build(model, data, dense)
    tracked.build(model, data, dense)
    equal()
    eta = wp.full(int(baseline.num_maxdims), 0.125, dtype=wp.float32, device=device)
    baseline.regularize(eta)
    tracked.regularize(eta)
    set_dims([1, 1, 1])
    baseline.build(model, data, dense)
    tracked.build(model, data, dense)
    equal()

    larger_model, larger_data, larger_jacobians, _, _ = _fixture(wp, [4, 5], device)
    tracked.finalize(larger_model, larger_data)
    baseline = DelassusOperator(larger_model, larger_data)
    tracked.build(larger_model, larger_data, larger_jacobians)
    baseline.build(larger_model, larger_data, larger_jacobians)
    equal()
    deferred = DelassusOperator()
    deferred.set_tracked_active_clear(True)
    deferred.finalize(larger_model, larger_data)
    deferred.build(larger_model, larger_data, larger_jacobians)
    np.testing.assert_array_equal(deferred.D.numpy(), baseline.D.numpy())


@pytest.mark.parametrize(
    "execution",
    [
        {"tracked_active_clear": True},
        {"tracked_active_clear": False},
        {"tracked_active_clear": True, "matrix_preconditioner_stride": 128},
        {"matrix_preconditioner_stride": 128},
    ],
)
def test_tracked_clear_bridge_is_independent_of_preconditioner_stride(execution):
    from humanoid_sim.physics.isaaclab import solver_cfg

    settings = {"integrator": "euler", "iterations": 9, "max_contacts_per_world": 2}
    baseline = solver_cfg("kamino", settings).to_solver_config()
    configured = solver_cfg("kamino", settings | execution).to_solver_config()
    assert baseline.dynamics.tracked_active_clear is False
    assert configured.dynamics.tracked_active_clear is execution.get("tracked_active_clear", False)
    assert configured.dynamics.matrix_preconditioner_stride == execution.get(
        "matrix_preconditioner_stride", 0
    )
    configured.dynamics.tracked_active_clear = False
    configured.dynamics.matrix_preconditioner_stride = 0
    assert configured == baseline


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_tracked_clear_requires_a_boolean(invalid):
    from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig

    with pytest.raises(ValueError, match="tracked_active_clear must be bool"):
        ConstrainedDynamicsConfig(tracked_active_clear=invalid)


def test_tracked_clear_requires_common_dense_dynamics_and_reaches_operator():
    import warp as wp
    from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
    from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig

    model, data, jacobians, _, _ = _fixture(wp, [3, 4, 5], "cuda:0")
    enabled = DualProblem.Config(dynamics=ConstrainedDynamicsConfig(tracked_active_clear=True))
    disabled = DualProblem.Config()
    with pytest.raises(ValueError, match="common to every world"):
        DualProblem(
            model=model, data=data, jacobians=jacobians, sparse=False, config=[enabled, disabled, enabled]
        )
    with pytest.raises(ValueError, match="requires dense dynamics"):
        DualProblem(model=model, data=data, jacobians=jacobians, sparse=True, config=enabled)
    default = DualProblem(model=model, data=data, jacobians=jacobians, sparse=False)
    configured = DualProblem(model=model, data=data, jacobians=jacobians, sparse=False, config=enabled)
    assert not default.delassus.tracked_active_clear
    assert configured.delassus.tracked_active_clear
