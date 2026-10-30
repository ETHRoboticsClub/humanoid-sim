"""Optional real-source lifecycle checks, imported only by executed physics tests."""

import unittest

import numpy as np
import warp as wp
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import SparseSystemJacobians
from newton._src.solvers.kamino._src.linalg.linear import LLTBlockedSolver
from newton._src.solvers.kamino.config import PADMMSolverConfig


def fixture(lazy, device):
    b = ModelBuilderKamino(default_world=False)
    w = b.add_world(name="owner-fixture")
    identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    bodies = [
        b.add_rigid_body(
            m_i=1.0, i_I_i=identity, q_i_0=wp.transformf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), world_index=w
        )
        for _ in range(2)
    ]
    b.add_joint(
        act_type=JointActuationType.PASSIVE,
        dof_type=JointDoFType.REVOLUTE,
        bid_B=bodies[0],
        bid_F=bodies[1],
        B_r_Bj=wp.vec3f(0.0),
        F_r_Fj=wp.vec3f(0.0),
        X_Bj=identity,
        a_j=0.1,
        b_j=1.0,
        f_j=0.1,
        world_index=w,
    )
    model = b.finalize(device)
    data = model.data()
    make_unilateral_constraints_info(model, data, None, None)
    jac = SparseSystemJacobians(model)
    p = DualProblem(
        model=model,
        data=data,
        jacobians=jac,
        solver=LLTBlockedSolver,
        sparse=False,
        defer_preconditioning=lazy,
    )
    d = p._private_data()
    rng = np.random.default_rng(191)
    raw = rng.normal(size=d.D.shape).astype(np.float32)
    P = rng.uniform(0.4, 2.0, size=d.P.shape).astype(np.float32)
    d.D.assign(raw)
    d.P.assign(P)
    d.v_f.fill_(0.125)
    return p, raw, model, data, jac


def bits(a, b):
    np.testing.assert_array_equal(a.numpy().view(np.uint32), b.numpy().view(np.uint32))


class Tests(unittest.TestCase):
    def test_config_default_and_scope(self):
        self.assertFalse(PADMMSolverConfig().reduced_lazy_preconditioning)
        for kwargs in (
            {"reduced_lazy_preconditioning": True},
            {"algorithm": "bilateral_reduced", "reduced_lazy_preconditioning": True},
            {"reduced_lazy_preconditioning": 1},
        ):
            with self.assertRaises(ValueError):
                PADMMSolverConfig(**kwargs)
        c = PADMMSolverConfig(
            algorithm="bilateral_reduced",
            reduced_ordering="rcm",
            reduced_dynamic_pattern=True,
            reduced_lazy_preconditioning=True,
        )
        self.assertTrue(c.reduced_lazy_preconditioning)

    def test_default_has_no_owner_and_actual_preconditioning_exact(self):
        a, *_ = fixture(False, self.device)
        b, raw, *_ = fixture(True, self.device)
        self.assertIsNone(a._preconditioning_owner)
        self.assertIsNone(a._delassus._logical_owner)
        for p in (a, b):
            p._apply_dual_preconditioner_to_dual()
        np.testing.assert_array_equal(b._private_data().D.numpy(), raw)
        bits(a.data.D, b.data.D)
        bits(a.data.v_f, b.data.v_f)

    def test_all_public_routes_preserve_objects_and_publish(self):
        for route in ("data", "D", "operator", "solver", "info"):
            p, *_ = fixture(True, self.device)
            q = p._private_data()
            d = p._delassus
            p._apply_dual_preconditioner_to_dual()
            expected = {
                "data": q,
                "D": q.D,
                "operator": d._operator,
                "solver": d._solver,
                "info": d._operator.info,
            }[route]
            got = p.data if route == "data" else getattr(d, route)
            self.assertIs(got, expected)
            self.assertTrue(p._preconditioning_owner.has_public_alias)
            self.assertEqual(int(p._preconditioning_owner.public_exposed.numpy()[0]), 1)

    def test_eager_default_public_getters_remain_allowed_during_capture(self):
        for route in ("data", "D", "operator", "solver", "info"):
            problem, *_ = fixture(False, self.device)
            problem._apply_dual_preconditioner_to_dual()
            data = problem.data
            operator = problem._delassus
            expected = {
                "data": data,
                "D": data.D,
                "operator": operator._operator,
                "solver": operator._solver,
                "info": operator._operator.info,
            }[route]
            aliases = []
            condition = wp.zeros(1, dtype=int, device=self.device)
            stream = wp.get_stream(self.device) if self.device.startswith("cuda") else None

            def access():
                aliases.append(problem.data if route == "data" else getattr(operator, route))

            with wp.ScopedCapture(device=self.device, stream=stream):
                wp.capture_if(condition, access, stream=stream)
            self.assertEqual(len(aliases), 1)
            self.assertIs(aliases[0], expected)
            self.assertIsNone(problem._preconditioning_owner)

    def test_all_public_first_getters_inside_false_capture_reject(self):
        for route in ("data", "D", "operator", "solver", "info"):
            p, *_ = fixture(True, self.device)
            p._apply_dual_preconditioner_to_dual()
            escaped = []
            errors = []
            flag = wp.zeros(1, dtype=int, device=self.device)

            def attempt():
                try:
                    escaped.append(p.data if route == "data" else getattr(p._delassus, route))
                except RuntimeError as e:
                    errors.append(str(e))

            with wp.ScopedCapture(
                device=self.device,
                stream=wp.get_stream(self.device) if self.device.startswith("cuda") else None,
            ):
                wp.capture_if(
                    flag,
                    attempt,
                    stream=wp.get_stream(self.device) if self.device.startswith("cuda") else None,
                )
            self.assertFalse(escaped)
            self.assertEqual(len(errors), 1)
            self.assertFalse(p._preconditioning_owner.has_public_alias)

    def test_data_rebinding_uses_eager_arrays_without_repair(self):
        for field in ("D", "P", "dim", "mio", "vio"):
            a, *_ = fixture(False, self.device)
            b, *_ = fixture(True, self.device)
            for p in (a, b):
                p._apply_dual_preconditioner_to_dual()
                d = p.data
                replacement = wp.clone(getattr(d, field))
                setattr(d, field, replacement)
                self.assertIs(getattr(p.data, field), replacement)
                if field in ("D", "P"):
                    replacement.fill_(0.75)
                elif field == "dim":
                    replacement.fill_(3)
                else:
                    d.dim = wp.clone(d.dim)
                    d.dim.fill_(3)
                    replacement.fill_(1)
                p._apply_dual_preconditioner_to_dual()
            self.assertIsNone(b._active_preconditioning_owner())
            bits(a.data.D, b.data.D)
            bits(a.data.v_f, b.data.v_f)
            bits(a._delassus._operator.mat, b._delassus._operator.mat)

    def test_operator_rebinding_preserves_detached_alias(self):
        a, *_ = fixture(False, self.device)
        b, *_ = fixture(True, self.device)
        for p in (a, b):
            p._apply_dual_preconditioner_to_dual()
            op = p.delassus.operator
            old = p.data.D
            replacement = wp.clone(op.mat)
            replacement.fill_(0.375)
            op.mat = replacement
            p._apply_dual_preconditioner_to_dual()
            self.assertIs(p.data.D, old)
            self.assertIs(op.mat, replacement)
        self.assertIsNone(b._active_preconditioning_owner())
        bits(a.data.D, b.data.D)
        bits(a.delassus.D, b.delassus.D)

    def test_public_regularize_and_zero_materialize_and_reset(self):
        a, *_ = fixture(False, self.device)
        b, *_ = fixture(True, self.device)
        for p in (a, b):
            p._apply_dual_preconditioner_to_dual()
            eta = wp.full(p._private_data().P.shape, 0.125, dtype=float, device=self.device)
            p.delassus.regularize(eta)
        bits(a.data.D, b.data.D)
        for p in (a, b):
            p.delassus.zero()
        bits(a.data.D, b.data.D)
        self.assertTrue(np.all(b.data.D.numpy() == 0))
        self.assertTrue(b._preconditioning_owner.has_public_alias)

    def test_actual_repeated_precondition(self):
        a, *_ = fixture(False, self.device)
        b, *_ = fixture(True, self.device)
        for p in (a, b):
            p._apply_dual_preconditioner_to_dual()
            p._private_data().P.fill_(1.25)
            p._apply_dual_preconditioner_to_dual()
        bits(a.data.D, b.data.D)
        bits(a.data.v_f, b.data.v_f)

    def test_actual_build_reset_and_additive_preserve_source_operator(self):
        for tracked in (False, True):
            for sequence in ((True,), (True, False), (True, False, False), (True, False, False, True)):
                fixtures = [fixture(False, self.device), fixture(True, self.device)]
                for p, raw, m, d, j in fixtures:
                    m.time.dt.fill_(0.001)
                    p._delassus.set_tracked_active_clear(tracked)
                    j.build(model=m, data=d, limits=None, contacts=None, reset_to_zero=True)
                    for reset in sequence:
                        p.build(model=m, data=d, jacobians=j, reset_to_zero=reset)
                a, b = fixtures[0][0], fixtures[1][0]
                # Observe only after the complete sequence: additive builds must
                # handle the prior pending P epoch themselves.
                b._materialize_matrix()
                bits(a._private_data().D, b._private_data().D)
                bits(a._private_data().P, b._private_data().P)
                bits(a._private_data().v_f, b._private_data().v_f)
                self.assertFalse(b._preconditioning_owner.has_public_alias)

    def test_native_sigma_status_and_repeated_proximal_updates(self):
        from newton._src.solvers.kamino._src.solvers.padmm.kernels import (
            _update_delassus_proximal_regularization,
        )
        from newton._src.solvers.kamino._src.solvers.padmm.types import PADMMStatus

        a, *_ = fixture(False, self.device)
        b, *_ = fixture(True, self.device)
        status = PADMMStatus()
        status.converged = 0
        statuses = wp.array([status], dtype=PADMMStatus, device=self.device)
        sigma = wp.array([wp.vec2f(0.05001, 0.0)], dtype=wp.vec2f, device=self.device)
        for p in (a, b):
            p._apply_dual_preconditioner_to_dual()
        for values, converged in (((0.05001, 0.0), 0), ((0.07501, 0.05001), 0), ((0.1, 0.07501), 1)):
            sigma.assign(np.array([values], np.float32))
            status.converged = converged
            statuses.assign([status])
            d = a._private_data()
            wp.launch(
                _update_delassus_proximal_regularization,
                dim=(1, d.D.shape[0]),
                inputs=[d.dim, d.mio, statuses, sigma, d.D],
                device=self.device,
            )
            b._preconditioning_owner.source_proximal_update(sigma, statuses)
        b._materialize_matrix()
        bits(a._private_data().D, b._private_data().D)
        self.assertFalse(b._preconditioning_owner.has_public_alias)

    def test_full_solver_on_unpublished_lazy_owner_matches_eager(self):
        from newton._src.solvers.kamino._src.solvers.padmm.solver import PADMMSolver

        captures = []
        for lazy in (False, True):
            p, raw, m, *_ = fixture(lazy, self.device)
            d = p._private_data()
            n = int(d.dim.numpy()[0])
            matrix = np.zeros(d.D.shape, np.float32)
            matrix[: n * n] = (np.eye(n, dtype=np.float32) * 2).ravel()
            d.D.assign(matrix)
            p._apply_dual_preconditioner_to_dual()
            solver = PADMMSolver(
                model=m, config=PADMMSolverConfig(max_iterations=7), use_graph_conditionals=False
            )
            solver.coldstart()
            solver.solve(p)
            wp.synchronize_device(self.device)
            record = {
                k: v.numpy().copy() for k, v in vars(solver.data.state).items() if isinstance(v, wp.array)
            }
            record["status"] = solver.data.status.numpy().copy()
            record["lambda"] = solver.data.solution.lambdas.numpy().copy()
            captures.append(record)
            if lazy:
                self.assertFalse(p._preconditioning_owner.has_public_alias)
        for key in captures[0]:
            self.assertEqual(captures[0][key].tobytes(), captures[1][key].tobytes(), key)

    def test_refinalize_replaces_owner_and_preserves_old_public_array(self):
        p, raw, m, d, j = fixture(True, self.device)
        p._apply_dual_preconditioner_to_dual()
        old = p.data.D
        oldvalues = old.numpy().copy()
        owner = p._preconditioning_owner
        p.finalize(model=m, data=d, jacobians=j, solver=LLTBlockedSolver)
        self.assertIsNot(p._preconditioning_owner, owner)
        self.assertIsNot(p._private_data().D, old)
        self.assertFalse(p._preconditioning_owner.has_public_alias)
        np.testing.assert_array_equal(old.numpy(), oldvalues)
