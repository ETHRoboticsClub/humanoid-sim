"""Fixed numerical references remain separate from robot and controller gains."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.devices.shaping import SolverGainReference
from humanoid_sim.model.description import Description
from humanoid_sim.physics.joint_properties import joint_properties
from humanoid_sim.replay.exc import DriveGains
from humanoid_sim.replay.gains import capture_solver_reference, check_run_gain_support, gain_support


def capture(desc, *, scale=1.0, name="fixture", zero_d=False):
    gains = {d: DriveGains(v.baseline_stiffness * scale, 0 if zero_d else v.baseline_damping * scale, 0, 0)
             for d, v in desc.drives.items()}
    return SimpleNamespace(dataset_id=name, gains=lambda: gains)


def test_default_is_numerically_unchanged_and_references_are_separate():
    desc = Description.load()
    cfg = PlantCfg(dtype=torch.float64)
    nominal = PlantCore(desc, list(desc.joints), 4, cfg)
    cap = capture(desc)
    equal = PlantCore(desc, list(desc.joints), 4, cfg, solver_gains=capture_solver_reference([cap], desc))
    torch.testing.assert_close(equal.k0, nominal.k0, atol=0, rtol=0)
    torch.testing.assert_close(equal.d0, nominal.d0, atol=0, rtol=0)
    higher = PlantCore(desc, list(desc.joints), 4, cfg,
                       solver_gains=capture_solver_reference([capture(desc, scale=10)], desc))
    torch.testing.assert_close(higher.drive_stiffness, nominal.drive_stiffness, atol=0, rtol=0)
    torch.testing.assert_close(higher.drive_damping, nominal.drive_damping, atol=0, rtol=0)
    torch.testing.assert_close(higher.k0, nominal.k0 * 10)
    torch.testing.assert_close(higher.d0, nominal.d0 * 10)
    props = joint_properties(higher, list(desc.joints))
    np.testing.assert_allclose(props.stiffness, higher.k0[0])
    np.testing.assert_array_equal(props.effort_limit, joint_properties(nominal, list(desc.joints)).effort_limit)


def test_capture_envelope_inventory_and_warning_reset():
    desc = Description.load()
    cap = capture(desc, scale=10)
    nominal = PlantCore(desc, list(desc.joints), 4)
    with pytest.raises(ValueError, match="exceed"):
        gain_support(cap, nominal)
    refs = capture_solver_reference([cap, capture(desc, scale=12, name="held")], desc)
    plant = PlantCore(desc, list(desc.joints), 4, solver_gains=refs)
    assert gain_support(cap, plant)["unsupported_drives"] == []
    p, d = refs.tensors(plant.map.drive_names, device="cpu", dtype=torch.float32)
    plant.submit(DriveDemand.pd(torch.zeros(4, plant.map.num_drives), p, d), torch.zeros(4, dtype=torch.int64))
    assert not plant.gain_warnings.any()
    plant.submit(DriveDemand.pd(torch.zeros(4, plant.map.num_drives), p * 3, d), torch.zeros(4, dtype=torch.int64))
    assert plant.gain_warnings.all()
    with pytest.raises(RuntimeError, match="support exceeded"):
        check_run_gain_support(plant)
    plant.reset(torch.arange(4))
    assert not plant.gain_warnings.any()
    assert check_run_gain_support(plant)["drive_warning"] is False
    with pytest.raises(ValueError, match="inventory"):
        PlantCore(desc, list(desc.joints), 4, solver_gains=SolverGainReference(("wrong",), (1,), (1,), "fixture"))


@pytest.mark.parametrize("zero_d", [False, True])
def test_actual_nonlinear_ankle_bounds_match_independent_eigenvalues(zero_d):
    desc = Description.load()
    cap = capture(desc, scale=7, zero_d=zero_d)
    plant = PlantCore(desc, list(desc.joints), 4, PlantCfg(dtype=torch.float64),
                       solver_gains=capture_solver_reference([cap], desc))
    q = torch.tensor([j.standing_offset_rad for j in desc.joints.values()], dtype=torch.float64).expand(4, -1).clone()
    for _, _, js, _ in plant.map.ankles:
        q[:, js] += torch.tensor([[0, 0], [0.02, 0.01], [-0.02, 0.015], [0.01, -0.01]])
    _, _, jacs = plant.map.drive_state(q, torch.zeros_like(q))
    plant._check_ankle_reference_ratios(jacs)
    expected = np.zeros(4)
    for (_, ds, js, _), jac in zip(plant.map.ankles, jacs, strict=True):
        for ref, diag in ((plant.solver_drive_stiffness, plant.k0), (plant.solver_drive_damping, plant.d0)):
            for e in range(4):
                k = jac[e].numpy().T @ np.diag(ref[ds].numpy()) @ jac[e].numpy()
                r = diag[e, js].numpy()
                if not np.any(r):
                    continue
                normal = np.diag(1 / np.sqrt(r))
                expected[e] = max(expected[e], np.linalg.eigvalsh(normal @ k @ normal).max())
    np.testing.assert_allclose(plant.ankle_reference_ratio_peak, expected, atol=1e-12, rtol=1e-12)
    assert np.isfinite(expected).all() and (expected < 2).all()
    # Same check must expose unsupported coupling, even without motor-space warnings.
    plant.k0 *= 0.1
    plant._check_ankle_reference_ratios(jacs)
    with pytest.raises(RuntimeError, match="coupled_ankle"):
        check_run_gain_support(plant)


@pytest.mark.parametrize("p,d", [(0, 1), (-1, 1), (1, -1), (float("nan"), 1)])
def test_invalid_reference_is_rejected(p, d):
    with pytest.raises(ValueError):
        SolverGainReference(("fixture",), (p,), (d,), "fixture")


def test_reference_order_and_envelope_not_selected_by_capture_order():
    desc = Description.load()
    a, b = capture(desc, scale=2), capture(desc, scale=3, name="held")
    one, two = capture_solver_reference([a, b], desc), capture_solver_reference([b, a], desc)
    assert one.stiffness == two.stiffness and one.damping == two.damping
    names = list(reversed(one.drives))
    p, _ = one.tensors(names, device="cpu", dtype=torch.float64)
    np.testing.assert_array_equal(p, list(reversed(one.stiffness)))
