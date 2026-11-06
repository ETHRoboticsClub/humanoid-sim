"""Per-env crank-stop stiffness and contact restitution: the solref arithmetic, the parameters' specs, the solver
support checks and the model readers/writers on a stand-in model (no Isaac Lab needed)."""

import types

import pytest
import torch

from humanoid_sim.model.asset import load_profile
from humanoid_sim.physics.crank_stops import StopSolref, check_stop_solref_support, scaled_solref
from humanoid_sim.randomization import PARAMS, ParamSpec, Randomizer

ROBOT = "ethrc_v1"
KNOBS = ("restitution", "crank_stop_stiffness", "crank_stop_damping")


def _mujoco_gains(solref):
    """MuJoCo's stiffness and damping of a time-constant solref, up to the impedance factor common to all."""
    tau, zeta = solref[..., 0], solref[..., 1]
    return 1.0 / (tau * zeta) ** 2, 2.0 / tau


def test_unit_scales_leave_solref_alone():
    base = torch.tensor([[[0.008, 1.0], [0.02, 0.7]]]).repeat(3, 1, 1)
    one = torch.ones(3, 1)
    torch.testing.assert_close(scaled_solref(base, one, one), base)


def test_stiffness_and_damping_scale_independently():
    base = torch.tensor([[[0.008, 1.0], [0.02, 0.7]]]).repeat(4, 1, 1)
    s = torch.tensor([[0.25], [4.0], [1.0], [0.5]])
    c = torch.tensor([[1.0], [1.0], [2.0], [0.5]])
    k0, b0 = _mujoco_gains(base)
    k, b = _mujoco_gains(scaled_solref(base, s, c))
    torch.testing.assert_close(k, k0 * s)
    torch.testing.assert_close(b, b0 * c)


def test_direct_solref_scales_as_given():
    base = torch.tensor([[[-2000.0, -30.0]]])
    out = scaled_solref(base, torch.tensor([[0.5]]), torch.tensor([[3.0]]))
    torch.testing.assert_close(out, torch.tensor([[[-1000.0, -90.0]]]))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_scales_raise(bad):
    base = torch.tensor([[[0.008, 1.0]]])
    ok = torch.ones(1, 1)
    with pytest.raises(ValueError, match="crank_stop_stiffness"):
        scaled_solref(base, torch.full((1, 1), bad), ok)
    with pytest.raises(ValueError, match="crank_stop_damping"):
        scaled_solref(base, ok, torch.full((1, 1), bad))


@pytest.mark.parametrize("solref", [[0.0, 0.0], [0.008, -1.0], [0.0, 1.0]])
def test_solref_without_a_scale_raises(solref):
    with pytest.raises(ValueError, match="solref"):
        scaled_solref(torch.tensor([[solref]]), torch.ones(1, 1), torch.ones(1, 1))


def test_parameters_are_per_env_physics_parameters_that_start_at_the_identity():
    for name in KNOBS:
        assert PARAMS[name].kind == "physics" and PARAMS[name].target == "env"
    specs = [ParamSpec(n, "log_uniform", (0.1, 1.0), "scale") for n in KNOBS[1:]]
    specs.append(ParamSpec("restitution", "uniform", (0.2, 0.8), "set"))
    nominal = {n: torch.ones(32, 1) for n in KNOBS} | {"restitution": torch.zeros(32, 1)}
    r = Randomizer(specs, nominal, {}, num_envs=32, seed=3)
    assert r.physics_params == sorted(KNOBS)
    assert ((r.values["crank_stop_stiffness"] >= 0.1) & (r.values["crank_stop_stiffness"] <= 1.0)).all()
    assert r.values["crank_stop_stiffness"].std() > 0 and r.values["restitution"].std() > 0
    assert r.values["crank_stop_stiffness"].shape == (32, 1)
    r.set_level(0.0)  # level 0 is the nominal: no softening, the restitution at the centre of its range
    torch.testing.assert_close(r.values["crank_stop_stiffness"], torch.ones(32, 1))
    torch.testing.assert_close(r.values["restitution"], torch.full((32, 1), 0.5))


def test_defaults_are_unchanged_without_a_spec():
    r = Randomizer([], {}, {}, num_envs=4)
    assert not set(KNOBS) & set(r.values)


def test_physics_scope_is_startup_only():
    with pytest.raises(ValueError, match="solver parameter"):
        ParamSpec("crank_stop_stiffness", "uniform", (0.5, 1.0), "scale", scope="reset")


def test_only_robots_with_ankle_linkages_have_stops_to_soften():
    physics = load_profile(ROBOT)["physics"]
    check_stop_solref_support(ROBOT, physics, "mjwarp")
    check_stop_solref_support(ROBOT, physics, "kamino")  # the native ankle's crank joints carry soft stops
    for solver in ("mjwarp", "kamino"):
        with pytest.raises(ValueError, match="no ankle linkages"):
            check_stop_solref_support("g1_29dof_rev_1_0", load_profile("g1_29dof_rev_1_0")["physics"], solver)
    with pytest.raises(ValueError, match="mjwarp or kamino"):
        check_stop_solref_support(ROBOT, physics, "other")


def _plant_stub(solver, manager=object()):
    from humanoid_sim.train.plant import RobotPlant

    plant = object.__new__(RobotPlant)
    plant._solver, plant._physics_manager = solver, manager
    plant.cfg = types.SimpleNamespace(robot=ROBOT)
    plant.profile = load_profile(ROBOT)
    return plant


def test_restitution_needs_kamino_and_the_physics_manager():
    _plant_stub("kamino")._contact_knobs({"restitution"})
    with pytest.raises(ValueError, match="no contact restitution"):
        _plant_stub("mjwarp")._contact_knobs({"restitution", "friction"})
    with pytest.raises(ValueError, match="physics manager"):
        _plant_stub("kamino", None)._contact_knobs({"restitution"})


def test_mjwarp_crank_stops_need_the_physics_manager_from_the_plant():
    _plant_stub("mjwarp")._contact_knobs({"crank_stop_stiffness"})
    _plant_stub("kamino")._contact_knobs({"crank_stop_damping"})
    with pytest.raises(ValueError, match="physics manager"):
        _plant_stub("mjwarp", None)._contact_knobs({"crank_stop_damping"})
    # Kamino's stops are torques on the crank joints: no solver model to write.
    _plant_stub("kamino", None)._contact_knobs({"crank_stop_damping"})
    with pytest.raises(ValueError, match="physics manager"):
        _plant_stub("kamino", None)._contact_knobs({"crank_stop_damping", "restitution"})
    _plant_stub("kamino")._contact_knobs({"friction"})  # the other parameters are not this check's business


def _tendon_model(num_envs, solref):
    import warp as wp

    stops = len(solref)
    labels = ["other"] + [f"/World/e{w}/crank_stop_{k}" for w in range(num_envs) for k in range(stops)]
    base = [[0.5, 0.5]] + [list(solref[k]) for _ in range(num_envs) for k in range(stops)]
    worlds = [-1] + [w for w in range(num_envs) for _ in range(stops)]
    mujoco = types.SimpleNamespace(
        tendon_label=labels, tendon_solref_limit=wp.array(base, dtype=wp.vec2, device="cpu"),
        tendon_world=wp.array(worlds, dtype=wp.int32, device="cpu"))
    return types.SimpleNamespace(mujoco=mujoco)


def test_tendon_stops_are_read_and_written_per_env_and_nothing_else_moves():
    pytest.importorskip("warp", reason="needs the newton extra")
    ModelFlags = pytest.importorskip("newton").ModelFlags

    desc = types.SimpleNamespace(usd_path=None)
    model = _tendon_model(3, [(0.008, 1.0), (0.008, 1.0)])
    stops = StopSolref(None, model, desc, 3)
    assert stops.flag == ModelFlags.TENDON_PROPERTIES
    read = stops.read()
    assert read.shape == (3, 2, 2)
    s, c = torch.tensor([[1.0], [0.25], [1.0]]), torch.tensor([[1.0], [1.0], [2.0]])
    stops.write(scaled_solref(read, s, c))
    after = stops.read()
    k0, b0 = _mujoco_gains(read)
    k, b = _mujoco_gains(after)
    torch.testing.assert_close(k, k0 * s)
    torch.testing.assert_close(b, b0 * c)
    import warp as wp

    torch.testing.assert_close(wp.to_torch(model.mujoco.tendon_solref_limit)[0], torch.tensor([0.5, 0.5]))


def test_tendon_stops_missing_or_out_of_world_order_raise():
    wp = pytest.importorskip("warp", reason="needs the newton extra")

    desc = types.SimpleNamespace(usd_path=None)
    with pytest.raises(RuntimeError, match="crank_stop_hook"):
        StopSolref(None, _tendon_model(3, [(0.008, 1.0)]), desc, 2)
    model = _tendon_model(2, [(0.008, 1.0)])
    model.mujoco.tendon_world = wp.array([-1, 1, 0], dtype=wp.int32, device="cpu")
    with pytest.raises(RuntimeError, match="world order"):
        StopSolref(None, model, desc, 2)
