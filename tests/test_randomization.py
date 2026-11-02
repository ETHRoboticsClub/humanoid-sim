"""Randomization specs and draws: parsing, selection, composition, scopes, levels, reproducibility."""

import pytest
import torch

from humanoid_sim.randomization import ParamSpec, Randomizer, parse_specs

NAMES = {"body": ["base", "torso", "arm"], "joint": ["hip", "knee"], "drive": ["d0", "d1", "d2"]}


def _nominal(e=64):
    return {
        "body_mass": torch.tensor([[10.0, 20.0, 2.0]]).repeat(e, 1),
        "body_com": torch.zeros(e, 3, 3),
        "joint_armature": torch.full((e, 2), 0.05),
        "friction": torch.ones(e, 1),
        "latency": torch.zeros(e, 3),
        "encoder_offset": torch.zeros(e, 3),
        "imu_gyro_bias": torch.zeros(e, 3),
    }


def test_parse_and_validate():
    s = ParamSpec.from_dict({"param": "body_mass", "bodies": "torso", "dist": "uniform", "range": [1, 5], "op": "add"})
    assert (s.select, s.kind, s.startup) == ("torso", "physics", True)
    d = ParamSpec.from_dict({"param": "latency", "drives": "d.*", "dist": "uniform", "range": [0.01, 0.02], "op": "set"})
    assert (d.kind, d.startup) == ("device", False)
    g = ParamSpec.from_dict({"param": "imu_gyro_bias", "select": "z", "dist": "uniform", "range": [-0.01, 0.01],
                             "op": "set"})
    assert g.select == "z"
    with pytest.raises(ValueError, match="'bodies'"):
        ParamSpec.from_dict({"param": "body_mass", "select": "torso", "dist": "uniform", "range": [1, 5], "op": "add"})
    with pytest.raises(ValueError, match="unknown parameter"):
        ParamSpec.from_dict({"param": "mass", "dist": "uniform", "range": [0, 1], "op": "add"})
    with pytest.raises(ValueError, match="solver parameter"):
        ParamSpec.from_dict({"param": "joint_armature", "dist": "uniform", "range": [0.8, 1.2], "op": "scale",
                             "scope": "reset"})
    with pytest.raises(ValueError, match="does not apply"):
        ParamSpec.from_dict({"param": "latency", "bodies": "x", "dist": "uniform", "range": [0, 1], "op": "set"})
    with pytest.raises(ValueError, match="log_uniform"):
        ParamSpec(param="latency", dist="log_uniform", range=(0.0, 1.0), op="set")


def test_startup_specs_compose_in_order_on_selected_targets():
    specs = parse_specs([
        {"param": "body_mass", "dist": "uniform", "range": [0.9, 1.1], "op": "scale"},
        {"param": "body_mass", "bodies": "torso", "dist": "uniform", "range": [1.0, 5.0], "op": "add"},
    ])
    r = Randomizer(specs, _nominal(), NAMES, num_envs=64)
    m = r.values["body_mass"]
    assert ((m[:, 0] >= 9.0) & (m[:, 0] <= 11.0)).all()
    assert ((m[:, 2] >= 1.8) & (m[:, 2] <= 2.2)).all()
    assert ((m[:, 1] >= 18.0 + 1.0) & (m[:, 1] <= 22.0 + 5.0)).all()
    assert m[:, 0].std() > 0  # per env and per target
    assert r.physics_params == ["body_mass"]


def test_reset_redraws_only_device_params_of_the_given_envs():
    specs = parse_specs([
        {"param": "joint_armature", "dist": "uniform", "range": [0.8, 1.2], "op": "scale"},
        {"param": "latency", "dist": "uniform", "range": [0.010, 0.020], "op": "set", "per_target": False},
    ])
    r = Randomizer(specs, _nominal(), NAMES, num_envs=64)
    arm, lat = r.values["joint_armature"].clone(), r.values["latency"].clone()
    assert (lat[:, 0] == lat[:, 2]).all()  # one draw shared by an env's drives
    ids = torch.tensor([3, 5])
    r.draw_reset(ids)
    assert torch.equal(r.values["joint_armature"], arm)
    keep = torch.ones(64, dtype=torch.bool)
    keep[ids] = False
    assert torch.equal(r.values["latency"][keep], lat[keep])
    assert not torch.equal(r.values["latency"][ids], lat[ids])


def test_level_shrinks_toward_identity_and_seed_reproduces():
    specs = parse_specs([
        {"param": "joint_armature", "dist": "uniform", "range": [0.5, 1.5], "op": "scale"},
        {"param": "encoder_offset", "dist": "normal", "range": [0.0, 0.05], "op": "add"},
        {"param": "friction", "dist": "uniform", "range": [0.3, 1.0], "op": "set", "levels": False},
    ])
    r0 = Randomizer(specs, _nominal(), NAMES, num_envs=64, level=0.0)
    assert torch.allclose(r0.values["joint_armature"], torch.full((64, 2), 0.05))
    assert torch.equal(r0.values["encoder_offset"], torch.zeros(64, 3))
    assert r0.values["friction"].std() > 0  # opted out of levels
    r1 = Randomizer(specs, _nominal(), NAMES, num_envs=64, level=1.0, seed=7)
    r2 = Randomizer(specs, _nominal(), NAMES, num_envs=64, level=1.0, seed=7)
    assert all(torch.equal(r1.values[k], r2.values[k]) for k in r1.values)
    half = Randomizer(specs, _nominal(), NAMES, num_envs=64, level=0.5, seed=7)
    dev = (half.values["joint_armature"] / 0.05 - 1).abs().max()
    assert dev <= 0.25 + 1e-9


def test_vector_params_and_bad_selection():
    specs = parse_specs([{"param": "body_com", "bodies": "torso", "dist": "uniform", "range": [-0.03, 0.03],
                          "op": "add"}])
    r = Randomizer(specs, _nominal(), NAMES, num_envs=64)
    com = r.values["body_com"]
    assert com[:, 1].abs().max() <= 0.03 and com[:, 1].abs().max() > 0
    assert torch.equal(com[:, [0, 2]], torch.zeros(64, 2, 3))
    with pytest.raises(ValueError, match="selects no"):
        Randomizer(parse_specs([{"param": "latency", "drives": "nope", "dist": "uniform", "range": [0, 1],
                                 "op": "set"}]), _nominal(), NAMES, num_envs=64)


def test_no_specs():
    r = Randomizer([], {}, NAMES, num_envs=4)
    assert r.values == {} and r.physics_params == []
    r.draw_reset(torch.tensor([1]))


def test_discrete_latency_shares_one_of_four_exact_durations_per_env():
    specs = parse_specs([{'param': 'latency', 'dist': 'discrete_uniform', 'range': [0., .015],
                          'quantum': .005, 'op': 'set', 'per_target': False}])
    r = Randomizer(specs, _nominal(), NAMES, num_envs=64, seed=4)
    value = r.values['latency']
    torch.testing.assert_close(value[:, 0], value[:, 2])
    assert sorted(torch.unique((value / .005).round()).tolist()) == [0., 1., 2., 3.]
    torch.testing.assert_close(value / .005, (value / .005).round())
    previous = value.clone()
    r.draw_reset(torch.tensor([1, 3]))
    torch.testing.assert_close(r.values['latency'][[0, 2]], previous[[0, 2]])


def test_shape_material_buckets_are_shared_and_startup_only():
    specs = parse_specs([{'param': 'shape_friction', 'dist': 'uniform', 'range': [.3, 1.],
                          'op': 'set', 'buckets': 8}])
    r = Randomizer(specs, {'shape_friction': torch.ones(16, 10)}, {'shape': [f'shape_{i}' for i in range(10)]},
                   num_envs=16)
    mu = r.values['shape_friction'].clone()
    assert torch.unique(mu).numel() == 8
    assert ((mu >= .3) & (mu <= 1.)).all()
    r.draw_reset(torch.arange(16))
    torch.testing.assert_close(mu, r.values['shape_friction'])


@pytest.mark.parametrize('overrides', [{'quantum': 0.}, {'quantum': .004}, {'buckets': 0}, {'buckets': True}])
def test_invalid_discrete_or_bucket_specs_fail(overrides):
    base = {'param': 'shape_friction', 'dist': 'discrete_uniform', 'range': [0., .015],
            'quantum': .005, 'op': 'set'}
    with pytest.raises(ValueError):
        ParamSpec.from_dict(base | overrides)


COM_AXES = [[-0.03, 0.12], [-0.01, 0.01], [0.02, 0.05]]


@pytest.mark.parametrize("dist,rng", [("uniform", COM_AXES), ("discrete_uniform", [[0.0, 0.1], [0.0, 0.02], [0.0, 0.05]]),
                                      ("log_uniform", [[0.1, 1.0], [1.0, 2.0], [5.0, 6.0]])])
def test_per_axis_draws_stay_in_each_axis_bounds(dist, rng):
    extra = {"quantum": 0.01} if dist == "discrete_uniform" else {}
    specs = parse_specs([{"param": "body_com", "bodies": "torso", "dist": dist, "range": rng, "op": "set", **extra}])
    com = Randomizer(specs, _nominal(256), NAMES, num_envs=256).values["body_com"][:, 1]
    for a, (lo, hi) in enumerate(rng):
        assert (com[:, a] >= lo - 1e-12).all() and (com[:, a] <= hi + 1e-12).all()
        assert com[:, a].max() - com[:, a].min() > 0.5 * (hi - lo)


def test_per_axis_normal_scale_and_levels():
    specs = parse_specs([{"param": "body_com", "bodies": "torso", "dist": "normal",
                          "range": [[0.0, 0.0], [1.0, 0.0], [0.0, 0.5]], "op": "set"}])
    com = Randomizer(specs, _nominal(256), NAMES, num_envs=256).values["body_com"][:, 1]
    assert (com[:, 0] == 0).all() and (com[:, 1] == 1).all() and com[:, 2].std() > 0.2
    half = Randomizer(specs, _nominal(256), NAMES, num_envs=256, level=0.5).values["body_com"][:, 1]
    assert torch.allclose(half[:, 1], torch.full((256,), 1.0, dtype=half.dtype))  # set: centre is the mean


def test_per_axis_levels_shrink_toward_each_axis_centre():
    specs = parse_specs([{"param": "body_com", "bodies": "torso", "dist": "uniform", "range": COM_AXES, "op": "set"}])
    com = Randomizer(specs, _nominal(256), NAMES, num_envs=256, level=0.0).values["body_com"][:, 1]
    centres = torch.tensor([0.5 * (lo + hi) for lo, hi in COM_AXES])
    assert torch.allclose(com, centres.expand(256, 3), atol=1e-6)


def test_per_axis_imu_range_is_taken_by_selected_axis():
    rng = [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]
    specs = parse_specs([{"param": "imu_gyro_bias", "select": "[yz]", "dist": "uniform", "range": rng, "op": "set"}])
    g = Randomizer(specs, _nominal(), NAMES, num_envs=64).values["imu_gyro_bias"]
    assert (g[:, 0] == 0).all()
    assert ((g[:, 1] >= 10) & (g[:, 1] <= 11)).all() and ((g[:, 2] >= 20) & (g[:, 2] <= 21)).all()


def test_scalar_pair_is_unchanged_by_per_axis_support():
    pair = {"param": "body_com", "bodies": "torso", "dist": "uniform", "op": "add"}
    a = Randomizer(parse_specs([{**pair, "range": [-0.03, 0.12]}]), _nominal(), NAMES, num_envs=64, seed=3)
    b = Randomizer(parse_specs([{**pair, "range": [[-0.03, 0.12]] * 3}]), _nominal(), NAMES, num_envs=64, seed=3)
    assert ParamSpec.from_dict({**pair, "range": [-0.03, 0.12]}).axis_ranges is None
    assert torch.equal(a.values["body_com"], b.values["body_com"])


@pytest.mark.parametrize("param,rng,match", [
    ("body_com", [[0, 1], [0, 1]], "needs 3 pairs"),
    ("body_com", [[0, 1], [0, 1], [0, 1], [0, 1]], "needs 3 pairs"),
    ("body_com", [[0, 1], [0, 1], [0]], "two numbers"),
    ("body_com", [[0, 1], [0, 1], [0, 1, 2]], "two numbers"),
    ("body_com", [[0, 1], [2, 1], [0, 1]], "low"),
    ("body_mass", [[0, 1]], "scalar parameter"),
    ("latency", [[0, 1], [0, 1], [0, 1]], "scalar parameter"),
])
def test_bad_per_axis_ranges_are_refused(param, rng, match):
    with pytest.raises(ValueError, match=match):
        ParamSpec.from_dict({"param": param, "dist": "uniform", "range": rng, "op": "add"})


def test_per_axis_log_uniform_and_quantum_are_checked_per_pair():
    with pytest.raises(ValueError, match="log_uniform"):
        ParamSpec.from_dict({"param": "body_com", "dist": "log_uniform", "range": [[1, 2], [0, 1], [1, 2]],
                             "op": "scale"})
    with pytest.raises(ValueError, match="whole number"):
        ParamSpec.from_dict({"param": "body_com", "dist": "discrete_uniform", "quantum": 0.01, "op": "set",
                             "range": [[0, 0.1], [0, 0.015], [0, 0.1]]})
