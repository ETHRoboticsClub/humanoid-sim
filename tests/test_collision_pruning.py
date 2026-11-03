"""Enclosure is a collision-space property, independent of camera visibility."""
from itertools import product

import numpy as np
import pytest

from humanoid_sim.model.collision_pruning import contained_hulls


@pytest.fixture
def shapes():
    newton = pytest.importorskip("newton")
    import warp as wp
    from scipy.spatial import ConvexHull

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=2.0)
    vertices = np.asarray(list(product((-1., 1.), repeat=3)), dtype=np.float32)
    mesh = newton.Mesh(vertices, ConvexHull(vertices).simplices.flatten(), compute_inertia=False)

    def add(name, scale, *, centre=(0., 0., 0.), target=body, mu=.5):
        return builder.add_shape_convex_hull(target, label=name, mesh=mesh, scale=(scale,) * 3,
            xform=wp.transform(centre, wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), .37)),
            cfg=newton.ModelBuilder.ShapeConfig(density=0., mu=mu))

    return builder, add


def test_prunes_only_strictly_enclosed_equivalent_hulls(shapes):
    builder, add = shapes
    outer = add("shell", 2.)
    add("interior", .2, centre=(.1, .1, .1))
    add("protruding", .4, centre=(2., 0., 0.))
    add("coincident", 2.)
    add("different_material", .3, mu=.9)
    other_body = builder.add_body(mass=1.)
    add("moving_inside", .2, target=other_body)
    filtered = add("filtered_inside", .2)
    neighbor = add("neighbor", .2, centre=(5., 0., 0.), target=other_body)
    builder.add_shape_collision_filter_pair(filtered, neighbor)
    before = np.asarray(builder.body_inertia).copy()
    result = contained_hulls(builder)
    assert [item["path"] for item in result] == ["interior"]
    assert result[0]["enclosed_by"] == builder.shape_label[outer]
    assert result[0]["clearance_m"] > 0
    np.testing.assert_array_equal(builder.body_inertia, before)


def test_native_pruned_asset_retains_visuals_inertias_and_contact_exterior(tmp_path):
    newton = pytest.importorskip("newton")
    from pxr import Usd, UsdGeom, UsdPhysics

    from humanoid_sim.model.asset import build_asset, load_profile
    from humanoid_sim.model.description import Description

    desc = Description.load()
    profile = load_profile(desc.robot)
    original = build_asset(desc, profile, tmp_path)
    profile['model']['prune_contained_hulls'] = True
    optimized = build_asset(desc, profile, tmp_path)
    assert original != optimized
    # Repeated construction resolves the prepared asset from its model-dependent key.
    assert build_asset(desc, profile, tmp_path) == optimized
    stages = [Usd.Stage.Open(str(path)) for path in (original, optimized)]
    def authored(stage):
        meshes = {str(p.GetPath()): np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
                  for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
        masses = {str(p.GetPath()): (UsdPhysics.MassAPI(p).GetMassAttr().Get(),
                    tuple(UsdPhysics.MassAPI(p).GetCenterOfMassAttr().Get()),
                    tuple(UsdPhysics.MassAPI(p).GetDiagonalInertiaAttr().Get()))
                  for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        return meshes, masses
    a, b = map(authored, stages)
    assert a[1] == b[1]
    assert a[0].keys() == b[0].keys()
    for name in a[0]:
        np.testing.assert_array_equal(a[0][name], b[0][name])
    builders = []
    for path in (original, optimized):
        builder = newton.ModelBuilder()
        builder.add_usd(str(path), floating=True, load_visual_shapes=False)
        builders.append(builder)
    expected = {item['path'] for item in contained_hulls(builders[0])}
    active = [{label for label, flags in zip(builder.shape_label, builder.shape_flags, strict=True)
               if flags & newton.ShapeFlags.COLLIDE_SHAPES} for builder in builders]
    assert active[0] - active[1] == expected
    assert active[1] <= active[0]
    for attr in ('body_mass', 'body_com', 'body_inertia', 'joint_type', 'joint_X_p', 'joint_X_c'):
        np.testing.assert_allclose(np.asarray(getattr(builders[0], attr)), np.asarray(getattr(builders[1], attr)))


def test_pruning_rejects_inferred_inertia():
    pytest.importorskip("newton")
    from pxr import Usd, UsdGeom, UsdPhysics

    from humanoid_sim.model.collision_pruning import prune_stage

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(1.)
    with pytest.raises(ValueError, match="authored mass and inertia"):
        prune_stage(stage)
