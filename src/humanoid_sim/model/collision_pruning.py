"""Cached asset preparation: omit convex colliders enclosed by equivalent colliders.

Visibility never determines collision eligibility. Only hulls strictly inside a
hull attached to the same rigid body, with matching contact settings and external
collision filters, are candidates. Delivered visuals and inertias stay authored.
"""

from collections import defaultdict

import numpy as np


def contained_hulls(builder, *, clearance_m: float = 1e-6) -> list[dict]:
    """Return redundant imported hulls; do not mutate the builder or its masses."""
    import newton
    import warp as wp
    from scipy.spatial import ConvexHull, QhullError

    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("collision-pruning clearance must be positive and finite")
    properties = ("shape_flags", "shape_collision_group", "shape_margin", "shape_gap",
                  "shape_material_ke", "shape_material_kd", "shape_material_kf", "shape_material_ka",
                  "shape_material_mu", "shape_material_restitution", "shape_material_mu_torsional",
                  "shape_material_mu_rolling", "shape_material_kh")
    filters = defaultdict(set)
    for a, b in builder.shape_collision_filter_pairs:
        if builder.shape_body[a] != builder.shape_body[b]:
            filters[a].add(b)
            filters[b].add(a)
    groups = defaultdict(list)
    for i, body in enumerate(builder.shape_body):
        if body < 0 or builder.shape_type[i] != newton.GeoType.CONVEX_MESH:
            continue
        if not builder.shape_flags[i] & newton.ShapeFlags.COLLIDE_SHAPES:
            continue
        points = np.asarray(builder.shape_source[i].vertices, dtype=np.float64)
        points = points * np.asarray(builder.shape_scale[i])
        frame = builder.shape_transform[i]
        rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(frame))).reshape(3, 3)
        points = points @ rotation.T + np.asarray(wp.transform_get_translation(frame))
        try:
            hull = ConvexHull(points)
        except QhullError:
            continue  # Degenerate surfaces are not closed volumes.
        settings = tuple(getattr(builder, name)[i] for name in properties)
        groups[(body, settings, frozenset(filters[i]))].append((i, points, hull))
    removed = []
    for shapes in groups.values():
        kept = []
        for i, points, hull in sorted(shapes, key=lambda item: (-item[2].volume, item[0])):
            for j, outer in kept:
                signed_distance = points @ outer.equations[:, :3].T + outer.equations[:, 3]
                margin = -float(signed_distance.max())
                if margin > clearance_m:
                    removed.append({"path": builder.shape_label[i], "enclosed_by": builder.shape_label[j],
                                    "clearance_m": margin})
                    break
            else:
                kept.append((i, hull))
    return removed


def prune_stage(stage) -> list[dict]:
    """Disable enclosed hulls in a prepared USD; preserve source prims for rendering."""
    import newton
    from pxr import UsdPhysics

    # Deleting collision contributions must never cause inferred mass changes.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            attrs = (mass.GetMassAttr(), mass.GetDiagonalInertiaAttr(),
                     mass.GetCenterOfMassAttr(), mass.GetPrincipalAxesAttr())
            if not (all(attr.HasAuthoredValueOpinion() for attr in attrs)
                    and mass.GetMassAttr().Get() > 0
                    and all(value > 0 for value in mass.GetDiagonalInertiaAttr().Get())):
                raise ValueError(f"collision pruning requires authored mass and inertia: {prim.GetPath()}")
    builder = newton.ModelBuilder()
    builder.add_usd(stage, floating=True, load_visual_shapes=False)
    removed = contained_hulls(builder)
    for item in removed:
        prim = stage.GetPrimAtPath(item["path"])
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(f"imported collider has no matching USD prim: {item['path']}")
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    return removed
