"""Prepare the delivered USD while preserving its closed ankle mechanisms."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import BASE_JOINTS, BASE_LINK_MASS_KG, BASE_ROOT


def source_digest(desc: Description) -> str:
    from pxr import Usd

    if desc.usd_path is None:
        raise ValueError(f"{desc.robot}: no delivered USD")
    stage = Usd.Stage.Open(str(desc.usd_path))
    digest = hashlib.sha256()
    for layer in sorted(stage.GetUsedLayers(), key=lambda layer: layer.identifier):
        if layer.anonymous:
            continue
        path = Path(layer.realPath)
        digest.update(path.relative_to(desc.package_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def asset_key(desc: Description, profile: dict, *, base_joints: bool = False) -> str:
    payload = {"schema": "humanoid-sim-native-usd/v4", "description": desc.digest,
               "source": source_digest(desc), "model": profile["model"], "base_joints": base_joints}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def crank_joints(desc: Description) -> dict[str, str]:
    return {drive: desc.ankle_model.payload["sides"][linkage.side]["usd_joints"][crank]
            for linkage in desc.linkages
            for crank, drive in (("upper", linkage.upper), ("lower", linkage.lower))}


def sole_boxes(path: Path, profile: dict) -> list[dict]:
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    joints = {prim.GetName(): UsdPhysics.Joint(prim) for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.Joint)}
    boxes = []
    for side, name in profile["model"].get("sole", {}).get("joints", {}).items():
        body_path = joints[name].GetBody1Rel().GetTargets()[0]
        bodies = {body_path}
        while True:
            children = {joint.GetBody1Rel().GetTargets()[0] for joint in joints.values()
                        if joint.GetPrim().IsA(UsdPhysics.FixedJoint)
                        and joint.GetBody0Rel().GetTargets()
                        and joint.GetBody0Rel().GetTargets()[0] in bodies
                        and joint.GetBody1Rel().GetTargets()}
            if children <= bodies:
                break
            bodies |= children
        inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path)).GetInverse()
        points = []
        for owner in bodies:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(owner)):
                if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                    continue
                transform = cache.GetLocalToWorldTransform(prim) * inverse
                points.extend(transform.Transform(Gf.Vec3d(point))
                              for point in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        if not points:
            raise ValueError(f"sole {side}: no collision mesh under {body_path}")
        points = np.asarray(points)
        lower, upper = points.min(axis=0), points.max(axis=0)
        boxes.append({"side": side, "body": body_path.name, "replaces_mesh": None,
                      "center": ((lower + upper) / 2).tolist(), "rotation": np.eye(3).tolist(),
                      "half_extents": ((upper - lower) / 2).tolist()})
    return boxes


def prepare(desc: Description, path: Path, *, base_joints: bool = False,
            collision_approximation: str = "convexHull") -> dict:
    from pxr import Usd, UsdPhysics

    if desc.usd_path is None:
        raise ValueError(f"{desc.robot}: no delivered USD")
    source = Usd.Stage.Open(str(desc.usd_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    source.Flatten().Export(str(path))
    stage = Usd.Stage.Open(str(path))
    cranks = crank_joints(desc)
    by_joint = {joint: desc.drives[drive] for drive, joint in cranks.items()}
    welded, closures = [], []
    for prim in list(stage.Traverse()):
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr(collision_approximation)
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        name = prim.GetName()
        if joint.GetExcludeFromArticulationAttr().Get():
            closures.append(str(prim.GetPath()))
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if name not in desc.joints and name not in by_joint:
            prim.SetTypeName("PhysicsFixedJoint")
            welded.append(name)
        else:
            limits = (by_joint[name].position_limit_device if name in by_joint
                      else desc.calibration_travel(name))
            revolute = UsdPhysics.RevoluteJoint(prim)
            revolute.CreateLowerLimitAttr(math.degrees(limits[0]))
            revolute.CreateUpperLimitAttr(math.degrees(limits[1]))
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            for prop in list(prim.GetProperties()):
                if prop.GetName().startswith("drive:angular:"):
                    prim.RemoveProperty(prop.GetName())
    if base_joints:
        _add_base_joints(stage)
    stage.GetRootLayer().Save()
    return {"source_usd": str(desc.usd_path.relative_to(desc.root)), "source_digest": source_digest(desc),
            "crank_joints": cranks, "loop_closures": closures, "welded_joints": welded,
            "base_joints": [name for name, _ in BASE_JOINTS] if base_joints else []}


def _add_base_joints(stage):
    from pxr import Gf, UsdGeom, UsdPhysics

    root = stage.GetDefaultPrim().GetPath()
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(bodies) != 1 or not bodies[0].HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("base joints require one rigid articulation root")
    body = bodies[0]
    body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    chain = [root.AppendChild(BASE_ROOT)] + [root.AppendChild(f"{name}_link")
                                             for name, _ in BASE_JOINTS[:-1]] + [body.GetPath()]
    for link in chain[:-1]:
        prim = UsdGeom.Xform.Define(stage, link).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(BASE_LINK_MASS_KG)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(chain[0]))
    for (name, axis), parent, child in zip(BASE_JOINTS, chain[:-1], chain[1:], strict=True):
        joint = UsdPhysics.RevoluteJoint.Define(stage, root.AppendChild("joints").AppendChild(name))
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateAxisAttr("XYZ"[axis.index(1)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateLowerLimitAttr(-180)
        joint.CreateUpperLimitAttr(180)
