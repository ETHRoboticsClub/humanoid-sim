"""Read a delivered USD description without converting it through URDF.

USD joint frames, body transforms and sensor mounts are authoritative. Angles returned
to callers are radians; USD's angular limits are degrees. Imports of OpenUSD are lazy.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


class UsdDescription:
    def __init__(self, path: Path):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.path = Path(path).resolve()
        self.stage = Usd.Stage.Open(str(self.path))
        if self.stage is None:
            raise ValueError(f"cannot open USD description {self.path}")
        if UsdGeom.GetStageMetersPerUnit(self.stage) != 1.0 or UsdGeom.GetStageUpAxis(self.stage) != "Z":
            raise ValueError(f"{self.path}: expected metres and Z up")
        self.bodies = {p.GetName(): p for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        self.joints = {p.GetName(): UsdPhysics.Joint(p) for p in self.stage.Traverse()
                       if p.IsA(UsdPhysics.Joint)}
        roots = [p.GetName() for p in self.bodies.values() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(roots) != 1:
            raise ValueError(f"{self.path}: expected one articulation root body, found {roots}")
        self.root_link = roots[0]

    def digest(self) -> str:
        """Hash every composed layer, including the mesh layer, with stable relative names."""
        h = hashlib.sha256()
        layers = [Path(layer.realPath) for layer in self.stage.GetUsedLayers() if layer.realPath]
        for path in sorted(layers):
            h.update(path.relative_to(self.path.parent).as_posix().encode())
            h.update(b"\0")
            h.update(path.read_bytes())
        return h.hexdigest()

    def mass_properties(self) -> dict[str, tuple[float, np.ndarray]]:
        """Delivered body masses and local centres of mass."""
        from pxr import UsdPhysics

        return {name: (float(UsdPhysics.MassAPI(body).GetMassAttr().Get()),
                       np.asarray(UsdPhysics.MassAPI(body).GetCenterOfMassAttr().Get()))
                for name, body in self.bodies.items()}

    def body_transform(self, name: str) -> np.ndarray:
        from pxr import Usd, UsdGeom

        return np.asarray(UsdGeom.Xformable(self.bodies[name]).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())).T.copy()

    def joint_bodies(self, name: str) -> tuple[str, str]:
        joint = self.joints[name]
        targets = [r.GetTargets() for r in (joint.GetBody0Rel(), joint.GetBody1Rel())]
        if any(len(t) != 1 for t in targets):
            raise ValueError(f"{name}: expected two body targets")
        return tuple(t[0].name for t in targets)

    @staticmethod
    def _frame(position, rotation) -> np.ndarray:
        from pxr import Gf

        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(rotation.GetReal(), Gf.Vec3d(*rotation.GetImaginary())))
        matrix.SetTranslateOnly(Gf.Vec3d(*position))
        return np.asarray(matrix).T.copy()

    def joint_frames(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        joint = self.joints[name]
        return (self._frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                self._frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()))

    def joint_axis(self, name: str) -> np.ndarray:
        from pxr import UsdPhysics

        prim = self.joints[name].GetPrim()
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            raise ValueError(f"{name} is not a revolute joint")
        return np.eye(3)["XYZ".index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())]

    def joint_limits(self, name: str) -> tuple[float, float]:
        from pxr import UsdPhysics

        joint = UsdPhysics.RevoluteJoint(self.joints[name].GetPrim())
        return tuple(math.radians(float(a.Get())) for a in (joint.GetLowerLimitAttr(), joint.GetUpperLimitAttr()))

    def imu_frame(self) -> str:
        sensors = [p for p in self.stage.Traverse()
                   if p.GetCustomDataByKey("fusion2usd:sensor") == "imu"]
        if len(sensors) != 1:
            raise ValueError(f"{self.path}: expected one IMU, found {len(sensors)}")
        return sensors[0].GetName()

    def imu_mount(self) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        return self.frame_mount(self.imu_frame())

    def frame_mount(self, name: str) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
        from pxr import Usd, UsdGeom, UsdPhysics

        from humanoid_sim.model.urdf import _rpy

        matches = [p for p in self.stage.Traverse() if p.GetName() == name]
        if len(matches) != 1:
            raise ValueError(f"{name}: expected one USD frame, found {len(matches)}")
        sensor = matches[0]
        body = sensor.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError(f"{sensor.GetPath()}: IMU has no rigid-body ancestor")
        world = np.asarray(UsdGeom.Xformable(sensor).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        local = np.linalg.solve(self.body_transform(body.GetName()), world)
        return body.GetName(), tuple(local[:3, 3]), _rpy(local[:3, :3])


    def sole_geometry(self, config: dict) -> list[dict]:
        """Sole contact boxes fitted to the delivered collision meshes, in their body frames."""
        from pxr import Usd, UsdGeom, UsdPhysics

        from humanoid_sim.model.urdf import fit_sole_plate

        result = []
        for side, name in config["joints"].items():
            body = self.joint_bodies(name)[1]
            body_prim = self.bodies[body]
            body_tf = self.body_transform(body)
            candidates = []
            for prim in Usd.PrimRange(body_prim):
                if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
                tf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                world = points @ tf[:3, :3].T + tf[:3, 3]
                candidates.append((world[:, 2].min(), str(prim.GetPath()), prim, world))
            if not candidates:
                raise ValueError(f"sole body {body} has no delivered collision meshes")
            _, _, mesh, points = min(candidates, key=lambda row: (row[0], row[1]))
            center, rotation, half = fit_sole_plate(points, config["box_thickness_m"], config["face_depth_m"])
            center = body_tf[:3, :3].T @ (center - body_tf[:3, 3])
            rotation = body_tf[:3, :3].T @ rotation
            result.append({"side": side, "body": body, "replaces_mesh": str(mesh.GetPath()),
                           "center": center.tolist(), "rotation": rotation.tolist(), "half_extents": half.tolist()})
        return result


def fixed_body_excludes(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """No internal contacts within a rigid assembly, including non-adjacent welded bodies."""
    parents = {name: name for pair in edges for name in pair}

    def root(name):
        while parents[name] != name:
            name = parents[name]
        return name

    for a, b in edges:
        parents[root(b)] = root(a)
    groups = {}
    for name in sorted(parents):
        groups.setdefault(root(name), []).append(name)
    return sorted(pair for names in groups.values() for pair in combinations(names, 2))


def asset_key(desc, profile: dict, *, base_joints: bool = False) -> str:
    record = {"schema": "humanoid-sim-native-usd/v5", "source": desc.usd_model.digest(),
              "description": desc.digest, "base_joints": base_joints,
              "actuators": hashlib.sha256(desc.actuators_path.read_bytes()).hexdigest(), "model": profile["model"]}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]


def build_asset(desc, profile: dict, out_root: Path, *, force=False, base_joints=False) -> Path:
    """Compose the delivered articulation and author simulator choices in a separate layer.

    Retain native crank bodies and all loop-closing joints. Weld only uncommanded
    revolute joints outside the ankle mechanisms; never edit the source delivery.
    """
    import shutil

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    from humanoid_sim.model.asset import ASSET_SCHEMA
    if profile["model"].get("collision_hulls") != "per_mesh":
        raise ValueError("native USD collision_hulls must be per_mesh")
    prune = profile["model"].get("prune_contained_hulls", False)
    if not isinstance(prune, bool):
        raise ValueError("prune_contained_hulls must be a boolean")
    key = asset_key(desc, profile, base_joints=base_joints)
    out = Path(out_root) / desc.robot / key
    meta_path = out / "asset.json"
    path = out / "robot.usda"
    if meta_path.is_file() and path.is_file() and not force:
        return path
    out.mkdir(parents=True, exist_ok=True)
    source = desc.usd_model
    for layer in source.stage.GetUsedLayers():
        if layer.realPath:
            origin = Path(layer.realPath)
            target = out / "source" / origin.relative_to(source.path.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    geometry = UsdGeom.Xform.Define(stage, "/Robot/Geometry")
    geometry.GetPrim().GetReferences().AddReference(f"source/{source.path.name}",
                                                  source.stage.GetDefaultPrim().GetPath())
    source_root = source.stage.GetDefaultPrim().GetPath()

    def mapped(prim):
        return prim.GetPath().ReplacePrefix(source_root, Sdf.Path("/Robot/Geometry"))

    for prim in source.stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            stage.GetPrimAtPath(mapped(prim)).RemoveAPI(UsdPhysics.DriveAPI, "angular")

    crank_limits = {}
    for spec in desc.ankle_model.payload["sides"].values():
        for role, drive in zip(("upper", "lower"), spec["motor_names"], strict=True):
            crank_limits[spec["usd_joints"][role]] = desc.drives[drive].shaft_limits
    welded = []
    for name, joint in source.joints.items():
        prim = joint.GetPrim()
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        target = mapped(prim)
        if name in desc.joints:
            bounds = desc.calibration_travel(name)
        elif name in crank_limits:
            bounds = crank_limits[name]
        elif profile["model"].get("weld_unlisted_joints", False):
            stage.OverridePrim(target).SetTypeName("PhysicsFixedJoint")
            welded.append(name)
            continue
        else:
            continue
        revolute = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(target))
        revolute.CreateLowerLimitAttr(math.degrees(bounds[0]))
        revolute.CreateUpperLimitAttr(math.degrees(bounds[1]))

    universal = _author_universal_rods(desc, profile, source, stage, mapped)

    excluded = []
    fixed_pairs = []
    if profile["model"].get("exclude_fixed_cluster_contacts", False):
        fixed = [source.joint_bodies(name) for name, joint in source.joints.items()
                 if name in welded or joint.GetPrim().IsA(UsdPhysics.FixedJoint)]
        fixed_pairs = fixed_body_excludes(fixed)
    contact_excludes = profile["model"].get("contact_excludes", [])
    for a, b in sorted({tuple(sorted(pair)) for pair in [*fixed_pairs, *contact_excludes]}):
        if a not in source.bodies or b not in source.bodies:
            raise ValueError(f"unknown native collision-exclusion bodies: {a}, {b}")
        pa, pb = mapped(source.bodies[a]), mapped(source.bodies[b])
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pa)).CreateFilteredPairsRel().AddTarget(pb)
        excluded.append((str(pa), str(pb)))

    # Keep each delivered mesh separate. Convex hulls are an explicit simulation
    # choice, avoiding an import-time decomposition with optional external tools.
    if profile["model"].get("collision_hulls") == "per_mesh":
        for prim in source.stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(mapped(prim))).CreateApproximationAttr(
                    "convexHull")

    sole_cfg = profile["model"].get("sole")
    soles = source.sole_geometry(sole_cfg) if sole_cfg else []
    for sole in soles:
        mesh = source.stage.GetPrimAtPath(sole["replaces_mesh"])
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(mapped(mesh))).CreateCollisionEnabledAttr(False)
        body_prim = source.bodies[sole["body"]]
        cube = UsdGeom.Cube.Define(stage, mapped(body_prim).AppendChild(f"sole_box_{sole['side']}"))
        cube.CreateSizeAttr(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*sole["center"]))
        cube.AddOrientOp().Set(Gf.Quatf(Gf.Matrix3d(np.asarray(sole["rotation"]).T).ExtractRotation().GetQuat()))
        cube.AddScaleOp().Set(Gf.Vec3f(*sole["half_extents"]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.CreatePurposeAttr("guide")
    pruned = []
    if prune:
        from humanoid_sim.model.collision_pruning import prune_stage

        pruned = prune_stage(stage)
    if base_joints:
        _add_base_joints(stage)
    from humanoid_sim.model.urdf import BASE_JOINTS

    closures = [str(mapped(joint.GetPrim())) for joint in source.joints.values()
                if joint.GetExcludeFromArticulationAttr().Get()]
    stage.GetRootLayer().Save()
    meta = {"schema": ASSET_SCHEMA, "key": key, "usd": path.name, **desc.identity(),
            "source_usd_sha256": source.digest(), "ankle_representation": "native",
            "base_joints": [name for name, _ in BASE_JOINTS] if base_joints else [],
            "loop_closures": closures, "crank_joints": crank_joints(desc), "universal_rod_joints": universal,
            "welded_joints": welded, "sole_boxes": soles, "merged_collision_bodies": [],
            "pruned_collision_hulls": pruned,
            "contact_excludes": contact_excludes, "self_collision_sets": None, "self_collision_excludes": excluded}
    meta_path.write_text(json.dumps(meta, indent=1))
    return path


def _author_universal_rods(desc, profile: dict, source, stage, mapped) -> list[str]:
    """With ``model.rod_joints: universal``, turn the open (articulated) ball joint of every ankle rod into a
    universal joint: rotation about the rod's own ball-to-ball axis locked, the two axes across it free. A rod
    between two balls spins freely about that axis on a tiny inertia; the spin carries no load. The joint frame
    turns by :func:`~humanoid_sim.model.native_state.universal_rod_frame` on both bodies, so the assembled pose
    stays the zero pose. Returns the joints changed."""
    kind = profile["model"].get("rod_joints", "spherical")
    if kind == "spherical":
        return []
    if kind != "universal":
        raise ValueError(f"rod_joints {kind!r}: spherical or universal")
    from pxr import Gf, UsdPhysics

    from humanoid_sim.model.native_state import NativeState, universal_rod_frame

    rods = NativeState(desc).rods
    if len(rods) != 2 * len(desc.linkages):
        raise ValueError(f"universal rods: found {len(rods)} loop-closing rods, expected two per ankle linkage")
    _check_universal_margin(desc)
    out = []
    for name, *_, rod_vector, _ in rods:
        joint = source.joints[name]
        if not joint.GetPrim().IsA(UsdPhysics.SphericalJoint) or joint.GetExcludeFromArticulationAttr().Get():
            raise ValueError(f"universal rods: {name} is not an articulated ball joint")
        if joint.GetPrim().GetAttribute("newton:armature").HasAuthoredValue():
            # Armature on the ball is the other fix for the same spin; keep one.
            raise ValueError(f"universal rods: {name} carries an authored newton:armature")
        turn = Gf.Matrix3d(*universal_rod_frame(rod_vector.numpy()).T.flatten()).ExtractRotation().GetQuat()
        prim = stage.OverridePrim(mapped(joint.GetPrim()))
        prim.SetTypeName("PhysicsJoint")
        target = UsdPhysics.Joint(prim)
        for get, create in ((joint.GetLocalRot0Attr, target.CreateLocalRot0Attr),
                            (joint.GetLocalRot1Attr, target.CreateLocalRot1Attr)):
            create(Gf.Quatf((Gf.Quatd(get().Get()) * turn).GetNormalized()))
        # A D6 joint: locked axes have low > high; free axes need a limit to be imported at all.
        for axis, (low, high) in {"transX": (1, -1), "transY": (1, -1), "transZ": (1, -1), "rotZ": (1, -1),
                                  "rotX": (-math.inf, math.inf), "rotY": (-math.inf, math.inf)}.items():
            limit = UsdPhysics.LimitAPI.Apply(prim, axis)
            limit.CreateLowAttr(float(low))
            limit.CreateHighAttr(float(high))
        out.append(name)
    return out


UNIVERSAL_MARGIN_RAD = math.radians(60.0)  # largest second angle allowed; the universal joint is singular at 90 deg


def _check_universal_margin(desc, samples: int = 4096) -> None:
    """Refuse universal rods whose second angle comes near its singularity anywhere in the ankle travel:
    candidates far past the travel, projected onto it."""
    import torch

    from humanoid_sim.model.native_state import NativeState

    state = NativeState(desc, rod_joints="universal")
    generator = torch.Generator().manual_seed(0)
    q = torch.zeros(samples, len(desc.joints), dtype=torch.float64)
    for linkage in desc.linkages:
        neutral = torch.as_tensor(desc.ankle_model.neutral(linkage.side), dtype=q.dtype)
        candidates = neutral + (torch.rand(samples, 2, generator=generator, dtype=q.dtype) - 0.5) * 3.0
        projected = desc.ankle_model.project(candidates.numpy(), linkage.side)
        q[:, [list(desc.joints).index(n) for n in linkage.joints]] = torch.as_tensor(projected, dtype=q.dtype)
    _, angles = state.coordinates(q)
    worst = max(float(a[:, 1].abs().max()) for a in angles.values())
    if worst > UNIVERSAL_MARGIN_RAD:
        raise ValueError(f"universal rods reach {math.degrees(worst):.1f} deg of their second axis "
                         f"(limit {math.degrees(UNIVERSAL_MARGIN_RAD):.0f}, singular at 90)")


def crank_joints(desc) -> dict[str, str]:
    return {drive: desc.ankle_model.payload["sides"][linkage.side]["usd_joints"][crank]
            for linkage in desc.linkages
            for crank, drive in (("upper", linkage.upper), ("lower", linkage.lower))}


def _add_base_joints(stage):
    from pxr import Gf, UsdGeom, UsdPhysics

    from humanoid_sim.model.urdf import BASE_JOINTS, BASE_LINK_MASS_KG, BASE_ROOT

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
