"""Read a delivered USD description without converting it through URDF.

USD joint frames, body transforms and sensor mounts are authoritative. Angles returned
to callers are radians; USD's angular limits are degrees. Imports of OpenUSD are lazy.
"""

from __future__ import annotations

import hashlib
import json
import math
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


def asset_key(desc, profile: dict, *, base_joints: bool = False) -> str:
    if base_joints:
        raise ValueError("prescribed base joints are not supported for a native USD articulation")
    record = {"schema": "humanoid-sim-native-usd/v1", "source": desc.usd_model.digest(),
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

    crank_limits = {}
    for spec in desc.ankle_model.payload["sides"].values():
        for role, drive in zip(("upper", "lower"), spec["motor_names"], strict=True):
            crank_limits[spec["usd_joints"][role]] = desc.drives[drive].position_limit_device
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
    stage.GetRootLayer().Save()
    meta = {"schema": ASSET_SCHEMA, "key": key, "usd": path.name, **desc.identity(),
            "source_usd_sha256": source.digest(), "ankle_representation": "native", "base_joints": [],
            "welded_joints": welded, "sole_boxes": soles, "merged_collision_bodies": [],
            "contact_excludes": [], "self_collision_sets": None, "self_collision_excludes": []}
    meta_path.write_text(json.dumps(meta, indent=1))
    return path
