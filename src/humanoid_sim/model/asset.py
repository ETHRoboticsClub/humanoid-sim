"""Robot asset: description + profile → USD, the model the physics and host kinematics load.

Isaac Lab spawns robots only from USD, so the canonical sim model is a USD built here:

1. :class:`~humanoid_sim.model.urdf.PreparedUrdf` (welds, sole boxes, collision meshes, absolute mesh paths);
2. Isaac Lab's URDF converter (``isaacsim-asset-isolated``, kitless): fixed joints merged, convex-hull
   colliders, self-collision on, no joint drives (gains are set at spawn from the plant);
3. ``physics:filteredPairs`` between bodies: the profile's contact excludes and every body pair the
   profile's self-collision sets do not pair up.

A variant with base joints (see :mod:`~humanoid_sim.model.urdf`) is a separate asset, for replays that
prescribe the base orientation. The output directory is keyed by the content digest, so a description or
profile change rebuilds. Runs in the Isaac Lab image (Linux); imports are lazy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import BASE_JOINTS, PreparedUrdf

ASSET_SCHEMA = "humanoid-sim-asset/v1"


def load_profile(robot: str, root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[3] / "profiles"
    profile = yaml.safe_load((root / f"{robot}.yaml").read_text())
    if profile.get("robot") != robot:
        raise ValueError(f"profile robot {profile.get('robot')!r} != {robot!r}")
    return profile


def asset_key(desc: Description, profile: dict, prepared: PreparedUrdf) -> str:
    h = hashlib.sha256()
    for part in (ASSET_SCHEMA, desc.digest, json.dumps(profile["model"], sort_keys=True), prepared.digest()):
        h.update(part.encode())
    return h.hexdigest()[:16]


def asset_dir(desc: Description, profile: dict, out_root: Path, *, base_joints: bool = False) -> Path:
    """Directory of the asset for this description and profile (exists only once built)."""
    prepared = PreparedUrdf(desc, profile).build(base_joints=base_joints)
    return Path(out_root) / desc.robot / asset_key(desc, profile, prepared)


def _body_prims(stage):
    from pxr import Usd, UsdPhysics

    return {p.GetName(): p for p in stage.Traverse(Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI)}


def add_contact_excludes(usd_path: Path, pairs: list[list[str]]) -> list[tuple[str, str]]:
    """Author ``physics:filteredPairs`` for each named body pair; returns the pairs as prim paths."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    bodies = _body_prims(stage)
    authored = []
    for a, b in pairs:
        missing = [n for n in (a, b) if n not in bodies]
        if missing:
            raise ValueError(f"contact exclude names unknown bodies {missing}; bodies: {sorted(bodies)}")
        api = UsdPhysics.FilteredPairsAPI.Apply(bodies[a])
        api.CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
        authored.append((str(bodies[a].GetPath()), str(bodies[b].GetPath())))
    stage.GetRootLayer().Save()
    return authored


def build_asset(desc: Description, profile: dict, out_root: Path, *, force: bool = False,
                base_joints: bool = False) -> Path:
    """Build (or reuse) the robot USD; returns its path. Writes ``asset.json`` with the provenance."""
    prepared = PreparedUrdf(desc, profile).build(base_joints=base_joints)
    key = asset_key(desc, profile, prepared)
    out = Path(out_root) / desc.robot / key
    meta_path = out / "asset.json"
    if meta_path.exists() and not force:
        return out / json.loads(meta_path.read_text())["usd"]
    urdf = prepared.write(out / "robot.urdf")

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    conv = UrdfConverter(UrdfConverterCfg(
        asset_path=str(urdf), usd_dir=str(out / "usd"), fix_base=False, merge_fixed_joints=True,
        collision_type="Convex Hull", self_collision=True, joint_drive=None, force_usd_conversion=True))
    usd = Path(conv.usd_path)
    excludes = add_contact_excludes(usd, profile["model"].get("contact_excludes", []))
    self_excludes = add_contact_excludes(usd, prepared.self_collision_excludes())
    meta = {
        "schema": ASSET_SCHEMA, "key": key, "usd": str(usd.relative_to(out)), **desc.identity(),
        "base_joints": [name for name, _ in BASE_JOINTS] if base_joints else [],
        "welded_joints": prepared.welded,
        "sole_boxes": [{"side": s.side, "body": s.body, "replaces_mesh": s.mesh,
                        "center": s.center.tolist(), "rotation": s.rotation.tolist(),
                        "half_extents": s.half_extents.tolist()} for s in prepared.soles],
        "merged_collision_bodies": sorted(prepared.merged),
        "contact_excludes": excludes,
        "self_collision_sets": prepared.body_sets() if "self_collision" in profile["model"] else None,
        "self_collision_excludes": self_excludes,
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return usd
