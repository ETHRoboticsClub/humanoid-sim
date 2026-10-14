"""Prepared URDF: the description's URDF plus this repo's modeling choices, ready for USD conversion.

Applied, in order:
1. weld every movable joint that has no actuators.yaml entry (the ORCA hands; Controls ``--fixed-orca-hands``);
2. joint ranges: the description's calibration travel (Controls ``ethrc_ssot/joint_travel.py`` rule);
3. sole boxes: on the body carrying each ``foot_roll`` joint, fit a thin box to the face of its lowest
   collision mesh (including meshes of fixed-joint children), add the box, drop that mesh's collision.
   Algorithm: Controls ``build_mjcf.py`` ``sole_plate`` @ 447b4acb, unchanged;
4. one hull per body (profile ``collision_hulls: per_body``): every remaining collision mesh of a
   body and of its fixed-joint children becomes one mesh in the body frame, so the converter makes one
   convex hull per body. Against the ground plane the hull of the union reaches exactly as far as the
   separate hulls did; between bodies it fills the gaps between parts. Primitive colliders stay;
5. resolve ``package://`` mesh URIs to absolute paths.

A *body* is a link that no fixed joint merges into a parent: the root and the child of every movable
joint. Self-collision sets (profile ``self_collision``) group bodies by the joint they hang from.

Nothing here invents geometry; every number comes from the meshes, the description or the profile.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.model.description import Description


@dataclass
class SoleBox:
    side: str
    body: str
    mesh: str  # file name of the plate mesh it replaces
    center: np.ndarray  # in the body frame (m)
    rotation: np.ndarray  # 3x3, box axes in the body frame (z up into the plate)
    half_extents: np.ndarray  # (m)


COLLISION_HULLS = ("per_mesh", "per_body")
MERGED_DIR = "collision"  # merged meshes, written next to the prepared URDF


def read_stl(path: Path) -> np.ndarray:
    """Vertices of a binary or ASCII STL as ``[N, 3]`` (duplicates kept; only extremes are used)."""
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:1024]:
        pts = [list(map(float, line.split()[1:4])) for line in data.decode(errors="ignore").splitlines()
               if line.strip().startswith("vertex")]
        return np.asarray(pts, dtype=np.float64)
    n = int.from_bytes(data[80:84], "little")
    tri = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                                ("a", "<u2")]))
    return tri["v"].reshape(-1, 3).astype(np.float64)


def write_stl(path: Path, triangles: np.ndarray) -> None:
    """Binary STL of a triangle soup ``[3T, 3]`` (consecutive vertex triplets, as :func:`read_stl` returns)."""
    tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, length, out=np.zeros_like(normal), where=length > 0)
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = normal, tri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 80 + len(tri).to_bytes(4, "little") + rec.tobytes())


def _origin(el) -> np.ndarray:
    """4x4 transform of a URDF ``<origin>`` (identity if absent)."""
    t = np.eye(4)
    if el is None:
        return t
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    r, p, y = (float(v) for v in el.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    t[:3, :3] = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                          [-sp, cp * sr, cp * cr]])
    t[:3, 3] = xyz
    return t


def _rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """URDF fixed-axis roll, pitch, yaw of a rotation matrix (R = Rz(y) Ry(p) Rx(r))."""
    p = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        return math.atan2(rot[2, 1], rot[2, 2]), p, math.atan2(rot[1, 0], rot[0, 0])
    return math.atan2(-rot[1, 2], rot[1, 1]), p, 0.0


def fit_sole_plate(points: np.ndarray, thickness: float, face_depth: float):
    """Box frame fitted to the lowest face of ``points`` (body frame). Controls ``sole_plate``, ported."""
    face = points[points[:, 2] < points[:, 2].min() + 0.0015]
    normal = np.array([0.0, 0.0, -1.0])
    for _ in range(5):
        normal = np.linalg.svd(face - face.mean(axis=0))[2][-1]
        if normal[2] > 0:
            normal = -normal
        depth = points @ normal
        refit = points[depth >= depth.max() - face_depth]
        if len(refit) == len(face) and np.allclose(refit, face):
            break
        face = refit
    if len(face) < 4:
        raise RuntimeError("sole plate face has too few coplanar vertices")
    centroid = face.mean(axis=0)
    in_plane = face - centroid
    a1 = np.linalg.svd(in_plane - np.outer(in_plane @ normal, normal))[2][0]
    a2 = np.cross(-normal, a1)
    e1, e2 = in_plane @ a1, in_plane @ a2
    half = np.array([0.5 * (e1.max() - e1.min()), 0.5 * (e2.max() - e2.min()), 0.5 * thickness])
    face_center = centroid + a1 * 0.5 * (e1.max() + e1.min()) + a2 * 0.5 * (e2.max() + e2.min())
    return face_center - normal * (0.5 * thickness), np.column_stack([a1, a2, -normal]), half


class PreparedUrdf:
    def __init__(self, desc: Description, profile: dict):
        self.desc = desc
        self.model_cfg = profile["model"]
        self.source = desc.package_dir / self.model_cfg["urdf"]
        self.tree = ET.parse(self.source)
        self.root = self.tree.getroot()
        self.welded: list[str] = []
        self.soles: list[SoleBox] = []
        self.merged: dict[str, np.ndarray] = {}  # body → triangle soup [3T, 3] in the body frame (m)
        self._links = {link.get("name"): link for link in self.root.findall("link")}
        self._joints = self.root.findall("joint")

    def _resolve(self, uri: str) -> Path:
        if uri.startswith("package://"):
            pkg, rel = uri[len("package://"):].split("/", 1)
            if pkg != self.desc.package_dir.name:
                raise ValueError(f"mesh from unknown package {pkg!r}")
            return self.desc.package_dir / rel
        return (self.source.parent / uri).resolve()

    def weld_unlisted(self) -> None:
        listed = set(self.desc.joints)
        for j in self._joints:
            if j.get("type") != "fixed" and j.get("name") not in listed:
                j.set("type", "fixed")
                for tag in ("axis", "limit", "dynamics", "mimic"):
                    for el in j.findall(tag):
                        j.remove(el)
                self.welded.append(j.get("name"))
        missing = listed - {j.get("name") for j in self._joints if j.get("type") != "fixed"}
        if missing:
            raise ValueError(f"actuated joints missing from the URDF: {sorted(missing)}")

    def set_joint_limits(self) -> None:
        """Every actuated joint becomes revolute with the description's calibration travel as its range
        (Controls' MJCF uses the same travel); effort and velocity are the joint's drive peak values."""
        for j in self._joints:
            name = j.get("name")
            if name not in self.desc.joints:
                continue
            lo, hi = self.desc.calibration_travel(name)
            jd = self.desc.joints[name]
            j.set("type", "revolute")
            for el in j.findall("limit"):
                j.remove(el)
            ET.SubElement(j, "limit", {"lower": f"{lo:.9g}", "upper": f"{hi:.9g}",
                                       "effort": f"{jd.effort_peak:.9g}", "velocity": f"{jd.velocity:.9g}"})

    def _merged_mesh(self, mesh) -> bool:
        return mesh.get("filename").startswith(f"{MERGED_DIR}/")

    def resolve_meshes(self) -> None:
        for mesh in self.root.iter("mesh"):
            if not self._merged_mesh(mesh):
                mesh.set("filename", str(self._resolve(mesh.get("filename"))))

    def _fixed_subtree(self, link: str, tf: np.ndarray):
        """(link, transform in the given link's frame) for the link and its fixed-joint descendants."""
        yield link, tf
        for j in self._joints:
            if j.get("type") == "fixed" and j.find("parent").get("link") == link:
                yield from self._fixed_subtree(j.find("child").get("link"), tf @ _origin(j.find("origin")))

    def add_sole_boxes(self) -> None:
        sole = self.model_cfg["sole"]
        for side, joint_name in sole["joints"].items():
            joint = next(j for j in self._joints if j.get("name") == joint_name)
            body = joint.find("child").get("link")
            candidates = []
            for link_name, tf in self._fixed_subtree(body, np.eye(4)):
                for col in self._links[link_name].findall("collision"):
                    pts = self._mesh_vertices(col, tf)
                    if pts is not None:
                        candidates.append((float(pts[:, 2].min()), link_name, col, col.find("geometry/mesh"), pts))
            if not candidates:
                raise RuntimeError(f"sole body {body} has no collision mesh")
            _, link_name, col, mesh, pts = min(candidates, key=lambda c: c[0])
            center, rot, half = fit_sole_plate(pts, float(sole["box_thickness_m"]), float(sole["face_depth_m"]))
            self._links[link_name].remove(col)
            box = ET.SubElement(self._links[body], "collision", {"name": f"sole_box_{side}"})
            r, p, y = _rpy(rot)
            ET.SubElement(box, "origin", {"xyz": " ".join(f"{v:.9g}" for v in center),
                                          "rpy": f"{r:.9g} {p:.9g} {y:.9g}"})
            geom = ET.SubElement(box, "geometry")
            ET.SubElement(geom, "box", {"size": " ".join(f"{2 * v:.9g}" for v in half)})
            self.soles.append(SoleBox(side, body, Path(mesh.get("filename")).name, center, rot, half))

    def _mesh_vertices(self, col, tf: np.ndarray) -> np.ndarray | None:
        """Vertices of a mesh collision element in the frame ``tf`` maps its link into (None: primitive)."""
        mesh = col.find("geometry/mesh")
        if mesh is None:
            return None
        scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
        v = read_stl(self._resolve(mesh.get("filename"))) * scale
        t = tf @ _origin(col.find("origin"))
        return v @ t[:3, :3].T + t[:3, 3]

    def body_tree(self) -> dict[str, tuple[str | None, str | None]]:
        """Bodies in URDF order, each with the movable joint above it and its parent body (None, None: root)."""
        parent_joint = {j.find("child").get("link"): j for j in self._joints}

        def body_of(link: str) -> str:
            while link in parent_joint and parent_joint[link].get("type") == "fixed":
                link = parent_joint[link].find("parent").get("link")
            return link

        out = {}
        for link in self._links:
            j = parent_joint.get(link)
            if j is None:
                out[link] = (None, None)
            elif j.get("type") != "fixed":
                out[link] = (j.get("name"), body_of(j.find("parent").get("link")))
        return out

    def merge_body_collisions(self) -> None:
        """One collision mesh per body: its own and its fixed-joint children's meshes, in the body frame."""
        for body in self.body_tree():
            parts = []
            for link_name, tf in self._fixed_subtree(body, np.eye(4)):
                link = self._links[link_name]
                for col in link.findall("collision"):
                    v = self._mesh_vertices(col, tf)
                    if v is not None:
                        parts.append(v)
                        link.remove(col)
            if not parts:
                continue
            self.merged[body] = np.concatenate(parts)
            col = ET.SubElement(self._links[body], "collision", {"name": f"{body}_collision"})
            ET.SubElement(col, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(ET.SubElement(col, "geometry"), "mesh", {"filename": f"{MERGED_DIR}/{body}.stl"})

    def body_sets(self) -> dict[str, str]:
        """Body → self-collision set: the nearest set whose root joint is at or above the body, else ``rest``."""
        cfg = self.model_cfg["self_collision"]
        roots = {joint: name for name, joint in cfg["sets"].items()}
        tree = self.body_tree()
        unknown = set(roots) - {joint for joint, _ in tree.values() if joint}
        if unknown:
            raise ValueError(f"self_collision sets name unknown joints {sorted(unknown)}")
        out = {}
        for body in tree:
            b, name = body, cfg["rest"]
            while b is not None:
                joint, parent = tree[b]
                if joint in roots:
                    name = roots[joint]
                    break
                b = parent
            out[body] = name
        return out

    def self_collision_excludes(self) -> list[tuple[str, str]]:
        """Body pairs (with colliders) whose sets the profile does not pair; ``pairs: all`` excludes none."""
        cfg = self.model_cfg.get("self_collision")
        if cfg is None or cfg["pairs"] == "all":
            return []
        sets = self.body_sets()
        names = set(cfg["sets"]) | {cfg["rest"]}
        allowed = set()
        for pair in cfg["pairs"]:
            unknown = set(pair) - names
            if len(pair) != 2 or unknown:
                raise ValueError(f"self_collision pair {pair}: expected two of {sorted(names)}")
            allowed.add(frozenset(pair))
        bodies = [b for b in self.body_tree()
                  if any(self._links[link].findall("collision") for link, _ in self._fixed_subtree(b, np.eye(4)))]
        return [(a, b) for i, a in enumerate(bodies) for b in bodies[i + 1:]
                if frozenset((sets[a], sets[b])) not in allowed]

    def build(self) -> PreparedUrdf:
        if self.model_cfg.get("weld_unlisted_joints", False):
            self.weld_unlisted()
        self.set_joint_limits()
        self.add_sole_boxes()
        hulls = self.model_cfg.get("collision_hulls", "per_mesh")
        if hulls not in COLLISION_HULLS:
            raise ValueError(f"collision_hulls {hulls!r}; expected one of {COLLISION_HULLS}")
        if hulls == "per_body":
            self.merge_body_collisions()
        # Digest before mesh URIs become absolute paths, so it is the same on every machine. Merged meshes
        # are a function of this text and the description's meshes.
        self._digest = hashlib.sha256(self.text().encode()).hexdigest()
        self.resolve_meshes()
        return self

    def text(self) -> str:
        return ET.tostring(self.root, encoding="unicode")

    def digest(self) -> str:
        """Content digest of the prepared URDF with package-relative mesh URIs (machine-independent)."""
        return self._digest

    def write(self, path: Path) -> Path:
        """Write the URDF and its merged meshes (absolute paths in the written file only)."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = [m for m in self.root.iter("mesh") if self._merged_mesh(m)]
        for mesh in merged:
            target = path.parent / mesh.get("filename")
            write_stl(target, self.merged[target.stem])
            mesh.set("filename", str(target))
        try:
            path.write_text(self.text())
        finally:
            for mesh in merged:
                mesh.set("filename", f"{MERGED_DIR}/{Path(mesh.get('filename')).name}")
        return path
