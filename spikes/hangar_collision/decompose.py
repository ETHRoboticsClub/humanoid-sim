"""Spike (issue #57): convex collision pieces from a scanned room's mesh, measured against the scan and the boxes.

Not imported by the package. Pipeline: read the scan OBJ, drop what lies entirely below ``--low-m`` (the floor's
reconstruction noise and objects a foot steps over: thickened into pieces they become slabs a foot catches on, the
reason Controls dropped its first convex hulls), what is out of the robot's reach and what moves (crop boxes),
voxelize so that every surface becomes a solid at least one voxel thick, extend what crosses ``--low-m`` straight
down to the floor (walls and furniture legs stand on it), split the solid into tiles and connected components,
decompose each with CoACD, drop any piece whose top is still below ``--low-m`` plus a voxel, and report piece count, vertices per piece, thinnest piece, how
much of the scan the pieces cover and how much of the pieces lies away from the scan. The same coverage and phantom
figures are given for the room's boxes file, for comparison.

    uv run --with coacd --with trimesh --with scipy --with scikit-image python spikes/hangar_collision/decompose.py \
        OBJ BOXES.npz --drop examples/hangar_crane_parts.json --out outputs/hangar_hulls.npz

Everything is in the scan file's frame (no spawn offset).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from humanoid_sim.world.boxes import box_corners
from humanoid_sim.world.scan import read_obj


def in_boxes(points: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Points inside any of the yawed boxes ``{"center", "half", "yaw"}``."""
    inside = np.zeros(len(points), dtype=bool)
    for b in boxes:
        d = points - np.asarray(b["center"])
        c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
        local = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
        inside |= (np.abs(local) <= np.asarray(b["half"])).all(axis=1)
    return inside


def cropped_scan(obj: Path, low_m: float, max_z: float, drop: list[dict]):
    """The scan's positions and the faces kept: those reaching above ``low_m``, centred below ``max_z`` and outside
    the crop boxes."""
    positions, _, fv, _ = read_obj(obj)
    tri = positions[fv]  # [F, 3, 3]
    centroid = tri.mean(axis=1)
    keep = (tri[:, :, 2].max(axis=1) > low_m) & (centroid[:, 2] < max_z) & ~in_boxes(centroid, drop)
    return positions, fv[keep]


def surface_voxels(positions, faces, pitch: float, chunk: int = 50_000):
    """Occupancy grid of the triangles: each triangle sampled on a barycentric lattice finer than half a voxel, in
    chunks so memory stays bounded. Returns the grid (one empty voxel of padding all round) and the scan-frame
    position of voxel (0, 0, 0)'s centre."""
    used = positions[np.unique(faces)]
    lo = used.min(axis=0) - 1.5 * pitch
    lo[2] = min(lo[2], -1.5 * pitch)  # the grid reaches below the floor, for the extrusion down to it
    shape = tuple(np.ceil((used.max(axis=0) + 1.5 * pitch - lo) / pitch).astype(int) + 1)
    occ = np.zeros(shape, dtype=bool)
    for start in range(0, len(faces), chunk):
        tri = positions[faces[start:start + chunk]]  # [C, 3, 3]
        edge = np.maximum(np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1), np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1))
        edge = np.maximum(edge, np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1))
        steps = np.ceil(edge / (0.5 * pitch)).astype(int).clip(1)
        for n in np.unique(steps):
            sel = tri[steps == n]
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = (i + j) <= n
            u, v = i[keep] / n, j[keep] / n  # [S]
            pts = (sel[:, None, 0] + u[None, :, None] * (sel[:, None, 1] - sel[:, None, 0])
                   + v[None, :, None] * (sel[:, None, 2] - sel[:, None, 0])).reshape(-1, 3)
            idx = np.floor((pts - lo) / pitch).astype(np.int64)
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ, lo + 0.5 * pitch


def solid_components(positions, faces, pitch: float, low_m: float):
    """Voxelize the triangles (surface voxels dilated by one voxel, holes filled), extend every column occupied at
    ``low_m`` down to the floor, and label connected components: labels on the voxel grid, voxel count per label
    (from 1), and the grid's index-to-scan-frame transform."""
    from scipy import ndimage

    occ, origin = surface_voxels(positions, faces, pitch)
    occ = ndimage.binary_fill_holes(ndimage.binary_dilation(occ, iterations=1))
    k_floor = int(np.ceil(-origin[2] / pitch))  # the first layer whose centre is at or above z = 0
    k_low = int(np.floor((low_m - origin[2]) / pitch))
    occ[:, :, :k_floor] = False  # nothing under the floor
    occ[:, :, k_floor:k_low] |= occ[:, :, k_low:k_low + 1]
    labels, _ = ndimage.label(occ)
    transform = np.eye(4)
    transform[:3, :3] *= pitch
    transform[:3, 3] = origin
    return labels, np.bincount(labels.ravel())[1:], transform


def component_mesh(mask: np.ndarray, transform: np.ndarray):
    """Closed surface of a voxel mask, in the scan frame."""
    import trimesh
    from skimage import measure

    padded = np.pad(mask, 1)
    verts, faces, _, _ = measure.marching_cubes(padded.astype(np.float32), 0.5)
    verts = verts - 1.0
    verts = trimesh.transform_points(verts, transform)
    return trimesh.Trimesh(verts, faces)


def decompose(mesh, threshold: float, max_hull_vertices: int):
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, decimate=True,
                            max_ch_vertex=max_hull_vertices, preprocess_mode="off")
    from humanoid_sim.world.hulls import MIN_THICKNESS_M

    return [p for p in (np.asarray(v) for v, _ in parts) if len(p) >= 4 and thickness(p) >= MIN_THICKNESS_M]


def thickness(points: np.ndarray) -> float:
    """Smallest extent of a point set along its principal axes."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ axes.T
    return float((proj.max(axis=0) - proj.min(axis=0)).min())


def hull_planes(points: np.ndarray):
    from scipy.spatial import ConvexHull

    return ConvexHull(points).equations  # [K, 4], outward normals: n.x + d <= 0 inside


def distance_outside(points: np.ndarray, planes: list[np.ndarray]) -> np.ndarray:
    """Lower bound of each point's distance to the union of the hulls (0 inside one): the largest plane violation,
    minimised over the hulls. Exact inside and along faces, an underestimate near edges and corners."""
    best = np.full(len(points), np.inf)
    for eq in planes:
        d = (points @ eq[:, :3].T + eq[:, 3]).max(axis=1)
        np.minimum(best, np.maximum(d, 0.0), out=best)
    return best


def surface_samples(planes_points: list[np.ndarray], per_m2: float, rng) -> np.ndarray:
    import trimesh

    out = []
    for pts in planes_points:
        hull = trimesh.convex.convex_hull(pts)
        n = max(8, int(hull.area * per_m2))
        out.append(trimesh.sample.sample_surface(hull, n, seed=int(rng.integers(1 << 31)))[0])
    return np.concatenate(out)


def report(name: str, pieces: list[np.ndarray], scan_points: np.ndarray, floor_z: float, rng) -> dict:
    from scipy.spatial import cKDTree

    planes = [hull_planes(p) for p in pieces]
    out = distance_outside(scan_points, planes)
    samples = surface_samples(pieces, 200.0, rng)
    samples = samples[samples[:, 2] > floor_z]  # the pieces' bottoms rest on the floor, which the scan cut out
    gap, _ = cKDTree(scan_points).query(samples)
    r = {
        "pieces": len(pieces),
        "vertices_max": int(max(len(p) for p in pieces)),
        "thinnest_m": round(min(thickness(p) for p in pieces), 3),
        "scan_within_3cm": round(float((out <= 0.03).mean()), 3),
        "scan_within_10cm": round(float((out <= 0.10).mean()), 3),
        "surface_beyond_10cm": round(float((gap > 0.10).mean()), 3),
        "surface_beyond_30cm": round(float((gap > 0.30).mean()), 3),
    }
    print(name, json.dumps(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj", type=Path)
    ap.add_argument("boxes", type=Path)
    ap.add_argument("--drop", type=Path, help="crop boxes JSON ({part: [box, ...]}); every part is dropped")
    ap.add_argument("--low-m", type=float, default=0.15, help="scan content entirely below this height is not "
                    "collided (floor noise, what a foot steps over); what crosses it is extended down to the floor")
    ap.add_argument("--max-z", type=float, default=2.5)
    ap.add_argument("--pitch", type=float, default=0.05)
    ap.add_argument("--min-voxels", type=int, default=8)
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--max-hull-vertices", type=int, default=64)
    ap.add_argument("--sample", type=int, default=200_000, help="scan vertices used for coverage")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stop-after", choices=("boxes", "voxels", "tiles"), default=None)
    ap.add_argument("--tile-m", type=float, default=3.0, help="side of the x, y tiles the solid is cut into (m)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    drop = []
    for part, boxes in (json.loads(args.drop.read_text()).items() if args.drop else ()):
        for b in boxes:
            if part == "drop":  # what the scan caught hanging in the gantry: cut down to the floor, its feet included
                top = b["center"][2] + b["half"][2]
                b = {**b, "center": [b["center"][0], b["center"][1], top / 2], "half": [b["half"][0], b["half"][1], top / 2]}
            drop.append(b)
    t = time.time()
    positions, faces = cropped_scan(args.obj, args.low_m, args.max_z, drop)
    used = np.unique(faces)
    print(f"scan: {len(faces)} triangles kept, {len(used)} vertices, bounds {positions[used].min(0).round(2)} "
          f"{positions[used].max(0).round(2)} ({time.time() - t:.1f} s)")
    scan_points = positions[rng.choice(used, min(args.sample, len(used)), replace=False)]

    with np.load(args.boxes) as d:
        n = len(d["center"])
        yaw = np.broadcast_to(d["yaw"], (n,))
        corners = box_corners(d["center"], d["half"], yaw)
    report("boxes", list(corners), scan_points, args.low_m, rng)
    if args.stop_after == "boxes":
        return

    t = time.time()
    labels, sizes, transform = solid_components(positions, faces, args.pitch, args.low_m)
    big = np.flatnonzero(sizes >= args.min_voxels) + 1
    print(f"voxels: grid {labels.shape}, {len(sizes)} components, {len(big)} with >= {args.min_voxels} voxels, "
          f"largest {np.sort(sizes)[::-1][:5].tolist()} ({time.time() - t:.1f} s)", flush=True)
    if args.stop_after == "voxels":
        return
    from scipy import ndimage

    # Cut the solid into square tiles in x, y (pieces end at tile edges) so that no single CoACD run sees the whole
    # room, then take each tile's connected components.
    occ = labels > 0
    tile = max(1, round(args.tile_m / args.pitch))
    meshes = []
    for x0 in range(0, occ.shape[0], tile):
        for y0 in range(0, occ.shape[1], tile):
            block = occ[x0:x0 + tile, y0:y0 + tile]
            if not block.any():
                continue
            sub, _ = ndimage.label(block)
            for k, sl in enumerate(ndimage.find_objects(sub), start=1):
                mask = sub[sl] == k
                if sl is None or mask.sum() < args.min_voxels:
                    continue
                start = np.array([x0 + sl[0].start, y0 + sl[1].start, sl[2].start], dtype=np.float64)
                local = transform.copy()
                local[:3, 3] += transform[:3, :3] @ start
                meshes.append(component_mesh(mask, local))
    print(f"tiles of {args.tile_m} m: {len(meshes)} solids, largest {max(len(m.faces) for m in meshes)} triangles",
          flush=True)
    if args.stop_after == "tiles":
        return

    results = {}
    for threshold in args.threshold:
        t = time.time()
        pieces = []
        for i, m in enumerate(meshes):
            pieces += [p for p in decompose(m, threshold, args.max_hull_vertices)
                       if p[:, 2].max() >= args.low_m + args.pitch]
            if i % 50 == 0:
                print(f"  {i}/{len(meshes)} solids, {len(pieces)} pieces ({time.time() - t:.0f} s)", flush=True)
        print(f"coacd threshold {threshold}: {time.time() - t:.1f} s", flush=True)
        results[threshold] = (pieces, report(f"hulls@{threshold}", pieces, scan_points, args.low_m, rng))
    if args.out:
        threshold = min(results, key=lambda k: abs(results[k][1]["pieces"] - 470))
        pieces = results[threshold][0]
        np.savez(args.out, points=np.concatenate(pieces), counts=np.array([len(p) for p in pieces]))
        print(f"wrote {args.out}: threshold {threshold}, {len(pieces)} pieces")


if __name__ == "__main__":
    main()
