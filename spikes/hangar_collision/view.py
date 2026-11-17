"""Spike (issue #57): a Rerun recording of a room's scan, its convex pieces and its boxes, to look at them together.

Entities: ``/scan`` (the scan's triangles, grey, shaded by height), ``/pieces/high`` (blue) and ``/pieces/low`` (red:
the pieces whose top is below ``--low-m``, where a foot catches), ``/boxes`` (green), ``/spawn/*`` (points). World
frame: the scan file's frame moved by ``-spawn``, as the sim places it.

    uv run --with scipy --with rerun-sdk python spikes/hangar_collision/view.py OBJ BOXES.npz PIECES.npz \
        --spawn -4.42 3.48 --rrd hangar_collision.rrd
"""

from __future__ import annotations

import argparse

import numpy as np

from humanoid_sim.world.boxes import StaticBoxes, box_mesh
from humanoid_sim.world.hulls import StaticHulls, hull_triangles
from humanoid_sim.world.scan import read_obj


def pieces_mesh(pieces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    verts, tris, base = [], [], 0
    for p in pieces:
        t = hull_triangles(p)
        verts.append(p)
        tris.append(t + base)
        base += len(p)
    return np.concatenate(verts), np.concatenate(tris)


def main() -> None:
    import rerun as rr

    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("boxes")
    ap.add_argument("pieces")
    ap.add_argument("--spawn", nargs=2, type=float, default=(0.0, 0.0))
    ap.add_argument("--other-spawn", nargs=2, type=float, default=(-7.24, -2.85), help="another spawn point to mark")
    ap.add_argument("--low-m", type=float, default=0.15)
    ap.add_argument("--rrd", required=True)
    args = ap.parse_args()

    rr.init("hangar collision", spawn=False)
    rr.save(args.rrd)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    offset = np.array([args.spawn[0], args.spawn[1], 0.0])

    positions, _, fv, _ = read_obj(args.obj)
    keep = positions[fv][:, :, 2].max(axis=1) < 3.0  # below the roof beams, to see inside
    z = np.clip(positions[:, 2] / 3.0, 0, 1)[:, None]
    grey = (np.hstack([0.45 + 0.4 * z] * 3) * 255).astype(np.uint8)
    rr.log("/scan", rr.Mesh3D(vertex_positions=(positions - offset).astype(np.float32), triangle_indices=fv[keep],
                              vertex_colors=grey), static=True)

    hulls = StaticHulls.load(args.pieces, args.spawn)
    pieces = hulls.pieces()
    top = np.array([p[:, 2].max() for p in pieces])
    for name, sel, rgb in (("high", top >= args.low_m, (70, 130, 220)), ("low", top < args.low_m, (230, 40, 40))):
        chosen = [p for p, s in zip(pieces, sel, strict=True) if s]
        if chosen:
            v, t = pieces_mesh(chosen)
            rr.log(f"/pieces/{name}", rr.Mesh3D(vertex_positions=v.astype(np.float32), triangle_indices=t,
                                                vertex_colors=np.tile(np.array(rgb, np.uint8), (len(v), 1))),
                   static=True)
        print(f"{name}: {len(chosen)} pieces")

    boxes = StaticBoxes.load(args.boxes, args.spawn)
    v, t, n, _ = box_mesh(boxes.corners(), boxes.rgb)
    rr.log("/boxes", rr.Mesh3D(vertex_positions=v.astype(np.float32), triangle_indices=t, vertex_normals=n,
                               vertex_colors=np.tile(np.array([60, 200, 90], np.uint8), (len(v), 1))), static=True)
    other = np.array(args.other_spawn) - np.array(args.spawn)
    rr.log("/spawn", rr.Points3D([[0, 0, 0.02], [other[0], other[1], 0.02]], radii=0.12,
                                 colors=[(255, 200, 0), (255, 120, 0)], labels=["spawn", "Controls spawn (gantry)"]),
           static=True)
    print(f"wrote {args.rrd}")


if __name__ == "__main__":
    main()
