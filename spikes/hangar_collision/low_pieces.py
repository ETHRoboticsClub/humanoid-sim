"""Spike (issue #57): the low pieces of a room's collision, the ones a foot catches on.

Controls replaced its first convex hulls of the hangar with boxes because they were "lumpy blobs with thin slabs lying
on the floor that a foot would catch on". This lists, for pieces files and for the room's boxes, the shapes whose top
is below ``--low-m`` (a foot's catch height): how many, how many of them thin, how many stand on open floor (outside
every box's footprint, so the boxes had nothing there), their footprint area, and the nearest to the spawn point.

    python spikes/hangar_collision/low_pieces.py BOXES.npz PIECES.npz [PIECES.npz ...] --spawn -4.42 3.48
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from humanoid_sim.world.boxes import StaticBoxes
from humanoid_sim.world.hulls import StaticHulls, convex_polygon, polygon_distance, thickness


def area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def summary(name: str, shapes: list[np.ndarray], boxes: StaticBoxes, low_m: float, thin_m: float) -> dict:
    footprints = [convex_polygon(s[:, :2]) for s in shapes]
    top = np.array([s[:, 2].max() for s in shapes])
    low = np.flatnonzero(top < low_m)
    rows = []
    for i in low:
        centre = shapes[i][:, :2].mean(axis=0)
        rows.append({
            "piece": int(i), "top_m": round(float(top[i]), 3), "thickness_m": round(thickness(shapes[i]), 3),
            "area_m2": round(area(footprints[i]), 3),
            "open_floor": boxes.clearance_m(centre) > 0.0,
            "from_spawn_m": round(polygon_distance(footprints[i], np.zeros(2)), 2),
        })
    thin = [r for r in rows if r["thickness_m"] < thin_m]
    open_floor = [r for r in rows if r["open_floor"]]
    out = {
        "file": name, "shapes": len(shapes), f"top_below_{low_m}m": len(rows), f"of_which_thinner_than_{thin_m}m": len(thin),
        "of_which_on_open_floor": len(open_floor),
        "open_floor_area_m2": round(sum(r["area_m2"] for r in open_floor), 2),
        "nearest_to_spawn": sorted(rows, key=lambda r: r["from_spawn_m"])[:5],
        "largest_on_open_floor": sorted(open_floor, key=lambda r: -r["area_m2"])[:5],
    }
    print(json.dumps(out, indent=1), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("boxes")
    ap.add_argument("pieces", nargs="+")
    ap.add_argument("--spawn", nargs=2, type=float, default=(0.0, 0.0))
    ap.add_argument("--low-m", type=float, default=0.15)
    ap.add_argument("--thin-m", type=float, default=0.05)
    args = ap.parse_args()
    boxes = StaticBoxes.load(args.boxes, args.spawn)
    summary("boxes", list(boxes.corners()), boxes, args.low_m, args.thin_m)
    for path in args.pieces:
        summary(path, StaticHulls.load(path, args.spawn).pieces(), boxes, args.low_m, args.thin_m)


if __name__ == "__main__":
    main()
