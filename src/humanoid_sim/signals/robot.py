"""The robot in Rerun: visual meshes logged once (static), then one transform per body per frame.

Backend-independent: frames are body poses ``[B, 7]`` (position, quaternion xyzw) in the model's body
order. Several robots (sim, real, …) share one set of meshes under different entity roots, e.g.
``/world/sim/<body>`` and ``/world/real/<body>``; a translucent color makes one a ghost of the other.
"""

from __future__ import annotations

import numpy as np

from humanoid_sim.signals.log import TIMELINE


class RobotMeshes:
    def __init__(self, model, *, world: int = 0):
        """Collect the visual meshes of one world of a Newton model (body index, local pose, mesh)."""
        import newton

        flags = model.shape_flags.numpy()
        body = model.shape_body.numpy()
        local = model.shape_transform.numpy()
        scale = model.shape_scale.numpy()
        sworld = model.shape_world.numpy()
        visible, collide = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
        bstart = int(np.flatnonzero(model.body_world.numpy() == world)[0])
        self.body_names = [lbl.rsplit("/", 1)[-1] for lbl in model.body_label][bstart:bstart + model.body_count
                                                                              // max(model.world_count, 1)]
        self.shapes = []
        for s in range(model.shape_count):
            src = model.shape_source[s]
            if sworld[s] != world or body[s] < 0 or not hasattr(src, "vertices"):
                continue
            if not (flags[s] & visible) or (flags[s] & collide):
                continue
            self.shapes.append((int(body[s]) - bstart, local[s], np.asarray(src.vertices) * scale[s],
                                np.asarray(src.indices).reshape(-1, 3)))

    def log_static(self, root: str, color: tuple[int, int, int, int]) -> None:
        import rerun as rr

        for i, (b, tf, verts, tris) in enumerate(self.shapes):
            path = f"{root}/{self.body_names[b]}/visual_{i}"
            rr.log(path, rr.Transform3D(translation=tf[:3], quaternion=tf[3:7]), static=True)
            rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=tris, albedo_factor=color),
                   static=True)

    def log_frames(self, root: str, t_ns: np.ndarray, body_q: np.ndarray) -> None:
        """All frames of one robot at once: ``body_q`` ``[T, B, 7]`` (columnar, cheap)."""
        import rerun as rr

        times = rr.TimeColumn(TIMELINE, timestamp=np.asarray(t_ns) / 1e9)
        for b, name in enumerate(self.body_names):
            rr.send_columns(f"{root}/{name}", indexes=[times],
                            columns=rr.Transform3D.columns(translation=body_q[:, b, :3],
                                                           quaternion=body_q[:, b, 3:7]))
