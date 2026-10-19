"""Many envs of a training scene in one Rerun recording: the ground and any static boxes once, each visual mesh of
the robot once, then every recorded env's body poses per frame as instance poses of those meshes.

A frame copies ``[envs, bodies, 7]`` poses from the GPU, so record at a low rate (every few policy steps) and for
debugging or showcases, not inside a training loop that has to stay free of host syncs. Entity paths:
``/world/ground``, ``/world/scene`` (the static boxes) and ``/world/robots/<body>/visual_<k>``. Quaternions are xyzw,
as Isaac Lab's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from humanoid_sim.signals.log import TIMELINE

ROBOT_COLOR = (96, 125, 170, 255)
GROUND_LOW, GROUND_HIGH = np.array([104, 112, 100]), np.array([178, 184, 168])  # shaded by height


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted unit normals ``[V, 3]`` of a triangle mesh."""
    face = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, tris[:, k], face)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def ground_colors(verts: np.ndarray) -> np.ndarray:
    """Vertex colors ``[V, 3]`` (uint8) from dark in the hollows to light on the bumps, scaled between the 1st and
    99th height percentiles so a few deep or tall vertices do not wash out centimetre roughness."""
    z = verts[:, 2]
    lo, hi = np.percentile(z, [1, 99])
    u = np.clip((z - lo) / max(float(hi - lo), 1e-3), 0.0, 1.0)
    return (GROUND_LOW + (GROUND_HIGH - GROUND_LOW) * u[:, None]).astype(np.uint8)


def log_boxes(rr, path: str, corners: np.ndarray, colors: np.ndarray) -> None:
    """Static boxes (corners ``[N, 8, 3]``, colours ``[N, 3]`` in [0, 1]) as one static mesh at ``path``."""
    from humanoid_sim.world.boxes import box_mesh

    verts, tris, normals, vcolors = box_mesh(corners, colors)
    rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=tris, vertex_normals=normals,
                           vertex_colors=vcolors), static=True)


def _rotate(quat_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat_xyzw).apply(v)


def ground_meshes(env) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ground the scene collides with, as world-frame triangle meshes read from the USD stage (the generated
    terrain), or for a ground plane a square at z = 0: 40 m, or 2 m beyond the static boxes."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        from humanoid_sim.world.boxes import stage_boxes

        corners, _ = stage_boxes()
        s = max(20.0, float(np.abs(corners[..., :2]).max()) + 2.0) if len(corners) else 20.0
        verts = np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]])
        return [(verts, np.array([[0, 1, 2], [0, 2, 3]]))]
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    out = []
    for path in terrain.terrain_prim_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            index = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            if not (counts == 3).all():
                raise ValueError(f"{prim.GetPath()}: only triangle meshes are supported")
            xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            world = points @ xf[:3, :3] + xf[3, :3]  # USD matrices are row-major, points as row vectors
            out.append((world, index.reshape(-1, 3)))
    return out


class SceneRecorder:
    """Records ``envs`` of ``env``'s robot (an Isaac Lab articulation named ``robot``) to ``rrd`` or a viewer at
    ``connect``. Call :meth:`frame` with the plant time whenever a frame should be recorded."""

    def __init__(self, env, usd: Path, *, rrd: str | None = None, connect: str | None = None,
                 envs: list[int] | None = None, app_id: str = "humanoid-sim scene"):
        import rerun as rr

        from humanoid_sim.model.kinematics import RobotKinematics
        from humanoid_sim.signals.robot import RobotMeshes

        self.rr = rr
        self.robot = env.scene["robot"]
        self.envs = list(range(self.robot.num_instances)) if envs is None else list(envs)
        rr.init(app_id)
        if rrd:
            rr.save(rrd)
        elif connect:
            rr.connect_grpc(connect)
        names = list(self.robot.body_names)
        kin = RobotKinematics(usd, list(self.robot.joint_names))
        meshes = RobotMeshes(kin.model)
        self.visuals = []  # (articulation body index, entity path)
        for k, (b, tf, verts, tris) in enumerate(meshes.shapes):
            body = meshes.body_names[b]
            if body not in names:
                continue
            local = _rotate(tf[3:7], verts) + tf[:3]  # the shape's pose in its body, baked into the vertices
            path = f"/world/robots/{body}/visual_{k}"
            rr.log(path, rr.Mesh3D(vertex_positions=local, triangle_indices=tris,
                                   vertex_normals=vertex_normals(local, tris), albedo_factor=ROBOT_COLOR),
                   static=True)
            self.visuals.append((names.index(body), path))
        for i, (verts, tris) in enumerate(ground_meshes(env)):
            rr.log(f"/world/ground/{i}", rr.Mesh3D(vertex_positions=verts, triangle_indices=tris,
                                                   vertex_normals=vertex_normals(verts, tris),
                                                   vertex_colors=ground_colors(verts)), static=True)
        from humanoid_sim.world.boxes import stage_boxes

        corners, colors = stage_boxes()
        if len(corners):
            log_boxes(rr, "/world/scene", corners, colors)
        self._ids = None

    def frame(self, t_ns: int) -> None:
        import torch

        if self._ids is None:
            self._ids = torch.as_tensor(self.envs, device=self.robot.device)
        poses = self.robot.data.body_link_pose_w.torch[self._ids].double().cpu().numpy()  # [N, B, 7]
        rr = self.rr
        rr.set_time(TIMELINE, duration=t_ns / 1e9)
        for b, path in self.visuals:
            rr.log(path, rr.InstancePoses3D(translations=poses[:, b, :3], quaternions=poses[:, b, 3:7]))

    def close(self) -> None:
        self.rr.disconnect()
