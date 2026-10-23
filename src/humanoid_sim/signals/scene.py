"""Many envs of a training scene in one Rerun recording: the ground and any static boxes (or the room's textured
scan) once, each visual mesh of the robot once, then every recorded env's body poses per frame as instance poses of
those meshes.

A frame copies ``[envs, bodies, 7]`` poses from the GPU, so record at a low rate (every few policy steps) and for
debugging or showcases, not inside a training loop that has to stay free of host syncs. Entity paths:
``/world/ground``, ``/world/scene/{boxes,scan,floor}`` and ``/world/robots/<body>/visual_<k>``. Quaternions are xyzw,
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


DRAWS = ("scan", "boxes")


def log_scene(rr, scene, *, draw: str | None = None, detail: str = "full",
              path: str = "/world/scene") -> set[str]:
    """Log ``scene`` (:class:`~humanoid_sim.world.boxes.StaticBoxes`) as static entities under ``path``: its scan
    and the scan's floor (``draw="scan"``, the default when the scene has a scan, at ``detail``, see
    :data:`~humanoid_sim.world.scan.DETAILS`) or its boxes (``"boxes"``). Returns the names logged: ``{"boxes"}``,
    or ``{"scan"}`` with ``"floor"`` when the scan has a floor, which then stands for the ground."""
    from humanoid_sim.world.scan import scan_meshes

    draw = draw or ("scan" if scene.scan is not None else "boxes")
    if draw not in DRAWS:
        raise ValueError(f"draw {draw!r}: one of {DRAWS}")
    if draw == "boxes" or scene.scan is None:
        if draw == "scan":
            raise ValueError(f"{scene.file}: the scene has no scan to draw")
        log_boxes(rr, f"{path}/boxes", scene.corners(), scene.rgb)
        return {"boxes"}
    meshes = scan_meshes(scene.scan, scene.spawn_m, scene.floor_corners, detail=detail)
    for name, mesh in meshes.items():
        rr.log(f"{path}/{name}", scan_mesh3d(rr, mesh), static=True)
    return set(meshes)


def scan_mesh3d(rr, mesh):
    """A :class:`~humanoid_sim.world.scan.ScanMesh` as an unlit ``Mesh3D``: no normals, its texture or colours."""
    kw = {"vertex_positions": mesh.positions}
    if mesh.triangles is not None:
        kw["triangle_indices"] = mesh.triangles
    if mesh.texture is not None:
        kw |= {"vertex_texcoords": mesh.texcoords, "albedo_texture": mesh.texture}
    if mesh.colors is not None:
        kw["vertex_colors"] = mesh.colors
    return rr.Mesh3D(**kw)


def scan_blueprint(origin: str = "/world"):
    """A default layout for a scan: the 3D view against the neutral sky, other views as Rerun picks them."""
    import rerun.blueprint as rrb

    from humanoid_sim.world.scan import SKY_RGB

    sky = [round(255 * c) for c in SKY_RGB]
    return rrb.Blueprint(rrb.Spatial3DView(origin=origin, background=rrb.Background(color=sky)), auto_views=True)


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


def find_gantry(env):
    """The :class:`~humanoid_sim.world.supports.Gantry` of the plant an action term of ``env`` owns
    (``plant.gantry``), or None."""
    manager = getattr(env, "action_manager", None)
    for term in getattr(manager, "_terms", {}).values():
        gantry = getattr(getattr(term, "plant", None), "gantry", None)
        if gantry is not None:
            return gantry
    return None


class SceneRecorder:
    """Records ``envs`` of ``env``'s robot (an Isaac Lab articulation named ``robot``) to ``rrd`` or a viewer at
    ``connect``. Call :meth:`frame` with the plant time whenever a frame should be recorded.

    The static boxes on the stage are drawn. With ``scene`` (the :class:`~humanoid_sim.world.boxes.StaticBoxes` the
    scene was built from) its scan is drawn instead, with its floor in place of the ground and a default layout
    against the scan's sky (``draw="boxes"`` keeps the boxes), at ``detail``: ``full`` when recording to a file,
    ``live`` when streaming, unless given.

    Each recorded env also shows its gantry (``/world/gantry/<env>``, ``/sim/<env>/gantry``, see
    :mod:`humanoid_sim.signals.gantry_view`): the rope coloured by what it carries, its anchor, a status label and
    the tension as signals, the crane's travel, bridge and trolley when one is mounted, or a "no gantry" label.
    ``gantry`` is the plant's :class:`~humanoid_sim.world.supports.Gantry`, found through the env's action terms
    (``plant.gantry``) unless given. ``notes`` maps an env to a string that follows its status label (an operator's
    phase or outcome, set by the caller)."""

    def __init__(self, env, usd: Path, *, rrd: str | None = None, connect: str | None = None,
                 envs: list[int] | None = None, app_id: str = "humanoid-sim scene", scene=None,
                 draw: str | None = None, detail: str | None = None, gantry=None):
        import rerun as rr

        from humanoid_sim.model.kinematics import RobotKinematics
        from humanoid_sim.signals.robot import RobotMeshes

        self.rr = rr
        self.robot = env.scene["robot"]
        self.envs = list(range(self.robot.num_instances)) if envs is None else list(envs)
        self.gantry = gantry if gantry is not None else find_gantry(env)
        self.notes: dict[int, str] = {}
        rr.init(app_id)
        draw = draw or ("scan" if scene is not None and scene.scan is not None else "boxes")
        # Given to the sink, not sent after it: Rerun 0.39 opened a file with the layout sent later as empty.
        layout = scan_blueprint() if scene is not None and draw == "scan" else None
        if rrd:
            rr.save(rrd, default_blueprint=layout)
        elif connect:
            rr.connect_grpc(connect, default_blueprint=layout)
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
        drawn = set()
        if scene is not None:
            drawn = log_scene(rr, scene, draw=draw, detail=detail or ("full" if rrd else "live"))
        else:
            from humanoid_sim.world.boxes import stage_boxes

            corners, colors = stage_boxes()
            if len(corners):
                log_boxes(rr, "/world/scene/boxes", corners, colors)
        if "floor" not in drawn:
            for i, (verts, tris) in enumerate(ground_meshes(env)):
                rr.log(f"/world/ground/{i}", rr.Mesh3D(vertex_positions=verts, triangle_indices=tris,
                                                       vertex_normals=vertex_normals(verts, tris),
                                                       vertex_colors=ground_colors(verts)), static=True)
        self._ids = None
        self._crane_offsets: dict[int, np.ndarray] = {}
        if self.gantry is not None and self.gantry.trolley is not None:
            from humanoid_sim.signals.gantry_view import log_travel

            trolley = self.gantry.trolley
            # Where each env's crane stands from where its rig was given: the trolley's origin, less the rig's own.
            home = np.array([*trolley.rig.center_m, trolley.rig.anchor_z_m])
            self._crane_offsets = {e: trolley.origin_w[e].cpu().numpy() - home for e in self.envs}
            for e in self.envs:
                log_travel(rr, f"/world/gantry/{e}/travel", trolley.rig, self._crane_offsets[e])

    def frame(self, t_ns: int) -> None:
        import torch

        if self._ids is None:
            self._ids = torch.as_tensor(self.envs, device=self.robot.device)
        poses = self.robot.data.body_link_pose_w.torch[self._ids].double().cpu().numpy()  # [N, B, 7]
        rr = self.rr
        rr.set_time(TIMELINE, duration=t_ns / 1e9)
        for b, path in self.visuals:
            rr.log(path, rr.InstancePoses3D(translations=poses[:, b, :3], quaternions=poses[:, b, 3:7]))
        self._gantry(t_ns, poses[:, 0, :3])

    def _gantry(self, t_ns: int, base_w: np.ndarray) -> None:
        """The recorded envs' ropes as the last physics step left them (``base_w`` ``[N, 3]``: their bases)."""
        from humanoid_sim.signals import gantry_view as gv
        from humanoid_sim.world.supports import GRAVITY_M_S2, NONE

        rr, g = self.rr, self.gantry
        if g is None:
            for n, e in enumerate(self.envs):
                gv.log_no_gantry(rr, f"/world/gantry/{e}", base_w[n])
            return
        import torch

        weight = self.robot.data.body_mass.torch.sum(dim=1) * GRAVITY_M_S2
        columns = [g.mode.double(), g.tension_n.double(), g.rest_length_m(t_ns).double(), g.stiffness.double(),
                   weight.double()]
        columns = [torch.stack(columns, dim=1), g.top_w.double(), g.harness_w.double()]
        if g.trolley is not None:
            columns.append(g.trolley.position.double())
        state = torch.cat(columns, dim=1)[self._ids].cpu().numpy()  # one copy: [N, 5 + 3 + 3 (+ 3 with a crane)]
        for n, e in enumerate(self.envs):
            mode, tension, rest, stiffness, weight_n = state[n, :5]
            world = f"/world/gantry/{e}"
            if int(mode) == NONE:
                gv.log_no_gantry(rr, world, base_w[n])
                continue
            stretch = float(gv.full_weight_stretch_m(weight_n, stiffness)) if stiffness > 0 else None
            gv.log_env(rr, world, f"/sim/{e}/gantry", top=state[n, 5:8], hook=state[n, 8:11], rest_m=float(rest),
                       tension_n=float(tension), weight_n=float(weight_n), stretch_m=stretch,
                       note=self.notes.get(e, ""), base_w=base_w[n])
            if g.trolley is not None:
                gv.log_crane(rr, world, g.trolley.rig, state[n, 11:14], self._crane_offsets[e])

    def close(self) -> None:
        self.rr.disconnect()
