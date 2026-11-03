"""Stairs with vertical risers as the ground: triangle-mesh terrain, and the same steps as static boxes.

The construction tests need Isaac Lab and run in the image on the CPU. The physics tests stand a robot on the stairs
on each solver (CUDA; Kamino needs it), and land it on them."""

import numpy as np
import pytest
import torch
from conftest import built_usd

from humanoid_sim.world.terrain_io import Ground

STEP_M = 0.15
STEP_WIDTH_M = 0.4
SIZE_M = 4.0
RINGS = 4  # Isaac Lab's count for this size and step width
PLATFORM_M = SIZE_M - 2 * RINGS * STEP_WIDTH_M  # the middle box
TOP_M = (RINGS + 1) * STEP_M  # the platform's height; ring k is (k + 1) steps high
ROBOTS = ["ethrc_v1", "g1_29dof_with_hand_rev_1_0"]
V1_MJWARP_MESH = pytest.mark.xfail(
    reason="MJWarp's soft contacts let V1's 5 mm sole boxes sink past a zero-thickness triangle on landing, where the "
           "contact normal flips and the robot falls through (docs/world.md)", strict=False)


def _stair_boxes():
    """The steps as boxes ``(centre, size)`` standing on z = 0, one on top of the next as Isaac Lab's stairs are."""
    boxes = [((0, 0, (k + 1) * STEP_M / 2), (SIZE_M - 2 * k * STEP_WIDTH_M,) * 2 + ((k + 1) * STEP_M,))
             for k in range(RINGS)]
    return boxes + [((0, 0, TOP_M / 2), (PLATFORM_M, PLATFORM_M, TOP_M))]


def _boxes_mesh(boxes):
    """Vertices and faces of axis-aligned boxes ``(centre, size)``, overlapping as they are."""
    vertices, faces = [], []
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) / 2
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    for centre, size in boxes:
        base = len(vertices) * 8
        vertices.append(np.asarray(centre) + corners * np.asarray(size))
        for a, b, c, d in quads:
            faces += [(base + a, base + b, base + c), (base + a, base + c, base + d)]
    return np.concatenate(vertices), np.array(faces)


def test_ground_reads_the_top_step_where_stair_boxes_overlap():
    """Each step is a box reaching down below the one inside it, so the surface at a point is the highest box over it
    (the ground height the runtime stands the robot on)."""
    ground = Ground(*_boxes_mesh(_stair_boxes()))
    r = SIZE_M / 2
    xy = [[0.0, 0.0], [r - 0.5 * STEP_WIDTH_M, 0.0], [r - 1.5 * STEP_WIDTH_M, 1.0], [-(r - 3.5 * STEP_WIDTH_M), 0.1]]
    assert ground.height(xy) == pytest.approx([TOP_M, STEP_M, 2 * STEP_M, 4 * STEP_M])


@pytest.fixture
def isaac_lab():
    pytest.importorskip("isaaclab")


def _stairs(**kw):
    from isaaclab.terrains.trimesh import MeshPyramidStairsTerrainCfg

    return {"stairs": MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(STEP_M, STEP_M),
                                                  step_width=STEP_WIDTH_M, platform_width=1.0, **kw)}


def _terrain(plane=False):
    from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg

    from humanoid_sim.world.terrain import mesh_terrain_cfg

    tiles = {"plane": MeshPlaneTerrainCfg(proportion=1.0)} if plane else _stairs()
    return mesh_terrain_cfg(tiles, size_m=SIZE_M, rows=1, cols=1, border_m=2.0, curriculum=False)


def test_mesh_terrain_cfg_builds_a_single_mesh_importer(isaac_lab):
    from humanoid_sim.world.terrain import mesh_terrain_cfg

    cfg = mesh_terrain_cfg(_stairs(), size_m=SIZE_M, rows=1, cols=1, border_m=2.0, curriculum=False, friction=0.8)
    assert cfg.terrain_generator.sub_terrains["stairs"].function.endswith("pyramid_stairs_terrain")
    assert cfg.physics_material.static_friction == 0.8 and cfg.physics_material.restitution == 0.0
    assert cfg.collision_group == -1
    assert not any(c.convert_to_heightfield for c in cfg.terrain_generator.sub_terrains.values())


def test_meshes_and_heightfields_go_to_their_own_builders(isaac_lab):
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg

    from humanoid_sim.world.terrain import heightfield_terrain_cfg, mesh_terrain_cfg

    bumps = {"bumps": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, 0.02), noise_step=0.005)}
    args = dict(size_m=SIZE_M, rows=1, cols=1, border_m=2.0, curriculum=False)
    with pytest.raises(ValueError, match="use mesh_terrain_cfg"):
        heightfield_terrain_cfg(_stairs(), **args)
    with pytest.raises(ValueError, match="use heightfield_terrain_cfg"):
        mesh_terrain_cfg(bumps, **args)
    mixed = mesh_terrain_cfg(_stairs() | bumps, **args)  # one collision mesh holding both
    assert set(mixed.terrain_generator.sub_terrains) == {"stairs", "bumps"}


class _Stairs:
    """A robot holding its home pose on a pyramid of stairs, three envs on the same stairs: on the platform, on the
    second ring from the edge, and with its soles over the platform's edge. ``world``: ``"mesh"`` (generated terrain)
    or a ``StaticBoxes`` of the steps on the flat ground."""

    def __init__(self, robot, solver, world):
        try:
            self._build(robot, solver, world)
        except BaseException:
            self.close()  # the simulation context outlives a failed build
            raise

    def _build(self, robot, solver, world):
        import isaaclab.sim as sim_utils
        from isaaclab.utils import instantiate

        from humanoid_sim.devices.demand import DriveDemand
        from humanoid_sim.model.asset import asset_meta
        from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg
        from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg

        usd = built_usd(robot)
        kw = {"terrain": _terrain(plane=world == "plane")} if world in ("mesh", "plane") else {"scene": world}
        self.sim = sim_utils.SimulationContext(robot_sim_cfg(robot, device="cuda:0", solver=solver))
        with robot_build(robot, solver=solver):
            self.scene = instantiate(robot_scene_cfg(3, usd, robot=robot, **kw))
            self.sim.reset()
        origin = self.scene.env_origins[0, :2].cpu().numpy()
        if world == "plane":  # a level mesh at z = 0: three robots on it
            self.ground = Ground(*_boxes_mesh([((0, 0, -0.5), (8, 8, 1))]))
            self.levels, self.xy = [0.0, 0.0, 0.0], [[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0]]
        else:
            self.ground = Ground(*_boxes_mesh(_stair_boxes()))
            ring = SIZE_M / 2 - 1.5 * STEP_WIDTH_M
            self.levels, self.xy = [TOP_M, 2 * STEP_M, TOP_M], [[0.0, 0.0], [ring, 0.0], [PLATFORM_M / 2 - 0.06, 0.0]]
        if world in ("mesh", "plane"):  # the generated terrain is the surface these tests expect
            probe = np.array([[0.0, 0.0], [1.7, 0.2], [1.3, -0.4], [-0.9, 1.5], [0.3, -1.1], [-1.5, -1.9]])
            assert Ground.from_stage().height(probe + origin) == pytest.approx(self.ground.height(probe), abs=1e-4)
        self.art = self.scene["robot"]
        self.plant = RobotPlant(self.art, TrainPlantCfg(robot=robot), imu=None, contacts=self.scene["contacts"])
        self.boxes = asset_meta(usd)["sole_boxes"]
        self.soles = [self.art.body_names.index(b["body"]) for b in self.boxes]
        default = self.art.data.default_joint_pos.torch
        q = default[:, [list(self.art.joint_names).index(n) for n in self.plant.core.map.joint_names]]
        drive_q, _, _ = self.plant.core.map.drive_state(q, torch.zeros_like(q))
        self.hold = DriveDemand.pd(drive_q, self.plant.core.drive_stiffness.expand(3, -1),
                                   self.plant.core.drive_damping.expand(3, -1))
        self.stand = self.art.data.default_root_pose.torch[0, 2].item()  # soles on the ground, upright

    def start(self, drop_m=0.0):
        """Env 0 on the platform, env 1 on ring 1 (three steps down), env 2 with its soles near the platform's edge;
        each ``drop_m`` above its rest height."""
        origins = self.scene.env_origins.cpu() * torch.tensor([1.0, 1.0, 0.0])
        level = torch.tensor(self.levels)
        pos = torch.cat([origins[:, :2] + torch.tensor(self.xy), (level + self.stand + 0.01 + drop_m)[:, None]], 1)
        root = torch.cat([pos, torch.tensor([[0.0, 0, 0, 1]] * 3)], 1).to(self.art.device)
        self.plant.reset(torch.arange(3, device=self.art.device), root_pose=root,
                         root_vel=torch.zeros(3, 6, device=self.art.device), demand=self.hold)

    def run(self, seconds):
        """Step; ``sunk`` is each env's deepest sole corner below the stairs' surface (m, 0 or less when none),
        ``trace`` keeps ``(t, [(base height, deepest corner)] per env)`` every 0.1 s."""
        from humanoid_sim.model.kinematics import _sole_corners

        sunk, self.trace = np.zeros(3), []
        steps = int(seconds * 1e9 / self.plant.period_ns)
        every = int(0.1e9 / self.plant.period_ns)
        for k in range(steps):
            self.plant.pre_physics()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.plant.period_ns / 1e9)
            if k % 20 == 0:
                poses = self.art.data.body_link_pose_w.torch.cpu().numpy()
                row = []
                for e in range(3):
                    corners = np.concatenate([_sole_corners(poses[e, b], box)
                                              for b, box in zip(self.soles, self.boxes, strict=True)])
                    xy = corners[:, :2] - self.scene.env_origins[e, :2].cpu().numpy()
                    depth = self.ground.height(xy) - corners[:, 2]  # NaN off the stairs
                    deepest = float(np.nanmax(depth)) if np.isfinite(depth).any() else float("inf")
                    sunk[e] = max(sunk[e], deepest)
                    row.append((round(float(poses[e, 0, 2]), 3), round(deepest, 3)))
                if k % every == 0:
                    self.trace.append((round(k * self.plant.period_ns / 1e9, 2), row))
        return sunk

    def close(self):
        import isaaclab.sim as sim_utils

        sim_utils.SimulationContext.clear_instance()


def _boxes_scene(tmp_path):
    from humanoid_sim.world.boxes import StaticBoxes

    boxes = _stair_boxes()
    np.savez(tmp_path / "stairs.npz", center=np.array([b[0] for b in boxes]), half=np.array([b[1] for b in boxes]) / 2,
             yaw=0.0)
    return StaticBoxes.load(tmp_path / "stairs.npz")


def _stairs_run(robot, solver, world, tmp_path, request):
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("the stairs tests run on CUDA (Kamino needs it)")
    run = _Stairs(robot, solver, _boxes_scene(tmp_path) if world == "boxes" else world)
    request.addfinalizer(run.close)
    return run


CASES = [
    pytest.param("ethrc_v1", "kamino", "mesh", id="v1-kamino-mesh"),
    pytest.param("g1_29dof_with_hand_rev_1_0", "kamino", "mesh", id="g1-kamino-mesh"),
    pytest.param("g1_29dof_with_hand_rev_1_0", "mjwarp", "mesh", id="g1-mjwarp-mesh"),
    pytest.param("ethrc_v1", "mjwarp", "mesh", id="v1-mjwarp-mesh", marks=V1_MJWARP_MESH),
    pytest.param("ethrc_v1", "kamino", "plane", id="v1-kamino-plane"),
    pytest.param("ethrc_v1", "mjwarp", "plane", id="v1-mjwarp-plane", marks=pytest.mark.xfail(
        reason="MuJoCo refuses to compile a flat, open mesh: mesh volume is too small", strict=False)),
    pytest.param("ethrc_v1", "kamino", "boxes", id="v1-kamino-boxes"),
    pytest.param("ethrc_v1", "mjwarp", "boxes", id="v1-mjwarp-boxes"),
]


@pytest.mark.physics
@pytest.mark.parametrize(("robot", "solver", "world"), CASES)
def test_a_robot_stands_on_stairs_and_does_not_sink_into_them(robot, solver, world, tmp_path, request):
    """Holding its home pose, the robot rests on the platform, on a lower ring and near an edge: no sole corner goes
    more than a few millimetres below the stairs; the V1 also stays at the height of the steps it stands on, its feet
    carrying the weight."""
    run = _stairs_run(robot, solver, world, tmp_path, request)
    run.start()
    sunk = run.run(1.5)
    print(robot, solver, world, "(t, [(base z, deepest sole corner below the surface)] per env):", *run.trace, sep="\n")
    base = run.art.data.root_link_pose_w.torch.cpu().numpy()[:, 2]
    assert (sunk[:2] < 0.01).all(), (sunk, run.trace)
    if robot == "ethrc_v1":  # the G1 does not hold its home pose for 1.5 s: it only has to stay on top
        assert sunk[2] < 0.01, (sunk, run.trace)
        assert base[0] - base[1] == pytest.approx(run.levels[0] - run.levels[1], abs=0.01)  # the steps' height
        assert base[2] > base[1] + 0.5 * (run.levels[2] - run.levels[1]) - 0.005  # near the edge: the upper side
        forces = run.plant.truth().contact_forces_w[..., 2].cpu()
        weight = float(run.art.data.body_mass.torch[0].sum()) * 9.81
        assert (forces[:, run.soles].sum(1) > 0.5 * weight).all(), forces[:, run.soles].sum(1) / weight


LANDINGS = [
    pytest.param("ethrc_v1", "kamino", "mesh", id="v1-kamino-mesh"),
    pytest.param("ethrc_v1", "mjwarp", "boxes", id="v1-mjwarp-boxes"),
    pytest.param("ethrc_v1", "mjwarp", "mesh", id="v1-mjwarp-mesh", marks=V1_MJWARP_MESH),
]


@pytest.mark.physics
@pytest.mark.parametrize(("robot", "solver", "world"), LANDINGS)
def test_a_robot_landing_on_stairs_does_not_go_through(robot, solver, world, tmp_path, request):
    """Dropped 0.3 m (2.4 m/s on landing) onto the platform and a ring, the robot lands on the stairs: MJWarp's soft
    contact lets the soles into a box a couple of centimetres for a moment, and a second later they rest on its
    surface. (Into a triangle they would go through: the MJWarp mesh case is expected to fail.)"""
    run = _stairs_run(robot, solver, world, tmp_path, request)
    run.start(drop_m=0.3)
    sunk = run.run(1.0)
    print(robot, solver, world, "landing:", *run.trace, sep="\n")
    assert (sunk[:2] < 0.04).all(), (sunk, run.trace)
    assert all(abs(depth) < 0.005 for _, depth in run.trace[-1][1][:2]), run.trace[-1]  # resting on the stairs


def test_a_tile_file_with_mesh_tiles_is_a_mesh_terrain(tmp_path, isaac_lab):
    import json

    from humanoid_sim.world.terrain_io import load_terrain

    spec = {"size_m": SIZE_M, "rows": 1, "cols": 1, "border_m": 2.0, "curriculum": False, "friction": 0.9,
            "tiles": {"stairs": {"type": "MeshPyramidStairsTerrainCfg", "proportion": 1.0,
                                 "step_height_range": [STEP_M, STEP_M], "step_width": STEP_WIDTH_M}}}
    path = tmp_path / "stairs.json"
    path.write_text(json.dumps(spec))
    terrain = load_terrain(path)
    assert terrain.describe()["type"] == "mesh" and terrain.describe()["tiles"] == ["stairs"]
    assert terrain.cfg.terrain_generator.sub_terrains["stairs"].step_height_range == (STEP_M, STEP_M)
    assert terrain.cfg.physics_material.static_friction == 0.9
    path.write_text(json.dumps({**spec, "horizontal_scale_m": 0.1}))
    with pytest.raises(ValueError, match="for heightfield tiles"):
        load_terrain(path)
    path.write_text(json.dumps({**spec, "tiles": {"x": {"type": "NotATerrainCfg"}}}))
    with pytest.raises(ValueError, match="not a sub-terrain"):
        load_terrain(path)
