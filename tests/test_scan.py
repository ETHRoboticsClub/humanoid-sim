"""A scan's look: drawn where and as Controls' MuJoCo ``hangar_scan`` environment draws it.

The oracle is MuJoCo itself: the scan mesh, its texture and the floor quad added to a MuJoCo world as Controls adds
them (``ethrc_sim/environments.py`` @ 44353e2b), compiled, and read back through MuJoCo's kinematics (vertex
positions) and its compiled texture coordinates and texture rows (which texel each face corner shows).
"""

import hashlib

import numpy as np
import pytest

from humanoid_sim.world.boxes import StaticBoxes
from humanoid_sim.world.scan import (
    FLOOR_LIFT_M,
    LIVE_CELL_M,
    ScanFiles,
    read_obj,
    rerun_texcoords,
    scan_meshes,
)

SPAWN = (-4.42, 3.48)
FLOOR = np.array([[-3.0, -2.0], [2.5, -2.2], [2.7, 1.9], [-3.1, 2.1]], dtype=np.float32)


def _png(path, h, w, seed):
    Image = pytest.importorskip("PIL.Image")
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
    return path


def _obj(path, positions, texcoords, faces):
    """An OBJ written as Controls' scan is: one normal, faces as v/vt/vn."""
    lines = ["vn 0 0 1"] + [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in positions]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in texcoords]
    lines += ["f " + " ".join(f"{a + 1}/{t + 1}/1" for a, t in zip(fv, ft, strict=True)) for fv, ft in faces]
    path.write_text("\n".join(lines) + "\n")
    return path


def _scene(tmp_path):
    """A boxes file with a floor outline, and a scan: a tetrahedron, every corner its own texture coordinate."""
    np.savez(tmp_path / "proxy.npz", center=np.array([[-4.0, 3.0, 0.5]]), half=np.array([[0.5, 0.5, 0.5]]),
             yaw=np.float32(0.03), floor_corners=FLOOR + np.array(SPAWN, dtype=np.float32))
    positions = np.array([[-3.1, 2.2, 0.3], [-2.0, 2.9, 0.1], [-3.6, 4.4, 0.2], [-3.0, 3.3, 2.4]])
    tris = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]
    rng = np.random.default_rng(0)
    texcoords = rng.uniform(0.02, 0.98, (12, 2))
    faces = [(f, (3 * k, 3 * k + 1, 3 * k + 2)) for k, f in enumerate(tris)]
    files = ScanFiles.of(_obj(tmp_path / "scan.obj", positions, texcoords, faces),
                         _png(tmp_path / "scan.png", 6, 8, 1), _png(tmp_path / "floor.png", 5, 7, 2))
    return StaticBoxes.load(tmp_path / "proxy.npz", SPAWN, scan=files)


def _controls_world(scene):
    """The scan and the floor in a MuJoCo world, built as Controls builds its hangar_scan scene."""
    mujoco = pytest.importorskip("mujoco")
    offset = [-SPAWN[0], -SPAWN[1], 0.0]
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    spec.add_texture(name="hangar_floor", type=mujoco.mjtTexture.mjTEXTURE_2D, file=scene.scan.floor_texture)
    floor_material = spec.add_material(name="hangar_floor", specular=0, shininess=0, rgba=[0.8, 0.8, 0.8, 1])
    floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hangar_floor"
    with np.load(scene.file) as data:
        corners = data["floor_corners"]
    floor_mesh = spec.add_mesh(name="hangar_floor", uservert=np.c_[corners, np.full(4, 0.001)].ravel().tolist(),
                               usertexcoord=[0, 0, 1, 0, 1, 1, 0, 1], userface=[0, 1, 2, 0, 2, 3])
    floor_mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    spec.worldbody.add_geom(name="hangar_floor", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hangar_floor",
                            material="hangar_floor", pos=offset, contype=0, conaffinity=0)
    spec.add_texture(name="hangar_visual", type=mujoco.mjtTexture.mjTEXTURE_2D, file=scene.scan.texture)
    material = spec.add_material(name="hangar_visual", specular=0, shininess=0, rgba=[0.7, 0.7, 0.7, 1])
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "hangar_visual"
    spec.add_mesh(name="hangar_visual", file=scene.scan.mesh)
    spec.worldbody.add_geom(name="hangar_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hangar_visual",
                            material="hangar_visual", pos=offset, contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    out = {}
    for name in ("hangar_visual", "hangar_floor"):
        g, mi = model.geom(name).id, model.mesh(name).id
        verts = model.mesh_vert[model.mesh_vertadr[mi]:model.mesh_vertadr[mi] + model.mesh_vertnum[mi]]
        faces = model.mesh_face[model.mesh_faceadr[mi]:model.mesh_faceadr[mi] + model.mesh_facenum[mi]]
        tc = model.mesh_texcoord[model.mesh_texcoordadr[mi]:model.mesh_texcoordadr[mi] + model.mesh_texcoordnum[mi]]
        ftc = model.mesh_facetexcoord[model.mesh_faceadr[mi]:model.mesh_faceadr[mi] + model.mesh_facenum[mi]]
        world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
        t = model.texture(name).id
        h, w = model.tex_height[t], model.tex_width[t]
        tex = model.tex_data[model.tex_adr[t]:model.tex_adr[t] + h * w * 3].reshape(h, w, 3)
        out[name] = {"corners": world[faces], "texcoords": tc[ftc], "texture": tex}
    return out


def _shown(texture, uv):
    """The texel each texture coordinate shows: row v H, column u W of a texture whose first row is at v = 0. That is
    how MuJoCo shows its compiled texture (OpenGL puts the first row of the data at v = 0) and how Rerun shows an
    image (its v = 0 is the image's top, its first row)."""
    h, w = texture.shape[:2]
    rows = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
    cols = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    return texture[rows, cols]


def test_the_scan_is_placed_and_textured_as_controls_draws_it(tmp_path):
    from humanoid_sim.world.scan import read_image

    scene = _scene(tmp_path)
    mj = _controls_world(scene)["hangar_visual"]
    mesh = scan_meshes(scene.scan, scene.spawn_m, scene.floor_corners, detail="full")["scan"]
    np.testing.assert_allclose(mesh.positions.reshape(-1, 3, 3), mj["corners"], atol=1e-5)
    np.testing.assert_array_equal(mj["texture"], read_image(scene.scan.texture))  # MuJoCo keeps the rows in order
    np.testing.assert_allclose(mesh.texcoords.reshape(-1, 3, 2), mj["texcoords"], atol=1e-6)
    shown_mj = _shown(mj["texture"], mj["texcoords"].reshape(-1, 2))
    shown_rr = _shown(mesh.texture, mesh.texcoords)
    np.testing.assert_array_equal(shown_rr, shown_mj)
    assert mesh.triangles is None and mesh.colors is None


def test_the_floor_is_drawn_as_controls_draws_it_on_the_boxes_floor(tmp_path):
    scene = _scene(tmp_path)
    mj = _controls_world(scene)["hangar_floor"]
    floor = scan_meshes(scene.scan, scene.spawn_m, scene.floor_corners, detail="full")["floor"]
    np.testing.assert_allclose(floor.positions[floor.triangles], mj["corners"], atol=1e-6)
    np.testing.assert_allclose(floor.texcoords[floor.triangles], mj["texcoords"], atol=1e-7)
    np.testing.assert_array_equal(floor.texture, mj["texture"])
    # The floor lies on the ground the boxes stand on, 1 mm up, over the outline moved like the boxes.
    box_bottom = (scene.center[:, 2] - scene.half[:, 2]).min()
    assert box_bottom == 0.0 and np.allclose(floor.positions[:, 2] - box_bottom, FLOOR_LIFT_M)
    np.testing.assert_allclose(floor.positions[:, :2], FLOOR, atol=1e-5)


def test_the_live_detail_merges_vertices_and_colours_them_from_the_texture(tmp_path):
    from humanoid_sim.world.scan import read_image

    # A 5 by 5 grid of vertices 0.1 m apart, two triangles per square, every corner its own texture coordinate.
    xy = np.stack(np.meshgrid(np.arange(5) * 0.1, np.arange(5) * 0.1, indexing="ij"), -1).reshape(-1, 2)
    positions = np.c_[xy + 0.01, np.zeros(25)]
    tris = []
    for i in range(4):
        for j in range(4):
            a, b, c, d = 5 * i + j, 5 * (i + 1) + j, 5 * (i + 1) + j + 1, 5 * i + j + 1
            tris += [(a, b, c), (a, c, d)]
    rng = np.random.default_rng(3)
    texcoords = rng.uniform(0.0, 1.0, (3 * len(tris), 2))
    faces = [(f, (3 * k, 3 * k + 1, 3 * k + 2)) for k, f in enumerate(tris)]
    files = ScanFiles.of(_obj(tmp_path / "grid.obj", positions, texcoords, faces), _png(tmp_path / "t.png", 16, 16, 4))
    live = scan_meshes(files, (0.0, 0.0), None, detail="live")["scan"]

    # Independently: each grid cell's members, their mean position, and the mean texel over their corners.
    texture = read_image(files.texture)
    cells = {}
    for k, (f, t) in enumerate(faces):
        for vi, ti in zip(f, t, strict=True):
            key = tuple(np.floor(positions[vi] / LIVE_CELL_M).astype(int))
            cell = cells.setdefault(key, {"verts": set(), "texels": []})
            cell["verts"].add(vi)
            u, v = texcoords[ti]
            cell["texels"].append(texture[min(int((1 - v) * 16), 15), min(int(u * 16), 15)])
    means = np.array([positions[sorted(c["verts"])].mean(0) for c in cells.values()])
    colors = np.array([np.round(np.mean(c["texels"], axis=0)) for c in cells.values()], dtype=np.uint8)
    assert len(live.positions) == len(means) < len(positions)
    for p, color in zip(live.positions, live.colors, strict=True):
        k = np.linalg.norm(means - p, axis=1).argmin()
        assert np.linalg.norm(means[k] - p) < 1e-5
        np.testing.assert_array_equal(color, colors[k])
    assert live.texture is None and live.triangles.max() < len(live.positions)


def test_the_descriptor_names_the_scan_files_and_their_digests(tmp_path):
    scene = _scene(tmp_path)
    scan = scene.describe()["scan"]
    for key in ("mesh", "texture", "floor_texture"):
        f = scan[key]["file"]
        assert scan[key]["sha256"] == hashlib.sha256(open(f, "rb").read()).hexdigest()


def test_scan_files_that_cannot_be_drawn_are_refused(tmp_path):
    quads = tmp_path / "quads.obj"
    quads.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\n")
    with pytest.raises(ValueError, match="triangles"):
        read_obj(quads)
    bare = tmp_path / "bare.obj"
    bare.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nf 1//1 2//1 3//1\n")
    with pytest.raises(ValueError, match="texture coordinate"):
        read_obj(bare)
    pointer = tmp_path / "pointer.png"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 8\n")
    with pytest.raises(ValueError, match="git-lfs"):
        ScanFiles.of(quads, pointer)


def test_rerun_texture_coordinates_count_rows_from_the_top():
    np.testing.assert_allclose(rerun_texcoords(np.array([[0.25, 0.0], [0.5, 0.9]])), [[0.25, 1.0], [0.5, 0.1]])
