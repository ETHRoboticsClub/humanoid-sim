"""Model build contract: everything the prepared URDF changes, checked against the description."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import joints_below

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import PreparedUrdf, fit_sole_plate


@pytest.fixture(scope="module")
def built():
    desc = Description.load("ethrc_v1")
    return desc, PreparedUrdf(desc, _profile()).build()


def test_inventory(built):
    desc, _ = built
    cranks = {name for name, d in desc.drives.items() if d.crank is not None}
    assert cranks == {c for lk in desc.linkages for c in (lk.upper, lk.lower)}
    for lk in desc.linkages:
        assert desc.ankle_cranks(lk.side) == (lk.upper, lk.lower)


def test_unlisted_joints_welded_and_actuated_joints_limited(built):
    desc, p = built
    source = ET.parse(desc.urdf_path).getroot()
    unlisted = {j.get("name") for j in source.findall("joint") if j.get("type") != "fixed"} - set(desc.joints)
    assert set(p.welded) == unlisted  # URDF_6.urdf ships the ORCA hands fixed already
    movable = {j.get("name"): j for j in p.root.findall("joint") if j.get("type") != "fixed"}
    assert set(movable) == set(desc.joints)
    for name, j in movable.items():
        lim = j.find("limit")
        assert (float(lim.get("lower")), float(lim.get("upper"))) == pytest.approx(desc.calibration_travel(name))
    # An ankle hinge's travel is the hull of its cranks' stops (ankle_model.bounds), with no limit of its own.
    assert desc.calibration_travel(desc.linkages[0].joints[0]) == pytest.approx(desc.ankle_model.bounds("l")[0])


def test_sole_plate_fit_recovers_a_plate():
    # A 249.9 x 90 x 10 mm plate turned 0.3 rad about z: the box takes its bottom face's footprint, the configured
    # thickness, and sits on that face inside the plate.
    yaw = 0.3
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    half = np.array([0.12495, 0.045, 0.005])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    offset = np.array([0.05, -0.02, -0.04])
    center, rotation, got = fit_sole_plate(corners @ rot.T + offset, 0.005, 0.001)
    assert got * 2e3 == pytest.approx([249.9, 90.0, 5.0])
    assert abs(rotation[:, 0] @ rot[:, 0]) == pytest.approx(1.0)
    assert rotation[:, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert center == pytest.approx(offset + [0.0, 0.0, 0.0025 - half[2]])


def test_sole_boxes_on_the_profiles_feet(built):
    desc, p = built
    sole = p.model_cfg["sole"]
    child = {j.get("name"): j.find("child").get("link") for j in p.root.findall("joint")}
    assert {s.body for s in p.soles} == {child[j] for j in sole["joints"].values()}
    for s in p.soles:
        assert s.half_extents[2] * 2 == pytest.approx(sole["box_thickness_m"])
        assert np.allclose(s.rotation.T @ s.rotation, np.eye(3), atol=1e-9)
        assert s.rotation[2, 2] > 0.99  # box z (into the plate) points up


def test_digest_is_machine_independent(built):
    desc, p = built
    again = PreparedUrdf(desc, _profile()).build()
    assert p.digest() == again.digest()
    assert "package://" not in p.text()  # meshes resolved for the converter


def _profile(**model):
    profile = load_profile("ethrc_v1")
    desc = Description.load("ethrc_v1")
    profile["model"] = {"urdf": str(desc.urdf_path), "weld_unlisted_joints": True,
                        "collision_hulls": "per_body", "sole": profile["model"]["sole"],
                        "self_collision": {"sets": {"leg_l": "hip_flexion_l", "leg_r": "hip_flexion_r",
                                                    "arm_l": "shoulder_flexion_l", "arm_r": "shoulder_flexion_r"},
                                           "rest": "trunk", "pairs": "all"}, **model}
    return profile


def _mesh_triangles(p, links):
    """Triangles of the mesh colliders of ``links`` in a per-mesh build."""
    from humanoid_sim.model.urdf import read_stl

    count = 0
    for link in links:
        for col in p._links[link].findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is not None:
                count += len(read_stl(p._resolve(mesh.get("filename")))) // 3
    return count


def test_body_tree(built):
    desc, p = built
    tree = p.body_tree()
    root = p.root_link()
    assert len(tree) == len(desc.joints) + 1 and tree[root] == (None, None)
    child = {j.get("name"): j.find("child").get("link") for j in p.root.findall("joint")}
    for body, (joint, parent) in tree.items():
        if body != root:
            assert joint in desc.joints and child[joint] == body and parent in tree
    # Links behind welded joints (the ORCA hands) are merged into their parent body.
    assert not {child[j] for j in p.welded} & set(tree)


def test_one_collision_mesh_per_body(built):
    desc, p = built
    per_mesh = PreparedUrdf(desc, _profile(collision_hulls="per_mesh")).build()
    for body in p.body_tree():
        meshes = [c for c in p._links[body].findall("collision") if c.find("geometry/mesh") is not None]
        subtree = [link for link, _ in p._fixed_subtree(body, np.eye(4))]
        assert len(meshes) == (1 if body in p.merged else 0)
        assert all(not p._links[link].findall("collision") for link in subtree[1:])
        # Every triangle of the per-mesh colliders is in the merged mesh, and nothing else.
        assert len(p.merged.get(body, [])) // 3 == _mesh_triangles(per_mesh, subtree)
    # The sole boxes stay primitives on the feet, beside the description's own boxes, and the sole plates are not
    # merged back in.
    def boxes(links):
        return [c for link in links for c in link.findall("collision") if c.find("geometry/box") is not None]
    assert len(boxes(p._links.values())) == len(p.soles) + len(boxes(ET.parse(desc.urdf_path).getroot().iter("link")))
    soles = {s.body: s for s in p.soles}
    for body, sole in soles.items():
        z_min = p.merged[body][:, 2].min()
        plate_bottom = (sole.center - sole.rotation[:, 2] * sole.half_extents[2])[2]
        assert z_min > plate_bottom  # the merged foot hull ends above the box's ground face


def test_written_meshes_round_trip(built, tmp_path):
    from humanoid_sim.model.urdf import read_stl

    _, p = built
    text = p.text()
    urdf = p.write(tmp_path / "robot.urdf")
    assert p.text() == text  # writing does not change the prepared model
    written = urdf.read_text()
    for body, tris in p.merged.items():
        stl = tmp_path / "collision" / f"{body}.stl"
        assert f'filename="{stl}"' in written
        assert np.allclose(read_stl(stl), tris, atol=1e-6)


def test_self_collision_sets_and_excludes(built):
    desc, p = built
    sets = p.body_sets()
    count = {name: sum(s == name for s in sets.values()) for name in set(sets.values())}
    # A set holds the body below its root joint and every body below that; the rest is everything else.
    cfg = p.model_cfg["self_collision"]
    below = {joint: body for body, (joint, _) in p.body_tree().items()}
    expect = {name: 1 + len(joints_below(desc.robot, below[joint]) & set(desc.joints))
              for name, joint in cfg["sets"].items()}
    expect[cfg["rest"]] = len(sets) - sum(expect.values())
    assert count == expect
    assert p.self_collision_excludes() == []  # the profile collides every pair
    pairs = [["leg_l", "leg_r"], ["arm_l", "trunk"], ["arm_r", "trunk"]]
    subset = PreparedUrdf(desc, _profile(self_collision={**p.model_cfg["self_collision"], "pairs": pairs})).build()
    allowed = {frozenset(pair) for pair in pairs}
    excluded = {frozenset(pair) for pair in subset.self_collision_excludes()}
    bodies = [b for b in p.body_tree() if p._links[b].findall("collision")]
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            assert (frozenset((a, b)) in excluded) == (frozenset((sets[a], sets[b])) not in allowed)


def test_collision_config_is_checked(built):
    desc, p = built
    with pytest.raises(ValueError, match="collision_hulls"):
        PreparedUrdf(desc, _profile(collision_hulls="per_part")).build()
    sc = p.model_cfg["self_collision"]
    with pytest.raises(ValueError, match="unknown joints"):
        PreparedUrdf(desc, _profile(self_collision={**sc, "sets": {"leg_l": "no_such_joint"}})).build().body_sets()
    with pytest.raises(ValueError, match="expected two of"):
        PreparedUrdf(desc, _profile(self_collision={**sc, "pairs": [["leg_l", "tail"]]})).build() \
            .self_collision_excludes()


def test_base_joints_chain_world_to_root(built):
    from humanoid_sim.model.urdf import BASE_JOINTS, BASE_ROOT

    desc, p = built
    root = p.root_link()
    g = PreparedUrdf(desc, _profile()).build(base_joints=True)
    assert g.root_link() == BASE_ROOT
    chain = {j.get("name"): j for j in g.root.findall("joint") if j.get("name") in dict(BASE_JOINTS)}
    assert [chain[n].find("axis").get("xyz") for n, _ in BASE_JOINTS] == ["0 0 1", "0 1 0", "1 0 0"]
    assert chain["base_roll"].find("child").get("link") == root
    assert g.digest() != p.digest()
    tree = g.body_tree()
    assert tree[root] == ("base_roll", "base_pitch_link")
    assert g.self_collision_excludes() == []  # the chain's links have no colliders
