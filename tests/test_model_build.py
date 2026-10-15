"""Model build contract: everything the prepared URDF changes, checked against the description."""

import numpy as np
import pytest

from humanoid_sim.model.asset import load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import PreparedUrdf


@pytest.fixture(scope="module")
def built():
    desc = Description.load("ethrc_v1")
    return desc, PreparedUrdf(desc, load_profile("ethrc_v1")).build()


def test_inventory(built):
    desc, _ = built
    assert len(desc.joints) == 22 and len(desc.drives) == 22
    assert sum(d.crank is not None for d in desc.drives.values()) == 4
    assert desc.ankle_cranks("l") == ("maxonHEJ50L_ANK1", "maxonHEJ50L_ANK2")


def test_unlisted_joints_welded_and_actuated_joints_limited(built):
    desc, p = built
    assert len(p.welded) == 34  # ORCA hands, as Controls --fixed-orca-hands
    movable = {j.get("name"): j for j in p.root.findall("joint") if j.get("type") != "fixed"}
    assert set(movable) == set(desc.joints)
    for name, j in movable.items():
        lim = j.find("limit")
        assert (float(lim.get("lower")), float(lim.get("upper"))) == pytest.approx(desc.calibration_travel(name))
    assert desc.calibration_travel("knee_l")[1] == 0.0  # recorded mechanical stop


def test_sole_boxes_match_controls_fit(built):
    _, p = built
    # Controls build_mjcf.py @ 447b4acb reports 249.9 x 90.0 mm, 5 mm thick, on both feet.
    for s in p.soles:
        assert s.half_extents * 2e3 == pytest.approx([249.9, 90.0, 5.0], abs=0.15)
        assert np.allclose(s.rotation.T @ s.rotation, np.eye(3), atol=1e-9)
        assert s.rotation[2, 2] > 0.99  # box z (into the plate) points up
    assert {s.body for s in p.soles} == {"bottom_universal_joint_l_1", "bottom_universal_joint_l_Mirror__1"}


def test_digest_is_machine_independent(built):
    desc, p = built
    again = PreparedUrdf(desc, load_profile("ethrc_v1")).build()
    assert p.digest() == again.digest()
    assert "package://" not in p.text()  # meshes resolved for the converter


def _profile(**model):
    profile = load_profile("ethrc_v1")
    profile["model"] = {**profile["model"], **model}
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
    _, p = built
    tree = p.body_tree()
    assert len(tree) == 23 and tree["base_link"] == (None, None)
    assert tree["Component171_1"] == ("waist_upperbody_yaw", "base_link")
    # A foot hangs from the ankle cross; the ORCA hand links are merged into the forearm body.
    assert tree["bottom_universal_joint_l_Mirror__1"] == ("foot_roll_l", "universal_joint_cross_l_Mirror__1")
    assert not any(b.startswith(("right_", "left_")) for b in tree)


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
    # The sole boxes stay primitives on the feet, and the sole plates are not merged back in.
    boxes = [c for link in p._links.values() for c in link.findall("collision") if c.find("geometry/box") is not None]
    assert len(boxes) == 2
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
    assert count == {"leg_l": 6, "leg_r": 6, "arm_l": 4, "arm_r": 4, "trunk": 3}
    assert p.self_collision_excludes() == []  # the profile collides every pair
    pairs = [["leg_l", "leg_r"], ["arm_l", "trunk"], ["arm_r", "trunk"]]
    subset = PreparedUrdf(desc, _profile(self_collision={**p.model_cfg["self_collision"], "pairs": pairs})).build()
    allowed = {frozenset(pair) for pair in pairs}
    excluded = {frozenset(pair) for pair in subset.self_collision_excludes()}
    bodies = list(p.body_tree())
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            assert (frozenset((a, b)) in excluded) == (frozenset((sets[a], sets[b])) not in allowed)
    assert frozenset(("bottom_universal_joint_l_1", "bottom_universal_joint_l_Mirror__1")) not in excluded
    assert frozenset(("connector_elbow_forearm_l_1", "connector_elbow_forearm_r_1")) in excluded


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
