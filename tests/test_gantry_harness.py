"""Where the gantry ropes attach: the description's gantry_attachment_link frames, else the profile's harness."""

import math
from types import SimpleNamespace

import pytest
import torch

from humanoid_sim.world.harness import Harness, attachment_link_names, resolve_harness
from humanoid_sim.world.supports import Gantry

PROFILE_HARNESS = {"body": "torso", "shoulders": ["shoulder_l", "shoulder_r"]}


def _desc(tmp_path, *links):
    """A description (as far as the harness reads one) whose URDF has a base, a torso behind a revolute joint and
    ``links``: (name, parent, xyz) fixed frames."""
    rows = "".join(f'<link name="{n}"/><joint name="{n}_joint" type="fixed"><parent link="{p}"/><child link="{n}"/>'
                   f'<origin xyz="{x} {y} {z}" rpy="0 0 0"/></joint>' for n, p, (x, y, z) in links)
    urdf = tmp_path / "fixture.urdf"
    urdf.write_text('<robot name="fixture"><link name="pelvis"/><link name="torso"/>'
                    '<link name="shoulder_l"/><link name="shoulder_r"/>'
                    '<joint name="waist" type="revolute"><parent link="pelvis"/><child link="torso"/>'
                    '<origin xyz="0 0 0.2"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
                    f'</joint>{rows}</robot>')
    import xml.etree.ElementTree as ET

    names = {link.get("name") for link in ET.parse(urdf).getroot().findall("link")}
    return SimpleNamespace(usd_model=None, urdf_path=urdf, link_names=lambda: names)


# -- discovery ------------------------------------------------------------------------------------


def test_no_links_is_an_empty_list():
    assert attachment_link_names(["base_link", "torso", "gantry_attachment", "my_gantry_attachment_link"]) == []


def test_one_link():
    assert attachment_link_names(["torso", "gantry_attachment_link"]) == ["gantry_attachment_link"]
    assert attachment_link_names(["gantry_attachment_link_3"]) == ["gantry_attachment_link_3"]


def test_numbered_links_sort_numerically_whatever_the_input_order():
    names = [f"gantry_attachment_link_{n}" for n in (10, 2, 1)]
    assert attachment_link_names(names) == [f"gantry_attachment_link_{n}" for n in (1, 2, 10)]
    assert attachment_link_names(reversed(names)) == attachment_link_names(names)


def test_the_bare_name_is_the_first_rope_ahead_of_every_numbered_link():
    names = ["gantry_attachment_link_2", "gantry_attachment_link_0", "gantry_attachment_link"]
    assert attachment_link_names(names) == ["gantry_attachment_link", "gantry_attachment_link_0",
                                            "gantry_attachment_link_2"]


def test_two_names_with_the_same_number_are_refused():
    with pytest.raises(ValueError, match="number the same rope"):
        attachment_link_names(["gantry_attachment_link_1", "gantry_attachment_link_01"])


# -- resolution -----------------------------------------------------------------------------------


def test_without_links_the_profile_harness_applies_unchanged(tmp_path):
    harness = resolve_harness(_desc(tmp_path), PROFILE_HARNESS)
    assert harness == Harness("torso", shoulders=("shoulder_l", "shoulder_r"))
    assert harness.attachment() == {"attachment_bodies": ["shoulder_l", "shoulder_r"]}
    assert harness.describe() == PROFILE_HARNESS  # the runtime descriptor is what it was
    assert resolve_harness(_desc(tmp_path), None) is None


def test_one_link_is_a_point_in_the_frame_of_the_body_that_carries_it(tmp_path):
    desc = _desc(tmp_path, ("gantry_attachment_link", "torso", (0.0, 0.1, 0.3)))
    harness = resolve_harness(desc, PROFILE_HARNESS)
    assert harness.body == "torso" and harness.links == ("gantry_attachment_link",)
    assert harness.points_b == ((0.0, 0.1, 0.3),)
    assert harness.describe() == {"body": "torso", "attachment_links": ["gantry_attachment_link"]}


def test_the_links_win_over_the_profile_harness_and_serve_a_profile_without_one(tmp_path):
    desc = _desc(tmp_path, ("gantry_attachment_link_2", "torso", (0.0, -0.2, 0.3)),
                 ("gantry_attachment_link_1", "torso", (0.0, 0.2, 0.3)))
    for profile in (PROFILE_HARNESS, None):
        harness = resolve_harness(desc, profile)
        assert harness.links == ("gantry_attachment_link_1", "gantry_attachment_link_2")
        assert harness.points_b == ((0.0, 0.2, 0.3), (0.0, -0.2, 0.3))


def test_links_on_different_bodies_are_refused(tmp_path):
    desc = _desc(tmp_path, ("gantry_attachment_link_1", "torso", (0, 0, 0.3)),
                 ("gantry_attachment_link_2", "pelvis", (0, 0, 0.3)))
    with pytest.raises(ValueError, match="different bodies"):
        resolve_harness(desc, PROFILE_HARNESS)


# -- a hung robot ---------------------------------------------------------------------------------


class _Robot:
    """Body names and link poses ``[1, B, 7]`` (xyzw); keeps the forces as apply() hands them over."""

    body_names = ["pelvis", "torso", "shoulder_l", "shoulder_r"]
    num_instances, device = 1, "cpu"

    def __init__(self, poses):
        self.data = SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=torch.tensor([poses], dtype=torch.float32)),
            body_link_vel_w=SimpleNamespace(torch=torch.zeros(1, len(poses), 6)))
        self.forces = []  # (force [3], position [3], body ids), one per rope point slot
        robot = self

        class Composer:
            def add_forces_and_torques_index(self, *, forces, torques, positions, body_ids, env_ids, is_global):
                robot.forces.append((forces[0, 0].clone(), positions[0, 0].clone(), body_ids.tolist()))

        self.instantaneous_wrench_composer = Composer()

    def pulls(self):
        """(position, force) of every rope point that pulls."""
        return [(p, f) for f, p, _ in self.forces if float(f.norm()) > 0]


# The torso 1 m up, turned 90 deg about z (so a point's offset in its frame differs from the world's).
S = math.sqrt(0.5)
POSES = [[0, 0, 0.8, 0, 0, 0, 1], [0, 0, 1.0, 0, 0, S, S], [0, 0, 1.3, 0, 0, 0, 1], [0, 0, 1.3, 0, 0, 0, 1]]
LINKS_ONE = [("gantry_attachment_link", "torso", (0.1, 0.0, 0.3))]
LINKS_TWO = [("gantry_attachment_link_2", "torso", (0.0, 0.0, 0.3)),
             ("gantry_attachment_link_1", "torso", (0.0, 0.2, 0.3))]


@pytest.mark.parametrize("links, world", [
    (LINKS_ONE, [[0.0, 0.1, 1.3]]),  # the torso's x is the world's y
    (LINKS_TWO, [[-0.2, 0.0, 1.3], [0.0, 0.0, 1.3]]),  # _1 first, then _2
])
def test_a_tension_rope_pulls_at_each_link_with_the_load_split_equally(tmp_path, links, world):
    robot = _Robot(POSES)
    harness = resolve_harness(_desc(tmp_path, *links), PROFILE_HARNESS)
    gantry = Gantry(robot, harness=harness)
    gantry.hang_tension(torch.tensor([0]), 0, [(0.0, 100.0)], **harness.attachment())
    gantry.apply(0)
    n = len(links)
    assert gantry.body_id == robot.body_names.index("torso")
    pulls = robot.pulls()
    assert [p.tolist() for p, _ in pulls] == [pytest.approx(w, abs=1e-6) for w in world]
    assert all(f.tolist() == pytest.approx([0.0, 0.0, 100.0 / n], abs=1e-5) for _, f in pulls)


def test_a_cable_hangs_a_strap_from_each_link_and_splits_stiffness_equally(tmp_path):
    robot = _Robot(POSES)
    harness = resolve_harness(_desc(tmp_path, *LINKS_TWO), PROFILE_HARNESS)
    gantry = Gantry(robot, harness=harness)
    gantry.hang_cable(torch.tensor([0]), 0, overhead_m=1.0, length_breakpoints_s_m=[(0.0, 0.98)], stiffness_n_m=1e4,
                      damping_n_s_m=0.0, **harness.attachment())
    gantry.apply(0)
    pulls = robot.pulls()
    assert [p.tolist() for p, _ in pulls] == [pytest.approx(w, abs=1e-6) for w in ([-0.2, 0, 1.3], [0, 0, 1.3])]
    # The hook 1 m over the midpoint, the bar's ends over the links; 2 cm stretch on 1/2 of the stiffness each.
    assert gantry.anchor_w[0].tolist() == pytest.approx([-0.1, 0.0, 2.3], abs=1e-6)
    assert gantry.strap_bottom_w[0].tolist() == [pytest.approx([-0.2, 0, 1.3], abs=1e-6),
                                                 pytest.approx([0, 0, 1.3], abs=1e-6)]
    stretch = 0.02  # each strap hangs straight from the bar's end over its link
    assert float(gantry.tension_n[0]) == pytest.approx(1e4 * stretch, rel=1e-4)
    assert all(float(f.norm()) == pytest.approx(0.5e4 * stretch, rel=1e-4) for _, f in pulls)
    assert gantry.harness_hook_w().tolist() == pytest.approx([-0.1, 0.0, 1.3], abs=1e-6)


def test_more_links_than_the_default_widen_the_gantry(tmp_path):
    links = [(f"gantry_attachment_link_{n}", "torso", (0.0, 0.1 * n, 0.3)) for n in (1, 2, 3)]
    gantry = Gantry(_Robot(POSES), harness=resolve_harness(_desc(tmp_path, *links), None))
    assert gantry.points_b.shape[1] == 3


def test_without_links_the_gantry_is_what_the_profile_gave(tmp_path):
    """The shoulder harness through the new path hangs the very ropes the harness body name did."""
    harness = resolve_harness(_desc(tmp_path), PROFILE_HARNESS)
    ropes = []
    for build in (lambda r: Gantry(r, harness="torso"), lambda r: Gantry(r, harness=harness)):
        robot = _Robot(POSES)
        gantry = build(robot)
        gantry.hang_cable(torch.tensor([0]), 0, overhead_m=1.0, length_breakpoints_s_m=[(0.0, 0.98)],
                          stiffness_n_m=1e4, damping_n_s_m=10.0, swing_damping_n_s_m=5.0, attachment_bodies=[
                              "shoulder_l", "shoulder_r"])
        gantry.apply(0)
        ropes.append((gantry.body_id, gantry.points_b.clone(), gantry.anchor_w.clone(), gantry.top_offset_w.clone(),
                      [(f.tolist(), p.tolist()) for f, p, _ in robot.forces]))
    assert ropes[0][0] == ropes[1][0] and ropes[0][4] == ropes[1][4]
    assert all(torch.equal(a, b) for a, b in zip(ropes[0][1:4], ropes[1][1:4], strict=True))
    assert harness.attachment() == {"attachment_bodies": ["shoulder_l", "shoulder_r"]}


@pytest.mark.parametrize("robot", ["ethrc_v1", "g1_29dof_rev_1_0", "g1_29dof_with_hand_rev_1_0"])
def test_the_delivered_descriptions_resolve_to_their_profile_harness(robot):
    """None of the delivered descriptions carries attachment links yet: every robot keeps its profile's harness, the
    runtime descriptor's included. (Once a description gains them, its profile can drop the hand-written harness.)"""
    from humanoid_sim.model.asset import load_profile
    from humanoid_sim.model.description import Description

    profile = load_profile(robot)["support"]["harness"]
    harness = resolve_harness(Description.load(robot), profile)
    assert harness.links == () and harness.body == profile["body"]
    assert harness.shoulders == tuple(profile["shoulders"])
    assert harness.attachment() == {"attachment_bodies": profile["shoulders"]}
    assert harness.describe() == profile


def test_a_usd_description_gives_the_links_in_the_frame_of_their_body(tmp_path):
    """The links of a USD are frames under the rigid body that carries them (not bodies the physics keeps)."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "fixture.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    base = UsdGeom.Xform.Define(stage, "/robot/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    base.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    for name, xyz in (("gantry_attachment_link_10", (0.0, -0.2, 0.4)), ("gantry_attachment_link_2", (0.0, 0.2, 0.4))):
        UsdGeom.Xform.Define(stage, f"/robot/base/{name}").AddTranslateOp().Set(Gf.Vec3d(*xyz))
    stage.Save()
    from humanoid_sim.model.usd import UsdDescription

    desc = SimpleNamespace(usd_model=UsdDescription(path), urdf_path=None)
    harness = resolve_harness(desc, None)
    assert harness.body == "base" and harness.links == ("gantry_attachment_link_2", "gantry_attachment_link_10")
    assert [list(p) for p in harness.points_b] == [pytest.approx([0.0, 0.2, 0.4], abs=1e-9),
                                                  pytest.approx([0.0, -0.2, 0.4], abs=1e-9)]
