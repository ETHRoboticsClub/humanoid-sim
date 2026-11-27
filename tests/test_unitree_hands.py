"""Unitree hands (``unitree-hands.yaml``): welded with the other unlisted joints, or each motor a 1:1 drive of its own
(profile ``model.unitree_hands``). Expectations come from the description's own files at run time."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
import yaml

import humanoid_sim.model.asset as asset
from humanoid_sim.model.description import HAND_REGION, Description
from humanoid_sim.model.urdf import PreparedUrdf

ROBOT = "g1_29dof_with_hand_rev_1_0"


def _hands_file(desc: Description) -> dict:
    return yaml.safe_load((desc.robot_dir / "unitree-hands.yaml").read_text())


def _urdf_limits(desc: Description) -> dict[str, dict[str, float]]:
    joints = ET.parse(desc.urdf_path).getroot().findall("joint")
    return {j.get("name"): {k: float(j.find("limit").get(k)) for k in ("lower", "upper", "effort", "velocity")}
            for j in joints if j.find("limit") is not None}


def _with_hands(monkeypatch, hands: dict | None):
    """The robot's profile with ``model.unitree_hands`` replaced (None: removed, the defaults' welded mode)."""
    load = asset.load_profile

    def patched(robot, *args, **kw):
        profile = load(robot, *args, **kw)
        profile["model"] = dict(profile["model"])
        if hands is None:
            profile["model"].pop("unitree_hands", None)
        else:
            profile["model"]["unitree_hands"] = hands
        return profile

    monkeypatch.setattr(asset, "load_profile", patched)


def _driven_terms() -> dict:
    return {"mode": "driven", "armature_kg_m2": 1e-3, "viscous_damping_nm_s_per_rad": 0.0, "friction_nm": 0.01}


def test_driven_hand_motors_are_drives_of_their_own(monkeypatch):
    _with_hands(monkeypatch, _driven_terms())
    desc = Description.load(ROBOT)
    document, urdf = _hands_file(desc), _urdf_limits(desc)
    motors = [m for hand in document["hands"].values() for m in hand["motors"]]
    body = list(desc.table["joints"])
    # Body drives first, in description order, then every hand's motors in slot order.
    assert list(desc.drives) == body + motors
    assert list(desc.joints) == body + motors
    for side, hand in document["hands"].items():
        assert desc.hands[side].joints == tuple(hand["motors"])
        for name in hand["motors"]:
            drive, joint, limit = desc.drives[name], desc.joints[name], urdf[name]
            assert drive.family == hand["model"] and drive.joints == (name,) and drive.sign == 1
            assert drive.motor_index is None and drive.crank is None
            assert drive.baseline_stiffness == document["kp_nm_per_rad"]
            assert drive.baseline_damping == document["kd_nm_s_per_rad"]
            assert drive.effort_peak == joint.effort_peak == limit["effort"]
            assert drive.velocity == joint.velocity == limit["velocity"]
            assert joint.region == HAND_REGION and joint.sign == 1 and joint.standing_offset_rad == 0.0
            assert (joint.armature, joint.damping, joint.friction) == (1e-3, 0.0, 0.01)
            assert desc.calibration_travel(name) == (limit["lower"], limit["upper"])
            assert desc.drive_torque_speed(name) == [(0.0, limit["effort"]), (limit["velocity"], limit["effort"])]
    assert desc.hand_families == {hand["model"] for hand in document["hands"].values()}
    for family in desc.hand_families:
        assert desc.drive_delay_ns(family) is None


def test_driven_hand_joints_stay_movable_with_their_urdf_limits(monkeypatch):
    _with_hands(monkeypatch, _driven_terms())
    desc = Description.load(ROBOT)
    prepared = PreparedUrdf(desc, asset.load_profile(ROBOT)).build()
    joints = {j.get("name"): j for j in prepared.root.findall("joint")}
    urdf = _urdf_limits(desc)
    assert not set(prepared.welded) & set(desc.hand_joints)
    for name in desc.hand_joints:
        limit = joints[name].find("limit")
        assert joints[name].get("type") == "revolute"
        for key in ("lower", "upper", "effort", "velocity"):
            assert float(limit.get(key)) == pytest.approx(urdf[name][key])


def test_welded_hands_leave_only_the_body_drives(monkeypatch):
    _with_hands(monkeypatch, None)
    desc = Description.load(ROBOT)
    motors = {m for hand in _hands_file(desc)["hands"].values() for m in hand["motors"]}
    assert desc.hand_plant is None and desc.hand_joints == {} and desc.hand_families == set()
    assert set(desc.drives) == set(desc.table["joints"])
    assert desc.hands  # still declared, only not driven
    prepared = PreparedUrdf(desc, asset.load_profile(ROBOT)).build()
    assert motors <= set(prepared.welded)


@pytest.mark.parametrize("hands, match", [
    ({"mode": "kinematic"}, "mode"),
    ({"mode": "driven", "viscous_damping_nm_s_per_rad": 0.0, "friction_nm": 0.0}, "armature_kg_m2"),
    ({**_driven_terms(), "armature_kg_m2": 0.0}, "armature_kg_m2"),
    ({**_driven_terms(), "friction_nm": -1.0}, "friction_nm"),
    ({**_driven_terms(), "viscous_damping_nm_s_per_rad": float("nan")}, "viscous_damping"),
])
def test_the_hand_mode_and_its_terms_are_checked(monkeypatch, hands, match):
    _with_hands(monkeypatch, hands)
    with pytest.raises(ValueError, match=match):
        Description.load(ROBOT).drives


def test_driving_hands_a_description_does_not_declare_raises(monkeypatch):
    _with_hands(monkeypatch, _driven_terms())
    desc = Description.load("ethrc_v1")
    assert desc.hands == {}
    with pytest.raises(ValueError, match="declares none"):
        desc.joints
