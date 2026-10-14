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
