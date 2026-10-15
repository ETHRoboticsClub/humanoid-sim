"""The mirror map: a mirrored joint deviation moves the partner's side of the body in the mirrored direction
(URDF forward kinematics about the standing pose), and the drive map agrees with the joint map."""

import numpy as np
import pytest

from humanoid_sim.model.description import Description
from humanoid_sim.model.mirror import _joint_positions, mirror_map


@pytest.fixture(scope="module")
def setup():
    desc = Description.load()
    return desc, mirror_map(desc)


def test_mirrored_deviations_move_the_body_in_mirrored_directions(setup):
    desc, m = setup
    n = np.zeros(3)
    n[m.lateral_axis] = 1.0
    reflect = np.eye(3) - 2 * np.outer(n, n)
    c = np.array(m.center)
    base = _joint_positions(desc.urdf_path, dict(zip(m.joint_names, c, strict=True)))
    checked = 0
    for i, name in enumerate(m.joint_names):
        dq = np.zeros(len(c))
        dq[i] = 0.05
        q, q_m = c + dq, m.mirror_joints(c + dq)
        moved = _joint_positions(desc.urdf_path, dict(zip(m.joint_names, q, strict=True)))
        moved_m = _joint_positions(desc.urdf_path, dict(zip(m.joint_names, q_m, strict=True)))
        partner_of = {a: m.joint_names[m.joint_perm[k]] for k, a in enumerate(m.joint_names)}
        for j in m.joint_names:
            d = moved[j] - base[j]
            if np.linalg.norm(d) < 2e-3:
                continue  # upstream, or on the joint's axis (the ankle roll frame sits ~1 mm off the pitch axis)
            d_m = moved_m[partner_of[j]] - base[partner_of[j]]
            cos = float((reflect @ d) @ d_m / (np.linalg.norm(d) * np.linalg.norm(d_m)))
            assert cos > 0.9, f"{name}: {j} moves {d}, its partner {d_m}"
            checked += 1
    assert checked > 30


def test_mirror_is_an_involution_and_drives_follow_joints(setup):
    desc, m = setup
    x = np.arange(len(m.joint_names), dtype=float) + 1
    np.testing.assert_allclose(m.mirror_joints(m.mirror_joints(x)), x)
    y = np.arange(len(m.drive_names), dtype=float) + 1
    np.testing.assert_array_equal(m.mirror_drives(m.mirror_drives(y)), y)
    for i, name in enumerate(m.drive_names):
        d = desc.drives[name]
        if d.crank is None:
            j = m.joint_names.index(d.joints[0])
            assert desc.drives[m.drive_names[m.drive_perm[i]]].joints[0] == m.joint_names[m.joint_perm[j]]
        else:
            assert desc.drives[m.drive_names[m.drive_perm[i]]].crank == d.crank
    assert m.center == [desc.joints[n].standing_offset_rad for n in m.joint_names]
    assert 0.0 < m.standing_asymmetry_m < 0.1


def test_torch_and_numpy_agree(setup):
    torch = pytest.importorskip("torch")
    _, m = setup
    q = np.random.default_rng(0).normal(size=(4, len(m.joint_names)))
    np.testing.assert_allclose(m.mirror_joints(torch.tensor(q)).numpy(), m.mirror_joints(q))
