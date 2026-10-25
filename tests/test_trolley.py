import math

import numpy as np
import pytest
import torch

from humanoid_sim.world.trolley import GantryRig, Trolley

MS = 1_000_000


def rig(**kw) -> GantryRig:
    base = dict(center_m=(0.0, 0.0), yaw_rad=0.0, anchor_z_m=3.0, travel_x_m=(-2.0, 2.0), travel_y_m=(-1.0, 1.0),
                hook_z_m=(0.5, 2.5), max_speed_m_s=(0.5, 0.5, 0.2), max_accel_m_s2=(1.0, 1.0, 0.5),
                home_m=(0.0, 0.0, 2.5))
    return GantryRig(**(base | kw))


def run(trolley: Trolley, seconds: float, step_ns: int = 4 * MS):
    """Advance in physics steps, returning positions and velocities [steps, E, 3]."""
    t0 = trolley.t_ns or 0
    trolley.advance(t0)
    ps, vs = [], []
    for k in range(1, int(seconds * 1e9) // step_ns + 1):
        trolley.advance(t0 + k * step_ns)
        ps.append(trolley.position.clone())
        vs.append(trolley.velocity.clone())
    return torch.stack(ps), torch.stack(vs)


def test_the_crane_lags_its_command_within_its_speed_and_acceleration_limits():
    t = Trolley(rig(), 1, "cpu")
    t.set_command(torch.arange(1), torch.tensor([1.5, 0.0, 2.5]))
    assert t.command[0].tolist() == [1.5, 0.0, 2.5] and t.position[0].tolist() == [0.0, 0.0, 2.5]
    p, v = run(t, 5.0)
    x, vx = p[:, 0, 0], v[:, 0, 0]
    assert vx.max() <= 0.5 + 1e-6
    assert (vx.diff().abs() / 0.004).max() <= 1.0 + 1e-3
    # 0.5 s up to speed, 2.5 s at 0.5 m/s, 0.5 s braking: there after about 3.5 s, at most 1 mm past it.
    assert x[int(3.4 / 0.004)] < 1.5
    assert x[int(3.7 / 0.004):].tolist() == pytest.approx([1.5] * (len(x) - int(3.7 / 0.004)), abs=1e-5)
    assert x.max() <= 1.501


def test_commands_and_positions_stay_in_the_travel():
    t = Trolley(rig(), 2, "cpu")
    t.set_command(torch.arange(2), torch.tensor([[5.0, -5.0, 9.0], [-5.0, 0.3, 0.0]]))
    assert t.command.tolist() == [[2.0, -1.0, 2.5], [-2.0, pytest.approx(0.3), 0.5]]
    t.place(torch.tensor([1]), torch.tensor([0.0, 0.0, 0.1]))
    assert t.position[1].tolist() == [0.0, 0.0, 0.5] and t.velocity[1].tolist() == [0.0, 0.0, 0.0]


def test_the_hoist_sets_the_rope_length_and_the_rig_its_top():
    t = Trolley(rig(center_m=(1.0, 2.0), yaw_rad=math.pi / 2), 2, "cpu",
                origin_w=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    t.place(torch.arange(2), torch.tensor([0.5, 0.2, 1.0]))
    # Rig x is world y at 90 degrees; rig y is world -x.
    assert t.anchor_w.tolist() == [[pytest.approx(0.8), pytest.approx(2.5), 3.0],
                                   [pytest.approx(10.8), pytest.approx(2.5), 3.0]]
    assert t.length_m.tolist() == [2.0, 2.0]
    r = t.rig
    assert np.allclose(r.to_world((0.5, 0.2, 1.0)), [0.8, 2.5, 3.0])
    assert r.from_world([0.8, 2.5, 1.0]) == pytest.approx((0.5, 0.2, 1.0))


def test_a_rig_is_checked_and_placed_like_the_scene():
    with pytest.raises(ValueError, match="below the hoist"):
        rig(hook_z_m=(0.5, 3.0))
    with pytest.raises(ValueError, match="outside the travel"):
        rig(home_m=(3.0, 0.0, 2.5))
    with pytest.raises(ValueError, match="unknown gantry keys"):
        GantryRig.from_dict(rig().describe() | {"color": "red"})
    r = GantryRig.from_dict(rig(center_m=(-7.0, -3.0)).describe())
    assert r.placed((-4.0, 3.0)).center_m == (-3.0, -6.0)


def test_the_crane_parts_move_from_where_they_were_scanned():
    r = rig(yaw_rad=math.pi / 2, scan_m=(0.5, 0.0, 2.0))
    off = r.part_offsets((1.5, 0.25, 1.0))
    # Rig x is world y, rig y world -x.
    assert off["bridge"].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert off["trolley"].tolist() == pytest.approx([-0.25, 1.0, 0.0])
    assert off["hook"].tolist() == pytest.approx([-0.25, 1.0, -1.0])
    assert r.travel_outline_w()[:, 2].tolist() == [3.0] * 4


def test_the_front_view_faces_the_crane_s_long_side_from_minus_y():
    r = rig(center_m=(1.0, 2.0), yaw_rad=math.pi / 2)
    eye, target = r.front_view(distance_m=4.0, height_m=2.0)
    # Rig x is world y, rig y world -x: the eye stands 4 m out on world +x, looking back at the centre.
    assert target.tolist() == pytest.approx([1.0, 2.0, 0.9])
    assert eye.tolist() == pytest.approx([5.0, 2.0, 2.0])


def test_the_hangar_example_rig_loads_with_its_home_and_scan_position_in_reach():
    from pathlib import Path

    rig = GantryRig.load(Path(__file__).parents[1] / "examples" / "hangar_gantry.json")
    assert rig.contains(rig.home_m) and rig.contains(rig.scan_m)


def test_the_hangar_crane_starts_clear_of_the_robot_the_scan_caught():
    import json
    from pathlib import Path

    from humanoid_sim.world.splat import in_boxes

    examples = Path(__file__).parents[1] / "examples"
    rig = GantryRig.load(examples / "hangar_gantry.json")
    drop = json.loads((examples / "hangar_crane_parts.json").read_text())["drop"]
    # The hook's column, from the floor to the hoist: a robot hung from it at home stands clear of the scanned one.
    column = np.array([rig.to_world(rig.home_m)[:2].tolist() + [z] for z in np.linspace(0.0, rig.anchor_z_m, 31)])
    assert not in_boxes(column, drop).any()
    assert in_boxes(np.array([rig.to_world(rig.scan_m)[:2].tolist() + [1.0]]), drop).all()


def test_a_crane_over_a_hook_stretches_its_rope_by_the_given_length():
    from pathlib import Path

    rig = GantryRig.load(Path(__file__).parents[1] / "examples" / "hangar_gantry.json")
    hook = rig.to_world((0.5, -0.2, 0.0))
    hook[2] = 1.6  # to_world gives the rope's top over the point; the hook hangs below it
    x, y, z = rig.over(hook, 0.02)
    assert (x, y) == pytest.approx((0.5, -0.2), abs=1e-9)
    assert z == pytest.approx(1.62)  # the rope anchor - z long, the hook anchor - 1.6 below its top: 2 cm stretched
    outside = rig.to_world((rig.travel_x_m[1] + 0.5, 0.0, 0.0))
    outside[2] = 1.6
    assert rig.over(outside, 0.02) is None
