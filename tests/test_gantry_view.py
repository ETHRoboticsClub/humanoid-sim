"""The gantry in the viewers: the rope runs from the anchor to the harness hook, its colour class follows what it
carries, and a robot with no gantry says so. The expected points and tensions are worked out by hand here, not read
back from the code under test; Rerun is replaced by a recorder of what is logged."""

import math

import numpy as np
import pytest
import torch

from humanoid_sim.signals import gantry_view as gv
from humanoid_sim.signals.scene import SceneRecorder, find_gantry
from humanoid_sim.world.supports import GRAVITY_M_S2, Gantry
from humanoid_sim.world.trolley import GantryRig

MASS_KG = 10.0
WEIGHT_N = MASS_KG * 9.81  # g typed here, not read from the module under test
STIFFNESS = 2000.0
HOOK_Z = 1.3  # the base is 1 m up and the rope point 0.3 m above it
ANCHOR = [0.0, 0.0, 3.3]
MS = 1_000_000


class FakeRerun:
    """Records ``log`` calls as ``(path, kind, kwargs)``; the archetypes are plain records."""

    def __init__(self):
        self.logged = []

    def log(self, path, entity, static=False):
        self.logged.append((path, entity, static))

    def last(self, path):
        return [e for p, e, _ in self.logged if p == path][-1]

    def scalars(self, path):
        return [e[1][0] for p, e, _ in self.logged if p == path and e[0] == "scalars"]

    def LineStrips3D(self, strips, colors=None, radii=None):
        return ("lines", [np.asarray(s, dtype=float) for s in strips], colors, radii)

    def Points3D(self, positions, radii=None, colors=None, labels=None):
        return ("points", np.asarray(positions, dtype=float), colors, labels)

    def Scalars(self, value):
        return ("scalars", [value])

    def Clear(self, recursive=False):
        return ("clear",)


class Robot:
    """Two envs' articulations: a base 1 m up and 10 kg, ropes pulling on it, forces collected and dropped."""

    def __init__(self):
        self.num_instances, self.device, self.body_names = 2, "cpu", ["base"]
        pose = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).expand(2, 1, 7)
        vel = torch.zeros(2, 1, 6)
        self.data = type("Data", (), {"body_link_pose_w": type("T", (), {"torch": pose})(),
                                      "body_link_vel_w": type("T", (), {"torch": vel})(),
                                      "body_mass": type("T", (), {"torch": torch.full((2, 1), MASS_KG)})()})()

        class Composer:
            def add_forces_and_torques_index(self, **kw):
                pass

        self.instantaneous_wrench_composer = Composer()


def recorder(gantry, robot, envs=(0, 1)) -> tuple[SceneRecorder, FakeRerun]:
    """A SceneRecorder without a scene: only what its gantry drawing reads."""
    rec = object.__new__(SceneRecorder)
    rec.rr, rec.robot, rec.gantry = FakeRerun(), robot, gantry
    rec.envs, rec._ids, rec.notes, rec._crane_offsets = list(envs), torch.tensor(list(envs)), {}, {}
    return rec, rec.rr


BASES = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])


def hang(robot, rest_m, anchors=(ANCHOR, ANCHOR)):
    g = Gantry(robot)
    n = 2
    g.hang_cables(torch.arange(n), 0, point_b=torch.tensor([0.0, 0.0, 0.3]), anchor_w=torch.tensor(anchors),
                  times_ns=torch.zeros(n, 1, dtype=torch.int64), lengths_m=torch.full((n, 1), float(rest_m)),
                  stiffness_n_m=torch.full((n,), STIFFNESS), damping_n_s_m=torch.zeros(n),
                  swing_damping_n_s_m=torch.zeros(n))
    return g


def test_the_gravity_the_viewers_use_is_the_sims():
    assert GRAVITY_M_S2 == 9.81


def test_classes_for_known_tensions():
    w = WEIGHT_N
    cases = [(0.0, gv.SLACK), (0.5 * w, gv.TAUT), (w, gv.TAUT), (1.01 * w, gv.OVER), (1.9 * w, gv.OVER),
             (2.01 * w, gv.CAUGHT)]
    for tension, expected in cases:
        assert gv.rope_state(tension, w) == expected, tension


def test_classes_for_known_lengths():
    w, stretch = WEIGHT_N, WEIGHT_N / STIFFNESS  # 4.905 cm under the full weight
    for extension, expected in [(-0.1, gv.SLACK), (0.0, gv.SLACK), (0.5 * stretch, gv.TAUT),
                                (0.99 * stretch, gv.TAUT), (1.01 * stretch, gv.OVER)]:
        tension = max(0.0, STIFFNESS * extension)
        assert gv.rope_state(tension, w, extension_m=extension, stretch_m=stretch) == expected, extension
    # A damper can hide the stretch from the tension: the length still says over-stretched.
    assert gv.rope_state(0.5 * w, w, extension_m=2 * stretch, stretch_m=stretch) == gv.OVER


def test_the_full_weight_stretch_is_weight_over_stiffness():
    assert gv.full_weight_stretch_m(98.1, 2000.0) == pytest.approx(0.04905)


def test_status_texts():
    assert gv.status_text(gv.TAUT, 0.88 * WEIGHT_N, WEIGHT_N) == "gantry: taut 0.88 W"
    assert gv.status_text(gv.SLACK, 0.0, WEIGHT_N) == "gantry: slack"
    assert gv.status_text(gv.CAUGHT, 2.5 * WEIGHT_N, WEIGHT_N) == "gantry: caught 2.50 W"
    assert gv.status_text(gv.OVER, 1.4 * WEIGHT_N, WEIGHT_N, "stand") == "gantry: over-stretched 1.40 W | stand"


@pytest.mark.parametrize("rest_m, color, tension_n", [
    (2.0, gv.STATE_RGB[gv.SLACK], 0.0),  # 2.0 m of rope for a 2.0 m gap: slack
    (2.1, gv.STATE_RGB[gv.SLACK], 0.0),
    (2.0 - 0.5 * WEIGHT_N / STIFFNESS, gv.STATE_RGB[gv.TAUT], 0.5 * WEIGHT_N),  # carries half the weight
    (2.0 - 1.5 * WEIGHT_N / STIFFNESS, gv.STATE_RGB[gv.OVER], 1.5 * WEIGHT_N),  # one and a half weights
])
def test_the_rope_joins_the_anchor_to_the_hook_in_the_colour_of_its_load(rest_m, color, tension_n):
    robot = Robot()
    g = hang(robot, rest_m)
    g.apply(0)
    rec, rr = recorder(g, robot)
    rec._gantry(0, BASES)
    for env in (0, 1):
        _, strips, colors, _ = rr.last(f"/world/gantry/{env}/rope")
        assert strips[0] == pytest.approx(np.array([ANCHOR, [0.0, 0.0, HOOK_Z]]), abs=1e-5)  # anchor to hook, by hand
        assert colors == [color]
        assert rr.scalars(f"/sim/{env}/gantry/tension_n")[-1] == pytest.approx(tension_n, rel=1e-4, abs=1e-9)
        assert rr.scalars(f"/sim/{env}/gantry/tension_share")[-1] == pytest.approx(tension_n / WEIGHT_N, abs=1e-4)
        assert rr.scalars(f"/sim/{env}/gantry/length_m")[-1] == pytest.approx(ANCHOR[2] - HOOK_Z)
        assert rr.scalars(f"/sim/{env}/gantry/rest_length_m")[-1] == pytest.approx(rest_m)
        assert rr.last(f"/world/gantry/{env}/anchor")[1] == pytest.approx(np.array([ANCHOR]), abs=1e-5)


def test_a_rope_hung_from_the_wrong_anchor_is_slanted_and_red():
    """The training bug: the anchor above the default spawn, the robot elsewhere: the rope leans and over-stretches."""
    robot = Robot()
    stretched = [1.0, 0.0, 3.3]  # 1 m aside; the rope is sqrt(1 + 2 ** 2) long for a 2 m rest length
    g = hang(robot, 2.0, anchors=(ANCHOR, stretched))
    g.apply(0)
    rec, rr = recorder(g, robot)
    rec._gantry(0, BASES)
    _, strips, colors, _ = rr.last("/world/gantry/1/rope")
    assert strips[0] == pytest.approx(np.array([stretched, [0.0, 0.0, HOOK_Z]]), abs=1e-5)
    assert colors == [gv.STATE_RGB[gv.CAUGHT]]  # 1.5 kN against a 98 N weight
    expected = STIFFNESS * (math.sqrt(1.0 + 2.0 ** 2) - 2.0)
    assert rr.scalars("/sim/1/gantry/tension_n")[-1] == pytest.approx(expected, rel=1e-4)
    assert "caught" in rr.last("/world/gantry/1/status")[3][0]
    assert rr.last("/world/gantry/0/status")[3][0] == "gantry: slack"
    # The plumb line stays straight down from the anchor, so the lean shows against it.
    assert rr.last("/world/gantry/1/plumb")[1][0] == pytest.approx(np.array([stretched, [1.0, 0.0, 3.3 - 2.0]]), abs=1e-5)


def test_envs_without_a_rope_say_no_gantry():
    robot = Robot()
    g = hang(robot, 2.0)
    g.release(torch.tensor([1]))
    g.apply(0)
    rec, rr = recorder(g, robot)
    rec._gantry(0, BASES)
    assert rr.last("/world/gantry/1/status")[3] == ["no gantry"]
    assert rr.last("/world/gantry/0/status")[3] == ["gantry: slack"]
    assert rr.last("/world/gantry/1/rope") == ("clear",)
    rec, rr = recorder(None, robot)  # no gantry at all
    rec._gantry(0, BASES)
    assert [rr.last(f"/world/gantry/{e}/status")[3] for e in (0, 1)] == [["no gantry"]] * 2
    assert rr.last("/world/gantry/0/status")[1][0].tolist() == [0.0, 0.0, 1.0 + gv.LABEL_LIFT_M]


def test_the_callers_note_follows_the_status():
    robot = Robot()
    g = hang(robot, 2.0)
    g.apply(0)
    rec, rr = recorder(g, robot)
    rec.notes[1] = "released"
    rec._gantry(0, BASES)
    assert rr.last("/world/gantry/0/status")[3] == ["gantry: slack"]
    assert rr.last("/world/gantry/1/status")[3] == ["gantry: slack | released"]


def test_a_tension_program_has_a_label_and_no_rope():
    robot = Robot()
    g = Gantry(robot)
    g.hang_tension(torch.arange(2), 0, [(0.0, 0.5 * WEIGHT_N)], attachment_b=[0.0, 0.0, 0.3])
    g.apply(0)
    rec, rr = recorder(g, robot)
    rec._gantry(0, BASES)
    assert rr.last("/world/gantry/0/rope") == ("clear",)
    assert rr.last("/world/gantry/0/status")[3] == ["gantry: taut 0.50 W"]


def test_rest_length_follows_the_program_or_the_hoist_and_is_nan_without_a_cable():
    robot = Robot()
    g = Gantry(robot)
    g.hang_cable(torch.tensor([0]), 0, overhead_m=2.0, length_breakpoints_s_m=[(0.0, 1.0), (2.0, 3.0)],
                 stiffness_n_m=1000.0, damping_n_s_m=0.0, attachment_b=[0.0, 0.0, 0.3])
    rest = g.rest_length_m(1_000_000_000)  # halfway through the ramp
    assert rest[0].item() == pytest.approx(2.0)
    assert math.isnan(rest[1].item())
    rig = GantryRig(center_m=(0.0, 0.0), yaw_rad=0.0, anchor_z_m=3.3, travel_x_m=(-2.0, 2.0), travel_y_m=(-1.0, 1.0),
                    hook_z_m=(0.5, 2.5), max_speed_m_s=(0.5, 0.5, 0.2), max_accel_m_s2=(1.0, 1.0, 0.5),
                    home_m=(0.0, 0.0, 1.32))
    g.mount(rig)
    g.hang_hoist(torch.tensor([1]), attachment_b=[0.0, 0.0, 0.3], stiffness_n_m=1000.0, damping_n_s_m=0.0)
    assert g.rest_length_m(0)[1].item() == pytest.approx(3.3 - 1.32)


def test_the_crane_is_drawn_per_env_where_its_rig_stands():
    robot = Robot()
    g = Gantry(robot)
    rig = GantryRig(center_m=(1.0, 2.0), yaw_rad=0.0, anchor_z_m=3.0, travel_x_m=(-2.0, 2.0), travel_y_m=(-1.0, 1.0),
                    hook_z_m=(0.5, 2.5), max_speed_m_s=(0.5, 0.5, 0.2), max_accel_m_s2=(1.0, 1.0, 0.5),
                    home_m=(0.0, 0.0, 1.7))
    g.mount(rig, origin_w=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))  # env 1 stands 10 m along x
    g.trolley.place(torch.arange(2), torch.tensor([0.5, 0.25, 1.7]))
    g.hang_hoist(torch.arange(2), attachment_b=[0.0, 0.0, 0.3], stiffness_n_m=STIFFNESS, damping_n_s_m=0.0)
    g.apply(0)
    rec, rr = recorder(g, robot)
    home = np.array([1.0, 2.0, 3.0])
    rec._crane_offsets = {e: g.trolley.origin_w[e].numpy() - home for e in (0, 1)}
    rec._gantry(0, BASES)
    for env, dx in ((0, 0.0), (1, 10.0)):
        # The trolley is at rig (0.5, 0.25): world (1.5 + dx, 2.25), under the hoist at z 3; the bridge spans y 1 to 3.
        assert rr.last(f"/world/gantry/{env}/trolley")[1].tolist() == [[1.5 + dx, 2.25, 3.0]]
        assert rr.last(f"/world/gantry/{env}/bridge")[1][0].tolist() == [[1.5 + dx, 1.0, 3.0], [1.5 + dx, 3.0, 3.0]]
        assert rr.last(f"/world/gantry/{env}/rope")[1][0][0].tolist() == [1.5 + dx, 2.25, 3.0]
    gv.log_travel(rr, "/travel", rig, rec._crane_offsets[1])
    corners = rr.last("/travel")[1][0]
    assert corners.tolist() == [[9.0, 1.0, 3.0], [13.0, 1.0, 3.0], [13.0, 3.0, 3.0], [9.0, 3.0, 3.0], [9.0, 1.0, 3.0]]


def test_the_recorder_finds_the_gantry_of_a_plant_action_term():
    gantry = object()
    term = type("Term", (), {"plant": type("Plant", (), {"gantry": gantry})()})()
    env = type("Env", (), {"action_manager": type("M", (), {"_terms": {"plant": term}})()})()
    assert find_gantry(env) is gantry
    assert find_gantry(type("Env", (), {})()) is None


def test_the_weight_must_be_positive():
    with pytest.raises(ValueError, match="weight"):
        gv.log_env(FakeRerun(), "/w", "/s", top=[0, 0, 3], hook=[0, 0, 1], rest_m=2.0, tension_n=0.0, weight_n=0.0)


def test_a_real_recording_takes_the_rope(tmp_path):
    rr = pytest.importorskip("rerun")
    rr.init("test-gantry-view")
    rr.save(str(tmp_path / "g.rrd"))
    gv.log_env(rr, "/world/gantry/0", "/sim/0/gantry", top=ANCHOR, hook=[0, 0, HOOK_Z], rest_m=1.9, tension_n=50.0,
               weight_n=WEIGHT_N)
    gv.log_no_gantry(rr, "/world/gantry/1", [0.0, 0.0, 1.0])
    rr.disconnect()
    assert (tmp_path / "g.rrd").stat().st_size > 0


def test_view_runtime_draws_the_rope_from_the_truth_and_says_no_gantry_without_one():
    from humanoid_sim.runtime.view import RuntimeView

    view = object.__new__(RuntimeView)
    view.rr, view.rig = FakeRerun(), None
    truth = {"gantry_top_m": [0.0, 0.0, 3.3], "gantry_harness_m": [0.0, 0.0, 1.3], "gantry_tension_n": 0.88 * WEIGHT_N,
             "mass_kg": MASS_KG, "base_position_m": [0.0, 0.0, 1.0], "gantry_position_m": [math.nan] * 3,
             "gantry_command_m": [math.nan] * 3}
    view._gantry(truth)
    rr = view.rr
    assert rr.last("/world/gantry/rope")[1][0].tolist() == [[0.0, 0.0, 3.3], [0.0, 0.0, 1.3]]
    assert rr.last("/world/gantry/rope")[2] == [gv.STATE_RGB[gv.TAUT]]
    assert rr.last("/world/gantry/status")[3] == ["gantry: taut 0.88 W"]
    assert rr.scalars("/sim/0/gantry/tension_share")[-1] == pytest.approx(0.88)
    truth |= {"gantry_top_m": [math.nan] * 3, "gantry_harness_m": [math.nan] * 3, "gantry_tension_n": 0.0}
    view._gantry(truth)
    assert rr.last("/world/gantry/status")[3] == ["no gantry"]
    assert rr.last("/world/gantry/rope") == ("clear",)
