"""The native V1 with self-collision on, on both solvers: standing on its feet in a gantry rope, a scripted march and
random thrashing stay finite and bounded.

The march is a stand-in for walking: a gait-like hip and knee swing in antiphase between the legs about the standing
pose, big enough to bring the legs and feet close and into the ground, with no policy behind it. The thrash puts every
drive on a fresh random target every 20 ms, which throws the robot about on the ground (the case that once took the
ankle rods to non-finite values). Both only ask that nothing goes non-finite or spins up.
"""
import math

import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand

ENVS = 16
POLICY_PERIOD_NS = 20_000_000
STAND_S, MARCH_S, THRASH_S = 3.0, 6.0, 6.0
MARCH_HZ, HIP_RAD, KNEE_RAD, THRASH_RAD = 1.5, 0.35, 0.5, 0.4
ROPE_SHARE = 0.7  # of the weight, in the standing check
MAX_BODY_SPIN_RAD_S = 500.0  # the ankle rods once spun past 1600 rad/s on the way to non-finite values


@pytest.fixture(autouse=True)
def clean_solver_output(capfd):
    yield
    output = "".join(capfd.readouterr())
    assert "possibly invalid inertia tensor" not in output
    assert "exceeded MJWarp limit" not in output
    assert "Contact buffer overflowed" not in output


def _targets(desc, drive_q, mode, t_s, noise):
    """Drive position targets ``[E, D]`` at time ``t_s``: the standing ones, with the march's swing or the thrash's
    noise on top."""
    names = list(desc.drives)
    out = drive_q.clone()
    if mode == "march":
        w = 2 * math.pi * MARCH_HZ * t_s
        for i, name in enumerate(names):
            side = -1.0 if name.endswith("_r") else 1.0
            if name.startswith("hip_flexion"):
                out[:, i] += side * HIP_RAD * math.sin(w)
            elif name.startswith("knee_flexion"):
                out[:, i] += KNEE_RAD * 0.5 * (1.0 - side * math.cos(w))
    elif mode == "thrash":
        out += THRASH_RAD * noise
    return out


def _rollout(solver, mode, seconds, *, self_collision=True, pose="standing", device="cuda:0"):
    """Step ``ENVS`` robots in the training scene for ``seconds``; returns what the checks read. The simulation is
    cleared at the end (one per process)."""
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate

    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg
    from humanoid_sim.train.scene import robot_build, robot_scene_cfg, robot_sim_cfg

    usd = built_usd("ethrc_v1")
    sim = sim_utils.SimulationContext(robot_sim_cfg("ethrc_v1", device=device, solver=solver))
    try:
        with robot_build("ethrc_v1", solver=solver):
            scene = instantiate(robot_scene_cfg(ENVS, usd, pose=pose, self_collision=self_collision))
            sim.reset()
        robot = scene["robot"]
        plant = RobotPlant(robot, TrainPlantCfg(reporting=False, feedback_sampling=False),
                           imu=scene["imu"], contacts=scene["contacts"], physics_manager=sim.physics_manager)
        core = plant.core
        q, qd = plant.joint_state()
        stand, _, _ = core.map.drive_state(q.clone(), qd)
        stiff, damp = core.drive_stiffness.expand(ENVS, -1), core.drive_damping.expand(ENVS, -1)
        plant.reset(torch.arange(ENVS, device=device), demand=DriveDemand.pd(stand, stiff, damp))  # drives on
        if mode == "stand":
            # Held at the baseline gains the native V1 does not balance on its own (a policy does); a rope that takes
            # part of its weight keeps it on its feet, with every contact between its shapes and the ground live.
            weight_n = float(robot.data.body_mass.torch[0].sum()) * 9.81
            plant.gantry.hang_tension(torch.arange(ENVS, device=device), 0, [(0.0, ROPE_SHARE * weight_n)],
                                      **plant.gantry.harness.attachment())
        start = robot.data.root_link_pose_w.torch[:, 2].clone()
        generator = torch.Generator(device=device).manual_seed(0)
        every = POLICY_PERIOD_NS // plant.period_ns
        finite, spin = True, 0.0
        noise = torch.zeros_like(stand)
        for step in range(int(seconds * 1e9) // plant.period_ns):
            if step % every == 0 and mode != "stand":  # standing holds the demand the reset put in force
                noise = torch.randn(stand.shape, generator=generator, device=device, dtype=stand.dtype)
                target = _targets(core.desc, stand, mode, step * plant.period_ns * 1e-9, noise)
                plant.submit(DriveDemand.pd(target, stiff, damp),
                             torch.full((ENVS,), plant.t_ns, dtype=torch.int64, device=device))
            plant.pre_physics()
            scene.write_data_to_sim()
            sim.step()
            scene.update(plant.period_ns / 1e9)
            if step % 50 == 0:
                truth = plant.truth()
                finite &= all(bool(torch.isfinite(t).all()) for t in
                              (truth.joint_pos, truth.joint_vel, truth.root_pose_w))
                spin = max(spin, float(robot.data.body_link_vel_w.torch[..., 3:].norm(dim=-1).max()))
        end = plant.truth().root_pose_w
        x, y = end[:, 3], end[:, 4]
        return {"finite": finite, "spin": spin, "height": end[:, 2], "start_height": start,
                "tilt_deg": torch.rad2deg(2 * torch.asin(torch.hypot(x, y).clamp(max=1.0)))}
    finally:
        sim_utils.SimulationContext.clear_instance()


def _cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("native collision qualification requires CUDA")


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
def test_native_standing_with_self_collision_holds_the_pose(solver):
    _cuda()
    r = _rollout(solver, "stand", STAND_S)
    assert r["finite"]
    assert r["spin"] < MAX_BODY_SPIN_RAD_S
    assert (r["height"] > 0.9 * r["start_height"]).all(), "a robot sank"
    assert r["tilt_deg"].max() < 10.0


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
@pytest.mark.parametrize("mode,seconds", [("march", MARCH_S), ("thrash", THRASH_S)])
def test_native_moving_with_self_collision_stays_finite(solver, mode, seconds):
    _cuda()
    r = _rollout(solver, mode, seconds)
    assert r["finite"]
    assert r["spin"] < MAX_BODY_SPIN_RAD_S
