"""Independent translating-support reference on the pinned Newton stack.

The synthetic three-joint fixture has declared test dimensions, not robot facts.
All runs use exactly four environments. No production robot limits/gains change.
"""

import json
import math

import numpy as np
import pytest
import torch


@pytest.mark.physics
def test_inertial_loading_against_static_torque_and_translating_support(tmp_path, request):
    pytest.importorskip("isaaclab_newton")
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.utils import configclass, instantiate

    from humanoid_sim.model.asset import load_profile
    from humanoid_sim.physics.isaaclab import IsaacLabRobot, articulation_cfg, simulation_cfg
    from humanoid_sim.physics.joint_properties import JointProperties

    # A moving support, heading joint, and a 2 kg arm with CoM 0.3 m from its hinge.
    urdf = tmp_path / "inertial-reference.urdf"
    links = ""
    for name, mass, com in [("support", 1, "0 0 0"), ("carriage", 0.1, "0 0 0"),
                            ("heading", 0.1, "0 0 0"), ("arm", 2, "0.3 0 0")]:
        links += f'''<link name="{name}"><inertial><origin xyz="{com}"/>
          <mass value="{mass}"/><inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
          </inertial></link>'''
    joints = ""
    for name, kind, parent, child, axis in [("slide", "prismatic", "support", "carriage", "1 0 0"),
                                           ("yaw", "revolute", "carriage", "heading", "0 0 1"),
                                           ("hinge", "revolute", "heading", "arm", "0 1 0")]:
        joints += f'''<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>
          <axis xyz="{axis}"/><limit lower="-3" upper="3" effort="1000000" velocity="100"/></joint>'''
    urdf.write_text(f'<robot name="inertial-reference">{links}{joints}</robot>')
    asset = UrdfConverter(UrdfConverterCfg(asset_path=str(urdf), usd_dir=str(tmp_path / "usd"),
                                          fix_base=False, merge_fixed_joints=True, joint_drive=None))
    props = JointProperties(["slide", "yaw", "hinge"], [20000, 20000, 40], [200, 200, 4],
                            [0] * 3, [0] * 3, [0] * 3, [1e6] * 3, [0, 0.4, 0.3])

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = articulation_cfg(asset.usd_path, props, fixed_base=True, self_collision=False, base_height_m=2)

    device = request.config.getoption("--replay-device")
    physics = load_profile("ethrc_v1")["physics"]
    context = SimulationContext(simulation_cfg(physics, device=device, period_s=0.004))
    request.addfinalizer(SimulationContext.clear_instance)
    scene = instantiate(SceneCfg(num_envs=4, env_spacing=3, replicate_physics=True))
    context.reset()
    robot = scene["robot"]
    ids = [robot.joint_names.index(n) for n in props.names]
    field = IsaacLabRobot.__new__(IsaacLabRobot)
    field.robot = robot
    field._inertial_bodies = [i for i, n in enumerate(robot.body_names) if n != "support"]
    initial = robot.data.default_joint_pos.torch.clone()
    initial[:, ids] = torch.tensor([0, 0.4, 0.3], device=device)
    dt_ns = 4_000_000
    dt = dt_ns * 1e-9

    def reset():
        robot.write_joint_position_to_sim_index(position=initial)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial))
        scene.write_data_to_sim()
        scene.update(dt)

    def step(target, velocity, acceleration=None):
        robot.actuators.target_command.set_position_index(value=target)
        robot.actuators.target_command.set_velocity_index(value=velocity)
        if acceleration is not None:
            field.apply_base_acceleration(acceleration)
        scene.write_data_to_sim()
        context.step(render=False)
        scene.update(dt)
        return robot.data.joint_pos.torch[:, ids].clone(), robot.data.joint_vel.torch[:, ids].clone()

    reset()
    acceleration = torch.tensor([[0, 0, 0], [0, 0, 1.962], [0, 0, -1.962], [2, 0, 0]], device=device)
    t_ns = 0
    while t_ns < 2_000_000_000:
        q, v = step(initial, torch.zeros_like(initial), acceleration)
        t_ns += dt_ns
    angle, heading = q[:, 2], q[:, 1]
    # Independent static balance: motor torque must oppose r x m(g-a).
    expected = -2 * 0.3 * ((9.81 + acceleration[:, 2]) * angle.cos()
                           + acceleration[:, 0] * heading.cos() * angle.sin())
    motor = 40 * (0.3 - angle) - 4 * v[:, 2]
    static_error = float((motor - expected).abs().max())
    assert static_error < 0.01 * float(expected.abs().max()) + 1e-3
    assert float(v[:, 2].abs().max()) < 1e-3

    # Actual reference acceleration comes from the translating support's measured
    # velocity increment, not the new IMU conversion or force helper.
    reset()
    ref_q, ref_v, slider_v = [], [], [torch.zeros((), device=device)]
    t_ns = 0
    while t_ns < 2_000_000_000:
        t = t_ns * 1e-9
        target, velocity = initial.clone(), torch.zeros_like(initial)
        target[:, ids[0]] = 0.025 * (1 - math.cos(2 * math.pi * 1.3 * t))
        velocity[:, ids[0]] = 0.025 * 2 * math.pi * 1.3 * math.sin(2 * math.pi * 1.3 * t)
        q, v = step(target, velocity)
        ref_q.append(q[0, 2])
        ref_v.append(v[0, 2])
        slider_v.append(v[0, 0])
        t_ns += dt_ns
    ref_q, ref_v = torch.stack(ref_q), torch.stack(ref_v)
    support_acc = torch.diff(torch.stack(slider_v)) / dt
    reset()
    observed_q, observed_v, support_yaw = [], [], []
    t_ns = 0
    while t_ns < 2_000_000_000:
        k, t = t_ns // dt_ns, t_ns * 1e-9
        target, velocity = initial.clone(), torch.zeros_like(initial)
        target[0, ids[0]] = 0.025 * (1 - math.cos(2 * math.pi * 1.3 * t))
        velocity[0, ids[0]] = 0.025 * 2 * math.pi * 1.3 * math.sin(2 * math.pi * 1.3 * t)
        correction = torch.zeros(4, 3, device=device)
        correction[1, 0], correction[3, 0] = support_acc[k], 0.5 * support_acc[k]
        q, v = step(target, velocity, correction)
        observed_q.append(q[:, 2])
        observed_v.append(v[:, 2])
        support_yaw.append(q[:, 1])
        t_ns += dt_ns
    observed_q, observed_v = torch.stack(observed_q), torch.stack(observed_v)
    errors = torch.sqrt(((observed_q - ref_q[:, None]) ** 2).mean(0))
    ref_motor = 40 * (0.3 - ref_q) - 4 * ref_v
    motors = 40 * (0.3 - observed_q) - 4 * observed_v
    torque_errors = torch.sqrt(((motors - ref_motor[:, None]) ** 2).mean(0))
    metrics = {"envs": 4, "static_torque_error_nm": static_error,
               "dynamic_angle_rms_rad": errors.tolist(), "dynamic_torque_rms_nm": torque_errors.tolist(),
               "reference_acceleration_peak_m_s2": float(support_acc.abs().max()),
               "heading_tracking_peak_rad": float((torch.stack(support_yaw) - 0.4).abs().max())}
    print(json.dumps(metrics), flush=True)
    (tmp_path / "reference-metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(tmp_path / "reference-traces.npz", q=observed_q.cpu(), ref_q=ref_q.cpu(),
             motor=motors.cpu(), ref_motor=ref_motor.cpu(), support_acc=support_acc.cpu())
    # Frozen before first execution: numerical repeatability, absolute resolution,
    # and a material improvement over omitting the measured base acceleration.
    assert float(errors[0]) < 1e-5
    assert float(errors[1]) < 2e-3 and float(errors[1]) < 0.25 * float(errors[2])
    assert float(torque_errors[1]) < 0.25 * float(torque_errors[2])
    assert metrics["heading_tracking_peak_rad"] < 1e-3
