"""Independent high-gain direct and coupled-rotor fixtures; not V1 parameters.

Four environments, unchanged production solver/timestep. The coupled fixture has
constant known inertia and a declared virtual crank matrix, permitting a dense
continuous-time reference independent of target shaping. Actual nonlinear V1
ankle mapping is checked separately on CPU and by the V1 smoke gate.
"""

import json

import numpy as np
import pytest
import torch


@pytest.mark.physics
def test_direct_and_coupled_high_gain_reference(tmp_path, request):
    pytest.importorskip("isaaclab_newton")
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.utils import configclass, instantiate

    from humanoid_sim.devices.shaping import shaped_position_target
    from humanoid_sim.model.asset import load_profile
    from humanoid_sim.physics.isaaclab import articulation_cfg, simulation_cfg
    from humanoid_sim.physics.joint_properties import JointProperties

    links = ""
    for name, inertia in (("root", 1), ("direct", 0.5), ("outer", 1.0), ("inner", 0.7)):
        links += f'''<link name="{name}"><inertial><mass value="1"/>
          <inertia ixx="{inertia}" iyy="{inertia}" izz="{inertia}" ixy="0" ixz="0" iyz="0"/>
          </inertial></link>'''
    joints = ""
    for name, parent, child in (("direct", "root", "direct"), ("outer", "root", "outer"), ("inner", "outer", "inner")):
        joints += f'''<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>
          <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1000000" velocity="100"/></joint>'''
    path = tmp_path / "rotors.urdf"
    path.write_text(f'<robot name="rotors">{links}{joints}</robot>')
    asset = UrdfConverter(UrdfConverterCfg(asset_path=str(path), usd_dir=str(tmp_path / "usd"),
                                          fix_base=False, merge_fixed_joints=True, joint_drive=None))
    matrix = np.array([[1., 0.6], [0.4, 1.1]])
    k = matrix.T @ np.diag([480., 600.]) @ matrix
    d = matrix.T @ np.diag([8., 12.]) @ matrix
    mass = np.array([[1.7, 0.7], [0.7, 0.7]])  # Two coaxial rotors, inertia 1 and .7.
    props = JointProperties(["direct", "outer", "inner"], [480, *np.diag(k)], [8, *np.diag(d)],
                            [0]*3, [0]*3, [0]*3, [1e6]*3, [-0.01, 0.02, -0.015])

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
    initial = robot.data.default_joint_pos.torch.clone()
    initial[:, ids] = torch.tensor(props.standing, device=device)
    robot.write_joint_position_to_sim_index(position=initial)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial))
    scene.write_data_to_sim()
    scene.update(0.004)
    j = torch.tensor(matrix, device=device, dtype=torch.float32)
    kt, dt = torch.tensor(k, device=device, dtype=torch.float32), torch.tensor(d, device=device, dtype=torch.float32)
    k0, d0 = torch.diagonal(kt), torch.diagonal(dt)
    goal = torch.tensor([-0.01, 0.025], device=device)
    frames = []
    elapsed_ns = 0
    while elapsed_ns < 2_000_000_000:
        q = robot.data.joint_pos.torch[:, ids]
        v = robot.data.joint_vel.torch[:, ids]
        target = initial.clone()
        target[:, ids[0]] = 0.015  # Independent native implicit PD reference.
        force = 480 * (0.015 - q[1, 0]) - 8 * v[1, 0]
        target[1, ids[0]] = shaped_position_target(q[1, 0], v[1, 0], force, 480, 8)
        # Explicit per-crank law then transpose mapping, distinct from dense oracle.
        crank_force = torch.tensor([480., 600.], device=device) * (j @ (goal - q[2, 1:]))
        crank_force -= torch.tensor([8., 12.], device=device) * (j @ v[2, 1:])
        target[2, ids[1:]] = shaped_position_target(q[2, 1:], v[2, 1:], j.T @ crank_force, k0, d0)
        dense_force = kt @ (goal - q[3, 1:]) - dt @ v[3, 1:]
        target[3, ids[1:]] = q[3, 1:] + (dense_force + d0 * v[3, 1:]) / k0
        robot.actuators.target_command.set_position_index(value=target)
        robot.actuators.target_command.set_velocity_index(value=torch.zeros_like(target))
        scene.write_data_to_sim()
        context.step(render=False)
        scene.update(0.004)
        frames.append(robot.data.joint_pos.torch[:, ids].clone())
        elapsed_ns += 4_000_000
    frames = torch.stack(frames).cpu().numpy()
    a = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.linalg.solve(mass, k), -np.linalg.solve(mass, d)]])
    eigenvalues, eigenvectors = np.linalg.eig(a)
    target = np.array([-0.01, 0.025])
    coeff = np.linalg.solve(eigenvectors, np.r_[np.array([0.02, -0.015]) - target, np.zeros(2)])
    times = np.arange(1, len(frames)+1) * 0.004
    exact = target + np.real((np.exp(times[:, None] * eigenvalues) * coeff) @ eigenvectors.T)[:, :2]
    metrics = {"envs": 4, "direct_native_shaped_max_rad": float(np.max(np.abs(frames[:, 0, 0]-frames[:, 1, 0]))),
               "coupled_dense_parity_max_rad": float(np.max(np.abs(frames[:, 2, 1:]-frames[:, 3, 1:]))),
               "coupled_continuous_reference_max_rad": float(np.max(np.abs(frames[:, 2, 1:]-exact))),
               "coupled_final_error_rad": float(np.max(np.abs(frames[-1, 2, 1:]-target)))}
    (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(tmp_path / "traces.npz", actual=frames, continuous_reference=exact)
    print(json.dumps(metrics), flush=True)
    # Frozen before first execution; not a claim of exact continuous-time integration.
    assert np.isfinite(frames).all()
    assert metrics["direct_native_shaped_max_rad"] < 2e-6
    assert metrics["coupled_dense_parity_max_rad"] < 2e-6
    assert metrics["coupled_continuous_reference_max_rad"] < 0.003
    assert metrics["coupled_final_error_rad"] < 0.001
