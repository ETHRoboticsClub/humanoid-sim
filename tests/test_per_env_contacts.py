"""Per-env contact restitution (Kamino) and crank-stop stiffness (MJWarp) in the solver. Runs in the Isaac Lab image
on CUDA; the arithmetic and the support checks are in test_crank_stop_solref.py, Kamino's crank stops in
test_kamino_crank_stops_physics.py."""

import pytest
import torch
from conftest import built_usd

from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.randomization import ParamSpec

pytestmark = pytest.mark.physics
DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def cuda():
    pytest.importorskip("isaaclab_newton")
    if not torch.cuda.is_available():
        pytest.skip("solver parameters need CUDA")


def _sim(request, solver, num_envs, *, fixed_base):
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd("ethrc_v1"), device=DEVICE, num_envs=num_envs, fixed_base=fixed_base, solver=solver,
                  cuda_graph=False)
    request.addfinalizer(sim.close)
    return sim


def _plant(sim, specs):
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    return RobotPlant(sim.backend.robot, TrainPlantCfg(reporting=False, randomization=specs),
                      physics_manager=sim.backend.sim.physics_manager)


def _redraw(plant, **values):
    """Set the realized per-env values and write them to the solver, as a level change does."""
    for name, v in values.items():
        plant.randomizer.values[name] = torch.as_tensor(v, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
    plant._write_physics()


def test_kamino_takes_the_larger_restitution_of_a_contact(request):
    import warp as wp

    sim = _sim(request, "kamino", 2, fixed_base=False)
    plant = _plant(sim, [ParamSpec("restitution", "uniform", (0.2, 0.8), "set")])
    model = sim.backend.sim.physics_manager.get_model()
    shapes = wp.to_torch(model.shape_material_restitution)
    robot_shapes = shapes[shapes != 0]
    assert robot_shapes.numel() > 0
    realized = plant.params()["restitution"][:, 0]
    assert realized.min() >= 0.2 and realized.max() <= 0.8
    per_env = wp.to_torch(plant.robot._root_view.get_attribute("shape_material_restitution", model))[:, 0]
    torch.testing.assert_close(per_env, realized[:, None].expand_as(per_env))
    from isaaclab_newton.physics import NewtonManager

    assert NewtonManager._solver is not None
    materials = sim.backend.sim.cfg.physics.solver_cfg.materials
    assert (materials.friction_mix_mode, materials.restitution_mix_mode) == ("max", "max")


def test_restitution_makes_the_robot_rebound_on_kamino(request):
    sim = _sim(request, "kamino", 3, fixed_base=False)
    plant = _plant(sim, [ParamSpec("restitution", "uniform", (0.0, 1.0), "set")])
    _redraw(plant, restitution=[0.0, 0.5, 1.0])
    robot = sim.backend.robot
    root = robot.data.default_root_pose.torch.clone()
    root[:, 2] += 0.4
    robot.write_root_pose_to_sim_index(root_pose=root)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(3, 6, device=DEVICE))
    height = []
    for _ in range(400):
        plant.pre_physics()
        sim.backend.step()
        height.append(robot.data.root_pos_w.torch[:, 2].clone())
    z = torch.stack(height)
    after = z[z.argmin(0).min():]
    rebound = after.max(0).values - z.min(0).values
    assert rebound[2] > rebound[0] + 0.02


@pytest.mark.parametrize("solver", ["mjwarp"])
def test_mjwarp_has_no_restitution_to_randomize(request, solver):
    sim = _sim(request, solver, 2, fixed_base=True)
    with pytest.raises(ValueError, match="no contact restitution"):
        _plant(sim, [ParamSpec("restitution", "uniform", (0.0, 1.0), "set")])


def _crank_overshoot_rad(sim, plant, steps=150, past=0.3):
    """How far past its upper stop each env's upper crank sits while its drive pushes ``past`` rad beyond."""
    from humanoid_sim.model.usd import crank_joints

    drives = list(crank_joints(sim.desc))
    ids = [plant.drives.index(d) for d in drives]
    upper = torch.tensor([sim.desc.drives[d].position_limit_device[1] for d in drives], device=DEVICE)
    fb = plant.truth().drive
    target = fb.position_rad.clone()
    target[:, ids] = upper + past
    n = target.shape[0]
    plant.submit(DriveDemand.pd(target, plant.core.drive_stiffness, plant.core.drive_damping),
                 torch.zeros(n, dtype=torch.int64, device=DEVICE))
    for _ in range(steps):
        plant.pre_physics()
        sim.backend.step()
    return (plant.truth().drive.position_rad[:, ids] - upper).clamp(min=0)


def test_softer_crank_stops_give_way_further_on_mjwarp(request):
    sim = _sim(request, "mjwarp", 3, fixed_base=True)
    plant = _plant(sim, [ParamSpec("crank_stop_stiffness", "uniform", (0.05, 1.0), "scale"),
                         ParamSpec("crank_stop_damping", "uniform", (0.5, 2.0), "scale")])
    nominal = plant._stops_solref0
    _redraw(plant, crank_stop_stiffness=[1.0, 0.25, 0.05], crank_stop_damping=[1.0, 1.0, 1.0])
    written = plant._stops.read()
    torch.testing.assert_close(written[0], nominal[0])
    # Softer at a fixed damping is a larger damping ratio, the time constant staying put.
    torch.testing.assert_close(written[..., 0], nominal[..., 0])
    assert (written[2, :, 1] > written[1, :, 1]).all() and (written[1, :, 1] > written[0, :, 1]).all()
    over = _crank_overshoot_rad(sim, plant)
    assert (over[1] > over[0]).any() and (over[2] > over[1]).any()
