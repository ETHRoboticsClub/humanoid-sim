"""Angular momentum of the free-flying robot under fast joint motion. Runs in the Isaac Lab image.

Released high above the ground, the robot falls freely: gravity is uniform, so it puts no torque about the centre of
mass, and every other force (drive torques, joint friction, stops, loop closures) acts between the robot's own
bodies. Its angular momentum about the centre of mass therefore stays exactly zero from rest, whatever the joints
do; any that appears is the solver's. A first-order integrator (MJWarp's euler, implicit and implicitfast) leaves
several percent at 4 ms, which a hung robot shows as a spin about the vertical."""

import math

import pytest
import torch
from conftest import PHYSICS_ROBOTS, built_usd

from humanoid_sim.timing.clock import NS_PER_S

DURATION_NS = NS_PER_S
FREQUENCY_HZ = 3.0  # each 1:1 joint swings about its start position at this rate
MAX_AMPLITUDE_RAD = 0.5  # and by up to this much, inside its travel
TRAVEL_FRACTION = 0.5  # of the margin to its nearer end of travel
RELEASE_HEIGHT_M = 20.0  # it falls 5 m in a second: never near the ground
QUEUE_CAPACITY = 64  # one demand per step, held for the drive delay
# Allowed net angular momentum, as a fraction of the momentum the bodies carry about the centre of mass: RK4 at 4 ms
# leaves well under it, a first-order integrator at 4 ms several times it.
TOLERANCE = 0.01


def _matrix_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` of unit quaternions ``[..., 4]`` (x, y, z, w)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def angular_momentum(data) -> tuple[torch.Tensor, torch.Tensor]:
    """The robot's angular momentum about its centre of mass ``[E, 3]`` (world frame, kg m²/s), and the sum of its
    bodies' magnitudes ``[E]``: the momentum moving around inside the robot, the scale the net is measured against."""
    mass = data.body_mass.torch.double()  # [E, B]
    pos = data.body_com_pos_w.torch.double()  # [E, B, 3]
    vel = data.body_com_lin_vel_w.torch.double()
    omega = data.body_com_ang_vel_w.torch.double()
    rot = _matrix_xyzw(data.body_com_quat_w.torch.double())  # [E, B, 3, 3]
    inertia = data.body_inertia.torch.double().reshape(*mass.shape, 3, 3)  # about the COM, in the COM frame
    total = mass.sum(1, keepdim=True)
    com = (mass.unsqueeze(-1) * pos).sum(1, keepdim=True) / total.unsqueeze(-1)
    com_vel = (mass.unsqueeze(-1) * vel).sum(1, keepdim=True) / total.unsqueeze(-1)
    spin = rot @ inertia @ rot.transpose(-1, -2) @ omega.unsqueeze(-1)
    each = mass.unsqueeze(-1) * torch.linalg.cross(pos - com, vel - com_vel) + spin.squeeze(-1)
    return each.sum(1), each.norm(dim=-1).sum(1)


def swing_amplitudes(sim) -> torch.Tensor:
    """Per-joint swing amplitudes ``[E, J]``: every 1:1 drive's joint, alternating in sign so that the motion is
    asymmetric, by a share of the margin to its nearer end of travel; cranked joints hold still."""
    joints = list(sim.desc.joints)
    q0, _ = sim.joint_state()
    amplitude = torch.zeros_like(q0)
    for sign, drive in zip([1, -1] * len(sim.desc.drives), sim.desc.drives.values()):
        if len(drive.joints) != 1 or drive.crank is not None:
            continue
        j = joints.index(drive.joints[0])
        lo, hi = sim.desc.calibration_travel(drive.joints[0])
        margin = min(float(q0[0, j]) - lo, hi - float(q0[0, j]))
        amplitude[:, j] = sign * min(MAX_AMPLITUDE_RAD, TRAVEL_FRACTION * max(margin, 0.0))
    return amplitude


def swing_demand(sim, q0: torch.Tensor, amplitude: torch.Tensor):
    """The PD demand at the baseline gains that puts every swinging joint at its point of the swing now."""
    from humanoid_sim.devices.demand import DriveDemand

    p = sim.plant
    phase = 2 * math.pi * FREQUENCY_HZ * sim.t_ns / NS_PER_S
    target, _, _ = p.map.drive_state(q0 + amplitude * math.sin(phase), torch.zeros_like(q0))
    n = q0.shape[0]
    return DriveDemand.pd(target, p.drive_stiffness.expand(n, -1), p.drive_damping.expand(n, -1))


@pytest.mark.physics
@pytest.mark.parametrize("solver", ["mjwarp", "kamino"])
@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_free_fall_conserves_zero_angular_momentum(request, robot, solver):
    """From rest, every 1:1 joint swings fast about its start position while cranked joints hold theirs, at the
    profile's step and integrator: the robot's net angular momentum about its centre of mass stays zero to within a
    small fraction of what its bodies carry."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.sim import SimCore

    sim = SimCore(built_usd(robot), device="cpu", robot=robot, solver=solver, base_height_m=RELEASE_HEIGHT_M,
                  self_collision=False, queue_capacity=QUEUE_CAPACITY)
    request.addfinalizer(sim.close)
    data = sim.backend.robot.data
    q0, _ = sim.joint_state()
    amplitude = swing_amplitudes(sim)
    assert amplitude.abs().max() > 0
    worst, scale = 0.0, 0.0
    while sim.t_ns < DURATION_NS:
        sim.plant.submit(swing_demand(sim, q0, amplitude), torch.full((1,), sim.t_ns, dtype=torch.long))
        sim.step()
        net, carried = angular_momentum(data)
        assert torch.isfinite(net).all(), f"diverged at {sim.t_ns / NS_PER_S:.3f} s"
        worst, scale = max(worst, float(net.norm())), max(scale, float(carried[0]))
    assert scale > 0
    assert worst <= TOLERANCE * scale, (
        f"{sim.solver} {sim.integrator} at {sim.period_ns / 1e6:g} ms: net angular momentum reached {worst:.3e} "
        f"kg m²/s, {worst / scale:.3%} of the {scale:.3e} its bodies carry")
