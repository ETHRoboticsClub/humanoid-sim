"""Captured training drive math preserves arrivals, previous feedback and partial resets."""
from dataclasses import fields

import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import LAW_FREEZE, DriveDemand
from humanoid_sim.model.description import Description
from humanoid_sim.transmissions.drive_map import native_joint_names


@pytest.mark.physics
@pytest.mark.parametrize('compile', [False, True])
@pytest.mark.parametrize('queued_before_capture', [False, True])
def test_training_drive_graph_matches_eager(compile, queued_before_capture):
    if not torch.cuda.is_available():
        pytest.skip('CUDA graphs need a GPU')
    from humanoid_sim.train.drive_step import DriveStepGraph

    desc = Description.load()
    names = native_joint_names(desc)
    period_ns = 1_000_000
    cfg = PlantCfg(device='cuda:0', physics_period_ns=period_ns, queue_capacity=8)
    eager, captured = [PlantCore(desc, list(desc.joints), 4, cfg, solver_joint_names=names) for _ in range(2)]
    q = torch.zeros(4, len(names), device='cuda:0')
    qd = torch.zeros_like(q)
    if queued_before_capture:
        for core in (eager, captured):
            target, _, _ = core.solver_map.drive_state(q, qd)
            demand = DriveDemand.pd(target + .02, core.drive_stiffness.expand(4, -1),
                                    core.drive_damping.expand(4, -1))
            core.submit(demand, torch.zeros(4, device=q.device, dtype=torch.int64))
            core.pre_physics(0, q, qd)
            demand.law[:, 0] = LAW_FREEZE
            core.submit(demand, torch.full((4,), 2_500_000, device=q.device, dtype=torch.int64))
    graph = DriveStepGraph(captured, q, qd, compile=compile)
    for step in range(40):
        t_ns = step * period_ns
        q.fill_(.01 * (step % 3))
        qd.fill_(.02 * (step % 5))
        if step == 19:
            ids = torch.tensor([1, 3], device=q.device)
            for core in (eager, captured):
                core.reset(ids)
                core.bank.p_scale[ids] = .9
                core.bank.d_scale[ids] = 1.1
                core.bank.position_offset[ids] = .007
        if step % 4 == 0:
            for core in (eager, captured):
                target, _, _ = core.solver_map.drive_state(q, qd)
                demand = DriveDemand.pd(target + .015, core.drive_stiffness.expand(4, -1),
                                        core.drive_damping.expand(4, -1))
                if step % 12 == 0:
                    demand.law[:, 0] = LAW_FREEZE
                arrival = torch.arange(target.shape[1], device=q.device).expand(4, -1) % 3 * 750_000 + t_ns
                core.submit(demand, arrival)
        fb = eager.feedback(q, qd)
        expected_fb = {f.name: getattr(fb, f.name).clone() for f in fields(fb)}
        expected = eager.pre_physics(t_ns, q, qd, eager.post_state)
        actual, previous = graph.step(t_ns, q, qd)
        tol = 3e-5 if compile else 1e-6
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        for f in fields(previous):
            torch.testing.assert_close(getattr(previous, f.name), expected_fb[f.name], atol=tol, rtol=tol)
        for name in ('applied', 'demanded', 'enabled_fraction', 'target_velocity'):
            torch.testing.assert_close(getattr(captured.last, name), getattr(eager.last, name), atol=tol, rtol=tol)
        for name in ('arrival', 'count', 'overflows'):
            torch.testing.assert_close(getattr(captured.bank.queue, name), getattr(eager.bank.queue, name), atol=0, rtol=0)


def test_compile_requires_explicit_graph_execution():
    from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg

    with pytest.raises(ValueError, match='compile requires cuda_graph'):
        RobotPlant(None, TrainPlantCfg(compile=True))


def test_drive_graph_rejects_cpu_without_falling_back():
    from humanoid_sim.train.drive_step import DriveStepGraph

    q = torch.zeros(1, 1)
    with pytest.raises(ValueError, match='requires CUDA'):
        DriveStepGraph(None, q, q)
