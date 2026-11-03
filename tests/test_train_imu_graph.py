"""Sensor capture must retain sample delivery, reset behavior and RNG progression."""
from dataclasses import fields

import pytest
import torch

from humanoid_sim.sensors.imu import ImuDelivery, ImuDeliveryCfg


@pytest.mark.physics
@pytest.mark.parametrize('compile', [False, True])
def test_training_imu_graph_matches_eager(compile):
    if not torch.cuda.is_available():
        pytest.skip('CUDA graphs need a GPU')
    from humanoid_sim.train.imu_step import ImuStepGraph

    cfg = ImuDeliveryCfg(sample_period_ns=800_000, burst_period_ns=(2_000_000, 4_000_000),
                         in_burst_spacing_ns=1000, gyro_noise_density=.01, accel_noise_density=.1,
                         gyro_turn_on_bias=.02, accel_turn_on_bias=.03,
                         gyro_bias_instability=.001, accel_bias_instability=.002)
    eager, captured = [ImuDelivery(cfg, 8, device='cuda:0', seed=42) for _ in range(2)]
    ids = torch.arange(8, device='cuda:0')
    for imu in (eager, captured):
        imu.reset(ids, 0)
    graph = ImuStepGraph(captured, 1_000_000, compile=compile)
    t_ns = 0
    for step in range(80):
        if step:
            t_ns += 2_500_000 if step == 35 else 1_000_000
        gyro = torch.full((8, 3), step * .02, device='cuda:0')
        acc = torch.full_like(gyro, step * .1)
        if step == 21:
            for imu in (eager, captured):
                imu.reset(ids[::2], t_ns)
                imu.gyro_bias[ids[::2]] = .03
                imu.misalignment[ids[::2]] = .01
        eager.step(t_ns, gyro, acc)
        graph.step(t_ns, gyro, acc)
        for name, value in vars(eager).items():
            if isinstance(value, torch.Tensor):
                tol = 3e-6 if compile and value.is_floating_point() else 0
                torch.testing.assert_close(getattr(captured, name), value, atol=tol, rtol=tol)
        for f in fields(eager.samples):
            value = getattr(eager.samples, f.name)
            tol = 3e-6 if compile and value.is_floating_point() else 0
            torch.testing.assert_close(getattr(captured.samples, f.name), value, atol=tol, rtol=tol)
        assert captured._prev_host == eager._prev_host
    torch.testing.assert_close(captured.generator.get_state(), eager.generator.get_state(), atol=0, rtol=0)
