"""Capture the existing IMU delivery model without changing its random-number stream."""
from dataclasses import fields

import torch

from humanoid_sim.sensors.imu import ImuDelivery


class ImuStepGraph:
    """Fixed-period replay; irregular boundaries retain the ordinary sensor path."""

    def __init__(self, imu: ImuDelivery, period_ns: int, *, compile: bool = False):
        self.imu, self.period_ns = imu, period_ns
        self.t_ns = torch.zeros((), dtype=torch.int64, device=imu.device)
        self.gyro = torch.zeros(imu.num_envs, 3, dtype=imu.dtype, device=imu.device)
        self.acc = torch.zeros_like(self.gyro)
        state = [v for v in vars(imu).values() if isinstance(v, torch.Tensor)]
        state.extend(getattr(imu.samples, f.name) for f in fields(imu.samples))
        saved = [v.clone() for v in state]
        rng = imu.generator.get_state()
        run = torch.compile(self._run, dynamic=False) if compile else self._run
        side = torch.cuda.Stream(device=imu.device)
        side.wait_stream(torch.cuda.current_stream(imu.device))
        try:
            with torch.cuda.stream(side):
                for _ in range(3):
                    run()
            torch.cuda.current_stream(imu.device).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            self.graph.register_generator_state(imu.generator)
            with torch.cuda.graph(self.graph, stream=side):
                run()
        finally:
            torch.cuda.current_stream(imu.device).wait_stream(side)
            for dst, src in zip(state, saved, strict=True):
                dst.copy_(src)
            imu.generator.set_state(rng)

    def _run(self):
        self.imu.step(self.t_ns, self.gyro, self.acc, period_ns=self.period_ns)

    def step(self, t_ns: int, gyro: torch.Tensor, acc: torch.Tensor):
        if t_ns - self.imu._prev_host != self.period_ns:
            self.imu.step(t_ns, gyro, acc)
            return
        self.t_ns.fill_(t_ns)
        self.gyro.copy_(gyro)
        self.acc.copy_(acc)
        self.graph.replay()
        self.imu._prev_host = t_ns
