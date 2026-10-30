"""Captured drive math for training; physics, sensors and task accounting remain with their owners."""
from __future__ import annotations

from dataclasses import fields

import torch

from humanoid_sim.core import PlantCore
from humanoid_sim.devices.feedback import DriveFeedback


class DriveStepGraph:
    """Feedback and target shaping on stable buffers, with a device nanosecond clock.

    Capture warmup preserves the queue, freeze latches and previous drive outputs. The
    feedback returned by :meth:`step` belongs to the preceding step, exactly as when
    ``RobotPlant`` samples feedback before advancing its drive bank. It is copied before
    ``PlantCore.pre_physics`` updates the bank's persistent output buffers.
    """

    def __init__(self, core: PlantCore, q: torch.Tensor, qd: torch.Tensor, *, compile: bool = False):
        if q.device.type != 'cuda':
            raise ValueError('DriveStepGraph requires CUDA')
        self.core = core
        self.q, self.qd = q.clone(), qd.clone()
        self.t_ns = torch.zeros((), dtype=torch.int64, device=q.device)
        bank, queue = core.bank, core.bank.queue
        mutable = [bank.frozen, bank.last_kind, queue.arrival, queue.count, queue.overflows,
                   *queue.active.values(), *queue.pending.values(), core.passive_effort]
        saved = [t.clone() for t in mutable]
        previous = None if core.last is None else [getattr(core.last, f.name).clone() for f in fields(core.last)]
        # Materialize PlantCore's persistent outputs before tracing; no allocation/rebinding
        # of these buffers may occur on the first replay.
        core.feedback(self.q, self.qd)
        core.pre_physics(self.t_ns, self.q, self.qd, core.post_state)
        last = [getattr(core.last, f.name) for f in fields(core.last)]
        if previous is None:
            previous = [torch.zeros_like(t) for t in last]
        mutable.extend(last)
        saved.extend(previous)
        run = torch.compile(self._run, dynamic=False, fullgraph=True) if compile else self._run
        side = torch.cuda.Stream(device=q.device)
        side.wait_stream(torch.cuda.current_stream(q.device))
        try:
            with torch.cuda.stream(side):
                for _ in range(3):
                    run()
            torch.cuda.current_stream(q.device).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                self.target, self.previous_feedback = run()
        finally:
            torch.cuda.current_stream(q.device).wait_stream(side)
            for dst, src in zip(mutable, saved, strict=True):
                dst.copy_(src)

    def _run(self) -> tuple[torch.Tensor, DriveFeedback]:
        fb = self.core.feedback(self.q, self.qd)
        previous = DriveFeedback(fb.position_rad.clone(), fb.velocity_rad_s.clone(),
                                 fb.applied_torque_nm.clone(), fb.demanded_torque_nm.clone())
        target = self.core.pre_physics(self.t_ns, self.q, self.qd, self.core.post_state)
        return target, previous

    def step(self, t_ns: int, q: torch.Tensor, qd: torch.Tensor) -> tuple[torch.Tensor, DriveFeedback]:
        self.q.copy_(q)
        self.qd.copy_(qd)
        self.t_ns.fill_(t_ns)
        self.graph.replay()
        return self.target, self.previous_feedback
