"""Per-env FIFO of records that take effect at an arrival time.

A drive demand leaves Controls at some time and reaches the drive one bus/drive latency later. The
latency is not a multiple of the physics step, so a demand usually lands *inside* a step. The queue
reports, for one step ``[t0, t1)``, every record that is in force during the step together with the
fraction of the step it covers. The device layer evaluates its force law once per record and blends the
results by those weights, which is exact for a force law evaluated at the step-start state.

With ``lanes`` > 1 every field is ``[E, lanes, ...]`` and each lane has its own arrival time (a drive's
latency), so lane ``l`` of a record takes effect at ``arrival[e, k, l]``. A record stays queued until it has
arrived on every lane; a lane where it arrived already gives it the time from the step start.

Everything is batched over envs and stays on the device: no host syncs. State is updated in place and times
may be device tensors, so a step can be captured in a CUDA graph.
"""

from __future__ import annotations

import torch

_EMPTY = torch.iinfo(torch.int64).max


class ArrivalQueue:
    """Fixed-capacity, per-env FIFO of tensor records with integer-ns arrival stamps.

    Args:
        initial: the record in force before anything arrives, ``{field: tensor[E, ...]}`` (``[E, lanes, ...]``
            with lanes).
        capacity: pending records per env. Size it as ``ceil(max_latency / submit_period) + 1``.
            When an env's queue is full, its oldest pending record is dropped and ``overflows`` counts it.
        lanes: independent arrival times per record (1: one per env).
    """

    def __init__(self, initial: dict[str, torch.Tensor], capacity: int, lanes: int = 1):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        first = next(iter(initial.values()))
        self.num_envs = first.shape[0]
        self.device = first.device
        self.capacity, self.lanes = capacity, lanes
        if lanes > 1 and any(v.dim() < 2 or v.shape[1] != lanes for v in initial.values()):
            raise ValueError(f"with {lanes} lanes every field must be [E, {lanes}, ...]")
        self.active = {k: v.clone() for k, v in initial.items()}
        self.pending = {k: v.unsqueeze(1).repeat_interleave(capacity, dim=1).clone() for k, v in initial.items()}
        self.arrival = torch.full((self.num_envs, capacity, lanes), _EMPTY, dtype=torch.int64, device=self.device)
        self.count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.overflows = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._slots = torch.arange(capacity, device=self.device)

    def reset(self, env_ids: torch.Tensor, record: dict[str, torch.Tensor]) -> None:
        """Clear the queue of ``env_ids`` and put ``record`` (rows for those envs) in force. The emptied slots
        hold ``record`` too, so a reset queue is the same whatever was queued before it."""
        for k, v in record.items():
            self.active[k][env_ids] = v
            self.pending[k][env_ids] = v.unsqueeze(1)
        self.arrival.index_fill_(0, env_ids, _EMPTY)
        self.count.index_fill_(0, env_ids, 0)
        self.overflows.index_fill_(0, env_ids, 0)
    def submit(self, record: dict[str, torch.Tensor], arrival_ns: torch.Tensor,
               env_mask: torch.Tensor | None = None) -> None:
        """Queue one record per env (rows where ``env_mask`` is true) arriving at ``arrival_ns`` [E] (or
        ``[E, lanes]``).

        Arrivals are kept in submission order per lane: a record never overtakes an earlier one, so an
        arrival earlier than the last queued arrival is moved to it.
        """
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        arrival_ns = arrival_ns.to(torch.int64)
        if arrival_ns.dim() == 1:
            arrival_ns = arrival_ns.unsqueeze(1).expand(-1, self.lanes)
        full = (env_mask & (self.count >= self.capacity)).long()
        self._shift(full)  # drop the oldest pending record of full queues
        self.overflows += full
        idx = (self.count - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.lanes)
        last = torch.where((self.count > 0).unsqueeze(1), self.arrival.gather(1, idx).squeeze(1),
                           torch.full_like(arrival_ns, torch.iinfo(torch.int64).min))
        arrival = torch.maximum(arrival_ns, last)
        slot = self.count.clamp(max=self.capacity - 1)
        write = env_mask.unsqueeze(1) & (self._slots.unsqueeze(0) == slot.unsqueeze(1))
        self.arrival.copy_(torch.where(write.unsqueeze(2), arrival.unsqueeze(1), self.arrival))
        for k, v in record.items():
            mask = write.reshape(write.shape + (1,) * (v.dim() - 1))
            self.pending[k].copy_(torch.where(mask, v.unsqueeze(1), self.pending[k]))
        self.count.add_(env_mask.long())

    def weights(self, t0_ns: int | torch.Tensor, t1_ns: int | torch.Tensor) -> torch.Tensor:
        """Fraction of ``[t0, t1)`` covered by the active record and each pending slot, ``[E, 1 + K, lanes]``.
        Times are ints or 0-d int64 device tensors."""
        if isinstance(t0_ns, int) and isinstance(t1_ns, int) and t1_ns <= t0_ns:
            raise ValueError("empty step")
        c = self.arrival.clamp(min=t0_ns, max=t1_ns)
        zero = torch.zeros_like(c[:, :1])
        edges = torch.cat([zero + t0_ns, c, zero + t1_ns], dim=1)
        return (edges[:, 1:] - edges[:, :-1]).to(torch.float64) / (t1_ns - t0_ns)

    def records(self) -> dict[str, torch.Tensor]:
        """Active record followed by the pending slots, ``{field: [E, 1 + K, ...]}``, matching ``weights``."""
        return {k: torch.cat([self.active[k].unsqueeze(1), self.pending[k]], dim=1) for k in self.active}

    def advance(self, t1_ns: int | torch.Tensor) -> None:
        """Put every record that arrived by ``t1`` in force (per lane); drop those arrived on every lane."""
        arrived = (self.arrival <= t1_ns).sum(dim=1)  # [E, lanes]; arrivals ascend along the queue
        took = arrived > 0
        last = (arrived - 1).clamp(min=0)
        for k in self.active:
            if self.lanes == 1:
                src = self.pending[k][self._rows, last[:, 0]]
                mask = took[:, 0].reshape((-1,) + (1,) * (src.dim() - 1))
            else:
                v = self.pending[k]
                idx = last.reshape(last.shape[0], 1, self.lanes, *(1,) * (v.dim() - 3)).expand(
                    -1, 1, -1, *v.shape[3:])
                src = v.gather(1, idx).squeeze(1)
                mask = took.reshape(took.shape + (1,) * (src.dim() - 2))
            self.active[k].copy_(torch.where(mask, src, self.active[k]))
        self._shift(arrived.min(dim=1).values)

    def blend(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum over the record axis (dim 1) of per-record values ``[E, 1 + K, ...]``."""
        if self.lanes == 1:
            weights = weights[:, :, 0]
        w = weights.to(values.dtype).reshape(weights.shape + (1,) * (values.dim() - weights.dim()))
        return (values * w).sum(dim=1)

    def _shift(self, n: torch.Tensor) -> None:
        """Drop the first ``n[e]`` pending slots of each env."""
        src = self._slots.unsqueeze(0) + n.unsqueeze(1)
        valid = src < self.capacity
        src = src.clamp(max=self.capacity - 1)
        lane_src = src.unsqueeze(2).expand(-1, -1, self.lanes)
        self.arrival.copy_(torch.where(valid.unsqueeze(2), self.arrival.gather(1, lane_src),
                                       torch.full_like(self.arrival, _EMPTY)))
        for v in self.pending.values():
            idx = src.reshape(src.shape + (1,) * (v.dim() - 2)).expand_as(v)
            v.copy_(v.gather(1, idx))
        self.count.copy_((self.count - n).clamp(min=0))
