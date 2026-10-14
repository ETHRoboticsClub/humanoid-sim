"""Per-env FIFO of records that take effect at an arrival time.

A drive demand leaves Controls at some time and reaches the drive one bus/drive latency later. The
latency is not a multiple of the physics step, so a demand usually lands *inside* a step. The queue
reports, for one step ``[t0, t1)``, every record that is in force during the step together with the
fraction of the step it covers. The device layer evaluates its force law once per record and blends the
results by those weights, which is exact for a force law evaluated at the step-start state.

Everything is batched over envs and stays on the device: no host syncs.
"""

from __future__ import annotations

import torch

_EMPTY = torch.iinfo(torch.int64).max


class ArrivalQueue:
    """Fixed-capacity, per-env FIFO of tensor records with integer-ns arrival stamps.

    Args:
        initial: the record in force before anything arrives, ``{field: tensor[E, ...]}``.
        capacity: pending records per env. Size it as ``ceil(max_latency / submit_period) + 1``.
            When an env's queue is full, its oldest pending record is dropped and ``overflows`` counts it.
    """

    def __init__(self, initial: dict[str, torch.Tensor], capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        first = next(iter(initial.values()))
        self.num_envs = first.shape[0]
        self.device = first.device
        self.capacity = capacity
        self.active = {k: v.clone() for k, v in initial.items()}
        self.pending = {k: v.unsqueeze(1).repeat_interleave(capacity, dim=1).clone() for k, v in initial.items()}
        self.arrival = torch.full((self.num_envs, capacity), _EMPTY, dtype=torch.int64, device=self.device)
        self.count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.overflows = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._slots = torch.arange(capacity, device=self.device)

    def reset(self, env_ids: torch.Tensor, record: dict[str, torch.Tensor]) -> None:
        """Clear the queue of ``env_ids`` and put ``record`` (rows for those envs) in force."""
        for k, v in record.items():
            self.active[k][env_ids] = v
        self.arrival[env_ids] = _EMPTY
        self.count[env_ids] = 0
        self.overflows[env_ids] = 0

    def submit(self, record: dict[str, torch.Tensor], arrival_ns: torch.Tensor,
               env_mask: torch.Tensor | None = None) -> None:
        """Queue one record per env (rows where ``env_mask`` is true) arriving at ``arrival_ns`` [E].

        Arrivals are kept in submission order: a record never overtakes an earlier one, so an arrival
        earlier than the last queued arrival is moved to it.
        """
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        full = (env_mask & (self.count >= self.capacity)).long()
        self._shift(full)  # drop the oldest pending record of full queues
        self.overflows += full
        last = torch.where(self.count > 0,
                           self.arrival.gather(1, (self.count - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                           torch.full_like(self.count, torch.iinfo(torch.int64).min))
        arrival = torch.maximum(arrival_ns.to(torch.int64), last)
        slot = self.count.clamp(max=self.capacity - 1)
        write = env_mask.unsqueeze(1) & (self._slots.unsqueeze(0) == slot.unsqueeze(1))
        self.arrival = torch.where(write, arrival.unsqueeze(1), self.arrival)
        for k, v in record.items():
            mask = write.reshape(write.shape + (1,) * (v.dim() - 1))
            self.pending[k] = torch.where(mask, v.unsqueeze(1), self.pending[k])
        self.count = self.count + env_mask.long()

    def weights(self, t0_ns: int, t1_ns: int) -> torch.Tensor:
        """Fraction of ``[t0, t1)`` covered by the active record and each pending slot, ``[E, 1 + K]``."""
        if t1_ns <= t0_ns:
            raise ValueError("empty step")
        c = self.arrival.clamp(min=t0_ns, max=t1_ns)
        edges = torch.cat([torch.full_like(c[:, :1], t0_ns), c, torch.full_like(c[:, :1], t1_ns)], dim=1)
        return (edges[:, 1:] - edges[:, :-1]).to(torch.float64).div(t1_ns - t0_ns)

    def records(self) -> dict[str, torch.Tensor]:
        """Active record followed by the pending slots, ``{field: [E, 1 + K, ...]}``, matching ``weights``."""
        return {k: torch.cat([self.active[k].unsqueeze(1), self.pending[k]], dim=1) for k in self.active}

    def advance(self, t1_ns: int) -> None:
        """Put every record that arrived by ``t1`` in force and drop it from the queue."""
        arrived = (self.arrival <= t1_ns).sum(dim=1)
        took = arrived > 0
        last = (arrived - 1).clamp(min=0)
        for k in self.active:
            src = self.pending[k][self._rows, last]
            mask = took.reshape((-1,) + (1,) * (src.dim() - 1))
            self.active[k] = torch.where(mask, src, self.active[k])
        self._shift(arrived)

    def blend(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum over the record axis (dim 1) of per-record values ``[E, 1 + K, ...]``."""
        w = weights.to(values.dtype).reshape(weights.shape + (1,) * (values.dim() - 2))
        return (values * w).sum(dim=1)

    def _shift(self, n: torch.Tensor) -> None:
        """Drop the first ``n[e]`` pending slots of each env."""
        src = self._slots.unsqueeze(0) + n.unsqueeze(1)
        valid = src < self.capacity
        src = src.clamp(max=self.capacity - 1)
        self.arrival = torch.where(valid, self.arrival.gather(1, src), torch.full_like(self.arrival, _EMPTY))
        for k, v in self.pending.items():
            idx = src.reshape(src.shape + (1,) * (v.dim() - 2)).expand_as(v)
            self.pending[k] = v.gather(1, idx)
        self.count = (self.count - n).clamp(min=0)
