"""Draw :class:`~humanoid_sim.randomization.spec.ParamSpec` lists into per-env tensors, batched on the device.

Every parameter starts at its nominal ``[E, N]`` (``[E, N, W]`` for vector parameters). Startup specs apply
first, in order, giving the startup value; reset specs then apply on top of it, in order, for the envs being
reset. Draws use the plant's own generator, so a seed reproduces them.
"""

from __future__ import annotations

import math
import re

import torch

from humanoid_sim.randomization.spec import PARAMS, ParamSpec

AXES = ("x", "y", "z")


class Randomizer:
    def __init__(self, specs: list[ParamSpec], nominal: dict[str, torch.Tensor], names: dict[str, list[str]], *,
                 num_envs: int, device="cpu", seed: int = 0, level: float = 1.0):
        """``nominal``: every parameter the specs use, ``[E, N(, W)]``. ``names``: target names for ``body``,
        ``joint`` and ``drive`` (``env`` and ``axis`` are implicit)."""
        self.specs = list(specs)
        self.num_envs, self.device = num_envs, torch.device(device)
        self.nominal = {k: v.clone() for k, v in nominal.items()}
        self.names = {**names, "env": ["env"], "axis": list(AXES)}
        self._cols: list[torch.Tensor] = []
        for spec in self.specs:
            p = PARAMS[spec.param]
            if spec.param not in self.nominal:
                raise ValueError(f"no nominal value for {spec.param}")
            pool = self.names[p.target]
            cols = [i for i, n in enumerate(pool) if re.fullmatch(spec.select, n)]
            if not cols:
                raise ValueError(f"{spec.param}: {spec.select!r} selects no {p.target} of {pool}")
            want = (self.num_envs, len(pool)) + ((p.width,) if p.width > 1 else ())
            if tuple(self.nominal[spec.param].shape) != want:
                raise ValueError(f"{spec.param}: nominal shape {tuple(self.nominal[spec.param].shape)} != {want}")
            self._cols.append(torch.tensor(cols, device=self.device))
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.level = float(level)
        self.startup_values: dict[str, torch.Tensor] = {}
        self.values: dict[str, torch.Tensor] = {}
        self.draw_startup()

    @property
    def physics_params(self) -> list[str]:
        return sorted({s.param for s in self.specs if s.kind == "physics"})

    def draw_startup(self) -> None:
        """Startup draws for every env (and the reset draws on top). Called at construction and per level."""
        self.startup_values = {k: v.clone() for k, v in self.nominal.items()}
        everyone = torch.arange(self.num_envs, device=self.device)
        for spec, cols in zip(self.specs, self._cols, strict=True):
            if spec.startup:
                self._apply(spec, cols, self.startup_values, everyone)
        self.values = {k: v.clone() for k, v in self.startup_values.items()}
        self.draw_reset(everyone)

    def draw_reset(self, env_ids: torch.Tensor) -> None:
        """Redraw the reset-scope parameters of ``env_ids`` (device parameters only)."""
        touched = {s.param for s in self.specs if not s.startup}
        for k in touched:
            self.values[k][env_ids] = self.startup_values[k][env_ids]
        for spec, cols in zip(self.specs, self._cols, strict=True):
            if not spec.startup:
                self._apply(spec, cols, self.values, env_ids)

    def set_level(self, level: float) -> None:
        self.level = float(level)
        self.draw_startup()

    def _apply(self, spec: ParamSpec, cols: torch.Tensor, values: dict[str, torch.Tensor], env_ids: torch.Tensor):
        value = values[spec.param]
        width = value.shape[2:] if value.dim() == 3 else ()
        shape = (len(env_ids), len(cols) if spec.per_target else 1) + tuple(width)
        if spec.buckets is None:
            x = self._sample(spec, shape)
        else:
            table = self._sample(spec, (spec.buckets,))
            indices = torch.randint(spec.buckets, shape, generator=self.generator, device=self.device)
            x = table[indices]
        x = x.expand(len(env_ids), len(cols), *width)
        current = value[env_ids][:, cols]
        if spec.op == "scale":
            new = current * x
        elif spec.op == "add":
            new = current + x
        else:
            new = x.to(current.dtype).clone()
        rows = env_ids.unsqueeze(1)
        value[rows, cols.unsqueeze(0)] = new.to(value.dtype)

    def _sample(self, spec: ParamSpec, shape: tuple[int, ...]) -> torch.Tensor:
        lo, hi = spec.range
        u = torch.rand(shape, generator=self.generator, device=self.device, dtype=torch.float64)
        if spec.dist == "discrete_uniform":
            count = round((hi - lo) / spec.quantum) + 1
            x = lo + torch.floor(u * count) * spec.quantum
            centre = 0.5 * (lo + hi)
        elif spec.dist == "uniform":
            x = lo + (hi - lo) * u
            centre = 0.5 * (lo + hi)
        elif spec.dist == "log_uniform":
            x = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)
            centre = math.sqrt(lo * hi)
        else:
            x = lo + hi * torch.randn(shape, generator=self.generator, device=self.device, dtype=torch.float64)
            centre = lo
        if spec.levels:
            identity = {"scale": 1.0, "add": 0.0, "set": centre}[spec.op]
            x = identity + self.level * (x - identity)
        return x
