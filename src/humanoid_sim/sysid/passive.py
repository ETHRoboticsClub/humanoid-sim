"""Fit passive joint parameters (viscous damping, Coulomb friction, armature) to excitation captures.

Each env of a batched replay (:func:`~humanoid_sim.replay.replay.replay_position_rms`) runs the same capture
with its own parameters, written to the solver before the run. A candidate is a vector of log multipliers on
start values, one per (joint group, parameter); a group ties joints that share a value (left and right). Start
values are the description's unless given (``FitSpace.start``, e.g. a datasheet armature); parameters not in
``FitSpace.params`` stay at their start values. The objective is the mean position RMS of the scored drives over the captures. CMA-ES with the
population as the number of envs; a candidate outside the multiplier bounds is evaluated at the bound and
penalized by its squared distance from it, so the search does not drift through the flat region beyond.

The description's values are the starting point and the result is reported against them; nothing is written
back here (robot facts land in humanoid-urdf).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

PASSIVE_PARAMS = ("damping", "friction", "armature")


@dataclass(frozen=True)
class Group:
    name: str
    joints: tuple[str, ...]


@dataclass
class FitSpace:
    groups: list[Group]
    params: tuple[str, ...] = PASSIVE_PARAMS  # fitted
    log_bound: float = math.log(10.0)  # multipliers in [1/10, 10]
    start: dict[str, float] = field(default_factory=dict)  # "group.param" -> value for every joint of the group

    def __post_init__(self):
        unknown = set(self.params) - set(PASSIVE_PARAMS)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}; known: {PASSIVE_PARAMS}")
        names = {f"{g.name}.{p}" for g in self.groups for p in PASSIVE_PARAMS}
        if set(self.start) - names:
            raise ValueError(f"start values for unknown group parameters {sorted(set(self.start) - names)}")

    def start_value(self, desc, group: Group, joint: str, param: str) -> float:
        from humanoid_sim.physics.joint_properties import joint_dynamics

        dynamics = joint_dynamics(desc)
        nominal = dynamics["viscous_damping" if param == "damping" else param][dynamics["joint_names"].index(joint)]
        return float(self.start.get(f"{group.name}.{param}", nominal))

    @property
    def names(self) -> list[str]:
        return [f"{g.name}.{p}" for g in self.groups for p in self.params]

    @property
    def dim(self) -> int:
        return len(self.groups) * len(self.params)

    def multipliers(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(x, -self.log_bound, self.log_bound))

    def out_of_bounds(self, x: np.ndarray) -> np.ndarray:
        """Squared distance of each candidate from the bounds (log units), ``[E]``."""
        return ((x - np.clip(x, -self.log_bound, self.log_bound)) ** 2).sum(axis=-1)


def _write(robot, desc, space: FitSpace, x: np.ndarray) -> None:
    """Write candidate ``x[e]`` to env ``e``: start value × multiplier per joint (start value alone for the
    parameters not fitted)."""
    m = space.multipliers(x)  # [E, G*P]
    e = m.shape[0]
    joints = [j for g in space.groups for j in g.joints]
    ids = torch.tensor([robot.joint_names.index(j) for j in joints], device=robot.device)
    cols = {p: [] for p in PASSIVE_PARAMS}
    nominal = {p: [] for p in PASSIVE_PARAMS}
    for gi, g in enumerate(space.groups):
        for p in PASSIVE_PARAMS:
            col = m[:, gi * len(space.params) + space.params.index(p)] if p in space.params else np.ones(e)
            cols[p] += [col] * len(g.joints)
            nominal[p] += [space.start_value(desc, g, j, p) for j in g.joints]
    for p in PASSIVE_PARAMS:
        if p not in space.params and not any(k.endswith(f".{p}") for k in space.start):
            continue  # neither fitted nor moved from the description: leave the solver's value
        value = torch.tensor(np.stack(cols[p], axis=1) * np.asarray(nominal[p])[None], dtype=torch.float32,
                             device=robot.device)
        if p == "damping":
            robot.write_joint_viscous_friction_coefficient_to_sim_index(joint_viscous_friction_coeff=value,
                                                                        joint_ids=ids)
        elif p == "friction":
            robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=value, joint_ids=ids)
        else:
            robot.write_joint_armature_to_sim_index(armature=value, joint_ids=ids)


def evaluate(sim, captures, space: FitSpace, x: np.ndarray, scored: list[int], *, base_motion: str | None,
             duration_s: float | None, demand_delay_ns: int = 0) -> np.ndarray:
    """Objective per candidate ``[E]`` and the per-capture, per-drive RMS ``[C, E, D]``."""
    from humanoid_sim.replay.replay import replay_position_rms

    _write(sim.backend.robot, sim.desc, space, x)
    delays = np.full(len(x), demand_delay_ns) if demand_delay_ns else None
    per = np.stack([replay_position_rms(c, sim, base_motion=base_motion, duration_s=duration_s,
                                        delays_ns=delays).cpu().numpy() for c in captures])
    return per[:, :, scored].mean(axis=(0, 2)), per


class Cmaes:
    """(μ/μ_w, λ)-CMA-ES (Hansen's tutorial defaults), minimizing."""

    def __init__(self, x0: np.ndarray, sigma: float, popsize: int, seed: int = 0, scale: np.ndarray | None = None):
        """``scale``: per coordinate, the initial step relative to ``sigma`` (the covariance starts diagonal at
        ``scale²``); all ones by default."""
        n = len(x0)
        self.n, self.lam, self.mean, self.sigma = n, popsize, np.asarray(x0, float), sigma
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C = np.eye(n) if scale is None else np.diag(np.asarray(scale, dtype=float) ** 2)
        self.rng = np.random.default_rng(seed)
        self.gen = 0

    def ask(self) -> np.ndarray:
        vals, vecs = np.linalg.eigh(self.C)
        self._B, self._D = vecs, np.sqrt(np.maximum(vals, 1e-20))
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = z * self._D @ self._B.T
        return self.mean + self.sigma * self._y

    def tell(self, x: np.ndarray, f: np.ndarray) -> None:
        """Rank ``x`` by ``f``. A non-finite objective (a candidate whose replay diverged) is never selected: with
        fewer finite candidates than ``mu`` the update uses only those, its weights renormalized; with none the
        step size halves and nothing else changes."""
        f = np.asarray(f, dtype=float)
        finite = np.isfinite(f)
        k = min(self.mu, int(finite.sum()))
        if k == 0:
            self.sigma *= 0.5
            return
        order = np.argsort(np.where(finite, f, np.inf))
        w = self.w[:k] / self.w[:k].sum()
        mueff = 1.0 / (w ** 2).sum()
        y = (x[order[:k]] - self.mean) / self.sigma
        yw = w @ y
        self.mean = self.mean + self.sigma * yw
        c_inv_sqrt = self._B @ np.diag(1 / self._D) @ self._B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * mueff) * c_inv_sqrt @ yw
        self.gen += 1
        hsig = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chin < 1.4 + 2 / (
            self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * mueff) * yw
        rank_mu = (w[:, None, None] * y[:, :, None] * y[:, None, :]).sum(0)
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)
                  + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))


BOUND_PENALTY = 1e-3  # rad of objective per squared log-unit outside the bounds


def fit(sim, captures, space: FitSpace, scored: list[int], *, generations: int, sigma: float = 0.7,
        base_motion: str | None = "yaw", duration_s: float | None = None, seed: int = 0, demand_delay_ns: int = 0,
        log=print) -> dict:
    """CMA-ES from the description's values (x = 0). Env 0 of every generation re-evaluates the best so far."""
    e = sim.plant.num_envs
    es = Cmaes(np.zeros(space.dim), sigma, e - 1, seed)
    x0 = np.zeros((e, space.dim))
    f0, per0 = evaluate(sim, captures, space, x0, scored, base_motion=base_motion, duration_s=duration_s,
                        demand_delay_ns=demand_delay_ns)
    best_x, best_f, best_per = np.zeros(space.dim), float(f0[0]), per0[:, 0]
    history = [{"generation": 0, "best": best_f, "median": best_f}]
    log(f"gen 0 description values: {best_f * 1e3:.2f} mrad", flush=True)
    for g in range(1, generations + 1):
        xs = es.ask()
        f, per = evaluate(sim, captures, space, np.vstack([best_x, xs]), scored, base_motion=base_motion,
                          duration_s=duration_s, demand_delay_ns=demand_delay_ns)
        es.tell(xs, f[1:] + BOUND_PENALTY * space.out_of_bounds(xs))
        i = int(np.argmin(f))
        if f[i] < best_f:
            x = best_x if i == 0 else np.clip(xs[i - 1], -space.log_bound, space.log_bound)
            best_x, best_f, best_per = x.copy(), float(f[i]), per[:, i]
        history.append({"generation": g, "best": best_f, "median": float(np.median(f[1:])), "sigma": es.sigma})
        log(f"gen {g}: best {best_f * 1e3:.2f} mrad, median {np.median(f[1:]) * 1e3:.2f}, sigma {es.sigma:.3f}",
            flush=True)
    return {"names": space.names, "x": best_x.tolist(), "multipliers": space.multipliers(best_x).tolist(),
            "objective": best_f, "objective_description": float(f0[0]), "history": history,
            "rms_description": per0[:, 0].tolist(), "rms_best": best_per.tolist()}
