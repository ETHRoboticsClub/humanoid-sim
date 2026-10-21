"""PACE's actuator model (leggedrobotics/pace-sim2real 0.2.0), fitted on this plant by batched replay.

The parameters, in PACE's order with ``separate_dynamic_friction``:

* ``armature`` [J], kg m²: reflected rotor inertia;
* ``damping`` [J], N m s/rad: viscous joint friction;
* ``friction`` [J], N m: static Coulomb friction (the solver's);
* ``bias`` [D], rad: encoder bias, PACE's sign: the drive reads its truth minus the bias (``position_offset =
  -bias``). Per drive, so the ankles' entries are their cranks';
* ``delay``, one for every drive: from a command leaving the master to the drive acting on it (the plant's
  ``latency``), in ns here and in physics steps in PACE's files (``delay_mode="command"``);
* ``friction_ratio`` [J]: dynamic over static friction, in [0, 1] (:attr:`~humanoid_sim.core.PlantCore.friction_relief`).

A candidate is a point of ``[-1, 1]^dim``: ``start · bound^x`` for armature, damping and friction (start values
from the description), linear between its bounds for the others. The search starts at :meth:`PaceSpace.x0`: the
description's values (its friction ratio, 1 where it states no dynamic friction), zero bias and the middle of the
delay range; a fixed entry stays there. A candidate outside the box is evaluated at its edge and
penalized by its squared distance from it, as in :mod:`humanoid_sim.sysid.passive`.

Nothing is written back: the values are robot facts and land in humanoid-urdf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

PACE_VERSION = "0.2.0"
JOINT_PARAMS = ("armature", "damping", "friction")
LOG_PARAMS = JOINT_PARAMS


@dataclass
class PaceSpace:
    joints: list[str]  # the plant's joint order
    drives: list[str]  # the plant's drive order
    start: dict[str, np.ndarray]  # armature, damping, friction [J]; friction_ratio [J] (default 1)
    log_bound: float = math.log(10.0)  # multipliers on the start values in [1/10, 10]
    bias_bound_rad: float = 0.02
    delay_bounds_ns: tuple[int, int] = (0, 20_000_000)
    ratio_bounds: tuple[float, float] = (0.0, 1.0)
    separate_dynamic_friction: bool = True  # PACE's flag: without it the ratio is 1 and not a parameter
    fixed: dict[str, np.ndarray] = field(default_factory=dict)  # parameter -> [n] bool: held at its start

    def __post_init__(self):
        j = len(self.joints)
        for p in JOINT_PARAMS:
            if np.asarray(self.start[p]).shape != (j,):
                raise ValueError(f"start {p}: {j} values expected")
            if (np.asarray(self.start[p]) <= 0).any():
                raise ValueError(f"start {p} must be positive (a multiplier cannot move zero)")
        self.start = dict(self.start)
        self.start.setdefault("friction_ratio", np.ones(j))
        lo, hi = self.ratio_bounds
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f"friction ratio bounds {self.ratio_bounds}: need 0 <= low <= high <= 1")
        if self.delay_bounds_ns[0] > self.delay_bounds_ns[1]:
            raise ValueError(f"delay bounds {self.delay_bounds_ns}")
        for name, mask in self.fixed.items():
            if name not in self.sizes or np.asarray(mask).shape != (self.sizes[name],):
                raise ValueError(f"fixed {name!r}: a mask over one of {sorted(self.sizes)}")

    @property
    def sizes(self) -> dict[str, int]:
        j, d = len(self.joints), len(self.drives)
        out = {"armature": j, "damping": j, "friction": j, "bias": d, "delay": 1}
        return out | ({"friction_ratio": j} if self.separate_dynamic_friction else {})

    @property
    def dim(self) -> int:
        return sum(self.sizes.values())

    def slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for name, n in self.sizes.items():
            out[name] = slice(i, i + n)
            i += n
        return out

    @property
    def names(self) -> list[str]:
        labels = {"bias": self.drives, "delay": ["all"]}
        return [f"{p}.{n}" for p in self.sizes for n in labels.get(p, self.joints)]

    def x0(self) -> np.ndarray:
        x = np.zeros(self.dim)
        if self.separate_dynamic_friction:
            lo, hi = self.ratio_bounds
            r = np.clip(np.asarray(self.start["friction_ratio"], dtype=np.float64), lo, hi)
            x[self.slices()["friction_ratio"]] = 2 * (r - lo) / (hi - lo) - 1 if hi > lo else 0.0
        return x

    def out_of_bounds(self, x: np.ndarray) -> np.ndarray:
        return ((x - np.clip(x, -1.0, 1.0)) ** 2).sum(axis=-1)

    def decode(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Values of candidates ``x`` ``[E, dim]``: ``[E, n]`` per parameter, ``delay`` ``[E]`` (int ns)."""
        x = np.clip(np.array(x, dtype=np.float64, ndmin=2), -1.0, 1.0)
        s, x0 = self.slices(), self.x0()
        for name, mask in self.fixed.items():
            idx = np.arange(self.dim)[s[name]][np.asarray(mask, dtype=bool)]
            x[:, idx] = x0[idx]
        out = {p: np.asarray(self.start[p])[None] * np.exp(x[:, s[p]] * self.log_bound) for p in LOG_PARAMS}
        out["bias"] = x[:, s["bias"]] * self.bias_bound_rad
        lo, hi = self.delay_bounds_ns
        out["delay"] = np.round(lo + (x[:, s["delay"]][:, 0] + 1) / 2 * (hi - lo)).astype(np.int64)
        if self.separate_dynamic_friction:
            lo, hi = self.ratio_bounds
            out["friction_ratio"] = lo + (x[:, s["friction_ratio"]] + 1) / 2 * (hi - lo)
        else:
            out["friction_ratio"] = np.ones((len(x), len(self.joints)))
        return out

    def bounds(self, period_ns: int) -> np.ndarray:
        """PACE's ``bounds_params`` ``[dim, 2]``: absolute values, the delay in physics steps of ``period_ns``."""
        rows = []
        for p in LOG_PARAMS:
            start = np.asarray(self.start[p])
            rows += list(zip(start * math.exp(-self.log_bound), start * math.exp(self.log_bound), strict=True))
        rows += [(-self.bias_bound_rad, self.bias_bound_rad)] * len(self.drives)
        rows.append(tuple(b / period_ns for b in self.delay_bounds_ns))
        if self.separate_dynamic_friction:
            rows += [self.ratio_bounds] * len(self.joints)
        return np.asarray(rows, dtype=np.float64)

    def pace_params(self, values: dict[str, np.ndarray], period_ns: int) -> np.ndarray:
        """One candidate's values (``decode(x)`` row) as PACE's parameter vector, the delay in whole physics steps
        (PACE's delay buffer takes whole steps; the exact delay is in ``fit-pace.json``)."""
        ratio = [values["friction_ratio"]] if self.separate_dynamic_friction else []
        return np.concatenate([values["armature"], values["damping"], values["friction"], values["bias"],
                               [round(int(values["delay"]) / period_ns)], *ratio])


def write(sim, space: PaceSpace, values: dict[str, np.ndarray]) -> None:
    """Write every env's candidate (``decode`` output, ``[E, ...]``) to the solver and the plant. The delay is the
    replay's to apply (:func:`humanoid_sim.replay.replay.replay_position_rms` ``latency_ns``)."""
    robot, plant = sim.backend.robot, sim.plant
    if list(plant.map.joint_names) != space.joints or list(plant.map.drive_names) != space.drives:
        raise ValueError("the space's joints and drives are not the plant's")
    dev = robot.device
    ids = torch.tensor([robot.joint_names.index(j) for j in space.joints], device=dev)

    def t(v):
        return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev)

    robot.write_joint_armature_to_sim_index(armature=t(values["armature"]), joint_ids=ids)
    robot.write_joint_viscous_friction_coefficient_to_sim_index(joint_viscous_friction_coeff=t(values["damping"]),
                                                                joint_ids=ids)
    robot.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=t(values["friction"]), joint_ids=ids)
    relief = values["friction"] * (1.0 - values["friction_ratio"])
    plant.friction_relief.copy_(torch.as_tensor(relief, dtype=plant.cfg.dtype, device=plant.cfg.device))
    plant.bank.position_offset.copy_(torch.as_tensor(-values["bias"], dtype=plant.cfg.dtype, device=plant.cfg.device))


def start_values(desc, joints: list[str]) -> dict[str, np.ndarray]:
    """The description's armature, damping, static friction and friction ratio (dynamic over static)."""
    out = {p: np.array([getattr(desc.joints[j], p) for j in joints]) for p in JOINT_PARAMS}
    dynamic = np.array([desc.joints[j].dynamic_friction for j in joints])
    out["friction_ratio"] = np.where(out["friction"] > 0, dynamic / np.maximum(out["friction"], 1e-12), 1.0)
    return out


def evaluate(sim, captures, space: PaceSpace, x: np.ndarray, scored: list[int], *, base_motion: str | None,
             duration_s: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Objective per candidate ``[E]`` (mean position RMS over captures and scored drives, rad) and the
    per-capture, per-drive RMS ``[C, E, D]``. Every capture is replayed from its command frames."""
    from humanoid_sim.replay.replay import replay_position_rms

    values = space.decode(x)
    write(sim, space, values)
    per = np.stack([replay_position_rms(c, sim, base_motion=base_motion, duration_s=duration_s, source="command",
                                        latency_ns=values["delay"]).cpu().numpy() for c in captures])
    return per[:, :, scored].mean(axis=(0, 2)), per


SEARCHES = ("blocks", "joint")


def blocks(space: PaceSpace, plant, scored: list[int]) -> list[tuple[np.ndarray, list[int]]]:
    """The search's blocks: ``(entries of x, drives scoring them)``. A 1:1 drive's block is its joint's armature,
    damping, friction and ratio and its own bias, scored on that drive; an ankle's, both hinges' and both cranks',
    scored on the two cranks; the delay's, scored on every scored drive. Fixed entries and blocks without a scored
    drive are left out."""
    s = space.slices()
    per_joint = [p for p in (*JOINT_PARAMS, "friction_ratio") if p in s]
    fixed = np.zeros(space.dim, dtype=bool)
    for name, mask in space.fixed.items():
        fixed[np.arange(space.dim)[s[name]][np.asarray(mask, dtype=bool)]] = True

    def entries(js, ds):
        idx = [s[p].start + j for j in js for p in per_joint] + [s["bias"].start + d for d in ds]
        return np.array([i for i in idx if not fixed[i]], dtype=np.int64)

    groups = [([j], [d]) for d, j in zip(plant.map._lin_d.tolist(), plant.map._lin_j.tolist(), strict=True)]
    groups += [(j_idx.tolist(), d_idx.tolist()) for _, d_idx, j_idx, _ in plant.map.ankles]
    out = []
    for js, ds in groups:
        ds = [d for d in ds if d in scored]
        idx = entries(js, ds)
        if ds and idx.size:
            out.append((idx, ds))
    out.append((np.array([s["delay"].start], dtype=np.int64), list(scored)))
    return out


def fit(sim, captures, space: PaceSpace, scored: list[int], *, generations: int, sigma: float = 0.3,
        base_motion: str | None = "yaw", duration_s: float | None = None, seed: int = 0, search: str = "blocks",
        log=print) -> dict:
    """CMA-ES from the start values. ``search="joint"``: one CMA-ES over every parameter on the objective, as PACE
    searches; env 0 of every generation re-evaluates the best so far. ``search="blocks"``: one CMA-ES per block
    (:func:`blocks`), each ranked by its own drives' error, all sharing every generation's envs: on a fixed base a
    drive's error depends mostly on its own joint, so the small searches converge in far fewer generations than
    one over all parameters. Env 0 evaluates the blocks' means."""
    if search not in SEARCHES:
        raise ValueError(f"search {search!r}: one of {SEARCHES}")
    if sim.plant.num_envs < 3:
        raise ValueError("the fit needs at least 3 envs (the best so far and a population of 2)")
    if search == "blocks":
        return _fit_blocks(sim, captures, space, scored, generations=generations, sigma=sigma,
                           base_motion=base_motion, duration_s=duration_s, seed=seed, log=log)
    from humanoid_sim.sysid.passive import BOUND_PENALTY, Cmaes

    e = sim.plant.num_envs
    es = Cmaes(space.x0(), sigma, e - 1, seed)
    f0, per0 = evaluate(sim, captures, space, np.tile(space.x0(), (e, 1)), scored, base_motion=base_motion,
                        duration_s=duration_s)
    best_x, best_f, best_per = space.x0(), float(f0[0]), per0[:, 0]
    history = [{"generation": 0, "best": best_f, "median": best_f}]
    log(f"gen 0 start values: {best_f * 1e3:.3f} mrad", flush=True)
    for g in range(1, generations + 1):
        xs = es.ask()
        f, per = evaluate(sim, captures, space, np.vstack([best_x, xs]), scored, base_motion=base_motion,
                          duration_s=duration_s)
        es.tell(xs, f[1:] + BOUND_PENALTY * space.out_of_bounds(xs))
        i = int(np.argmin(f))
        if f[i] < best_f:
            x = best_x if i == 0 else np.clip(xs[i - 1], -1.0, 1.0)
            best_x, best_f, best_per = x.copy(), float(f[i]), per[:, i]
        history.append({"generation": g, "best": best_f, "median": float(np.median(f[1:])), "sigma": es.sigma})
        log(f"gen {g}: best {best_f * 1e3:.3f} mrad, median {np.median(f[1:]) * 1e3:.3f}, sigma {es.sigma:.3f}",
            flush=True)
    return {"x": best_x.tolist(), "objective": best_f, "objective_start": float(f0[0]), "history": history,
            "rms_start": per0[:, 0].tolist(), "rms_best": best_per.tolist(), "search": search}


def _fit_blocks(sim, captures, space: PaceSpace, scored: list[int], *, generations: int, sigma: float,
                base_motion: str | None, duration_s: float | None, seed: int, log) -> dict:
    from humanoid_sim.sysid.passive import BOUND_PENALTY, Cmaes

    e = sim.plant.num_envs
    parts = blocks(space, sim.plant, scored)
    searches = [Cmaes(space.x0()[idx], sigma, e - 1, seed + b) for b, (idx, _) in enumerate(parts)]

    def means():
        x = space.x0()
        for (idx, _), es in zip(parts, searches, strict=True):
            x[idx] = np.clip(es.mean, -1.0, 1.0)
        return x

    def run(x):
        return evaluate(sim, captures, space, x, scored, base_motion=base_motion, duration_s=duration_s)

    f0, per0 = run(np.tile(space.x0(), (e, 1)))
    best_x, best_f, best_per = space.x0(), float(f0[0]), per0[:, 0]
    history = [{"generation": 0, "best": best_f, "median": best_f}]
    log(f"gen 0 start values: {best_f * 1e3:.3f} mrad ({len(parts)} blocks)", flush=True)
    for g in range(1, generations + 1):
        x = np.tile(means(), (e, 1))
        asked = []
        for (idx, _), es in zip(parts, searches, strict=True):
            xs = es.ask()
            x[1:, idx] = xs
            asked.append(xs)
        f, per = run(x)
        rms = per.mean(axis=0)  # [E, D] over the captures
        for (idx, ds), es, xs in zip(parts, searches, asked, strict=True):
            fb = rms[1:, ds].mean(axis=1)
            es.tell(xs, fb + BOUND_PENALTY * ((xs - np.clip(xs, -1.0, 1.0)) ** 2).sum(axis=1))
        if f[0] < best_f:
            best_x, best_f, best_per = x[0].copy(), float(f[0]), per[:, 0]
        history.append({"generation": g, "best": best_f, "means": float(f[0]), "median": float(np.median(f[1:]))})
        log(f"gen {g}: means {f[0] * 1e3:.3f} mrad, best {best_f * 1e3:.3f}, median {np.median(f[1:]) * 1e3:.3f}",
            flush=True)
    f, per = run(np.tile(means(), (e, 1)))
    if f[0] < best_f:
        best_x, best_f, best_per = means(), float(f[0]), per[:, 0]
    return {"x": best_x.tolist(), "objective": best_f, "objective_start": float(f0[0]), "history": history,
            "rms_start": per0[:, 0].tolist(), "rms_best": best_per.tolist(), "search": "blocks",
            "blocks": [{"entries": [space.names[i] for i in idx], "drives": [space.drives[d] for d in ds]}
                       for idx, ds in parts]}


def save(out, space: PaceSpace, best: dict[str, np.ndarray], plant, period_ns: int, captures, meta: dict) -> None:
    """``out/``: ``fit-pace.json`` (every value in SI, the delay in ns, with the run's provenance) and PACE 0.2.0's
    own files, ``config.pt`` and ``mean_final.pt`` (``CMAESOptimizer.save_checkpoint``'s parameter vector), the
    delay in physics steps of ``period_ns``. PACE's ``joint_order`` is the plant's joints, the order of the
    per-joint parameters; the bias entries and the trajectory columns are per drive, in ``drive_order`` (an
    ankle's two are its cranks')."""
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    params = space.pace_params(best, period_ns)
    t, des, meas = _pace_data(captures[0], plant, period_ns)
    torch.save({"bounds": torch.as_tensor(space.bounds(period_ns)), "joint_order": space.joints,
                "drive_order": space.drives, "dof_pos": meas, "des_dof_pos": des, "time": t,
                "separate_dynamic_friction": space.separate_dynamic_friction, "delay_mode": "command",
                "sim_dt": period_ns / 1e9, "pace_version": PACE_VERSION}, out / "config.pt")
    torch.save(torch.as_tensor(params), out / "mean_final.pt")
    doc = {"pace_version": PACE_VERSION, "delay_mode": "command",
           "separate_dynamic_friction": space.separate_dynamic_friction, "period_ns": period_ns,
           "joint_order": space.joints, "drive_order": space.drives,
           "values": {"joints": {j: {p: float(best[p][i]) for p in (*JOINT_PARAMS, "friction_ratio")}
                                 | {"dynamic_friction": float(best["friction"][i] * best["friction_ratio"][i])}
                                 for i, j in enumerate(space.joints)},
                      "drives": {d: {"bias_rad": float(best["bias"][i])} for i, d in enumerate(space.drives)},
                      "delay_ns": int(best["delay"])},
           "pace_params": params.tolist(), "pace_names": space.names, "bounds": space.bounds(period_ns).tolist(),
           **meta}
    import json

    (out / "fit-pace.json").write_text(json.dumps(doc, indent=1))


def _pace_data(capture, plant, period_ns: int):
    """PACE's trajectory tensors of one capture, one row per physics step as its ``fit.py`` steps them: time from
    the first command frame (s), the targets in force (zero-order hold of the frames) and the drives' readings
    ``[T, D]``, drive order."""
    from humanoid_sim.replay.exc import sample_hold, sample_linear
    from humanoid_sim.replay.replay import drive_targets

    targets, t_cmd = drive_targets(plant, capture.commands())
    pdo = capture.pdo()
    t_end = min(int(t_cmd[-1]), capture.time_span_ns()[1])
    t = np.arange(int(t_cmd[0]), t_end + 1, period_ns, dtype=np.int64)
    des = np.stack([sample_hold(t_cmd, targets[:, i], t) for i in range(targets.shape[1])], axis=1)
    meas = np.stack([sample_linear(pdo[d].t_ns, pdo[d]["native_position_rad"], t) for d in plant.map.drive_names],
                    axis=1)
    return torch.as_tensor((t - t[0]) / 1e9), torch.as_tensor(des), torch.as_tensor(meas)


@dataclass
class SyntheticCapture:
    """A capture the plant recorded itself (:func:`synthetic_capture`): the interface the replay reads."""

    dataset_id: str
    name: str
    _commands: object
    _pdo: dict
    _gains: dict

    def commands(self):
        return self._commands

    def pdo(self):
        return self._pdo

    def gains(self):
        return self._gains

    def imu(self):
        raise ValueError("a synthetic capture has no IMU: replay it on a fixed base")

    def time_span_ns(self) -> tuple[int, int]:
        traces = self._pdo.values()
        return max(int(tr.t_ns[0]) for tr in traces), min(int(tr.t_ns[-1]) for tr in traces)


def chirp_commands(joints: list[str], *, duration_s: float, f0_hz: float, f1_hz: float, amplitude_rad: dict,
                   frame_period_ns: int = 20_000_000, t0_ns: int = 1_000_000_000_000_000_000, seed: int = 0):
    """Semantic command frames: every joint sweeps ``f0 → f1`` (linear chirp) about the standing pose with its
    ``amplitude_rad[joint]`` and a random phase, held at zero for the first second and starting from zero."""
    from humanoid_sim.replay.exc import CommandTrace

    rng = np.random.default_rng(seed)
    n = int(duration_s * 1e9 // frame_period_ns) + 1
    t = np.arange(n) * frame_period_ns / 1e9
    run = np.clip(t - 1.0, 0.0, None)
    span = max(duration_s - 1.0, 1e-9)
    phase = 2 * np.pi * (f0_hz * run + (f1_hz - f0_hz) * run ** 2 / (2 * span))
    q = np.stack([amplitude_rad[j] * (np.sin(phase + phi) - np.sin(phi)) * (t >= 1.0)
                  for j, phi in zip(joints, rng.uniform(0, 2 * np.pi, len(joints)), strict=True)], axis=1)
    return CommandTrace(t0_ns + np.arange(n, dtype=np.int64) * frame_period_ns, list(joints), q)


def synthetic_capture(sim, space: PaceSpace, values: dict[str, np.ndarray], commands) -> SyntheticCapture:
    """Run ``commands`` on a fixed base with ``values`` (``decode`` output, every env the same) at the baseline
    gains, and record what the drives would: readings (bias included), velocity, torque and the target in force
    as the position demand, one sample per physics step."""
    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.replay.exc import DriveGains, DriveTrace
    from humanoid_sim.replay.replay import drive_targets, initial_joint_state

    plant = sim.plant
    e, dev, dt = plant.num_envs, plant.cfg.device, plant.cfg.dtype
    write(sim, space, values)
    targets, t_cmd = drive_targets(plant, commands)
    rel = t_cmd - t_cmd[0]
    tg = torch.as_tensor(targets, dtype=dt, device=dev)
    kp, kd = plant.drive_stiffness.expand(e, -1), plant.drive_damping.expand(e, -1)
    q0 = initial_joint_state(plant, targets[0]).expand(e, -1)
    sim.write_joint_state(q0, torch.zeros_like(q0))
    plant.reset(torch.arange(e, device=dev), DriveDemand.pd(tg[0].expand(e, -1), kp, kd))
    sim.t_ns = 0
    latency = torch.as_tensor(values["delay"], dtype=torch.int64, device=dev)
    period, nxt = sim.period_ns, 1
    offset = plant.bank.position_offset[0]
    # The first sample is the start state at rest, so a replay starts exactly where this run did.
    zero = torch.zeros_like(tg[0])
    rows = {"native_position_rad": [tg[0] + offset], "velocity_rad_s": [zero], "torque_nm": [zero]}
    stamps, demand = [int(t_cmd[0])], [targets[0]]
    for _ in range(int(rel[-1] // period)):
        while nxt < len(rel) and rel[nxt] < sim.t_ns + period:
            plant.submit(DriveDemand.pd(tg[nxt].expand(e, -1), kp, kd), int(rel[nxt]) + latency)
            nxt += 1
        fb = sim.step()
        stamps.append(int(t_cmd[0]) + sim.t_ns)
        rows["native_position_rad"].append(fb.position_rad[0] + offset)
        rows["velocity_rad_s"].append(fb.velocity_rad_s[0].clone())
        rows["torque_nm"].append(fb.applied_torque_nm[0].clone())
        arrived = int(np.searchsorted(rel + int(values["delay"][0]), sim.t_ns, side="right")) - 1
        demand.append(targets[max(arrived, 0)])
    t = np.asarray(stamps, dtype=np.int64)
    arrays = {k: torch.stack(v).double().cpu().numpy() for k, v in rows.items()}
    arrays["position_demand_rad"] = np.asarray(demand, dtype=np.float64)
    nan = np.full(len(t), np.nan)
    pdo = {d: DriveTrace(d, t, {k: a[:, i] for k, a in arrays.items()}
                         | {"current_a": nan, "current_demand_a": nan, "motor_position_rad": nan})
           for i, d in enumerate(plant.map.drive_names)}
    gains = {d: DriveGains(float(plant.drive_stiffness[i]), float(plant.drive_damping[i]), 0, 0)
             for i, d in enumerate(plant.map.drive_names)}
    return SyntheticCapture("synthetic", "synthetic chirp", commands, pdo, gains)
