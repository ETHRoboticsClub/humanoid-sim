"""Drive-level replay of an excitation capture.

Each drive is fed its recorded position demand (0x6062, linearly interpolated from the ~200 Hz PDO
samples, arriving without added latency: it is already what the drive used) at the recorded gains, PD
law, no feed-forward or target velocity (as in the capture). The sim's drive-frame position, velocity
and applied torque are scored against the drive's own measurements at the same times.

Base: the capture was taken on a gantry that lets the base swing, mostly in yaw (tens of degrees, IMU
|ω| up to about 1.8 rad/s). A fixed-base replay ignores that motion; with a base mode (on the asset built with base
joints) the base follows the capture's IMU (:class:`~humanoid_sim.replay.base.BaseMotion`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from humanoid_sim.core import PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.replay.base import BaseMotion
from humanoid_sim.replay.exc import ExcitationCapture, sample_linear
from humanoid_sim.sim import SimCore
from humanoid_sim.timing.clock import NS_PER_S


@dataclass
class ReplayResult:
    dataset_id: str
    period_ns: int
    drives: list[str]
    t_ns: np.ndarray  # capture time of each recorded sample (after each step)
    sim: dict[str, np.ndarray]  # position, velocity, applied, demanded: [T, D]
    real: dict[str, np.ndarray]  # position, velocity, torque: [T, D]
    demand: np.ndarray  # [T, D]
    meta: dict = field(default_factory=dict)

    def real_joint_q(self, plant: PlantCore) -> np.ndarray:
        """URDF joint positions of the real robot at each sample (1:1 signs; ankle FK from the cranks)."""
        out = np.zeros((len(self.t_ns), len(plant.map.joint_names)))
        for k in range(len(self.t_ns)):
            out[k] = initial_joint_state(plant, self.real["position"][k])[0].cpu().numpy()
        return out

    def rms(self, key="position", *, skip_s: float = 1.0) -> dict[str, float]:
        sim_key = {"position": "position", "velocity": "velocity", "torque": "applied"}[key]
        keep = self.t_ns >= self.t_ns[0] + int(skip_s * NS_PER_S)
        err = self.sim[sim_key][keep] - self.real[key][keep]
        return {d: float(np.sqrt(np.nanmean(err[:, i] ** 2))) for i, d in enumerate(self.drives)}

    def tracking_rms(self, *, skip_s: float = 1.0) -> dict[str, float]:
        """RMS of measured position − demand: how hard the drive worked (a scale for the sim error)."""
        keep = self.t_ns >= self.t_ns[0] + int(skip_s * NS_PER_S)
        err = self.real["position"][keep] - self.demand[keep]
        return {d: float(np.sqrt(np.nanmean(err[:, i] ** 2))) for i, d in enumerate(self.drives)}


def initial_joint_state(plant: PlantCore, native: np.ndarray) -> torch.Tensor:
    """URDF joint positions ``[1, J]`` whose drive-frame positions are ``native`` ``[D]``."""
    desc = plant.desc
    q = torch.tensor([[desc.joints[n].standing_offset_rad for n in plant.map.joint_names]],
                     dtype=plant.cfg.dtype, device=plant.cfg.device)
    nat = torch.as_tensor(native, dtype=plant.cfg.dtype, device=plant.cfg.device).unsqueeze(0)
    q[:, plant.map._lin_j] = nat[:, plant.map._lin_d] * plant.map._lin_s
    for _, d_idx, j_idx, tr in plant.map.ankles:
        q[:, j_idx] = tr.joints_from_cranks(nat[:, d_idx], q[:, j_idx])
        resid = (tr.cranks(q[:, j_idx]) - nat[:, d_idx]).abs().max().item()
        if resid > 1e-6:
            raise RuntimeError(f"ankle start pose did not converge (residual {resid:.2e} rad)")
    return q


@dataclass
class _Run:
    t_grid: np.ndarray
    t_meas: np.ndarray
    demand: np.ndarray  # [T, D], or [T, E, D] with a delay per env
    real: dict[str, np.ndarray]
    base: BaseMotion | None


def _prepare(capture: ExcitationCapture, sim: SimCore, duration_s: float | None, base_motion: str | None,
             delays_ns: np.ndarray | None = None) -> _Run:
    """``delays_ns`` ``[E]``: env ``e`` applies the demand recorded at ``t`` at ``t + delay`` (negative: earlier)."""
    plant = sim.plant
    drives = plant.map.drive_names
    pdo = capture.pdo()
    t_start, t_end = capture.time_span_ns()
    if duration_s is not None:
        t_end = min(t_end, t_start + int(duration_s * NS_PER_S))
    period = sim.period_ns
    t_grid = np.arange(t_start, t_end - period, period, dtype=np.int64)
    t_meas = t_grid + period

    def series(name, t):
        return np.stack([sample_linear(pdo[d].t_ns, pdo[d][name], t) for d in drives], axis=1)

    real = {"position": series("native_position_rad", t_meas), "velocity": series("velocity_rad_s", t_meas),
            "torque": series("torque_nm", t_meas)}
    # Base targets at the step start: with the velocity target the base PD lands on the step-end orientation.
    base = BaseMotion.from_imu(capture.imu(), t_grid, base_motion) if base_motion else None
    if delays_ns is None:
        demand = series("position_demand_rad", t_grid)
    else:
        demand = np.stack([series("position_demand_rad", t_grid - int(d)) for d in delays_ns], axis=1)
    return _Run(t_grid, t_meas, demand, real, base)


def _run(capture: ExcitationCapture, sim: SimCore, run: _Run, on_step) -> None:
    """Every env replays the capture from its first sample; ``on_step(k, feedback)`` after each step."""
    plant = sim.plant
    drives, e, dev, dt = plant.map.drive_names, plant.num_envs, plant.cfg.device, plant.cfg.dtype
    gains = capture.gains()
    kp = torch.tensor([[gains[d].stiffness for d in drives]], dtype=dt, device=dev).expand(e, -1)
    kd = torch.tensor([[gains[d].damping for d in drives]], dtype=dt, device=dev).expand(e, -1)
    pdo = capture.pdo()
    native0 = np.array([sample_linear(pdo[d].t_ns, pdo[d]["native_position_rad"], run.t_grid[:1])[0] for d in drives])
    q0 = initial_joint_state(plant, native0).expand(e, -1)
    sim.write_joint_state(q0, torch.zeros_like(q0))
    demand = torch.as_tensor(run.demand, dtype=dt, device=dev)
    angles = rates = None
    if run.base is not None:
        angles = torch.as_tensor(run.base.angles, dtype=torch.float32, device=dev)
        rates = torch.as_tensor(run.base.rates, dtype=torch.float32, device=dev)
        sim.set_base(angles[0].expand(e, -1), rates[0].expand(e, -1), write_state=True)
    plant.reset(torch.arange(e, device=dev))
    sim.t_ns = 0
    for k in range(len(run.t_grid)):
        target = demand[k] if demand.dim() == 3 else demand[k].expand(e, -1)
        d = DriveDemand.pd(target, kp, kd)
        plant.submit(d, torch.full((e,), sim.t_ns, dtype=torch.int64, device=dev))
        if angles is not None:
            sim.set_base(angles[k].expand(e, -1), rates[k].expand(e, -1))
        on_step(k, sim.step())


def replay(capture: ExcitationCapture, sim: SimCore, *, duration_s: float | None = None,
           base_motion: str | None = None, demand_delay_ns: int = 0) -> ReplayResult:
    """Replay on env 0. ``base_motion``: None (fixed base) or a :data:`~humanoid_sim.replay.base.BASE_MODES`
    mode. ``demand_delay_ns``: apply the demand recorded at ``t`` at ``t + delay``."""
    plant = sim.plant
    delays = np.full(plant.num_envs, demand_delay_ns) if demand_delay_ns else None
    run = _prepare(capture, sim, duration_s, base_motion, delays)
    keys = ("position_rad", "velocity_rad_s", "applied_torque_nm", "demanded_torque_nm")
    acc = {k: torch.zeros(len(run.t_grid), plant.map.num_drives, device=plant.cfg.device) for k in keys}
    acc_q = torch.zeros(len(run.t_grid), len(plant.map.joint_names), device=plant.cfg.device)

    def record(k, fb):
        for key in keys:
            acc[key][k] = getattr(fb, key)[0]
        acc_q[k] = sim.joint_state()[0][0]

    _run(capture, sim, run, record)
    out = {name: acc[key].cpu().double().numpy() for name, key in
           zip(("position", "velocity", "applied", "demanded"), keys, strict=True)}
    out["joint_q"] = acc_q.cpu().double().numpy()
    out["base_angles"] = run.base.angles if run.base is not None else np.zeros((len(run.t_grid), 3))
    return ReplayResult(capture.dataset_id, sim.period_ns, plant.map.drive_names, run.t_meas, out, run.real,
                        run.demand[:, 0] if run.demand.ndim == 3 else run.demand,
                        meta={"name": capture.name, "base": base_motion or "fixed",
                              "demand_delay_ns": demand_delay_ns, **plant.desc.identity()})


def replay_position_rms(capture: ExcitationCapture, sim: SimCore, *, duration_s: float | None = None,
                        base_motion: str | None = None, skip_s: float = 1.0,
                        delays_ns: np.ndarray | None = None) -> torch.Tensor:
    """Every env replays the capture (with its own demand delay if ``delays_ns`` ``[E]`` is given); position RMS
    (sim − measured) per env and drive ``[E, D]``, after ``skip_s``. Accumulated on the device: nothing leaves it
    until the end."""
    plant = sim.plant
    run = _prepare(capture, sim, duration_s, base_motion, delays_ns)
    real = torch.as_tensor(run.real["position"], dtype=torch.float32, device=plant.cfg.device)
    first = int(np.searchsorted(run.t_meas, run.t_meas[0] + int(skip_s * NS_PER_S)))
    sq = torch.zeros(plant.num_envs, plant.map.num_drives, device=plant.cfg.device)
    valid = torch.isfinite(real)
    real = torch.nan_to_num(real)

    def record(k, fb):
        if k >= first:
            err = (fb.position_rad.float() - real[k]) * valid[k]
            sq.add_(err * err)

    _run(capture, sim, run, record)
    count = valid[first:].sum(0).clamp(min=1).float()
    return (sq / count).sqrt()
