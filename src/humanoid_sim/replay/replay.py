"""Drive-level replay of an excitation capture on a fixed base.

Each drive is fed its recorded position demand (0x6062, linearly interpolated from the ~200 Hz PDO
samples, arriving without added latency: it is already what the drive used) at the recorded gains, JVPT
mode, no feed-forward or target velocity (as in the capture). The sim's drive-frame position, velocity
and applied torque are scored against the drive's own measurements at the same times.

Fixed base: the capture was taken on a gantry whose base swings (IMU |ω| up to ~1 rad/s); PACE fits with a
fixed root too, so this is the same approximation as the sysid baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from humanoid_sim.core import PlantCore
from humanoid_sim.devices.demand import JVPT_PER_SI_GAIN, DriveDemand
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


def replay(capture: ExcitationCapture, sim: SimCore, *, duration_s: float | None = None) -> ReplayResult:
    plant = sim.plant
    drives = plant.map.drive_names
    pdo = capture.pdo()
    gains = capture.gains()
    t_start, t_end = capture.time_span_ns()
    if duration_s is not None:
        t_end = min(t_end, t_start + int(duration_s * NS_PER_S))
    period = sim.period_ns
    t_grid = np.arange(t_start, t_end - period, period, dtype=np.int64)
    t_meas = t_grid + period

    def series(name, t):
        return np.stack([sample_linear(pdo[d].t_ns, pdo[d][name], t) for d in drives], axis=1)

    demand = series("position_demand_rad", t_grid)
    real = {"position": series("native_position_rad", t_meas), "velocity": series("velocity_rad_s", t_meas),
            "torque": series("torque_nm", t_meas)}
    dev = plant.cfg.device
    kp = torch.tensor([[gains[d].p_gain_raw for d in drives]], dtype=torch.int64, device=dev)
    kd = torch.tensor([[gains[d].d_gain_raw for d in drives]], dtype=torch.int64, device=dev)

    native0 = np.array([sample_linear(pdo[d].t_ns, pdo[d]["native_position_rad"], np.array([t_start]))[0]
                        for d in drives])
    q0 = initial_joint_state(plant, native0)
    sim.write_joint_state(q0, torch.zeros_like(q0))
    plant.reset(torch.tensor([0], device=dev))
    sim.t_ns = 0

    out = {k: np.zeros((len(t_grid), len(drives))) for k in ("position", "velocity", "applied", "demanded")}
    out["joint_q"] = np.zeros((len(t_grid), len(plant.map.joint_names)))
    dt = plant.cfg.dtype
    for k in range(len(t_grid)):
        pos = torch.as_tensor(demand[k], dtype=dt, device=dev).unsqueeze(0)
        d = DriveDemand.jvpt(pos, kp.to(dt) / JVPT_PER_SI_GAIN, kd.to(dt) / JVPT_PER_SI_GAIN)
        d.p_gain_raw, d.d_gain_raw = kp.clone(), kd.clone()
        plant.submit(d, torch.tensor([sim.t_ns], device=dev))
        fb = sim.step()
        out["position"][k] = fb.position_rad[0].cpu().numpy()
        out["velocity"][k] = fb.velocity_rad_s[0].cpu().numpy()
        out["applied"][k] = fb.applied_torque_nm[0].cpu().numpy()
        out["demanded"][k] = fb.demanded_torque_nm[0].cpu().numpy()
        out["joint_q"][k] = sim.joint_state()[0][0].cpu().numpy()
    return ReplayResult(capture.dataset_id, period, drives, t_meas, out, real, demand,
                        meta={"name": capture.name, **plant.desc.identity()})
