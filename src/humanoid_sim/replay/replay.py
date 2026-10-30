"""Drive-level replay of an excitation capture.

Each drive is fed, at the recorded gains, PD law, no feed-forward or target velocity (as in the capture), one of:

* ``source="drive"``: its recorded position demand (0x6062, linearly interpolated from the ~200 Hz PDO samples,
  arriving without added latency: it is already what the drive used);
* ``source="command"``: the targets the EtherCAT master sent it, rebuilt from the joint-space command frames
  (:func:`drive_targets`), each frame submitted at its stamp and taking effect one latency later, as the plant
  applies demands in training (``latency_ns``, per env).

The sim's drive-frame position, velocity and applied torque are scored against the drive's own measurements at
the same times; positions as the drive reads them (its ``position_offset`` included).

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
from humanoid_sim.replay.exc import CommandTrace, ExcitationCapture, sample_hold, sample_linear
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


def initial_joint_state(plant: PlantCore, native: np.ndarray | torch.Tensor) -> torch.Tensor:
    """URDF joint positions ``[N, J]`` whose drive-frame positions are ``native`` ``[D]`` or ``[N, D]``."""
    from humanoid_sim.transmissions.drive_map import DriveJointMap

    desc, dev = plant.desc, plant.cfg.device
    exact = DriveJointMap(desc, plant.map.joint_names, plant.map.drive_names, device=dev, dtype=torch.float64)
    nat = torch.as_tensor(native, device=dev).to(torch.float64)
    nat = nat.unsqueeze(0) if nat.dim() == 1 else nat
    q = torch.tensor([[desc.joints[n].standing_offset_rad for n in plant.map.joint_names]],
                     dtype=torch.float64, device=dev).expand(nat.shape[0], -1).clone()
    q[:, exact._lin_j] = nat[:, exact._lin_d] * exact._lin_s + exact._lin_z
    for _, d_idx, j_idx, tr in exact.ankles:  # solved in float64, whatever the plant's dtype
        q[:, j_idx] = tr.joints_from_cranks(nat[:, d_idx], q[:, j_idx])
        resid = (tr.cranks(q[:, j_idx]) - nat[:, d_idx]).abs().max().item()
        if resid > 1e-6:
            raise RuntimeError(f"ankle start pose did not converge (residual {resid:.2e} rad)")
    return q.to(plant.cfg.dtype)


def drive_targets(plant: PlantCore, commands: CommandTrace) -> tuple[np.ndarray, np.ndarray]:
    """The drive-frame targets ``[F, D]`` the master sends for each command frame from the first that names every
    joint, and their stamps ``[F]``. As Controls' master (``standalone.cpp`` ``bodyDriveTargets``): a 1:1 drive gets
    ``sign · (q + standing_offset)``, an ankle's cranks the inverse kinematics of its hinges. The master's velocity
    slew is not modeled: it stages a target unchanged when one step reaches it (check with
    :func:`command_demand_rms`)."""
    joints = plant.map.joint_names
    missing = sorted(set(joints) - set(commands.joints))
    if missing:
        raise ValueError(f"the command frames never name joints {missing}")
    cols = [commands.joints.index(j) for j in joints]
    q = commands.positions_rad[:, cols]
    full = np.isfinite(q).all(axis=1)
    if not full.any():
        raise ValueError("no command frame names every joint")
    first = int(np.argmax(full))
    offsets = np.array([plant.desc.joints[j].standing_offset_rad for j in joints])
    from humanoid_sim.transmissions.drive_map import DriveJointMap

    exact = DriveJointMap(plant.desc, joints, plant.map.drive_names, dtype=torch.float64)
    qt = torch.as_tensor(q[first:] + offsets, dtype=torch.float64)
    qdrv, _, _ = exact.drive_state(qt, torch.zeros_like(qt))
    return qdrv.numpy(), commands.t_ns[first:]


def fit_commands(capture):
    """The command frames a command replay applies and what their stamps are: ``(frames, "applied")``, stamped when
    the controller applied each, when the capture has those acknowledgements, so a delay after them is the drives'
    own (``drive_delay_s``); else ``(frames, "frame")``, stamped when they left the loop host, so the delay also
    holds the software path."""
    applied = capture.applied_commands()
    return (applied, "applied") if applied is not None else (capture.commands(), "frame")


def command_demand_rms(capture, plant: PlantCore, shifts_ns: np.ndarray) -> dict:
    """How well the rebuilt targets explain the drives' own demands: per shift ``s`` and drive, the RMS of 0x6062
    minus the target sent ``s`` earlier (zero-order hold), and the mean at the best shift. Near zero at some shift
    when the rebuild matches the master, up to the drive's interpolation and any slew. A drive without 0x6062 is
    compared with its measured position instead, which adds its tracking error (``against``); a large mean there
    still shows a zero or sign that differs from the description's."""
    targets, t_cmd = drive_targets(plant, fit_commands(capture)[0])
    pdo = capture.pdo()
    drives = plant.map.drive_names
    rms = np.full((len(shifts_ns), len(drives)), np.nan)
    mean = np.full_like(rms, np.nan)
    against = {}
    for i, d in enumerate(drives):
        field = "position_demand_rad" if np.isfinite(pdo[d]["position_demand_rad"]).any() else "native_position_rad"
        against[d] = field
        t, ref = pdo[d].t_ns, pdo[d][field]
        keep = (t >= t_cmd[0] + int(max(shifts_ns))) & np.isfinite(ref)
        for k, s in enumerate(shifts_ns):
            err = ref[keep] - sample_hold(t_cmd, targets[:, i], t[keep] - int(s))
            if err.size:
                rms[k, i], mean[k, i] = float(np.sqrt(np.mean(err ** 2))), float(np.mean(err))
    if np.isnan(rms).all():
        raise ValueError("no feedback after the first command frame to check the rebuilt targets against")
    best = int(np.nanargmin(np.nanmean(rms, axis=1)))
    return {"shifts_ns": [int(s) for s in shifts_ns], "drives": drives, "against": against, "rms_rad": rms.tolist(),
            "best_shift_ns": int(shifts_ns[best]), "rms_at_best_rad": dict(zip(drives, rms[best].tolist(), strict=True)),
            "mean_at_best_rad": dict(zip(drives, mean[best].tolist(), strict=True))}


@dataclass
class _Run:
    t_grid: np.ndarray
    t_meas: np.ndarray
    demand: np.ndarray  # [T, D], or [T, E, D] with a delay per env; the targets [F, D] for source="command"
    real: dict[str, np.ndarray]
    base: BaseMotion | None
    command_t: np.ndarray | None = None  # [F] stamps of the targets, relative to t_grid[0]
    latency_ns: torch.Tensor | None = None  # [E]


def _prepare(capture: ExcitationCapture, sim: SimCore, duration_s: float | None, base_motion: str | None,
             delays_ns: np.ndarray | None = None, source: str = "drive", latency_ns=None) -> _Run:
    """``delays_ns`` ``[E]``: env ``e`` applies the demand recorded at ``t`` at ``t + delay`` (negative: earlier).
    ``source="command"``: the targets the master sent (:func:`fit_commands`), taking effect ``latency_ns`` ``[E]`` or
    ``[E, D]`` (per drive) after their stamp."""
    if source not in ("drive", "command"):
        raise ValueError(f"replay source {source!r}: drive or command")
    plant = sim.plant
    drives = plant.map.drive_names
    pdo = capture.pdo()
    t_start, t_end = capture.time_span_ns()
    if source == "command":
        targets, t_cmd = drive_targets(plant, fit_commands(capture)[0])
        t_start = max(t_start, int(t_cmd[0]))  # from the first frame naming every joint
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
    if source == "command":
        if delays_ns is not None:
            raise ValueError("source='command' takes latency_ns, not delays_ns")
        lat = torch.zeros(plant.num_envs, dtype=torch.int64, device=plant.cfg.device) if latency_ns is None else \
            torch.as_tensor(np.asarray(latency_ns), dtype=torch.int64, device=plant.cfg.device)
        if lat.shape not in ((plant.num_envs,), (plant.num_envs, len(drives))) or bool((lat < 0).any()):
            raise ValueError(f"latency_ns: [{plant.num_envs}] or [{plant.num_envs}, {len(drives)}] delays >= 0 expected")
        return _Run(t_grid, t_meas, targets, real, base, t_cmd - t_grid[0], lat)
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
    read0 = np.array([sample_linear(pdo[d].t_ns, pdo[d]["native_position_rad"], run.t_grid[:1])[0] for d in drives])
    # Each env starts where its drives read the capture's first sample: truth = reading - the drive's offset.
    offset = plant.bank.position_offset
    if bool((offset != 0).any()):
        q0 = initial_joint_state(plant, torch.as_tensor(read0, dtype=dt, device=dev).unsqueeze(0) - offset)
    else:
        q0 = initial_joint_state(plant, read0).expand(e, -1)
    sim.write_joint_state(q0, torch.zeros_like(q0))
    demand = torch.as_tensor(run.demand, dtype=dt, device=dev)
    angles = rates = None
    if run.base is not None:
        angles = torch.as_tensor(run.base.angles, dtype=torch.float32, device=dev)
        rates = torch.as_tensor(run.base.rates, dtype=torch.float32, device=dev)
        sim.set_base(angles[0].expand(e, -1), rates[0].expand(e, -1), write_state=True)
    sim.t_ns = 0
    if run.command_t is None:
        plant.reset(torch.arange(e, device=dev))
    else:
        # The frame in force at the start (the latest at or before it), then each frame at its stamp.
        nxt = int(np.searchsorted(run.command_t, 0, side="right"))
        plant.reset(torch.arange(e, device=dev), DriveDemand.pd(demand[max(nxt - 1, 0)].expand(e, -1), kp, kd))
    period = sim.period_ns
    for k in range(len(run.t_grid)):
        if run.command_t is None:
            target = demand[k] if demand.dim() == 3 else demand[k].expand(e, -1)
            plant.submit(DriveDemand.pd(target, kp, kd), torch.full((e,), sim.t_ns, dtype=torch.int64, device=dev))
        else:
            while nxt < len(run.command_t) and run.command_t[nxt] < sim.t_ns + period:
                plant.submit(DriveDemand.pd(demand[nxt].expand(e, -1), kp, kd),
                             int(run.command_t[nxt]) + run.latency_ns)
                nxt += 1
        if angles is not None:
            sim.set_base(angles[k].expand(e, -1), rates[k].expand(e, -1))
        on_step(k, sim.step())
    if bool((plant.bank.queue.overflows > 0).any()):
        raise RuntimeError("demands in flight overflowed the drive queue: latency too long for its capacity")


def replay(capture: ExcitationCapture, sim: SimCore, *, duration_s: float | None = None,
           base_motion: str | None = None, demand_delay_ns: int = 0, source: str = "drive",
           latency_ns: int = 0) -> ReplayResult:
    """Replay on env 0. ``base_motion``: None (fixed base) or a :data:`~humanoid_sim.replay.base.BASE_MODES`
    mode. ``demand_delay_ns`` (``source="drive"``): apply the demand recorded at ``t`` at ``t + delay``;
    ``latency_ns`` (``source="command"``): from a frame's stamp to its targets taking effect."""
    plant = sim.plant
    delays = np.full(plant.num_envs, demand_delay_ns) if demand_delay_ns else None
    lat = np.full(plant.num_envs, latency_ns) if source == "command" else None
    run = _prepare(capture, sim, duration_s, base_motion, delays, source, lat)
    keys = ("position_rad", "velocity_rad_s", "applied_torque_nm", "demanded_torque_nm")
    acc = {k: torch.zeros(len(run.t_grid), plant.map.num_drives, device=plant.cfg.device) for k in keys}
    acc_q = torch.zeros(len(run.t_grid), len(plant.map.joint_names), device=plant.cfg.device)

    offset0 = plant.bank.position_offset[0]

    def record(k, fb):
        for key in keys:
            acc[key][k] = getattr(fb, key)[0]
        acc["position_rad"][k] += offset0  # what the drive reads, as the capture recorded
        acc_q[k] = sim.joint_state()[0][0]

    _run(capture, sim, run, record)
    out = {name: acc[key].cpu().double().numpy() for name, key in
           zip(("position", "velocity", "applied", "demanded"), keys, strict=True)}
    out["joint_q"] = acc_q.cpu().double().numpy()
    out["base_angles"] = run.base.angles if run.base is not None else np.zeros((len(run.t_grid), 3))
    if run.command_t is None:
        demand = run.demand[:, 0] if run.demand.ndim == 3 else run.demand
    else:  # the target sent last at or before each step's start
        idx = np.clip(np.searchsorted(run.command_t, run.t_grid - run.t_grid[0], side="right") - 1, 0, None)
        demand = run.demand[idx]
    return ReplayResult(capture.dataset_id, sim.period_ns, plant.map.drive_names, run.t_meas, out, run.real, demand,
                        meta={"name": capture.name, "base": base_motion or "fixed", "source": source,
                              "demand_delay_ns": demand_delay_ns, "latency_ns": latency_ns,
                              **plant.desc.identity()})


def replay_position_rms(capture: ExcitationCapture, sim: SimCore, *, duration_s: float | None = None,
                        base_motion: str | None = None, skip_s: float = 1.0,
                        delays_ns: np.ndarray | None = None, source: str = "drive",
                        latency_ns: np.ndarray | None = None) -> torch.Tensor:
    """Every env replays the capture (with its own demand delay if ``delays_ns`` ``[E]`` is given, or, from the
    commands, its own ``latency_ns`` ``[E]``); position RMS (sim − measured) per env and drive ``[E, D]``, after
    ``skip_s``. Accumulated on the device: nothing leaves it until the end."""
    plant = sim.plant
    run = _prepare(capture, sim, duration_s, base_motion, delays_ns, source, latency_ns)
    real = torch.as_tensor(run.real["position"], dtype=torch.float32, device=plant.cfg.device)
    first = int(np.searchsorted(run.t_meas, run.t_meas[0] + int(skip_s * NS_PER_S)))
    sq = torch.zeros(plant.num_envs, plant.map.num_drives, device=plant.cfg.device)
    valid = torch.isfinite(real)
    real = torch.nan_to_num(real)

    offset = plant.bank.position_offset.float()

    def record(k, fb):
        if k >= first:
            err = (fb.position_rad.float() + offset - real[k]) * valid[k]
            sq.add_(err * err)

    _run(capture, sim, run, record)
    count = valid[first:].sum(0).clamp(min=1).float()
    return (sq / count).sqrt()
