"""Chatter gate: the scoring of a policy's closed-loop stand and walk runs, shared by every plant that runs one.

humanoid-sim has no policy loop of its own (the policy side, observations through joint targets, is Controls' policy
core, which humanoid-learning models on this plant). This module is the part that must be the same everywhere: the
numbers of one run and the checkpoint's verdict. It is NumPy only. The Controls MuJoCo gate
(``humanoid_learning.eval.sim_gate``, validated against the robot on 2026-11-17) and the Kamino gate
(``humanoid_learning.eval.kamino_gate``) both score through it.

Scenarios (:data:`SCENARIOS`): stand for 66 s (scored over 6 to 66 s) and walk forward at 0.2 m/s for 40 s (scored over
10 to 40 s), seeds 0 to 2, under a realistic IMU (``real``) and a clean one (``noimu``), command latency 10 to 20 ms.

Per run (:func:`score_run`), over the scored window, from the leg joint targets (the 12 hip, knee and ankle joints of the
action), the IMU body's gravity and gyro as the policy sees them, and the base:

- ``fell``: tilt above 45 degrees at any sample, the base below 0.6 m at the end, or the run's own fall flag;
- ``leg_target_above_3hz_rms_rad`` and ``leg_target_power_above_3hz_share``: the RMS in rad of the leg targets' content
  above 3 Hz (mean removed, over time and the 12 joints) and its share of their power;
- ``ankle_clip_tick_share``: the share of control ticks on which an ankle target was moved by the clip or the linkage
  projection (the final target differs from default plus scale times the action by more than 1e-4 rad);
- ``tilt_deg_mean`` and the base's drift (``base_drift_m``);
- the sagittal low-frequency measures, reported and not gating: ``pitch_sway_deg`` (the IMU pitch's standard deviation
  per 2 s window, averaged over the windows), ``pitch_rate_rms_rad_s`` (the gyro's pitch rate RMS about its mean per 2 s
  window, averaged), ``pitch_0p5_3hz_rms_deg`` and ``pitch_rate_0p5_3hz_rms_rad_s`` (the content in 0.5 to 3 Hz over the
  window) and ``pitch_dominant_hz`` (the strongest frequency of the pitch in 0.5 to 3 Hz). These are the numbers that
  separate a steady 1.4 Hz limit cycle, which the above-3 Hz measure does not see, from a quiet stand.

The per-run test (:func:`run_passes`) is no fall, chatter (share under 10% or RMS under 0.005 rad), ankle clip under 5%
and mean tilt under 3 degrees. The checkpoint verdict (:func:`verdict`) is the reading validated against the robot: it
passes when all stand runs have no fall and pass the chatter test and no walk run falls. Ankle clip, tilt, the walk's
chatter and the low-frequency measures are reported and do not decide: a walking leg has power above 3 Hz, and v11
11400 stands quietly on the robot while failing the per-run test on them.

:func:`score_csv` reads the Controls MuJoCo runner's ``--log-policy`` CSV and is the pinned reference for the Controls
gate (``tests/test_chatter_gate.py`` keeps it equal to humanoid-learning's original scoring).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
HF_HZ = 3.0
CHATTER_SHARE = 0.10
CHATTER_RMS_RAD = 0.005
ANKLE_CLIP_SHARE = 0.05
TILT_MEAN_DEG = 3.0
FALL_TILT_DEG = 45.0
FALL_HEIGHT_M = 0.6
CLIP_TOL_RAD = 1e-4
LOW_BAND_HZ = (0.5, 3.0)
SWAY_WINDOW_S = 2.0
LEG_PREFIXES = ("hip_", "knee_", "ankle_")
GATING_IMUS = ("real", "noimu")


@dataclass(frozen=True)
class Scenario:
    name: str
    seconds: float  # run length, s
    vx: float  # forward command, m/s
    window_s: tuple[float, float]  # the part that is scored


STAND = Scenario("stand", 66.0, 0.0, (6.0, 66.0))
WALK = Scenario("walk02", 40.0, 0.2, (10.0, 40.0))
SCENARIOS = (STAND, WALK)


def run_name(scenario: str, imu: str, seed: int) -> str:
    return f"{scenario}_{imu}_s{seed}"


def _highpass(x: np.ndarray, f: np.ndarray, hz: float) -> np.ndarray:
    spec = np.fft.rfft(x - x.mean(0), axis=0)
    spec[f <= hz] = 0
    return np.fft.irfft(spec, n=len(x), axis=0)


def _bandpass(x: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x - x.mean(0), axis=0)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, n=len(x), axis=0)


def _window_stats(x: np.ndarray, dt: float, seconds: float) -> float:
    """The mean over whole ``seconds`` windows of the RMS of ``x`` about its window mean (the standard deviation); the
    whole signal as one window when it is shorter."""
    n = max(2, round(seconds / dt))
    count = max(1, len(x) // n)
    return float(np.mean([x[k * n:(k + 1) * n if count > 1 else len(x)].std() for k in range(count)]))


def score_run(t, qdes, raw_target, gravity, gyro, joint_names, window_s, *, base_end_xyz=None,
              fell_flag: bool = False) -> dict:
    """The numbers of one run (the module docstring) and its per-run test, over ``window_s`` of run time.

    ``t`` ``[N]`` s; ``qdes`` ``[N, A]`` the final joint targets and ``raw_target`` ``[N, A]`` default plus scale times
    the action (before the clip and the ankle projection), both in action order; ``gravity`` ``[N, 3]`` the IMU body's
    gravity direction and ``gyro`` ``[N, 3]`` its angular velocity, native base (+x forward, +y left); ``joint_names``
    the action's joints; ``base_end_xyz`` the base's displacement and height at the end of the run (None: unknown);
    ``fell_flag``: the run's own fall (an episode that ended)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    m = (t >= window_s[0]) & (t < window_s[1])
    if m.sum() < 2:
        raise ValueError(f"no samples in the window {tuple(window_s)}")
    t = t[m]
    qdes, raw = np.asarray(qdes, dtype=float)[m], np.asarray(raw_target, dtype=float)[m]
    g, w = np.asarray(gravity, dtype=float)[m], np.asarray(gyro, dtype=float)[m]
    names = list(joint_names)
    fwd = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))
    lat = np.degrees(np.arctan2(-g[:, 1], -g[:, 2]))  # to the robot's right positive
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2] / np.linalg.norm(g, axis=1), -1, 1)))
    clipped = np.abs(raw - qdes) > CLIP_TOL_RAD
    ankles = [k for k, name in enumerate(names) if name.startswith("ankle_")]
    legs = [k for k, name in enumerate(names) if name.startswith(LEG_PREFIXES)]
    dt = float(np.median(np.diff(t)))
    f = np.fft.rfftfreq(len(t), dt)
    spec = np.fft.rfft(qdes[:, legs] - qdes[:, legs].mean(0), axis=0)
    power = np.abs(spec) ** 2
    share = float(power[f > HF_HZ].sum() / max(power[f > 0].sum(), 1e-12))
    per_joint = {names[k]: round(float(power[f > HF_HZ, i].sum() / max(power[f > 0, i].sum(), 1e-12)), 3)
                 for i, k in enumerate(legs)}
    spec[f <= HF_HZ] = 0  # the targets' part above 3 Hz in rad, the RMS over time and the leg joints
    rms = float(np.sqrt((np.fft.irfft(spec, n=len(qdes), axis=0) ** 2).mean()))
    base_end = None if base_end_xyz is None else [float(v) for v in base_end_xyz]
    fell = bool(fell_flag or tilt.max() > FALL_TILT_DEG or (base_end is not None and base_end[2] < FALL_HEIGHT_M))
    band = (f >= LOW_BAND_HZ[0]) & (f <= LOW_BAND_HZ[1])
    pitch_power = np.abs(np.fft.rfft(fwd - fwd.mean())) ** 2
    pitch_hz = float(f[band][np.argmax(pitch_power[band])]) if band.any() else None
    out = {"window_s": [float(window_s[0]), float(t[-1])], "fell": fell, "base_end_xyz_m": base_end,
           "base_drift_m": math.hypot(base_end[0], base_end[1]) if base_end else None,
           "lean_fwd_deg_mean": float(fwd.mean()), "lean_lat_deg_mean": float(lat.mean()),
           "tilt_deg_mean": float(tilt.mean()), "tilt_deg_p95": float(np.percentile(tilt, 95)),
           "tilt_deg_max": float(tilt.max()), "sway_fwd_deg_std": float(fwd.std()), "sway_lat_deg_std": float(lat.std()),
           "leg_target_power_above_3hz_share": share, "leg_target_above_3hz_rms_rad": rms,
           "ankle_clip_tick_share": float(clipped[:, ankles].any(1).mean()),
           "any_clip_tick_share": float(clipped.any(1).mean()),
           "per_joint_hf_share": per_joint,
           "pitch_sway_deg": _window_stats(fwd, dt, SWAY_WINDOW_S),
           "pitch_rate_rms_rad_s": _window_stats(w[:, 1], dt, SWAY_WINDOW_S),
           "pitch_0p5_3hz_rms_deg": float(np.sqrt((_bandpass(fwd, f, *LOW_BAND_HZ) ** 2).mean())),
           "pitch_rate_0p5_3hz_rms_rad_s": float(np.sqrt((_bandpass(w[:, 1], f, *LOW_BAND_HZ) ** 2).mean())),
           "pitch_dominant_hz": pitch_hz}
    out["chatter_pass"] = chatter_passes(out)
    out["gate"] = {"no_fall": not fell, "chatter_share_lt_10pct_or_rms_lt_0.005rad": out["chatter_pass"],
                   "ankle_clip_lt_5pct": out["ankle_clip_tick_share"] < ANKLE_CLIP_SHARE,
                   "tilt_mean_lt_3deg": out["tilt_deg_mean"] < TILT_MEAN_DEG}
    out["pass"] = run_passes(out)
    return out


def _col(d, prefix: str, n: int) -> np.ndarray:
    return np.stack([d[f"{prefix}{k}"] for k in range(n)], 1)


def score_csv(csv: Path, deploy: dict, log: str, window_s: tuple[float, float]) -> dict:
    """:func:`score_run` of a Controls MuJoCo runner run: its ``--log-policy`` CSV (columns ``t``, ``obs<k>``, ``act<k>``,
    ``qdes<k>``), the bundle's deploy.yaml as a dict and the runner's log text (its ``base pos (end)`` line)."""
    names = deploy["joint_names"]
    d = np.genfromtxt(csv, delimiter=",", names=True)
    obs = _col(d, "obs", int(deploy["obs_total_dim"]))
    act = _col(d, "act", len(names))
    layout = {x["name"]: x for x in deploy["obs_layout"]}
    end = layout["projected_gravity"]["end"]
    g = obs[:, end - 3:end]  # the newest frame of the IMU body's gravity: native base, +x forward, +y left
    gyro_layout = layout.get("base_ang_vel")
    gyro = (obs[:, gyro_layout["end"] - 3:gyro_layout["end"]] / float(gyro_layout.get("scale", 1.0))
            if gyro_layout else np.zeros_like(g))
    raw = np.array(deploy["default_angles"]) + float(deploy["action_scale"]) * act
    found = re.search(r"base pos \(end\) = x([-+0-9.]+) y\(fwd\)([-+0-9.]+) z([-+0-9.]+)", log)
    base_end = [float(found.group(i)) for i in (1, 2, 3)] if found else None
    return score_run(np.atleast_1d(d["t"]), _col(d, "qdes", len(names)), raw, g, gyro, names, window_s,
                     base_end_xyz=base_end)


def chatter_passes(score: dict) -> bool:
    return (score["leg_target_power_above_3hz_share"] < CHATTER_SHARE
            or score["leg_target_above_3hz_rms_rad"] < CHATTER_RMS_RAD)


def run_passes(score: dict) -> bool:
    """The per-run test: no fall, chatter, ankle clip under 5% and mean tilt under 3 degrees (the module docstring)."""
    return (not score["fell"] and chatter_passes(score) and score["ankle_clip_tick_share"] < ANKLE_CLIP_SHARE
            and score["tilt_deg_mean"] < TILT_MEAN_DEG)


def verdict(scores: dict[str, dict], crosscheck_pass: bool = True) -> tuple[bool, list[str]]:
    """The checkpoint verdict and the reasons it fails (the module docstring). ``scores`` are :func:`score_run`'s results
    by :func:`run_name`, or ``{"error": ...}`` where a run gave none."""
    if not crosscheck_pass:
        return False, ["the bundle does not match Controls' policy core (cross-check), or it could not run"]
    reasons = []
    for imu in GATING_IMUS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                name = run_name(scenario.name, imu, seed)
                s = scores.get(name)
                if s is None or "error" in s:
                    reasons.append(f"{name}: no result" + (f" ({s['error']})" if s else ""))
                elif s["fell"]:
                    reasons.append(f"{name}: fell")
                elif scenario is STAND and not s["chatter_pass"]:
                    reasons.append(f"{name}: chatter, {s['leg_target_above_3hz_rms_rad']:.4f} rad RMS above 3 Hz "
                                   f"({s['leg_target_power_above_3hz_share']:.0%} of the power)")
    return not reasons, reasons


def summary(scores: dict[str, dict], imus: tuple[str, ...] = ("real", "noimu")) -> dict:
    """Per scenario and IMU setting, the per-seed numbers: RMS above 3 Hz (rad), ankle clip share, mean tilt (degrees),
    base drift (m), the low-frequency pitch measures, falls and per-run passes."""
    keys = {"rms_above_3hz_rad": "leg_target_above_3hz_rms_rad", "ankle_clip_share": "ankle_clip_tick_share",
            "tilt_deg_mean": "tilt_deg_mean", "base_drift_m": "base_drift_m", "pitch_sway_deg": "pitch_sway_deg",
            "pitch_rate_rms_rad_s": "pitch_rate_rms_rad_s", "pitch_0p5_3hz_rms_deg": "pitch_0p5_3hz_rms_deg",
            "pitch_rate_0p5_3hz_rms_rad_s": "pitch_rate_0p5_3hz_rms_rad_s", "pitch_dominant_hz": "pitch_dominant_hz",
            "fell": "fell", "run_pass": "pass"}
    out: dict = {}
    for s in SCENARIOS:
        for imu in imus:
            runs = [scores.get(run_name(s.name, imu, seed), {}) for seed in SEEDS]
            out[f"{s.name}_{imu}"] = {k: [r.get(v) for r in runs] for k, v in keys.items()}
    return out
