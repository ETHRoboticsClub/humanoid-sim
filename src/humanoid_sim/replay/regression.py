"""Sim-versus-robot regression cases: recorded V1 bags, a window of each, the metrics of
:mod:`humanoid_sim.replay.metrics` and a pass band for each, in a versioned file (``regression/cases.yaml``).

A case names a rosbag2 recording (its S3 path, its sha256 and where it sits under the recordings directory), a time
window in seconds after the bag's start, and the bands. :func:`run_case` replays the window on the sim with each
COMMAND frame's targets applied as the robot's command path applies them (``devices.command_path.frame_to_drive_ns``
of the profile) and the base prescribed from the IMU, puts robot and sim on the 200 Hz grid by source stamps
(:mod:`humanoid_sim.replay.grid`) and compares them window by window.

A band is ``{ref, lo, hi}`` on the mean over the windows of the sim's value: ``ref`` is the robot's value when the
band was set (:func:`propose_bands`, from the tolerances in the file), and a robot value that has moved from its ``ref``
means the bag, the window or the metric changed and the bands are stale. Phase bands are circular in degrees. Metrics
that need the sim's own torso motion (the IMU and pitch-rate families) are skipped while the base is prescribed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from humanoid_sim.replay.bag import BagCapture
from humanoid_sim.replay.base import euler_zyx
from humanoid_sim.replay.grid import GRID_HZ, hold, interpolate, make_grid
from humanoid_sim.replay.metrics import (
    HipPair,
    MetricResult,
    Signals,
    compare,
    family,
    is_phase,
    needs_free_base,
    window_slices,
    wrap_deg,
)
from humanoid_sim.replay.replay import DriveCommands

FORMAT = "humanoid-sim-regression/v1"
SCORING = ("open_loop", "closed_loop")
DEFAULT_WARMUP_S = 2.0
MIN_BAND_WINDOWS = 3  # a metric the robot gives on fewer windows than this gets no band
STALE_TOLERANCE = 1e-6  # a robot value this far from its pinned ref means the bands are stale
DEFAULT_TOLERANCES = {
    "tracking_rms": {"rel": 0.25, "abs": 0.0005},  # rad
    "sim_real_rms": {"fraction_of_tracking": 0.5, "abs": 0.0005},  # rad
    "response_gain": {"rel": 0.30},
    "response_phase_deg": {"abs": 20.0},
    "imu_sway_deg": {"rel": 0.30, "abs": 0.05},
    "pitch_rate_rms": {"rel": 0.30, "abs": 0.005},  # rad/s
}


@dataclass
class Case:
    id: str
    description: str
    bag: dict  # s3, sha256, path
    window_s: tuple[float, float]
    warmup_s: float
    sources: dict[str, str]  # family -> where its bands come from
    bands: dict[str, dict]  # metric -> {ref, lo, hi}
    robot_windows: int = 0
    support: str = "gantry hang"  # how the robot was carried: "floor, rope slack", "gantry hang", ...
    scoring: str = "open_loop"  # open_loop: a held or hung base represents the recording; closed_loop: it does not


@dataclass
class Cases:
    version: int
    tolerances: dict
    cases: list[Case]


def load_cases(path: Path) -> Cases:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("format") != FORMAT:
        raise ValueError(f"{path}: format {raw.get('format')!r} != {FORMAT}")
    ids = [c["id"] for c in raw["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    cases = []
    for c in raw["cases"]:
        missing = {"id", "description", "bag", "window_s"} - set(c)
        if missing:
            raise ValueError(f"{path}: case {c.get('id')!r} lacks {sorted(missing)}")
        if {"s3", "sha256", "path"} - set(c["bag"]):
            raise ValueError(f"{path}: case {c['id']} bag needs s3, sha256 and path")
        a, b = c["window_s"]
        if not 0 <= a < b:
            raise ValueError(f"{path}: case {c['id']} window {c['window_s']} must be 0 <= start < end")
        if c.get("scoring", "open_loop") not in SCORING:
            raise ValueError(f"{path}: case {c['id']} scoring {c.get('scoring')!r} must be one of {SCORING}")
        if not c.get("support"):
            raise ValueError(f"{path}: case {c['id']} lacks its support state (floor, gantry hang, ...)")
        cases.append(Case(c["id"], c["description"], dict(c["bag"]), (float(a), float(b)),
                          float(c.get("warmup_s", DEFAULT_WARMUP_S)), dict(c.get("sources", {})),
                          {k: dict(v) for k, v in c.get("bands", {}).items()}, int(c.get("robot_windows", 0)),
                          str(c["support"]), c.get("scoring", "open_loop")))
    return Cases(int(raw["version"]), {**DEFAULT_TOLERANCES, **raw.get("tolerances", {})}, cases)


def resolve_bag(case: Case, recordings: Path) -> Path:
    """The case's bag directory under ``recordings``, its sha256 checked against the pinned one."""
    root = Path(recordings) / case.bag["path"]
    if not (root / "metadata.yaml").is_file():
        raise FileNotFoundError(f"{case.id}: no bag at {root} (S3: {case.bag['s3']})")
    h = hashlib.sha256()
    for p in sorted(root.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            with p.open("rb") as f:
                h.update(hashlib.file_digest(f, "sha256").digest())
    if h.hexdigest() != case.bag["sha256"]:
        raise ValueError(f"{case.id}: {root} has sha256 {h.hexdigest()}, the case pins {case.bag['sha256']}")
    return root


def hip_pair(desc, drives: list[str]) -> HipPair:
    """The symmetric hip flexion's drives, in the drive order, with their joint signs."""
    found = {}
    for name in drives:
        d = desc.drives[name]
        if d.crank is None and d.joints[0] in ("hip_flexion_l", "hip_flexion_r"):
            found[d.joints[0]] = (drives.index(name), float(d.sign))
    if len(found) != 2:
        raise ValueError("the description has no hip_flexion_l and hip_flexion_r drives")
    (li, ls), (ri, rs) = found["hip_flexion_l"], found["hip_flexion_r"]
    return HipPair(li, ri, ls, rs)


def drive_commands(bag: BagCapture, desc, drives: list[str]) -> DriveCommands:
    """The bag's COMMAND frames as drive-frame targets in ``drives`` order, from the first frame that names every
    joint, each at its frame stamp (what the loop host sent; the command path is the replay's latency)."""
    import torch

    from humanoid_sim.transmissions.drive_map import DriveJointMap

    trace = bag.commands()
    joints = list(desc.joints)
    missing = sorted(set(joints) - set(trace.joints))
    if missing:
        raise ValueError(f"the bag's COMMAND frames never name {missing}")
    q = trace.positions_rad[:, [trace.joints.index(j) for j in joints]]
    full = np.isfinite(q).all(axis=1)
    if not full.any():
        raise ValueError("no COMMAND frame names every joint")
    first = int(np.argmax(full))
    q = q[first:] + np.array([desc.joints[j].standing_offset_rad for j in joints])
    exact = DriveJointMap(desc, joints, drives, dtype=torch.float64)
    qt = torch.as_tensor(q, dtype=torch.float64)
    targets, _, _ = exact.drive_state(qt, torch.zeros_like(qt))
    return DriveCommands(tuple(drives), trace.t_ns[first:], targets.numpy(), "frame")


def robot_signals(bag: BagCapture, commands: DriveCommands, start_ns: int, end_ns: int) -> Signals:
    """The robot's signals on the 200 Hz grid over ``[start, end)``: each drive's position and torque and the IMU
    gyro and angles at their own source stamps (the IMU's are its receipt stamps), the targets held from their frame
    stamps."""
    grid = make_grid(start_ns, end_ns)
    pdo = bag.pdo()
    drives = list(commands.drives)
    position = np.stack([interpolate(pdo[d].t_ns, pdo[d]["native_position_rad"], grid) for d in drives], axis=1)
    torque = np.stack([interpolate(pdo[d].t_ns, pdo[d]["torque_nm"], grid) for d in drives], axis=1)
    imu = bag.imu()
    gyro = interpolate(imu.t_ns, imu.angular_velocity, grid)
    angles = interpolate(imu.t_ns, euler_zyx(imu.orientation_xyzw / np.linalg.norm(imu.orientation_xyzw, axis=1,
                                                                                  keepdims=True)), grid)
    return Signals(grid, drives, hold(commands.t_ns, commands.position_rad, grid), position, torque, gyro, angles)


def sim_signals(res, commands: DriveCommands, grid: np.ndarray) -> Signals:
    """The sim's signals of a replay result on the same grid. No gyro or angles: the base was prescribed."""
    position = interpolate(res.t_ns, res.sim["position"], grid)
    torque = interpolate(res.t_ns, res.sim["applied"], grid)
    return Signals(grid, list(commands.drives), hold(commands.t_ns, commands.position_rad, grid), position, torque)


@dataclass
class CaseRun:
    case: Case
    results: dict[str, MetricResult]
    meta: dict = field(default_factory=dict)
    replay: object = None  # the ReplayResult of a sim run


def command_latency_ns(sim) -> int:
    """The command path from a frame's stamp to the drive acting on it, from the profile's measurement."""
    path = sim.profile["devices"].get("command_path")
    if path is None:
        raise ValueError("the profile has no devices.command_path: no measured command latency to replay with")
    return int(path["frame_to_drive_ns"])


def window_ns(bag: BagCapture, window_s: tuple[float, float]) -> tuple[int, int]:
    return bag.start_ns + round(window_s[0] * 1e9), bag.start_ns + round(window_s[1] * 1e9)


def robot_only(case: Case, root: Path, desc, drives: list[str]) -> CaseRun:
    """The case's robot metrics, no sim: what :func:`propose_bands` and the stale check read."""
    bag = BagCapture(root, drives=drives)
    commands = drive_commands(bag, desc, drives)
    start, end = window_ns(bag, case.window_s)
    robot = robot_signals(bag, commands, start, end)
    return CaseRun(case, compare(robot, None, hip_pair(desc, drives)), {"bag": bag.dataset_id})


def run_case(case: Case, root: Path, sim, *, base: str = "orientation") -> CaseRun:
    """Replay the case's window on ``sim`` (built with base joints) and compare it with the robot."""
    from humanoid_sim.replay.replay import replay

    desc, drives = sim.desc, list(sim.plant.map.drive_names)
    start_s = case.window_s[0] - case.warmup_s
    if start_s < 0:
        raise ValueError(f"{case.id}: window start {case.window_s[0]} s leaves no room for {case.warmup_s} s of warm-up")
    bag = BagCapture(root, drives=drives, window_s=(start_s, case.window_s[1]))
    commands = drive_commands(bag, desc, drives)
    latency = command_latency_ns(sim)
    res = replay(bag, sim, base_motion=base, source="command", latency_ns=latency, commands=commands)
    start, end = window_ns(bag, case.window_s)
    robot = robot_signals(bag, commands, start, end)
    simulated = sim_signals(res, commands, robot.t_ns)
    results = compare(robot, simulated, hip_pair(desc, drives))
    meta = {"bag": bag.dataset_id, "latency_ns": latency, "base": base, "warmup_s": case.warmup_s,
            "period_ns": sim.period_ns, "physics": f"isaaclab/{sim.solver}", "integrator": sim.integrator,
            "solver_gain_support": res.meta.get("solver_gain_support"), **desc.identity()}
    return CaseRun(case, results, meta, res)


@dataclass
class Verdict:
    metric: str
    status: str  # pass, fail, skip, info, stale, missing
    reason: str
    result: MetricResult | None
    band: dict | None

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "band": self.band,
                **({} if self.result is None else self.result.as_dict())}


def _within(metric: str, value: float, band: dict) -> bool:
    if is_phase(metric):
        value = band["ref"] + float(wrap_deg(value - band["ref"]))
    return band["lo"] <= value <= band["hi"]


def judge(run: CaseRun, *, base_prescribed: bool = True) -> list[Verdict]:
    """A verdict for every metric the case has a band for, and ``info`` for the rest. A closed-loop case (the robot
    stood on the floor) is not scored while the base is prescribed: its bands are the robot's values that a closed-loop
    run is held to."""
    out = []
    closed = run.case.scoring == "closed_loop" and base_prescribed
    for metric, band in run.case.bands.items():
        r = run.results.get(metric)
        if closed:
            out.append(Verdict(metric, "skip", f"{run.case.support}: scored in closed loop only; a prescribed base "
                                                "misrepresents it", r, band))
            continue
        if needs_free_base(metric) and base_prescribed:
            out.append(Verdict(metric, "skip", "needs the sim's own torso motion; the base is prescribed", r, band))
            continue
        if r is None or (r.robot is None and family(metric) != "sim_real_rms"):
            out.append(Verdict(metric, "missing", "the robot has no value on any window", r, band))
            continue
        pinned = r.robot_all or r.robot  # the robot over all its windows, as the band was set
        if pinned is not None:
            drift = float(wrap_deg(pinned.mean - band["ref"])) if is_phase(metric) else pinned.mean - band["ref"]
            if abs(drift) > STALE_TOLERANCE * max(1.0, abs(band["ref"])):
                out.append(Verdict(metric, "stale", f"robot value {pinned.mean:.6g} is not the pinned ref "
                                                    f"{band['ref']:.6g}: bag, window or metric changed", r, band))
                continue
        if r.sim is None:
            out.append(Verdict(metric, "missing", "the sim has no value on the robot's windows", r, band))
            continue
        ok = _within(metric, r.sim.mean, band)
        out.append(Verdict(metric, "pass" if ok else "fail",
                           f"sim {r.sim.mean:.4g} {'inside' if ok else 'outside'} [{band['lo']:.4g}, {band['hi']:.4g}]",
                           r, band))
    for metric, r in run.results.items():
        if metric not in run.case.bands:
            out.append(Verdict(metric, "info", "no band", r, None))
    return out


def counts(verdicts: list[Verdict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in verdicts:
        out[v.status] = out.get(v.status, 0) + 1
    return out


def case_passed(verdicts: list[Verdict]) -> bool:
    c = counts(verdicts)
    return not (c.get("fail") or c.get("stale") or c.get("missing"))


def case_state(case: Case, verdicts: list[Verdict]) -> str:
    """PASS, FAIL, or NOT SCORED (nothing the sim was judged on: a closed-loop case on a prescribed base)."""
    if not case_passed(verdicts):
        return "FAIL"
    return "PASS" if counts(verdicts).get("pass") else "NOT SCORED"


def propose_bands(run: CaseRun, tolerances: dict) -> tuple[dict[str, dict], dict[str, str]]:
    """Bands from the robot's values of ``run`` and the tolerances; each family's source sentence."""
    t = tolerances
    bands: dict[str, dict] = {}
    tracking = {m: r.robot.mean for m, r in run.results.items() if family(m) == "tracking_rms" and r.robot}
    for metric, r in run.results.items():
        if r.robot is None or r.windows < MIN_BAND_WINDOWS:
            continue
        ref, fam = r.robot.mean, family(metric)
        if fam == "tracking_rms":
            tol = t["tracking_rms"]["rel"] * ref + t["tracking_rms"]["abs"]
            lo, hi = max(0.0, ref - tol), ref + tol
        elif is_phase(metric):
            tol = t["response_phase_deg"]["abs"]
            lo, hi = ref - tol, ref + tol
        elif metric.endswith(".gain"):
            lo, hi = ref * (1 - t["response_gain"]["rel"]), ref * (1 + t["response_gain"]["rel"])
        elif metric.startswith("imu.") and metric.endswith("_sway_deg"):
            tol = t["imu_sway_deg"]["rel"] * ref + t["imu_sway_deg"]["abs"]
            lo, hi = max(0.0, ref - tol), ref + tol
        elif metric.startswith("imu.pitch_rate_rms."):
            tol = t["pitch_rate_rms"]["rel"] * ref + t["pitch_rate_rms"]["abs"]
            lo, hi = max(0.0, ref - tol), ref + tol
        else:
            continue
        bands[metric] = {"ref": float(ref), "lo": float(lo), "hi": float(hi)}
    for metric in tracking if run.case.scoring == "open_loop" else ():  # sim minus robot: a fraction of tracking
        drive = metric.split(".", 1)[1]
        s = t["sim_real_rms"]
        bands[f"sim_real_rms.{drive}"] = {"ref": 0.0, "lo": 0.0,
                                          "hi": float(s["fraction_of_tracking"] * tracking[metric] + s["abs"])}
    a, b = run.case.window_s
    n = len(window_slices(round((b - a) * GRID_HZ)))
    win = f"{run.case.window_s[0]:g} to {run.case.window_s[1]:g} s of bag {run.meta.get('bag', '?')}"
    sources = {
        "tracking_rms": f"robot (measured - target) RMS, mean over the 2 s windows of {win}; "
                        f"tolerance {t['tracking_rms']['rel']:.0%} + {t['tracking_rms']['abs'] * 1e3:g} mrad",
        "sim_real_rms": f"a fraction {t['sim_real_rms']['fraction_of_tracking']:g} of the robot's tracking RMS "
                        f"(the sim explains at least {1 - t['sim_real_rms']['fraction_of_tracking'] ** 2:.0%} of its "
                        f"tracking-error power) + {t['sim_real_rms']['abs'] * 1e3:g} mrad, same windows",
        "response": f"robot response of the symmetric hip flexion, mean over the windows of {win}; gain "
                    f"+-{t['response_gain']['rel']:.0%}, phase +-{t['response_phase_deg']['abs']:g} deg",
        "imu": f"robot IMU, mean over the windows of {win}; sway {t['imu_sway_deg']['rel']:.0%} + "
               f"{t['imu_sway_deg']['abs']:g} deg, pitch-rate RMS {t['pitch_rate_rms']['rel']:.0%} + "
               f"{t['pitch_rate_rms']['abs']:g} rad/s",
    }
    sources["response"] += f"; metrics the robot gives on fewer than {MIN_BAND_WINDOWS} windows have no band"
    return bands, sources | {"windows": str(n)}


def report_lines(case: Case, verdicts: list[Verdict]) -> list[str]:
    c = counts(verdicts)
    head = ", ".join(f"{n} {k}" for k, n in sorted(c.items()))
    lines = [f"== {case.id} [{case.support}; {case.scoring}]: {case_state(case, verdicts)} ({head})"]
    for v in verdicts:
        if v.status in ("info",):
            continue
        r = v.result
        robot = "" if r is None or r.robot is None else f" robot {r.robot.mean:.4g}"
        sim = "" if r is None or r.sim is None else f" sim {r.sim.mean:.4g}"
        lines.append(f"  {v.status.upper():7s} {v.metric}:{robot}{sim}  {v.reason}")
    return lines


__all__ = ["Case", "Cases", "CaseRun", "Verdict", "load_cases", "resolve_bag", "run_case", "robot_only",
           "judge", "propose_bands", "report_lines", "case_passed", "case_state", "counts"]


HEADER = """\
# Sim-versus-robot regression cases (humanoid-sim regress). Read docs/replay-and-sysid.md, "Robot regression".
#
# Written by `humanoid-sim regress --propose`: it keeps each case's id, description, bag, window and warm-up, and
# rewrites `bands`, `sources` and `robot_windows` from the robot's values in the bag and the tolerances below.
# `support` is how the robot was carried; `scoring` is open_loop where a held or hung base represents the recording
# (gantry hang, excitations) and closed_loop where it does not (standing on the floor, rope slack): those cases are not
# scored by a replay with a prescribed base, and their bands are what a closed-loop run is held to.
# A band is {ref, lo, hi}: ref is the robot's value over the case's 2 s windows (mean over them), lo and hi bound the
# sim's value (same windows, mean over them). Phase bands are in degrees, circular around ref.
# Bags live on S3 (`s3`); `path` is where they sit under the recordings directory; `sha256` pins their contents.
"""


class _Flow(dict):
    pass


def _flow(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


yaml.add_representer(_Flow, _flow, Dumper=yaml.SafeDumper)


def dump_cases(cases: Cases) -> str:
    """The cases file's text."""
    doc = {
        "format": FORMAT,
        "version": cases.version,
        "tolerances": {k: _Flow(v) for k, v in cases.tolerances.items()},
        "cases": [{
            "id": c.id, "description": c.description, "support": c.support, "scoring": c.scoring,
            "bag": dict(c.bag), "window_s": list(c.window_s), "warmup_s": c.warmup_s, "robot_windows": c.robot_windows, "sources": c.sources,
            "bands": {m: _Flow({k: round(float(v), 9) for k, v in b.items()}) for m, b in sorted(c.bands.items())},
        } for c in cases.cases],
    }
    return HEADER + yaml.dump(doc, Dumper=yaml.SafeDumper, sort_keys=False, width=110, default_flow_style=False)


def propose_file(cases: Cases, recordings: Path, desc, drives: list[str]) -> Cases:
    """The cases with bands from the robot's values (no sim), for every case; bags are checked against their sha256."""
    out = []
    for case in cases.cases:
        run = robot_only(case, resolve_bag(case, recordings), desc, drives)
        bands, sources = propose_bands(run, cases.tolerances)
        windows = int(sources.pop("windows"))
        out.append(Case(case.id, case.description, case.bag, case.window_s, case.warmup_s, sources, bands, windows,
                        case.support, case.scoring))
    return Cases(cases.version, cases.tolerances, out)
