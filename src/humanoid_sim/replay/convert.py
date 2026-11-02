"""Convert an ``exc-*`` capture recorded on an older robot description into this description's joint names.

The robot's Controls may still run an older description than the one simulated here: at the time of writing it pins
humanoid-urdf URDF_6 (b14158e, ``legacy/urdf6-b14158e-actuators.yaml``, a byte copy of that revision's
``ethrc_v1/actuators.yaml``), while humanoid-sim simulates V1's native USD. Only ``commands.csv`` depends on the
description (joint names, signs, standing offsets, the ankle hinges); the drive streams are per drive and are copied
unchanged. The conversion is made at the drive level, so it never trusts one description's conventions over the
other's:

* a 1:1 joint is matched to the new one by its drive; the drive target the master sent,
  ``sign_old · (q_old + offset_old)``, is kept: ``q_new = sign_new · sign_old · (q_old + offset_old) - offset_new``
  (this is where URDF_6's ``waist_upperbody_yaw`` becomes ``torso_axial_rotation`` with the sign flipped);
* an ankle (``ankles="cranks"``) goes through the old description's crank transmission to the crank targets the
  master sent (a hardware check on 2026-10-29 confirmed URDF_6's transmission against the robot), and back through
  this description's transmission to its hinge angles. The crank targets are kept exactly whatever the two
  transmissions' conventions are: the new transmission is only the inverse of the one the replay rebuilds the crank
  targets with. Where it cannot reach a crank pair the conversion stops with the residual;
* ``ankles="hold"`` holds both ankles at the first converted frame for the whole capture instead, so the simulated
  feet stay still. The fit must then leave the ankle drives out (``fit-pace --skip-drives`` and ``--fix``).

:func:`convert_capture` checks the result: the drive targets this description's plant rebuilds from the converted
commands (:func:`~humanoid_sim.replay.replay.drive_targets`) against the old description's, per drive, and writes
both, with the inputs' digests, to ``conversion.json`` beside the converted capture.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

LEGACY = Path(__file__).resolve().parent / "legacy"
LEGACY_TABLES = {"urdf6-b14158e": LEGACY / "urdf6-b14158e-actuators.yaml"}
ANKLE_MODES = ("cranks", "hold")
CONVERTED_COLUMNS = ("q_nominal_rad", "q_mirror_rad")


def legacy_table(name_or_path: str) -> tuple[dict, Path]:
    """An older ``actuators.yaml``: a name in :data:`LEGACY_TABLES` or a path."""
    path = LEGACY_TABLES.get(name_or_path, Path(name_or_path))
    if not path.is_file():
        raise ValueError(f"no legacy description {name_or_path!r}: one of {sorted(LEGACY_TABLES)} or a path")
    return yaml.safe_load(path.read_text()), path


@dataclass
class JointMap:
    """How one description's command frames become another's: per 1:1 joint ``(new joint, factor, offset)``,
    per ankle side the old and the new ``(pitch, roll)`` hinges and their transmissions."""

    linear: dict[str, tuple[str, float, float]]
    old_drive: dict[str, tuple[str, float, float]]  # old 1:1 joint -> (drive, sign, standing offset) in the old table
    old_ankles: dict[str, tuple[str, str]]
    new_ankles: dict[str, tuple[str, str]]
    old_offsets: dict[str, np.ndarray]
    new_offsets: dict[str, np.ndarray]
    old_tr: dict = field(repr=False)
    new_tr: dict = field(repr=False)

    @property
    def old_joints(self) -> set[str]:
        return set(self.linear) | {j for pair in self.old_ankles.values() for j in pair}

    def ankle_cranks(self, side: str, q_old: np.ndarray) -> np.ndarray:
        """The crank targets ``[F, 2]`` (upper, lower) the old description sends for old hinge angles ``[F, 2]``."""
        q = torch.as_tensor(q_old + self.old_offsets[side], dtype=torch.float64)
        return self.old_tr[side].cranks(q).numpy()

    def ankle_joints(self, side: str, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """This description's hinge angles ``[F, 2]`` (semantic, minus its standing offsets) for crank targets
        ``[F, 2]``, and the largest crank residual (rad)."""
        tr = self.new_tr[side]
        th = torch.as_tensor(theta, dtype=torch.float64)
        q = tr.joints_from_cranks(th, torch.zeros(1, 2, dtype=torch.float64), iterations=30)
        resid = float((tr.cranks(q) - th).abs().max()) if len(th) else 0.0
        return q.numpy() - self.new_offsets[side], resid


def joint_map(desc, old: dict) -> JointMap:
    """The map from the older ``actuators.yaml`` table ``old`` to ``desc``, matched by drive."""
    from humanoid_sim.transmissions.ankle import AnkleTransmission, mirror_right

    def offset(v):
        return float(v.get("standing_offset_rad") or 0.0)

    old_by_dev = {v["device"]: (j, v) for j, v in old["joints"].items() if v.get("device")}
    linear, old_drive = {}, {}
    for name, joint in desc.joints.items():
        if joint.device is None:
            continue
        if joint.device not in old_by_dev:
            raise ValueError(f"drive {joint.device} ({name}) is not in the old description")
        oj, ov = old_by_dev[joint.device]
        s = float(joint.sign) * float(ov["sign"])
        linear[oj] = (name, s, s * offset(ov) - float(joint.standing_offset_rad or 0.0))
        old_drive[oj] = (joint.device, float(ov["sign"]), offset(ov))
    am = old.get("ankle_model")
    old_ankles, new_ankles, old_tr, new_tr, old_off, new_off = {}, {}, {}, {}, {}, {}
    for linkage in desc.linkages:
        side = linkage.side
        up, lo = old["drives"].get(linkage.upper), old["drives"].get(linkage.lower)
        if up is None or lo is None or up.get("crank") != "upper" or lo.get("crank") != "lower":
            raise ValueError(f"old description has no upper/lower crank drives {linkage.upper}, {linkage.lower}")
        if list(up["joints"]) != list(lo["joints"]):
            raise ValueError(f"old cranks of side {side} move different joints")
        if am is None:
            raise ValueError("old description has no ankle_model")
        old_ankles[side] = tuple(up["joints"])
        new_ankles[side] = tuple(linkage.joints)
        geometry = am["right_geometry"] if side == "r" else mirror_right(am["right_geometry"])
        old_tr[side] = AnkleTransmission(geometry, am["semantic_to_kinematic_sign"][side],
                                         am["semantic_neutral_rad"][side], dtype=torch.float64)
        new_tr[side] = desc.ankle_transmission(side, dtype=torch.float64)
        old_off[side] = np.array([offset(old["joints"][j]) for j in old_ankles[side]])
        new_off[side] = np.array([float(desc.joints[j].standing_offset_rad or 0.0) for j in new_ankles[side]])
    return JointMap(linear, old_drive, old_ankles, new_ankles, old_off, new_off, old_tr, new_tr)


def is_legacy(capture_dir: Path, desc) -> bool:
    """Whether the capture's commands name joints this description does not have."""
    with (Path(capture_dir) / "commands.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        names = {r["joint_id"] for _, r in zip(range(200), reader)}
    return not names <= set(desc.joints)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_capture(src: Path, dst_parent: Path, desc, old: dict | str = "urdf6-b14158e", *,
                    ankles: str = "cranks") -> Path:
    """Copy capture ``src`` into ``dst_parent/<src name>`` with ``commands.csv`` in ``desc``'s joint names
    (module docstring) and ``conversion.json`` beside it. Returns the new capture directory."""
    if ankles not in ANKLE_MODES:
        raise ValueError(f"ankles: one of {ANKLE_MODES}")
    src = Path(src)
    old_path = None
    if isinstance(old, str):
        old, old_path = legacy_table(old)
    jm = joint_map(desc, old)
    with (src / "commands.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    col = {n: i for i, n in enumerate(header)}
    unknown = sorted({r[col["joint_id"]] for r in rows} - jm.old_joints)
    if unknown:
        raise ValueError(f"{src.name}: commands name joints the old description does not map: {unknown}")

    # Ankle rows by (seq, stamp, side): a frame's pitch and roll convert together.
    out, frames = [], {}
    side_of = {j: s for s, pair in jm.old_ankles.items() for j in pair}
    for r in rows:
        j = r[col["joint_id"]]
        if j in jm.linear:
            nj, s, off = jm.linear[j]
            r = list(r)
            r[col["joint_id"]] = nj
            for c in CONVERTED_COLUMNS:
                if c in col and r[col[c]] != "":
                    r[col[c]] = repr(s * float(r[col[c]]) + off)
            out.append(r)
        else:
            frames.setdefault((r[col["seq"]], r[col["t_frame_ros_ns"]], side_of[j]), {})[j] = r
    report = {"ankles": ankles, "ankle_frames": {}, "ankle_residual_rad": {}}
    for side, (pitch, roll) in jm.old_ankles.items():
        keys = [k for k in frames if k[2] == side]
        keys.sort(key=lambda k: int(k[1]))
        for k in keys:
            if set(frames[k]) != {pitch, roll}:
                raise ValueError(f"{src.name}: frame {k[0]} names one hinge of ankle {side} without the other")
        converted = {}
        for c in CONVERTED_COLUMNS:
            if c not in col:
                continue
            present = [k for k in keys if frames[k][pitch][col[c]] != "" and frames[k][roll][col[c]] != ""]
            q_old = np.array([[float(frames[k][pitch][col[c]]), float(frames[k][roll][col[c]])] for k in present])
            if ankles == "hold" and len(present):
                q_old = np.repeat(q_old[:1], len(present), axis=0)
            q_new, resid = jm.ankle_joints(side, jm.ankle_cranks(side, q_old.reshape(-1, 2)))
            report["ankle_residual_rad"][f"{side}.{c}"] = resid
            if resid > 1e-6:
                raise ValueError(f"{src.name}: this description's ankle {side} cannot reach the crank targets of "
                                 f"{c} (residual {resid:.2e} rad); its transmission does not cover the old one's "
                                 "range (try ankles='hold')")
            converted[c] = dict(zip(present, q_new, strict=True))
        for k in keys:
            for i, nj in enumerate(jm.new_ankles[side]):
                r = list(frames[k][(pitch, roll)[i]])
                r[col["joint_id"]] = nj
                for c, values in converted.items():
                    if k in values:
                        r[col[c]] = repr(float(values[k][i]))
                out.append(r)
        report["ankle_frames"][side] = len(keys)

    dst = Path(dst_parent) / src.name
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    dst_parent = Path(dst_parent)
    dst_parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_parent / f".{src.name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("*.zip", "conversion.json"))
    with (tmp / "commands.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    report |= {
        "source": str(src.resolve()), "source_commands_sha256": _sha256(src / "commands.csv"),
        "old_description": {"robot": old.get("robot"), "urdf": old.get("urdf"),
                            "path": None if old_path is None else str(old_path),
                            "sha256": None if old_path is None else _sha256(old_path)},
        "description": desc.identity(),
        "linear": {oj: {"joint": nj, "factor": s, "offset_rad": off} for oj, (nj, s, off) in jm.linear.items()},
        "ankle_joints": {s: {"old": list(jm.old_ankles[s]), "new": list(jm.new_ankles[s])} for s in jm.old_ankles},
        "rows": {"in": len(rows), "out": len(out)},
    }
    report["drive_targets"] = check_targets(src, tmp, desc, jm)
    (tmp / "conversion.json").write_text(json.dumps(report, indent=1))
    tmp.rename(dst)
    return dst


def old_drive_targets(commands, jm: JointMap, drives: list[str], desc) -> tuple[np.ndarray, np.ndarray]:
    """The drive targets ``[F, D]`` the old description's master sends for old command frames, from the first
    frame that names every old joint, and their stamps."""
    old_joints = sorted(jm.old_joints)
    cols = [commands.joints.index(j) for j in old_joints]
    q = commands.positions_rad[:, cols]
    full = np.isfinite(q).all(axis=1)
    first = int(np.argmax(full))
    q = dict(zip(old_joints, q[first:].T, strict=True))
    out = np.zeros((len(q[old_joints[0]]), len(drives)))
    for oj, (drive, sign, offset) in jm.old_drive.items():
        out[:, drives.index(drive)] = sign * (q[oj] + offset)  # the old master: sign (q + standing offset)
    for linkage in desc.linkages:
        pitch, roll = jm.old_ankles[linkage.side]
        theta = jm.ankle_cranks(linkage.side, np.stack([q[pitch], q[roll]], axis=1))
        out[:, drives.index(linkage.upper)] = theta[:, 0]
        out[:, drives.index(linkage.lower)] = theta[:, 1]
    return out, commands.t_ns[first:]


def check_targets(src: Path, dst: Path, desc, jm: JointMap) -> dict:
    """Per drive, the largest difference (rad) between the targets this description's plant rebuilds from the
    converted capture and the old description's from the original, over the frames both name in full."""
    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.replay.exc import ExcitationCapture
    from humanoid_sim.replay.replay import drive_targets

    plant = PlantCore(desc, list(desc.joints), 1, PlantCfg(dtype=torch.float64))
    drives = list(plant.map.drive_names)
    new, t_new = drive_targets(plant, ExcitationCapture(dst).commands())
    old, t_old = old_drive_targets(ExcitationCapture(src).commands(), jm, drives, desc)
    common, i_new, i_old = np.intersect1d(t_new, t_old, return_indices=True)
    diff = np.abs(new[i_new] - old[i_old]) if len(common) else np.zeros((0, len(drives)))
    return {"frames": len(common), "max_abs_rad": dict(zip(drives, diff.max(axis=0).tolist() if len(common)
                                                           else [None] * len(drives), strict=True))}
