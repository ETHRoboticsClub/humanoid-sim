"""The holdout report of a set of ``fit-pace`` results (``fit-pace.json``), each fitted on one window of the
excitation and scored on the others (:mod:`humanoid_sim.replay.segments`).

For every fit: the error on each held-out window per joint group, at the description's values and at the fit's;
every searched entry at its bound; each value against the description's and, for the armature, the drive family's
datasheet. Across fits: how far the fitted values agree. A parameter set that only improves the window it was
fitted on, or that sits at its bounds, or that the fits disagree on by more than their bound allows, is not
identified by these captures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Reflected rotor inertia at the drive output (kg m^2), from the maxon datasheets as humanoid-urdf URDF_6 (b14158e)
# ``sources.datasheet`` quotes them. The HEJ 70's is a preliminary datasheet's.
DATASHEET_ARMATURE = {
    "HEJ90-48-140": (0.05, "maxon HEJ 90-48-140, Jan 2025 rev 1.5"),
    "HEJ70-48-50": (0.0055, "maxon HEJ 70-48-50, preliminary, 55 kgcm2"),
    "HEJ50-48-30": (0.0056, "maxon HEJ 50-48-30, Sep 2025 rev 0.9, 56 kgcm2"),
}
AT_BOUND = 0.98  # |x| at or beyond this is at the bound
PARAMS = ("armature", "damping", "friction", "friction_ratio")


def drive_groups(desc, drives: list[str]) -> dict[str, list[str]]:
    """Joint groups by drive: a 1:1 drive under its joint's name without the side, the ankles' cranks together."""
    out: dict[str, list[str]] = {}
    for d in drives:
        drive = desc.drives[d]
        if drive.crank is not None:
            name = "ankle_cranks"
        else:
            j = drive.joints[0]
            name = j[:-2] if j.endswith(("_l", "_r")) else j
        out.setdefault(name, []).append(d)
    return out


def joint_families(desc, joints: list[str]) -> dict[str, str | None]:
    """Each solver joint's drive family: a 1:1 joint's drive's, a crank shaft's crank's; None for a hinge."""
    out = {}
    shafts = {}
    if desc.usd_path is not None:
        from humanoid_sim.model.usd import crank_joints

        shafts = {joint: drive for drive, joint in crank_joints(desc).items()}
    for j in joints:
        if j in shafts:
            out[j] = desc.drives[shafts[j]].family
        elif j in desc.joints and desc.joints[j].device is not None:
            out[j] = desc.drives[desc.joints[j].device].family
        else:
            out[j] = None
    return out


def _group_mrad(drives, groups, rms) -> dict[str, float]:
    rms = np.asarray(rms)
    return {g: 1e3 * float(np.mean([rms[drives.index(d)] for d in ds])) for g, ds in groups.items()}


def summarize(fit_dirs: list[Path], desc) -> dict:
    """Everything the report prints, as data."""
    fits = {}
    for path in fit_dirs:
        fits[Path(path).name] = json.loads((Path(path) / "fit-pace.json").read_text())
    if not fits:
        raise ValueError("no fits")
    first = next(iter(fits.values()))
    drives, joints = first["drive_order"], first["joint_order"]
    groups = drive_groups(desc, drives)
    families = joint_families(desc, joints)
    scored = set(first.get("scored", drives))
    out = {"fits": {}, "groups": {g: ds for g, ds in groups.items()}, "drives": drives, "joints": joints}
    holdout_rows = []
    for name, d in fits.items():
        x = np.asarray(d["x"])
        free = set(d.get("free", d["names"]))
        f = {"fitted_on": d["captures"], "objective_start_mrad": 1e3 * d["objective_start"],
             "objective_mrad": 1e3 * d["objective"], "delay_ms": {k: v / 1e6 for k, v in d["values"]["delay_ns"].items()},
             "holdout": {}, "at_bound": []}
        h = d.get("holdout")
        if h:
            for c, cap in enumerate(h["captures"]):
                before = _group_mrad(drives, groups, h["rms_start"][c])
                after = _group_mrad(drives, groups, h["rms_fit"][c])
                total = (1e3 * float(np.mean([h["rms_start"][c][drives.index(x_)] for x_ in scored])),
                         1e3 * float(np.mean([h["rms_fit"][c][drives.index(x_)] for x_ in scored])))
                f["holdout"][cap] = {"scored_mrad": total, "groups_mrad": {g: (before[g], after[g]) for g in groups}}
                holdout_rows.append((name, cap, before, after))
        for i, n in enumerate(d["names"]):
            if n in free and abs(x[i]) >= AT_BOUND:
                f["at_bound"].append(f"{n} ({'upper' if x[i] > 0 else 'lower'})")
        f["values"] = {j: {p: d["values"]["joints"][j][p] for p in PARAMS} for j in joints}
        f["bias_mrad"] = {dr: 1e3 * d["values"]["drives"][dr]["bias_rad"] for dr in drives}
        out["fits"][name] = f
    start = first.get("start") or {}
    table = {}
    for j in joints:
        row = {"family": families[j], "description": start.get(j)}
        if families[j] in DATASHEET_ARMATURE:
            row["datasheet_armature"] = DATASHEET_ARMATURE[families[j]][0]
        for p in PARAMS:
            vals = [f["values"][j][p] for f in out["fits"].values()]
            pos = [v for v in vals if v > 0]
            row[f"{p}_spread"] = (max(pos) / min(pos)) if p != "friction_ratio" and len(pos) > 1 else None
        table[j] = row
    out["table"] = table
    agg = {}
    for g in groups:
        b = [r[2][g] for r in holdout_rows]
        a = [r[3][g] for r in holdout_rows]
        if b:
            agg[g] = {"before_mrad": float(np.mean(b)), "after_mrad": float(np.mean(a)),
                      "improved": int(sum(ai < bi for ai, bi in zip(a, b, strict=True))), "of": len(b)}
    out["holdout_by_group"] = agg
    out["identity"] = {k: first.get(k) for k in ("robot", "description_digest", "description_revision",
                                                 "description_root", "usd", "asset_key", "source_usd_sha256",
                                                 "source_usd", "self_collision", "frame_stamps", "solver",
                                                 "integrator", "period_ns", "num_envs", "generations", "search",
                                                 "scored", "fixed", "base")}
    return out


def render(summary: dict) -> str:
    """The report as Markdown."""
    s = summary
    lines = ["# PACE holdout report", ""]
    ident = s["identity"]
    lines += ["## Plant", ""] + [f"- {k}: `{v}`" for k, v in ident.items() if k not in ("scored", "fixed")]
    unscored = sorted(set(s["drives"]) - set(ident.get("scored") or s["drives"]))
    lines += [f"- drives left out of the objective: {', '.join(unscored) if unscored else 'none'}",
              f"- entries fixed by hand: {', '.join(ident.get('fixed') or []) or 'none'}", ""]
    lines += ["## Holdout error by joint group", "",
              "Mean over every fit and every window it was not fitted on; position RMS in mrad, description's "
              "values -> fit's.", "", "| group | before | after | change | fits improved |", "|---|---|---|---|---|"]
    for g, r in s["holdout_by_group"].items():
        ch = 100 * (r["after_mrad"] / r["before_mrad"] - 1) if r["before_mrad"] else math.nan
        lines.append(f"| {g} | {r['before_mrad']:.2f} | {r['after_mrad']:.2f} | {ch:+.0f}% | {r['improved']}/{r['of']} |")
    lines.append("")
    for name, f in s["fits"].items():
        lines += [f"## Fit `{name}`", "", f"Fitted on {', '.join(f['fitted_on'])}: {f['objective_start_mrad']:.2f} -> "
                  f"{f['objective_mrad']:.2f} mrad. Delays (ms): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in f["delay_ms"].items()) + ".", ""]
        if f["holdout"]:
            caps = list(f["holdout"])
            lines += ["| group | " + " | ".join(caps) + " |", "|---" * (len(caps) + 1) + "|"]
            lines.append("| scored drives | " + " | ".join(
                f"{f['holdout'][c]['scored_mrad'][0]:.2f} -> {f['holdout'][c]['scored_mrad'][1]:.2f}" for c in caps) + " |")
            for g in s["groups"]:
                lines.append(f"| {g} | " + " | ".join(
                    f"{f['holdout'][c]['groups_mrad'][g][0]:.2f} -> {f['holdout'][c]['groups_mrad'][g][1]:.2f}"
                    for c in caps) + " |")
            lines.append("")
        lines += [f"At a bound ({len(f['at_bound'])}): " + (", ".join(f["at_bound"]) if f["at_bound"] else "none"), ""]
    names = list(s["fits"])
    lines += ["## Values against the description and the datasheets", "",
              "Armature kg m^2, damping N m s/rad, friction N m; each fit's value, the description's, and for the "
              "armature the family's datasheet. Spread: the largest over the smallest fitted value.", "",
              "| joint | family | param | description | datasheet | " + " | ".join(names) + " | spread |",
              "|---" * (6 + len(names)) + "|"]
    for j, row in s["table"].items():
        for p in ("armature", "damping", "friction"):
            desc_v = (row["description"] or {}).get(p)
            if desc_v == 0.0:
                continue
            ds = row.get("datasheet_armature") if p == "armature" else None
            vals = [s["fits"][n]["values"][j][p] for n in names]
            spread = row[f"{p}_spread"]
            lines.append(f"| {j} | {row['family'] or ''} | {p} | {_g(desc_v)} | {_g(ds)} | "
                         + " | ".join(_g(v) for v in vals) + f" | {_g(spread)} |")
    lines += ["", "## Bias (mrad)", "", "| drive | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for dr in s["drives"]:
        lines.append(f"| {dr} | " + " | ".join(f"{s['fits'][n]['bias_mrad'][dr]:+.2f}" for n in names) + " |")
    return "\n".join(lines) + "\n"


def _g(v) -> str:
    return "" if v is None else f"{v:.4g}"
