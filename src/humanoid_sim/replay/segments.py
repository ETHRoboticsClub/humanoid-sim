"""Cut an ``exc-*`` capture into time windows, so a fit on one kind of excitation can be scored on the others.

A PACE fit that only ever sees one excitation can trade parameters against each other (armature against damping,
friction against bias) and still match it; scored on a window it was not fitted on, such a fit shows it. Windows are
seconds from the capture's first command frame, ``[start, end)``. :func:`plan_windows` chooses them:

* explicit windows, when given;
* else the capture's pieces, when it has several: command frames more than ``gap_s`` apart, or the signal clock
  (``t_signal_s``) starting over, separate one piece of the profile from the next (a slow segment, a second band);
* else, for one log chirp (``effective_action.sweep``: equal time per octave), ``bands`` windows of equal length,
  each covering an equal share of the octaves, named by their frequency range;
* else ``bands`` windows of equal length.

Windows shorter than ``min_s`` are dropped: the objective skips each replay's first second.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Each stream's own stamp column; rows outside a window are left out. Other files are copied.
STAMPS = {"commands.csv": "t_frame_ros_ns", "pdo_feedback.csv": "t_source_ros_ns", "imu.csv": "t_source_ros_ns",
          "acks.csv": "t_ack_ros_ns", "joints.csv": "t_source_ros_ns", "drives.csv": "t_source_ros_ns"}


@dataclass(frozen=True)
class Window:
    name: str
    start_s: float  # from the first command frame
    end_s: float
    note: str = ""


def parse_window(text: str) -> Window:
    """``NAME:START:END`` (seconds from the first command frame)."""
    name, start, end = text.split(":")
    return Window(name, float(start), float(end), "explicit")


def _frames(capture_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Each command frame's stamp (ns) and signal time (s, NaN when absent), in stamp order."""
    stamps: dict[int, float] = {}
    with (Path(capture_dir) / "commands.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = int(r["t_frame_ros_ns"])
            if t not in stamps:
                sig = r.get("t_signal_s", "")
                stamps[t] = float(sig) if sig not in ("", None) else math.nan
    t = np.array(sorted(stamps), dtype=np.int64)
    return t, np.array([stamps[int(v)] for v in t])


def plan_windows(capture_dir: Path, *, explicit: list[Window] | None = None, bands: int = 3, gap_s: float = 0.5,
                 min_s: float = 8.0) -> list[Window]:
    """The windows of one capture (module docstring)."""
    if explicit:
        return list(explicit)
    t_ns, sig = _frames(capture_dir)
    if len(t_ns) < 2:
        raise ValueError(f"{capture_dir}: fewer than two command frames")
    t = (t_ns - t_ns[0]) / 1e9
    breaks = np.flatnonzero((np.diff(t) > gap_s) | (np.diff(sig) < -gap_s)) + 1
    edges = [0, *breaks.tolist(), len(t)]
    pieces = [(t[a], t[b - 1] + (t[b] - t[b - 1] if b < len(t) else np.median(np.diff(t)))) for a, b in
              zip(edges[:-1], edges[1:], strict=True)]
    out = []
    if len(pieces) > 1:
        out = [Window(f"p{i}", float(a), float(b), "piece") for i, (a, b) in enumerate(pieces)]
    else:
        manifest = json.loads((Path(capture_dir) / "manifest.json").read_text())
        sweep = (manifest.get("effective_action") or {}).get("sweep")
        start, end = pieces[0]
        cuts = np.linspace(start, end, bands + 1)
        if sweep and sweep.get("f0_hz") and sweep.get("f1_hz") and sweep.get("duration"):
            f0, f1, duration_s = float(sweep["f0_hz"]), float(sweep["f1_hz"]), float(sweep["duration"]) / 1e9
            # Signal time 0 is the sweep's start; equal shares of the sweep are equal shares of its octaves.
            ok = np.isfinite(sig)
            t_sig0 = float(np.interp(0.0, sig[ok], t[ok])) if ok.any() else start
            cuts = np.clip(t_sig0 + np.linspace(0.0, duration_s, bands + 1), start, end)
            cuts[0], cuts[-1] = start, end
            for i in range(bands):
                fa = f0 * (f1 / f0) ** (i / bands)
                fb = f0 * (f1 / f0) ** ((i + 1) / bands)
                out.append(Window(f"b{i}", float(cuts[i]), float(cuts[i + 1]), f"log chirp {fa:.2g}-{fb:.2g} Hz"))
        else:
            out = [Window(f"w{i}", float(cuts[i]), float(cuts[i + 1]), "equal time") for i in range(bands)]
    return [w for w in out if w.end_s - w.start_s >= min_s]


def cut_window(src: Path, dst_parent: Path, window: Window) -> Path:
    """``dst_parent/<src name>-<window name>``: the capture's rows inside ``window``, its dataset id suffixed and
    the window recorded in the manifest (``segment``)."""
    src = Path(src)
    t0 = int(_frames(src)[0][0])
    lo, hi = t0 + int(round(window.start_s * 1e9)), t0 + int(round(window.end_s * 1e9))
    dst = Path(dst_parent) / f"{src.name}-{window.name}"
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    tmp = Path(dst_parent) / f".{dst.name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    counts = {}
    for path in sorted(src.iterdir()):
        if path.suffix == ".zip" or path.is_dir():
            continue
        if path.name in STAMPS:
            with path.open(newline="") as f, (tmp / path.name).open("w", newline="") as g:
                reader, writer = csv.reader(f), csv.writer(g)
                header = next(reader)
                i = header.index(STAMPS[path.name])
                writer.writerow(header)
                n = 0
                for row in reader:
                    if lo <= int(row[i]) < hi:
                        writer.writerow(row)
                        n += 1
            counts[path.name] = n
        elif path.name == "manifest.json":
            continue
        else:
            shutil.copy2(path, tmp / path.name)
    manifest = json.loads((src / "manifest.json").read_text())
    manifest["segment"] = asdict(window) | {"of": manifest["dataset_id"], "rows": counts}
    manifest["dataset_id"] = f"{manifest['dataset_id']}-{window.name}"
    manifest["display_name"] = f"{manifest.get('display_name', '')} {window.name} {window.start_s:g}-{window.end_s:g} s"
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tmp.rename(dst)
    return dst
