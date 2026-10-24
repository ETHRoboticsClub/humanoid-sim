"""The lidar leaves the cameras' splat as bright as it was: the G1 in a box room covered by a splat, with a scan mesh
that closes the room over the robot. With the lidar on, the walls and the mesh stay on the stage hidden from camera
rays (``sensors/room_rays.py``); with it off they are not there. Geometry hidden from the cameras that still cast
shadows would dim the splat by the share of the dome it hides, here all of it above the walls, so each camera's mean
luminance at the same plant time must not change with the lidar.

Each run is ``examples/rtx_sensors.py`` in a process of its own (Kit starts once a process), the splat made by
``humanoid-sim splat-usd``; this process starts neither Kit nor USD. Needs the Isaac Sim image
(``docker/Dockerfile.kit``) on an RTX GPU:

    pytest -m rtx tests/test_rtx_lidar_brightness.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("isaacsim", reason="RTX sensors need the Isaac Sim image (docker/Dockerfile.kit)")
if shutil.which("nvidia-smi") is None:
    pytest.skip("RTX sensors need an NVIDIA GPU with RT cores", allow_module_level=True)

from rtx_oracles import CAMERAS, G1, SETTLE_S, WALLS, write_box_obj  # noqa: E402

pytestmark = pytest.mark.rtx

REPO = Path(__file__).resolve().parents[1]
SECONDS = SETTLE_S + 0.4
# The scan mesh: a closed box just inside the walls and over their tops, so every Gaussian is under it.
SCAN_SHELL = ((0.0, 0.0, 1.6), (2.95, 2.45, 1.6))
# The splat: grey Gaussians every SPLAT_SPACING_M over the floor and the walls' inner faces.
SPLAT_SPACING_M = 0.1
SPLAT_SIGMA_M = 0.06  # more than half the spacing: the surface has no gaps
SPLAT_OPACITY = 0.95
SPLAT_GREY = 0.6  # linear base colour
SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic, 1 / (2 sqrt(pi)): colour = 0.5 + SH_C0 * f_dc (3DGS)
LUMA_BT601 = (0.299, 0.587, 0.114)
# A camera's mean luminance with and without the lidar (8-bit levels): the renders differ only by the renderer's
# sampling noise, averaged over at least 640 x 400 pixels to well under a level; hiding the dome from the splat
# darkens it by tens of levels.
SAME_LEVELS = 1.0
LIT_LEVELS = 20.0  # the splat must show: a black frame would match itself
PINNED_EXPOSURE = {"/rtx/post/histogram/enabled": False, "/rtx/post/tonemap/op": 6,  # ACES
                   "/rtx/post/tonemap/filmIso": 100.0, "/rtx/post/tonemap/exposureTime": 0.02,
                   "/rtx/post/tonemap/fNumber": 5.0}


def splat_positions() -> np.ndarray:
    """``[N, 3]``: the floor and the four walls' inner faces of rtx_oracles.WALLS, every SPLAT_SPACING_M."""
    x_in = min(c[0] - h[0] for c, h in WALLS if c[0] > 0)
    y_in = min(c[1] - h[1] for c, h in WALLS if c[1] > 0)
    top = max(c[2] + h[2] for c, h in WALLS)
    xs = np.arange(-x_in, x_in + 1e-9, SPLAT_SPACING_M)
    ys = np.arange(-y_in, y_in + 1e-9, SPLAT_SPACING_M)
    zs = np.arange(0.0, top + 1e-9, SPLAT_SPACING_M)
    floor = [(x, y, 0.0) for x in xs for y in ys]
    x_walls = [(s * x_in, y, z) for s in (-1, 1) for y in ys for z in zs]
    y_walls = [(x, s * y_in, z) for s in (-1, 1) for x in xs for z in zs]
    return np.array(floor + x_walls + y_walls)


def write_splat_ply(path: Path, positions: np.ndarray) -> None:
    """A degree-0 3DGS PLY (log scales, logit opacities, real-first quaternions) of isotropic grey Gaussians."""
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = np.zeros((len(positions), len(names)), dtype=np.float32)
    rows[:, 0:3] = positions
    rows[:, 3:6] = (SPLAT_GREY - 0.5) / SH_C0
    rows[:, 6] = np.log(SPLAT_OPACITY / (1 - SPLAT_OPACITY))
    rows[:, 7:10] = np.log(SPLAT_SIGMA_M)
    rows[:, 10] = 1.0  # identity rotation
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    path.write_bytes(header.encode("ascii") + rows.tobytes())


def luminance(png: Path) -> float:
    from PIL import Image

    image = np.asarray(Image.open(png), dtype=float)
    return float((image[..., :3] @ LUMA_BT601).mean() if image.ndim == 3 else image.mean())


def run(cmd: list[str], cwd: Path) -> None:
    done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, f"{' '.join(cmd)}\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("lidar_brightness")
    room, scan, ply, usd = d / "room.npz", d / "scan.obj", d / "room.ply", d / "room.usdc"
    np.savez(room, center=np.array([c for c, _ in WALLS]), half=np.array([h for _, h in WALLS]),
             yaw=np.zeros(len(WALLS)), wall=np.ones(len(WALLS), dtype=bool))
    write_box_obj(scan, *SCAN_SHELL)
    write_splat_ply(ply, splat_positions())
    run([sys.executable, "-m", "humanoid_sim.cli", "splat-usd", str(ply), str(usd)], REPO)
    out = {}
    for name, extra in (("on", []), ("off", ["--no-lidar"])):
        images, summary = d / name, d / f"{name}.json"
        run([sys.executable, "examples/rtx_sensors.py", "--robot", G1, "--sensors", CAMERAS, "--scene", str(room),
             "--scan-mesh", str(scan), "--splat", str(usd), "--seconds", str(SECONDS), "--images", str(images),
             "--summary", str(summary), *extra], REPO)
        out[name] = {"summary": json.loads(summary.read_text()),
                     "luminance": {p.name: luminance(p) for p in images.glob("*_last.png")
                                   if "depth" not in p.name and "_top_" not in p.name}}
    return out


def test_the_runs_differ_only_by_the_lidar(runs):
    on, off = runs["on"]["summary"], runs["off"]["summary"]
    assert on["lidars"] and not off["lidars"]
    assert on["scene"]["scan_mesh"] and off["scene"]["scan_mesh"] is None
    assert on["sim_s"] == off["sim_s"]
    assert runs["on"]["luminance"].keys() == runs["off"]["luminance"].keys() and runs["on"]["luminance"]


def test_the_lidar_leaves_every_camera_as_bright(runs):
    on, off = runs["on"]["luminance"], runs["off"]["luminance"]
    for stream in off:
        assert off[stream] > LIT_LEVELS, f"{stream}: mean {off[stream]:.1f}, the splat does not show"
        assert abs(on[stream] - off[stream]) < SAME_LEVELS, f"{stream}: {on[stream]:.2f} with the lidar, " \
                                                            f"{off[stream]:.2f} without"


def test_the_summary_records_the_pinned_exposure(runs):
    for name in ("on", "off"):
        post = runs[name]["summary"]["rtx_post"]
        for setting, value in PINNED_EXPOSURE.items():
            assert post[setting] == pytest.approx(value), f"{name}: {setting}"
