"""Renders the screenshots in docs/images from Rerun recordings, with a headless Rerun viewer.

    uv run python docs/images/capture.py                 # every shot whose recording exists
    uv run python docs/images/capture.py scene replay    # shots whose names start with these

The recordings come from `outputs/shots/` or `--recordings DIR` (SHOTS names the command that makes each;
`sky/e2e.sky.yaml` writes them all to `outputs/e2e/shots/`). Each shot opens its recording in a headless viewer,
sends the shot's layout and camera as a blueprint, moves the timeline to a moment and saves the viewer as a PNG.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

REPO = Path(__file__).resolve().parents[2]
SHOTS_DIR = REPO / "outputs" / "shots"
IMAGES = Path(__file__).resolve().parent
PORT = 9877


def eye(position, target, up=(0.0, 0.0, 1.0)):
    return rrb.archetypes.EyeControls3D(position=position, look_target=target, eye_up=up)


def panels(*views, time_panel: str = "hidden"):
    return rrb.Blueprint(*views, rrb.BlueprintPanel(state="hidden"), rrb.SelectionPanel(state="hidden"),
                         rrb.TimePanel(state=time_panel))


@dataclass
class Shot:
    name: str
    rrd: str  # file name in the recordings directory
    at_s: float  # timeline position, s
    blueprint: object
    size: str = "1600x900"
    note: str = ""  # how the recording is made


def scene_blueprint(position, target):
    return panels(rrb.Spatial3DView(name="Training scene", origin="/world", eye_controls=eye(position, target),
                                    background=rrb.Background(color=(232, 236, 240))))


def replay_blueprint(drive: str, joint: str):
    return panels(rrb.Horizontal(
        rrb.Spatial3DView(name="Overlay: sim (grey), real (orange)", origin="/world",
                          eye_controls=eye((2.0, -2.6, 1.45), (0.0, 0.0, 0.85))),
        rrb.Vertical(
            rrb.TimeSeriesView(name=f"{joint} ({drive}): position (rad), 2 s around the cursor", origin="/",
                               contents=[f"/sim/0/drives/{drive}/position", f"/real/drives/{drive}/position",
                                         f"/demand/drives/{drive}/position"],
                               time_ranges=rrb.VisibleTimeRange(
                                   "time", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-1.0),
                                   end=rrb.TimeRangeBoundary.cursor_relative(seconds=1.0))),
            rrb.TimeSeriesView(name="Position residual, every drive (sim - real, rad)", origin="/residual"),
        ),
        column_shares=[1, 1],
    ), time_panel="collapsed")


def replay_ghost_blueprint():
    return panels(rrb.Spatial3DView(name="Overlay: sim (grey), real (orange)", origin="/world",
                                    eye_controls=eye((1.9, -2.4, 1.4), (0.0, 0.0, 0.85))))


def runtime_blueprint(drives: list[str]):
    return panels(rrb.Horizontal(
        rrb.Spatial3DView(name="Robot (truth region)", origin="/world",
                          eye_controls=eye((1.7, -1.9, 1.5), (0.0, 0.0, 1.0))),
        rrb.Vertical(
            rrb.TimeSeriesView(name="Drive position (rad)", origin="/sim/0/drives",
                               contents=[f"/sim/0/drives/{d}/position" for d in drives]),
            rrb.TimeSeriesView(name="Applied torque (Nm)", origin="/sim/0/drives",
                               contents=[f"/sim/0/drives/{d}/applied_torque" for d in drives]),
            rrb.TimeSeriesView(name="IMU gyro (rad/s)", origin="/sim/0/imu/gyro"),
        ),
        column_shares=[1, 1],
    ), time_panel="collapsed")


def train_signals_blueprint():
    return panels(rrb.Horizontal(
        rrb.Spatial3DView(name="Training scene", origin="/world",
                          eye_controls=eye((3.4, -7.5, 1.3), (0.0, -2.0, 0.6)),
                          background=rrb.Background(color=(232, 236, 240))),
        rrb.Vertical(
            rrb.TimeSeriesView(name="knee_l of robot 0: reported vs true position (rad)", origin="/sim/0/drives",
                               contents=["/sim/0/drives/knee_l/position_reported",
                                         "/sim/0/drives/knee_l/position_true"]),
            rrb.TimeSeriesView(name="knee_l applied torque (Nm)", origin="/sim/0/drives",
                               contents=["/sim/0/drives/knee_l/applied_torque"]),
            rrb.TimeSeriesView(name="Base height (m)", origin="/sim/0/base"),
        ),
        column_shares=[3, 2],
    ), time_panel="collapsed")


def gantry_blueprint(envs: int):
    return panels(rrb.Horizontal(
        rrb.Spatial3DView(name="Training scene with gantry ropes", origin="/world",
                          eye_controls=eye((11.0, -3.0, 3.5), (0.0, -3.0, 1.2)),
                          background=rrb.Background(color=(232, 236, 240))),
        rrb.TimeSeriesView(name="Rope tension over the robot's weight, envs 0 to 3", origin="/sim",
                           contents=[f"/sim/{e}/gantry/tension_share" for e in range(envs)]),
        column_shares=[3, 2],
    ), time_panel="collapsed")


ARM = ["maxonHEJ70AL1", "maxonHEJ70AR1", "maxonHEJ50AL4", "maxonHEJ50AR4"]  # shoulder and elbow flexion, l and r
SHOTS = [
    Shot("scene-hero", "train_scene_6.rrd", 1.0,
         scene_blueprint((3.4, -7.5, 1.3), (0.0, -2.0, 0.8)), size="1800x800",
         note="python examples/train_scene.py --num-envs 6 --seconds 8 --rrd outputs/shots/train_scene_6.rrd"),
    Shot("scene-row", "train_scene_24.rrd", 2.2,
         scene_blueprint((7.0, -33.0, 4.0), (0.0, -18.0, 0.4)), size="1800x800",
         note="python examples/train_scene.py --num-envs 24 --seconds 8 --rrd outputs/shots/train_scene_24.rrd"),
    Shot("scene-gantry", "train_scene_gantry.rrd", 0.37, gantry_blueprint(4), size="1800x800",
         note="python examples/train_scene.py --num-envs 4 --seconds 4 --gantry --rrd outputs/shots/train_scene_gantry.rrd"),
    Shot("scene-signals", "train_scene_6.rrd", 2.6, train_signals_blueprint(),
         note="as scene-hero"),
    Shot("world-hangar", "hangar_v1.rrd", 4.0,
         panels(rrb.Spatial3DView(name="The V1 at Controls' hangar spawn point", origin="/world",
                                  eye_controls=eye((-4.5, -6.0, 4.0), (1.0, 2.0, 0.6)),
                                  background=rrb.Background(color=(232, 236, 240)))), size="1800x900",
         note="python examples/scene_stand.py --scene <controls hangar_proxy.npz> --spawn -4.42 3.48 --seconds 10 "
              "--device cuda:0 --rrd outputs/shots/hangar_v1.rrd"),
    Shot("world-hangar-scan", "hangar_v1_scan.rrd", 4.0,
         panels(rrb.Spatial3DView(name="The V1 in the scan of Controls' hangar", origin="/world",
                                  eye_controls=eye((-4.5, -6.0, 4.0), (1.0, 2.0, 0.6)),
                                  background=rrb.Background(color=(199, 196, 189)))), size="1800x900",
         note="python examples/scene_stand.py --scene <hangar_proxy.npz> --spawn -4.42 3.48 --scan-mesh "
              "<hangar_visual.obj> --scan-texture <hangar_visual.png> --scan-floor-texture <hangar_floor.png> "
              "--seconds 6 --device cuda:0 --rrd outputs/shots/hangar_v1_scan.rrd"),
    Shot("replay-overlay", "replay.rrd", 12.0, replay_blueprint("maxonHEJ70AL1", "shoulder_flexion_l"),
         note="humanoid-sim replay <exc> --base yaw --out outputs/replay; "
              "humanoid-sim view-replay outputs/replay --save outputs/shots/replay.rrd"),
    Shot("replay-ghost", "replay.rrd", 12.0, replay_ghost_blueprint(), size="1000x900",
         note="as replay-overlay"),
    Shot("runtime-live", "runtime.rrd", 12.0, runtime_blueprint(ARM),
         note="serve + examples/runtime_client.py + humanoid-sim view-runtime <segment> --save outputs/shots/runtime.rrd"),
]


def connect(port: int):
    from rerun.experimental import ViewerClient

    for _ in range(120):
        try:
            return ViewerClient.connect(f"rerun+http://127.0.0.1:{port}/proxy")
        except Exception:  # noqa: BLE001 - the viewer is still starting
            time.sleep(0.5)
    raise RuntimeError("the headless viewer did not start")


def longest(state):
    """The recording in the viewer with the longest ``time`` timeline (a file can hold more than one)."""
    def span(rec):
        return max((t.end - t.start for t in rec.timelines if t.name == "time"), default=-1)

    return max(state.recordings, key=span).store_id


def capture(shot: Shot, out: Path, recordings: Path = SHOTS_DIR, port: int = PORT, settle_s: float = 6.0) -> None:
    viewer = subprocess.Popen(["rerun", "--headless", "--window-size", shot.size, "--port", str(port),
                               "--hide-welcome-screen"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
    try:
        client = connect(port)
        client.open_url(str((recordings / shot.rrd).resolve()))
        time.sleep(settle_s)
        recording = longest(client.viewer_state())
        # Send the layout as the recording itself (the viewer's application and recording id for the loaded
        # file), so it replaces any layout the file carries and the viewer stays on the recording.
        _, app, rec = str(recording).split(":")
        stream = rr.RecordingStream(app, recording_id=rec)
        stream.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy")
        stream.send_blueprint(shot.blueprint, make_active=True, make_default=False)
        stream.flush()
        time.sleep(2.0)
        client.set_time("time", duration=shot.at_s, recording=recording)
        time.sleep(3.0)
        client.save_screenshot(str(out))
        time.sleep(2.0)
    finally:
        os.killpg(viewer.pid, signal.SIGTERM)  # the viewer binary, not only its launcher
        viewer.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("names", nargs="*", help="shot name prefixes (default: all)")
    p.add_argument("--at", type=float, default=None, help="override the timeline position, in s")
    p.add_argument("--out", default=None, help="write here instead of docs/images/<name>.png (one shot)")
    p.add_argument("--recordings", default=str(SHOTS_DIR), help="directory holding the .rrd files")
    args = p.parse_args(argv)
    recordings = Path(args.recordings)
    for shot in SHOTS:
        if args.names and not any(shot.name.startswith(n) for n in args.names):
            continue
        if not (recordings / shot.rrd).exists():
            print(f"{shot.name}: skipped, {recordings / shot.rrd} missing ({shot.note})")
            continue
        if args.at is not None:
            shot.at_s = args.at
        out = Path(args.out) if args.out else IMAGES / f"{shot.name}.png"
        capture(shot, out, recordings)
        print(f"{shot.name}: {out}" if out.exists() else f"{shot.name}: no screenshot written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
