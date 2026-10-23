"""A read-only Rerun view of a running runtime segment: the robot posed from the truth region, and its drive
feedback, IMU, soles, gantry and lateness as time series.

It only reads the segment (seqlock reads of the descriptor, feedback and truth regions); it never writes
demands or commands, so it can watch a session a client such as Controls is driving. Time is the sim's
``sim_time_ns``. Entity paths: ``/world/robot/<body>``, ``/world/scene/{scan,floor,boxes}`` (the host's scan, or its
static boxes, when its scene files are readable here with the same digests), ``/sim/0/drives/<drive>/{position,velocity,applied_torque,
demanded_torque}``, ``/sim/0/imu/{gyro,specific_force}/{x,y,z}``, ``/sim/0/soles/<side>/normal_force``,
``/sim/0/base/height``, ``/sim/0/gantry/tension``, ``/sim/0/host/{late_steps,keeping_up}``. A cable is drawn as
``/world/gantry/rope``; with a crane, ``/world/gantry/{travel,bridge,trolley}`` and
``/sim/0/gantry/{position,command}/{x,y,z}``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.signals.log import SignalLog
from humanoid_sim.signals.robot import RobotMeshes
from humanoid_sim.signals.scene import log_scene

ROBOT_COLOR = (190, 190, 200, 255)


def blueprint(drives: list[str], sky: tuple[float, float, float] | None = None):
    import rerun.blueprint as rrb

    background = None if sky is None else rrb.Background(color=[round(255 * c) for c in sky])
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Robot", origin="/world", background=background),
            rrb.Vertical(
                rrb.Tabs(
                    rrb.TimeSeriesView(name="Drive position", origin="/sim/0/drives",
                                       contents=[f"/sim/0/drives/{d}/position" for d in drives]),
                    rrb.TimeSeriesView(name="Applied torque", origin="/sim/0/drives",
                                       contents=[f"/sim/0/drives/{d}/applied_torque" for d in drives]),
                    rrb.TimeSeriesView(name="Demanded torque", origin="/sim/0/drives",
                                       contents=[f"/sim/0/drives/{d}/demanded_torque" for d in drives]),
                ),
                rrb.Tabs(
                    rrb.TimeSeriesView(name="IMU gyro", origin="/sim/0/imu/gyro"),
                    rrb.TimeSeriesView(name="IMU specific force", origin="/sim/0/imu/specific_force"),
                    rrb.TimeSeriesView(name="Sole forces", origin="/sim/0/soles"),
                    rrb.TimeSeriesView(name="Base, gantry, host", origin="/sim/0",
                                       contents=["/sim/0/base/**", "/sim/0/gantry/**", "/sim/0/host/**"]),
                ),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="collapsed"),
        collapse_panels=True,
    )


def _compose(base: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Body poses ``[B, 7]`` (xyz, quaternion xyzw) relative to a root at the origin, placed at root pose
    ``base`` ``[7]``."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_quat(base[3:])
    out = np.empty_like(poses)
    out[:, :3] = r.apply(poses[:, :3]) + base[:3]
    out[:, 3:] = (r * Rotation.from_quat(poses[:, 3:])).as_quat()
    return out


class RuntimeView:
    """Reads one segment and logs what it shows, one frame per :meth:`frame` call."""

    def __init__(self, segment: str | Path, usd: Path, log: SignalLog, *, wait_s: float = 60.0,
                 robot: str = "ethrc_v1", draw: str | None = None, detail: str = "live"):
        from humanoid_sim.model.description import Description
        from humanoid_sim.model.kinematics import RobotKinematics

        self.path = Path(segment)
        desc = Description.load(robot)
        # The sim must describe the robot this view draws: same robot id and description digest.
        self._identity = {"robot_id": desc.robot, "description_digest": desc.digest}
        self.seg = abi.Segment.open(self.path, wait_s=wait_s, **self._identity)
        desc = self.seg.descriptor()
        self.drives, self.joints = desc["drives"], desc["joints"]
        self.soles = desc.get("soles", [])
        from humanoid_sim.world.trolley import GantryRig

        self.rig = None if not desc.get("gantry") else GantryRig.from_dict(desc["gantry"])
        self.scene = self._scene(desc.get("scene"))
        # The scan when the host has one readable here and nothing else is asked for, else the boxes.
        self.draw = None if self.scene is None else (
            draw or ("scan" if self.scene.scan is not None else "boxes"))
        if self.draw == "scan" and self.scene.scan is None:
            raise ValueError("the host's scene has no scan readable here; draw its boxes")
        self.detail = detail
        self.kin = RobotKinematics(usd, self.joints, base_height_m=0.0)
        self.meshes = RobotMeshes(self.kin.model)
        self.log, self.rr = log, log.rr
        self.session_id = self.seg.session_id
        self._last_truth_seq = -1
        self._last_feedback_seq = -1
        self.frames = 0

    def log_static(self) -> None:
        rr = self.rr
        self.meshes.log_static("/world/robot", ROBOT_COLOR)
        if self.scene is not None:
            log_scene(rr, self.scene, draw=self.draw, detail=self.detail)
        for d in self.drives:
            for name in ("position", "velocity", "applied_torque", "demanded_torque"):
                rr.log(f"/sim/0/drives/{d}/{name}", rr.SeriesLines(names=[d]), static=True)
        if self.rig is not None:
            outline = self.rig.travel_outline_w()
            rr.log("/world/gantry/travel", rr.LineStrips3D([np.vstack([outline, outline[:1]])],
                                                           colors=[(120, 120, 120)]), static=True)
        for group in ("gyro", "specific_force"):
            for axis, color in zip("xyz", ((220, 80, 80), (80, 180, 80), (80, 120, 220))):
                rr.log(f"/sim/0/imu/{group}/{axis}", rr.SeriesLines(names=[axis], colors=[color]), static=True)

    @staticmethod
    def _scene(record: dict | None):
        """The host's scene, when its boxes file is readable here with the same digest; with its scan when those
        files are readable here with the same digests too."""
        from humanoid_sim.world.boxes import StaticBoxes
        from humanoid_sim.world.scan import ScanFiles

        if not record or not Path(record["file"]).is_file():
            return None
        scan = None
        files = {k: v for k, v in (record.get("scan") or {}).items() if v}
        if files and all(Path(v["file"]).is_file() for v in files.values()):
            got = ScanFiles.of(files["mesh"]["file"], files["texture"]["file"],
                               files["floor_texture"]["file"] if "floor_texture" in files else None)
            if all(got.sha256[v["file"]] == v["sha256"] for v in files.values()):
                scan = got
        boxes = StaticBoxes.load(record["file"], record["spawn_m"], scan=scan)
        return boxes if boxes.sha256 == record["sha256"] else None

    def _read(self, region: str):
        try:
            return self.seg.regions[region].read()
        except abi.StaleRead:
            return None

    def frame(self) -> bool:
        """Log the latest truth and feedback if they are new; False once the session has ended."""
        if self.seg.state == abi.STATE_ENDED or self.seg.session_id != self.session_id:
            return False
        rr = self.rr
        truth = self._read("truth")
        if truth is not None and truth[0] != self._last_truth_seq:
            self._last_truth_seq, t = truth
            self.log.at(int(t["sim_time_ns"]))
            q = torch.as_tensor(np.asarray(t["joint_pos_rad"][: len(self.joints)], dtype=np.float32))[None]
            base = np.concatenate([t["base_position_m"], t["base_quat_xyzw"]]).astype(np.float64)
            poses = _compose(base, self.kin.body_poses(q).astype(np.float64))
            for b, name in enumerate(self.meshes.body_names):
                rr.log(f"/world/robot/{name}", rr.Transform3D(translation=poses[b, :3], quaternion=poses[b, 3:]))
            rr.log("/sim/0/base/height", rr.Scalars(float(t["base_position_m"][2])))
            rr.log("/sim/0/gantry/tension", rr.Scalars(float(t["gantry_tension_n"])))
            self._gantry(t)
            for k, side in enumerate(("left", "right")[: len(self.soles)]):
                rr.log(f"/sim/0/soles/{side}/normal_force", rr.Scalars(float(t["sole"]["normal_force_n"][k])))
            self.frames += 1
        fb = self._read("feedback")
        if fb is not None and fb[0] != self._last_feedback_seq:
            self._last_feedback_seq, f = fb
            self.log.at(int(f["sim_time_ns"]))
            n = len(self.drives)
            drive = f["drive"]
            for i, d in enumerate(self.drives[:n]):
                rr.log(f"/sim/0/drives/{d}/position", rr.Scalars(float(drive["position_rad"][i])))
                rr.log(f"/sim/0/drives/{d}/velocity", rr.Scalars(float(drive["velocity_rad_s"][i])))
                rr.log(f"/sim/0/drives/{d}/applied_torque", rr.Scalars(float(drive["applied_torque_nm"][i])))
                rr.log(f"/sim/0/drives/{d}/demanded_torque", rr.Scalars(float(drive["demanded_torque_nm"][i])))
            if int(f["imu_count"]) > 0:
                imu = f["imu"][0]
                for axis, k in zip("xyz", range(3)):
                    rr.log(f"/sim/0/imu/gyro/{axis}", rr.Scalars(float(imu["gyro_rad_s"][k])))
                    rr.log(f"/sim/0/imu/specific_force/{axis}", rr.Scalars(float(imu["specific_force_m_s2"][k])))
            rr.log("/sim/0/host/late_steps", rr.Scalars(float(f["late_steps"])))
            rr.log("/sim/0/host/keeping_up", rr.Scalars(float(f["keeping_up"])))
        return True

    def _gantry(self, t) -> None:
        """The rope while a cable hangs, and the crane where it is."""
        rr = self.rr
        top, harness = np.asarray(t["gantry_top_m"]), np.asarray(t["gantry_harness_m"])
        if np.isfinite(top).all() and np.isfinite(harness).all():
            rr.log("/world/gantry/rope", rr.LineStrips3D([[top, harness]], colors=[(230, 160, 40)], radii=0.006))
        else:
            rr.log("/world/gantry/rope", rr.Clear(recursive=False))
        position = np.asarray(t["gantry_position_m"])
        if self.rig is None or not np.isfinite(position).all():
            return
        x, y = position[:2]
        y0, y1 = self.rig.travel_y_m
        rr.log("/world/gantry/bridge", rr.LineStrips3D([[self.rig.to_world((x, y0, 0)), self.rig.to_world((x, y1, 0))]],
                                                       colors=[(90, 90, 200)], radii=0.03))
        rr.log("/world/gantry/trolley", rr.Points3D([self.rig.to_world((x, y, 0))], radii=0.08, colors=[(90, 90, 200)]))
        for name, value in (("position", position), ("command", np.asarray(t["gantry_command_m"]))):
            for axis, v in zip("xyz", value):
                rr.log(f"/sim/0/gantry/{name}/{axis}", rr.Scalars(float(v)))

    def reopen(self, wait_s: float) -> None:
        """Follow the next session at the same path (``serve --persist`` opens a new segment there); plant time
        runs on across sessions, so the recording continues on the same timeline."""
        self.seg.close()
        self.seg = abi.Segment.open(self.path, wait_s=wait_s, **self._identity)
        self.session_id = self.seg.session_id
        self._last_truth_seq = self._last_feedback_seq = -1
        self.rr.log("/sim/0/host/session", self.rr.TextLog(f"session {self.session_id:#x}"))

    def close(self) -> None:
        self.seg.close()


FOLLOW_S = 10.0  # how long to wait for the next session of a `serve --persist` host


def view_runtime(segment: str | Path, usd: Path, *, rrd: str | None = None, connect: str | None = None,
                 fps: float = 30.0, seconds: float | None = None, wait_s: float = 60.0,
                 robot: str = "ethrc_v1", draw: str | None = None, detail: str | None = None) -> int:
    """Watch ``segment`` (waiting up to ``wait_s`` for it to appear) until its host is gone or for ``seconds``;
    a new session at the same path within ``FOLLOW_S`` of the last one ending is followed. Returns the number of
    robot frames logged. Writes to ``rrd``, streams to ``connect``, or spawns a viewer when neither is given. The
    host's scene is drawn as ``draw`` says (``scan`` or ``boxes``; the scan when it has one readable here) at
    ``detail`` (``full`` into a file, ``live`` otherwise, unless given)."""
    from humanoid_sim.world.scan import SKY_RGB

    log = SignalLog("humanoid-sim runtime", init=True)
    view = RuntimeView(segment, usd, log, wait_s=wait_s, robot=robot, draw=draw,
                       detail=detail or ("full" if rrd else "live"))
    bp = blueprint(view.drives, sky=SKY_RGB if view.draw == "scan" else None)
    if rrd:
        log.rr.save(rrd, default_blueprint=bp)
    elif connect:
        log.rr.connect_grpc(connect, default_blueprint=bp)
    else:
        log.rr.spawn(default_blueprint=bp)
    view.log_static()
    period = 1.0 / fps
    end = None if seconds is None else time.monotonic() + seconds
    try:
        while end is None or time.monotonic() < end:
            if not view.frame():
                try:
                    view.reopen(FOLLOW_S)
                except (abi.AbiError, FileNotFoundError):
                    break  # the server is gone
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        frames = view.frames
        if view.seg.buf is not None:
            view.close()
        log.rr.disconnect()
    return frames
