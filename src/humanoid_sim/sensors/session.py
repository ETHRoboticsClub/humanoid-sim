"""A robot with its RTX sensors in a scene: builds the simulation with the rig's cameras and lidars on it, steps the
physics, renders when a sensor is due and hands the result to its outputs (the camera feed rings, Unitree's lidar
topic, a Rerun recording, PNG frames). Kit must already be up (``rtx.start_kit``).

The sensors are the robot's rig (``sensors/rigs.py``), with an extra rig's added when one is named
(``SessionCfg.sensors``: one of ``rigs.EXTRAS``, for sensors the robot's rig does not declare). Every camera and
lidar the RTX sensors can simulate (``Sensor.rtx_ready``) is simulated; the others are listed in
``SensorSession.unsimulated``.

Timing: the physics steps every period (4 ms) and the stage renders every camera period (33 ms at 30 Hz; the lidar's
sweep period without cameras) at the plant time, which Kit's timeline follows. Each render gives every camera stream
a frame; a lidar fires its pattern over the time since the previous render and hands over a sweep whenever one
completes. Stamps are the realtime clock's reading at the frame's or sweep's plant time (``feeds.ros_time_s``). The
robot is welded at its spawn point in its home pose (a G1 topples on a fixed-gain hold), feet just above the floor,
unless ``fixed_base`` is off.

With ``physics`` off the session never steps: the caller poses the robot (:meth:`SensorSession.pose`, the
runtime's truth) and renders at plant times it chooses (:meth:`SensorSession.render_at`), telling the session where
the plant is now (``clock_ns``) so stamps map onto the realtime clock (``runtime/render.py``).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.sensors.models import RTX_KINDS
from humanoid_sim.sensors.room_rays import RoomRays, room_rays
from humanoid_sim.sensors.rtx import DOME_INTENSITY

LOG = logging.getLogger(__name__)


@dataclass
class SessionCfg:
    robot: str = "g1_29dof_with_hand_rev_1_0"
    sensors: str | None = None  # an extra rig (rigs.EXTRAS) whose sensors join the robot's
    scene: str | None = None  # static boxes (.npz)
    spawn_m: tuple[float, float] = (0.0, 0.0)
    splat: str | None = None  # the room's splat as USD
    splat_intensity: float = 1.0  # its emission (rtx.set_splat_intensity)
    scan_mesh: str | None = None  # the room's scan (OBJ), surfaces for the lidar
    cameras: bool = True
    lidar: bool = True
    feeds: str | None = None
    lidar_domain: int | None = None
    rrd: str | None = None
    images: str | None = None
    lidar_noise: bool = True  # the lidar model's range and angular errors; off for exact geometry
    light: float = DOME_INTENSITY  # the dome light (nits); 0 for none
    lights: str | None = None  # ceiling lights (JSON, sensors/lights.py), placed with spawn_m; may set the dome
    keep_sweeps: int = 0  # the last sweeps to keep in ``SensorSession.sweeps``
    device: str = "cuda:0"
    assets: str = "outputs/assets"
    fixed_base: bool = True
    physics: bool = True  # False: posed from outside (pose, render_at), never stepped


class SensorSession:
    def __init__(self, cfg: SessionCfg):
        from humanoid_sim.devices.demand import DriveDemand
        from humanoid_sim.model.asset import asset_dir, load_profile
        from humanoid_sim.model.description import Description
        from humanoid_sim.model.poses import STANDING_CLEARANCE_M, initial_pose
        from humanoid_sim.sensors import rigs, rtx
        from humanoid_sim.sensors.geometry import Mount, camera_spec
        from humanoid_sim.sensors.lights import CeilingLights, ceiling_light_prims, dome_nits
        from humanoid_sim.sim import SimCore
        from humanoid_sim.world.boxes import StaticBoxes

        self.cfg = cfg
        desc = Description.load(cfg.robot)
        out = asset_dir(desc, load_profile(cfg.robot), Path(cfg.assets))
        self.usd = out / json.loads((out / "asset.json").read_text())["usd"]
        self.rig = rigs.load(cfg.robot, desc, cfg.sensors)
        ready = [s for s in self.rig.sensors.values() if s.kind in RTX_KINDS and s.rtx_ready]
        cameras = [s for s in ready if s.kind == "camera"] if cfg.cameras else []
        lidars = [s for s in ready if s.kind == "lidar"] if cfg.lidar else []
        simulated = {s.name for s in cameras + lidars} | {n for n, s in self.rig.sensors.items() if s.simulated}
        self.unsimulated = [n for n in self.rig.sensors if n not in simulated]
        self.mounts = {s.name: Mount.from_description(desc, s.frame) for s in cameras + lidars}
        self.camera_specs = [camera_spec(s) for s in cameras]
        bodies = rtx.body_paths(self.usd, {m.link for m in self.mounts.values()})
        entities = rtx.camera_entities([(spec, self.mounts[s.name]) for spec, s in zip(self.camera_specs, cameras,
                                                                                     strict=True)], bodies)
        ceiling = CeilingLights.load(cfg.lights) if cfg.lights else None
        self.dome_nits = dome_nits(cfg.light, ceiling)
        if self.dome_nits:
            entities["dome_light"] = rtx.light_entity(self.dome_nits)
        if cfg.splat:
            entities["splat"] = rtx.splat_entity(cfg.splat)
        self.boxes = StaticBoxes.load(cfg.scene, cfg.spawn_m) if cfg.scene else None
        pose, height = initial_pose(self.usd, cfg.robot, "home")
        self.sim = SimCore(self.usd, device=cfg.device, robot=cfg.robot, fixed_base=cfg.fixed_base and cfg.physics,
                           base_height_m=height + STANDING_CLEARANCE_M, scene=self.boxes, entities=entities,
                           render_each_step=False)
        self.scan_mesh = cfg.scan_mesh if cfg.scan_mesh and lidars else None  # the lidar's surfaces, when loaded
        # SimCore built the physics model above: what follows changes what rays see, never what collides.
        self.room_rays = self._apply_room_rays(bool(cameras), bool(lidars))
        if cfg.splat and cfg.splat_intensity != 1.0:
            import isaaclab.sim as sim_utils

            rtx.set_splat_intensity(sim_utils.get_current_stage(), cfg.splat_intensity)
        if ceiling is not None:
            import isaaclab.sim as sim_utils

            ceiling_light_prims(sim_utils.get_current_stage(), ceiling, cfg.spawn_m)
        plant = self.sim.plant
        q = pose.to(device=plant.cfg.device, dtype=self.sim.joint_state()[0].dtype).unsqueeze(0)
        self.sim.write_joint_state(q, torch.zeros_like(q))
        hold, _, _ = plant.map.drive_state(q, torch.zeros_like(q))
        plant.submit(DriveDemand.pd(hold, plant.drive_stiffness.expand(1, -1), plant.drive_damping.expand(1, -1)),
                     torch.zeros(1, dtype=torch.long, device=hold.device))
        self.cameras = rtx.RtxCameras(self.sim.backend, self.camera_specs) if self.camera_specs else None
        self.lidars = []
        if lidars:
            from humanoid_sim.sensors.lidar import RtxLidar

            self.lidars = [RtxLidar(s, self.mounts[s.name], f"{rtx.ROBOT_ROOT}/{bodies[self.mounts[s.name].link]}",
                                    noise=cfg.lidar_noise) for s in lidars]
        color = any(st.kind != "depth" for c in self.camera_specs for st in c.streams)
        self.render = rtx.KitRender(self.sim.backend.sim, color=color)
        self.last_images: dict[tuple[str, str], np.ndarray] = {}
        self.sweeps = deque(maxlen=cfg.keep_sweeps) if cfg.keep_sweeps else None
        self.feeds = {}
        if cfg.feeds and self.camera_specs:
            from humanoid_sim.sensors.feeds import CameraFeed

            self.feeds = {spec.camera_id: CameraFeed(cfg.feeds, spec) for spec in self.camera_specs}
        self.publisher = None
        if cfg.lidar_domain is not None and self.lidars:
            from humanoid_sim.sensors.unitree_lidar import UNITREE_MODEL, UnitreeLidarPublisher

            # Unitree's service publishes its robot's Mid-360: the rig lidar of that model.
            if any(lidar.sensor.model == UNITREE_MODEL for lidar in self.lidars):
                self.publisher = UnitreeLidarPublisher(cfg.lidar_domain)
        self.recorder = None
        if cfg.rrd:
            from humanoid_sim.signals.scene import SceneRecorder

            self.recorder = SceneRecorder(self.sim.backend, self.usd, rrd=cfg.rrd, app_id="humanoid-sim rtx sensors",
                                          scene=self.boxes, draw="boxes" if self.boxes else None,
                                          gantry=self.sim.gantry)
        self.counts = {"steps": 0, "renders": 0, "camera_frames": 0, "lidar_sweeps": 0, "lidar_points": 0}
        self.ms = {"physics": [], "render": [], "camera_read": [], "lidar_read": [], "publish": []}
        self._saved: dict[str, bool] = {}  # every image stream seen: whether it is depth
        self._last_sweep: dict[str, tuple] = {}
        self.clock_ns: int | None = None  # where the plant is now, when something else steps it
        self.render_ns = min([int(round(1e9 / s.rate_hz)) for c in self.camera_specs for s in c.streams]
                             or [lidar.period_ns for lidar in self.lidars] or [0])

    def _apply_room_rays(self, cameras: bool, lidar: bool) -> RoomRays | None:
        """Add the scan mesh for the lidar and hide each of the room's surfaces from the rays that do not see it
        (``sensors/room_rays.py``)."""
        if self.boxes is None and self.scan_mesh is None:
            return None
        import isaaclab.sim as sim_utils

        from humanoid_sim.sensors import rtx
        from humanoid_sim.world.boxes import box_prim_path

        splat = bool(self.cfg.splat)
        boxes = self.boxes
        rays = room_rays(None if boxes is None else boxes.wall, 0 if boxes is None else len(boxes), cameras=cameras,
                         splat=splat, lidar=lidar, scan=self.scan_mesh is not None)
        if self.scan_mesh and not rays.walls_known:
            LOG.warning("%s has no wall flags: every box is kept as a wall, so the lidar sees furniture boxes in "
                        "front of the scan mesh %s", self.cfg.scene, self.scan_mesh)
        if self.scan_mesh:
            from humanoid_sim.sensors.lidar import add_scan_surfaces

            add_scan_surfaces(self.scan_mesh, self.cfg.spawn_m, hidden_from_cameras=rays.scan_hidden_from_cameras)
        stage = sim_utils.get_current_stage()
        for i in rays.hidden:
            rtx.hide_from_all_rays(stage, box_prim_path(i))
        for i in rays.lidar_only:
            rtx.hide_from_cameras(stage, box_prim_path(i))
        if splat and boxes is not None:
            rtx.hide_from_cameras(stage, GROUND_PRIM)  # the splat has the floor
        return rays

    @property
    def now_ns(self) -> int:
        return self.sim.t_ns if self.clock_ns is None else self.clock_ns

    def pose(self, t_ns: int, q: torch.Tensor, base_pose: torch.Tensor) -> None:
        """Put the robot at joint positions ``q`` ``[1, J]`` (the description's joint order) and base pose
        ``base_pose`` ``[1, 7]`` (position, quaternion xyzw, world) at plant time ``t_ns``, without stepping: the
        bodies follow by forward kinematics."""
        from isaaclab_newton.physics import NewtonManager

        robot = self.sim.backend.robot
        self.sim.write_joint_state(q, torch.zeros_like(q))
        robot.write_root_pose_to_sim_index(root_pose=base_pose.to(robot.data.root_link_pose_w.torch.dtype))
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=base_pose.device))
        self.sim.backend.scene.write_data_to_sim()
        NewtonManager.forward()
        self.sim.t_ns = t_ns

    # -- loop -------------------------------------------------------------------------------------------------

    def run(self, seconds: float) -> None:
        """Step ``seconds`` of plant time, rendering every camera period (or every lidar sweep without cameras)."""
        period_ns = self.sim.period_ns
        next_render = self.sim.t_ns + self.render_ns
        for _ in range(int(seconds * 1e9) // period_ns):
            t = time.perf_counter()
            self.sim.step()
            self.ms["physics"].append(1e3 * (time.perf_counter() - t))
            self.counts["steps"] += 1
            now = self.sim.t_ns
            if now < next_render:
                continue
            next_render += self.render_ns
            self.render_at(now, self.render_ns)

    def render_at(self, now_ns: int, dt_ns: int) -> None:
        """Render the stage at plant time ``now_ns``, ``dt_ns`` after the previous render, and hand every camera's
        frames and every completed lidar sweep to the outputs."""
        t = time.perf_counter()
        timeline_ns = self.lidars[0].render_time(now_ns) if self.lidars else now_ns
        self.render(timeline_ns)
        if self.cameras:
            self.cameras.update(dt_ns / 1e9)
        self.ms["render"].append(1e3 * (time.perf_counter() - t))
        self.counts["renders"] += 1
        if self.recorder:
            self.recorder.frame(now_ns)
        if self.cameras:
            self._cameras(now_ns)
        for lidar in self.lidars:
            t = time.perf_counter()
            sweeps = lidar.read(now_ns, timeline_ns)
            self.ms["lidar_read"].append(1e3 * (time.perf_counter() - t))
            for sweep in sweeps:
                self._lidar(now_ns, lidar, sweep)

    def _cameras(self, now_ns: int) -> None:
        from humanoid_sim.sensors.feeds import depth_mm, ros_time_s

        stamp = ros_time_s(now_ns, self.now_ns)
        for cam in self.camera_specs:
            for s in cam.streams:
                t = time.perf_counter()
                image = self.cameras.image(cam, s)
                self.ms["camera_read"].append(1e3 * (time.perf_counter() - t))
                frame = depth_mm(image, s.range_m) if s.kind == "depth" else image
                if cam.camera_id in self.feeds:
                    t = time.perf_counter()
                    self.feeds[cam.camera_id].publish(s.name, stamp, frame)
                    self.ms["publish"].append(1e3 * (time.perf_counter() - t))
                self.counts["camera_frames"] += 1
                self.last_images[(cam.camera_id, s.name)] = image
                self._record_image(f"{cam.camera_id}/{s.name}", image, now_ns, depth=s.kind == "depth")

    def _record_image(self, key: str, image: np.ndarray, now_ns: int, *, depth: bool) -> None:
        if self.recorder:
            rr = self.recorder.rr
            rr.set_time("time", duration=now_ns / 1e9)
            if depth:
                rr.log(f"/sensors/{key}", rr.DepthImage(np.nan_to_num(image, posinf=0.0), meter=1.0))
            else:  # JPEG: a second of six raw streams is about 100 MB
                rr.log(f"/sensors/{key}", rr.Image(image).compress(jpeg_quality=RRD_JPEG_QUALITY))
        if self.cfg.images and key not in self._saved:
            self._save_png(key, image, depth, "first")
        self._saved[key] = depth

    def _save_png(self, key: str, image: np.ndarray, depth: bool, which: str) -> None:
        from PIL import Image

        d = Path(self.cfg.images)
        d.mkdir(parents=True, exist_ok=True)
        if depth:
            image = np.clip(np.nan_to_num(image, posinf=0) / DEPTH_IMAGE_RANGE_M * 255, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(d / f"{key.replace('/', '_')}_{which}.png")

    def _lidar(self, now_ns: int, lidar, sweep) -> None:
        from humanoid_sim.sensors.feeds import ros_time_s
        from humanoid_sim.sensors.unitree_lidar import UNITREE_MODEL, pack_sweep

        self.counts["lidar_sweeps"] += 1
        self.counts["lidar_points"] += int(sweep.hit.sum())
        self._last_sweep[lidar.sensor.name] = (lidar, sweep)
        if self.sweeps is not None:
            self.sweeps.append(sweep)
        if self.publisher and lidar.sensor.model == UNITREE_MODEL:
            t = time.perf_counter()
            stamp_ns = int(ros_time_s(sweep.start_ns, self.now_ns) * 1e9)
            self.publisher.publish(pack_sweep(sweep.points_m, sweep.hit, sweep.time_s, stamp_ns))
            self.ms["publish"].append(1e3 * (time.perf_counter() - t))
        if self.recorder:
            rr = self.recorder.rr
            rr.set_time("time", duration=now_ns / 1e9)
            rr.log(f"/sensors/{lidar.sensor.name}/points",
                   rr.Points3D(lidar.to_world(sweep), radii=0.01, colors=(255, 120, 0)))

    # -- summary ----------------------------------------------------------------------------------------------

    def summary(self, wall_s: float) -> dict:
        def stats(values):
            if not values:
                return None
            a = np.asarray(values)
            return {"n": len(a), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                    "max": float(a.max())}

        sim_s = self.sim.t_ns / 1e9
        streams = sum(len(c.streams) for c in self.camera_specs)
        return {"robot": self.cfg.robot, "sim_s": sim_s, "wall_s": wall_s, "real_time_factor": sim_s / wall_s,
                "cameras": {c.camera_id: [s.name for s in c.streams] for c in self.camera_specs},
                "lidars": [lidar.sensor.name for lidar in self.lidars], "unsimulated": self.unsimulated,
                "scene": {"boxes": self.cfg.scene, "spawn_m": list(self.cfg.spawn_m), "scan_mesh": self.scan_mesh,
                          "splat": self.cfg.splat, "splat_intensity": self.cfg.splat_intensity,
                          "lights": self.cfg.lights, "dome_light_nits": self.dome_nits,
                          "rays": None if self.room_rays is None else self.room_rays.describe()},
                "rtx_post": self.render.post_settings(),
                "counts": self.counts, "ms": {k: stats(v) for k, v in self.ms.items()},
                "camera_frame_rate_hz": self.counts["camera_frames"] / max(streams, 1) / wall_s,
                "lidar_sweep_rate_hz": self.counts["lidar_sweeps"] / max(len(self.lidars), 1) / wall_s,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "outputs": {"rrd": self.cfg.rrd, "images": self.cfg.images, "feeds": self.cfg.feeds,
                            "lidar_domain": self.cfg.lidar_domain}}

    def close(self) -> None:
        if self.cfg.images:
            for (camera_id, stream), image in self.last_images.items():
                key = f"{camera_id}/{stream}"
                self._save_png(key, image, self._saved[key], "last")
            for name, (lidar, sweep) in self._last_sweep.items():
                self._save_png(f"{name}/top", top_view(lidar.to_world(sweep), sweep.origin_w), False, "last")
        for feed in self.feeds.values():
            feed.close()
        if self.recorder:
            self.recorder.close()
        self.sim.close()


def top_view(points_w: np.ndarray, centre_w: np.ndarray, half_m: float = 12.0, px: int = 800) -> np.ndarray:
    """A lidar sweep seen from above: ``px`` square around ``centre_w`` (x right, y up), each return a dot coloured by
    its height (blue low, red high)."""
    image = np.full((px, px, 3), 16, dtype=np.uint8)
    rel = points_w - centre_w
    u = ((rel[:, 0] + half_m) / (2 * half_m) * px).astype(int)
    v = ((half_m - rel[:, 1]) / (2 * half_m) * px).astype(int)
    keep = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    z = points_w[keep, 2]
    h = np.clip((z - z.min()) / max(np.ptp(z), 1e-6), 0, 1) if len(z) else z
    colour = np.stack([255 * h, 80 + 100 * (1 - abs(2 * h - 1)), 255 * (1 - h)], axis=1)
    image[v[keep], u[keep]] = colour.astype(np.uint8)
    return image


DEPTH_IMAGE_RANGE_M = 10.0  # depth PNGs: black at 0 m, white at this
RRD_JPEG_QUALITY = 90
GROUND_PRIM = "/World/Ground"  # physics/isaaclab.ground_cfg

__all__ = ["SensorSession", "SessionCfg"]
