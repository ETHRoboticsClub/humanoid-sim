"""The Mid-360 as an RTX lidar, and the scanned room's surfaces it sees.

* :class:`RtxLidar` puts a rig lidar on the robot as an RTX solid-state lidar (Isaac Sim's ``OmniLidar``), on the
  body its rig frame is merged into, at the frame's mount, with its model's range, field and noise; and makes each
  sweep from one render's output: points in the sensor frame by emitter. Its emitters fire the directions of
  :func:`~humanoid_sim.sensors.geometry.scan_pattern`: one RTX tick per sweep and one emitter state per tick for
  ``PATTERN_SWEEPS`` sweeps, after which the pattern repeats.
* :func:`scan_mesh_prim` writes a room's scan (the OBJ ``world/scan.py`` reads) as a USD mesh in the world frame
  (moved by ``-spawn`` like the room's boxes), for the lidar to hit. With a splat the cameras see it instead and
  the mesh is hidden from camera rays (``rtx.hide_from_cameras``, ``sensors/room_rays.py``).

The lidar fires on Kit's timeline, which :class:`~humanoid_sim.sensors.rtx.KitRender` advances by the plant time
between renders. A render's output holds the rays fired since the previous render, all from one emitter state and all
traced at the pose of that render (no motion BVH): the RTX sensor clock's sweep boundaries fall anywhere inside it.
So a sweep is one render's output, and the lidar must render once per sweep period (its own pass, without cameras):
:meth:`RtxLidar.read` stamps it with the render's plant time and pose and every ray's time 0, which is when RTX traced
it. Splicing outputs at the sensor clock's boundaries instead mixes two states traced at two poses under one pose.
When renders fall further apart than a period, :meth:`RtxLidar.render_time` advances the timeline by one period only,
so an output stays one whole pattern and the plant time skipped is a gap between sweeps.

Every ray that meets a surface within the model's range returns: the scene's surfaces carry no lidar reflectance
(RTX non-visual materials), so a real sensor's reflectivity-dependent reach (the Mid-360's 70 m is at 80 %) and its
intensities are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_sim.sensors.geometry import Mount, scan_pattern

SCAN_PRIM = "/World/ScanSurfaces"

PATTERN_SWEEPS = 10
LINES = 100  # the RTX model's emitter banks: rows of rays, a sweep's rays split evenly over them
CORE = "omni:sensor:Core:"
RTX_EXTENSION = "isaacsim.sensors.experimental.rtx"  # Isaac Sim's RTX lidar; Isaac Lab's Kit app leaves it off
VALID = 64  # GenericModelOutput ElementFlags.VALID: the ray returned


@dataclass(frozen=True)
class Sweep:
    """One sweep, one render's output: the render's plant time (ns), every ray's point in the sensor frame ``[N, 3]``
    by emitter (zero where it did not return), whether it returned ``[N]`` and when it was traced after ``start_ns``
    ``[N]`` (s, 0: RTX traces a render's rays at once); and the sensor's world pose at that render."""

    start_ns: int
    points_m: np.ndarray
    hit: np.ndarray
    time_s: np.ndarray
    origin_w: np.ndarray  # [3]
    rotation_w: np.ndarray  # [3, 3]


def pattern_states(model, sweeps: int = PATTERN_SWEEPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The RTX emitter states that fire :func:`scan_pattern` of a lidar model for ``sweeps`` sweeps: azimuth and
    elevation (deg) ``[sweeps, N]`` and each ray's firing time after the sweep's start (ns) ``[N]``. RTX measures an
    emitter's azimuth clockwise and reports returns counter-clockwise, so the states hold the azimuth negated."""
    dirs, times = zip(*(scan_pattern(model, s) for s in range(sweeps)), strict=True)
    d = np.stack(dirs)
    az = -np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1.0, 1.0)))
    fire_ns = np.round(times[0] * 1e9).astype(np.int64)
    return az.astype(np.float32), el.astype(np.float32), fire_ns


def sweep_rate_hz(model) -> float:
    return 1e9 / model.frame_period_ns


def lidar_attributes(model, *, noise: bool) -> dict:
    """The ``OmniLidar`` prim's core attributes for a lidar model: solid state, one tick per sweep, one return per
    ray, every ray reported (returned or not) as it fires; with ``noise`` the model's range and angular errors."""
    n = len(scan_pattern(model, 0)[1])
    if n % LINES:
        raise ValueError(f"{n} rays a sweep do not split into {LINES} lines")
    rate = sweep_rate_hz(model)
    if rate != int(rate):
        raise ValueError(f"RTX fires whole sweeps per second: {rate} Hz")
    angular_deg = float(np.degrees(model.angular_sigma_rad)) if noise else 0.0
    return {
        CORE + "scanType": "SOLID_STATE", CORE + "numberOfEmitters": n, CORE + "numberOfChannels": n,
        CORE + "numLines": LINES, CORE + "numRaysPerLine": [n // LINES] * LINES,
        CORE + "scanRateBaseHz": int(rate), CORE + "patternFiringRateHz": int(rate), CORE + "stateResolutionStep": 1,
        CORE + "nearRangeM": model.range_min_m, CORE + "farRangeM": model.range_max_m, CORE + "minReflectance": 0.0,
        CORE + "rangeAccuracyM": model.range_sigma_m if noise else 0.0, CORE + "azimuthErrorStd": angular_deg,
        CORE + "elevationErrorStd": angular_deg, CORE + "maxReturns": 1, CORE + "skipDroppingInvalidPoints": True,
        CORE + "accumulateOutputs": False, CORE + "elementsCoordsType": "SPHERICAL",
        CORE + "outputFrameOfReference": "SENSOR", "omni:sensor:tickRate": float(rate),
    }


class RtxLidar:
    """The rig lidar ``sensor`` (a :class:`~humanoid_sim.sensors.models.Lidar`) mounted at ``mount`` under the body
    prim ``body_prim``; :meth:`read` after every render."""

    def __init__(self, sensor, mount: Mount, body_prim: str, *, noise: bool = True):
        from isaaclab.sim.utils import enable_extension

        enable_extension(RTX_EXTENSION)
        import isaacsim.sensors.experimental.rtx as rtx

        self.sensor = sensor
        self.path = f"{body_prim}/{sensor.name}"
        self._rtx = rtx
        az, el, fire_ns = pattern_states(sensor)
        self.pattern = {"azimuth_deg": -az, "elevation_deg": el, "fire_ns": fire_ns}
        self.fire_ns = fire_ns
        self.period_ns = sensor.frame_period_ns
        attributes = lidar_attributes(sensor, noise=noise)
        self.lidar = rtx.Lidar(self.path, attributes=attributes, accumulate_outputs=None, aux_output_level="BASIC")
        x, y, z, w = mount.rotation_xyzw
        self.lidar.set_local_poses(translations=np.array([mount.translation_m]), orientations=np.array([[w, x, y, z]]))
        prim = self.lidar.prims[0]
        n = len(fire_ns)
        line = (np.arange(n) // (n // LINES)).tolist()
        for s in range(len(az)):
            name = f"s{s + 1:03d}"
            prim.ApplyAPI("OmniSensorGenericLidarCoreEmitterStateAPI", name)
            state = f"{CORE}emitterState:{name}:"
            prim.GetAttribute(state + "azimuthDeg").Set(az[s].tolist())
            prim.GetAttribute(state + "elevationDeg").Set(el[s].tolist())
            prim.GetAttribute(state + "fireTimeNs").Set(fire_ns.tolist())
            prim.GetAttribute(state + "channelId").Set(list(range(1, n + 1)))
            prim.GetAttribute(state + "bank").Set(line)
        self.runtime = rtx.LidarSensor(self.lidar, annotators=["generic-model-output"])
        self._init_reads()

    def _init_reads(self) -> None:
        self._last: tuple[int, int] | None = None  # the last render's plant and timeline times
        self._whole = False  # whether an output has held the whole pattern yet
        self.dropped = 0  # partial outputs while the sensor starts, before its first whole one
        self.reads = {"renders": 0, "outputs": 0, "rays": 0}

    def render_time(self, now_ns: int) -> int:
        """The timeline time to render plant time ``now_ns`` at: the last render's, advanced by the plant time since
        it, at most one sweep period."""
        if self._last is None:
            return now_ns
        plant_ns, timeline_ns = self._last
        return timeline_ns + min(now_ns - plant_ns, self.period_ns)

    def read(self, now_ns: int, timeline_ns: int | None = None) -> list[Sweep]:
        """The sweep the render at plant time ``now_ns`` traced (on the timeline at ``timeline_ns``, by default
        ``now_ns``): none while the sensor starts. Refuses an output that is not one whole pattern once one was."""
        self._last = (now_ns, now_ns if timeline_ns is None else timeline_ns)
        self.reads["renders"] += 1
        data, _ = self.runtime.get_data("generic-model-output")
        if data is None or getattr(data, "size", 0) == 0:
            return self._partial(now_ns, 0)
        g = self._rtx.parse_generic_model_output_data(data)
        if g.numElements == 0:
            return self._partial(now_ns, 0)
        self.reads["outputs"] += 1
        self.reads["rays"] += int(g.numElements)
        n = len(self.fire_ns)
        emitter = np.asarray(g.emitterId, dtype=np.int64)
        if emitter.min() < 0 or emitter.max() >= n:
            raise ValueError(f"lidar output names emitters {emitter.min()}..{emitter.max()} of {n}")
        # A ray fired on a render's boundary can come twice: keep its first report (np.unique sorts by emitter).
        _, first = np.unique(emitter, return_index=True)
        if len(first) != n:
            return self._partial(now_ns, len(first))
        self._whole = True
        pose = (np.asarray(g.frameEnd.posM, dtype=np.float64), _quat_matrix(g.frameEnd.orientation))
        return [self._sweep(now_ns, np.asarray(g.x, dtype=np.float64)[first], np.asarray(g.y, dtype=np.float64)[first],
                            np.asarray(g.z, dtype=np.float64)[first], np.asarray(g.flags)[first], pose)]

    def _partial(self, now_ns: int, rays: int) -> list[Sweep]:
        if self._whole:
            raise RuntimeError(f"the render at {now_ns} ns traced {rays} of the lidar's {len(self.fire_ns)} rays: "
                               "render the lidar once per sweep period, in its own pass")
        self.dropped += 1
        return []

    @staticmethod
    def _sweep(stamp_ns, az_deg, el_deg, r, flags, pose) -> Sweep:
        hit = ((flags & VALID) > 0) & (r > 0)
        r = np.where(hit, r, 0.0)
        az, el = np.radians(az_deg), np.radians(el_deg)
        points = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], axis=1)
        return Sweep(start_ns=int(stamp_ns), points_m=points.astype(np.float32), hit=hit, time_s=np.zeros(len(r)),
                     origin_w=pose[0], rotation_w=pose[1])

    @staticmethod
    def to_world(sweep: Sweep) -> np.ndarray:
        """The sweep's returns in the world, at the sensor's pose at the render that traced them."""
        return sweep.points_m[sweep.hit].astype(np.float64) @ sweep.rotation_w.T + sweep.origin_w


def _quat_matrix(q) -> np.ndarray:
    """Rotation matrix of a GMO frame orientation (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def scan_mesh_prim(stage, obj_path: str | Path, spawn_m: tuple[float, float], prim_path: str = SCAN_PRIM,
                   hidden_from_cameras: bool = True):
    """Author the scan at ``obj_path`` as a ``Mesh`` at ``prim_path`` on ``stage``, its point ``spawn_m`` at the
    origin; returns the prim."""
    from pxr import UsdGeom, Vt

    from humanoid_sim.world.scan import read_obj

    positions, _, faces, _ = read_obj(obj_path)
    points = (positions - np.array([spawn_m[0], spawn_m[1], 0.0])).astype(np.float32)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([tuple(map(float, lo)), tuple(map(float, hi))]))
    if hidden_from_cameras:
        from humanoid_sim.sensors.rtx import hide_from_cameras

        hide_from_cameras(stage, prim_path)
    return mesh.GetPrim()


def add_scan_surfaces(obj_path: str | Path, spawn_m: tuple[float, float], *, hidden_from_cameras: bool):
    """:func:`scan_mesh_prim` on the running simulation's stage."""
    import isaaclab.sim as sim_utils

    return scan_mesh_prim(sim_utils.get_current_stage(), obj_path, spawn_m, hidden_from_cameras=hidden_from_cameras)
