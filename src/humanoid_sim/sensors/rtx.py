"""RTX sensors on a robot in humanoid-sim: Isaac Sim's (Kit's) RTX renderer draws the cameras and casts the lidar
while Newton/MJWarp runs the physics. The Isaac RTX renderer copies every body's pose from the physics into Fabric
before it renders (Isaac Lab ``isaaclab_physx/renderers/fabric.py``), so the sensors see the robot where Newton put
it. Needs the Isaac Sim image (``docker/Dockerfile.kit``); training and the kitless image never import this module.

* :func:`start_kit` starts Kit with cameras enabled, headless or with its window, before the simulation is built.
* :func:`camera_entities` turns rig cameras' streams (``geometry.camera_spec``) into Isaac Lab cameras on the robot
  body each is mounted on (its rig frame, merged into that body), with the stream's lens: OpenCV's fisheye or
  pinhole model, which RTX renders natively (``omni:lensdistortion``).
* :func:`splat_entity` places a room's Gaussian splat (``world/splat.py``) for the cameras, :func:`light_entity` a
  dome light for everything else.
* :class:`KitRender` renders the stage at a plant time: Kit's timeline at that time (RTX lidars fire on it), the
  physics' body poses copied into Fabric, one Kit update.
* :class:`RtxCameras` reads each stream of the last render as its feed wants it.

The physics steps without rendering (``IsaacLabRobot.step``); the caller renders when a sensor is due.
"""

from __future__ import annotations

import contextlib
import re
import sys

import numpy as np

from humanoid_sim.sensors.geometry import CameraSpec, Lens, Mount, Stream, quat_xyzw

ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"
ROBOT_ROOT = "/World/envs/env_0/Robot"
DOME_INTENSITY = 600.0  # nits
DISABLE_COLOR_RENDER = "/rtx/sdg/force/disableColorRender"
# Camera exposure is not modeled (docs/rtx-sensors.md): every camera renders at one fixed exposure. RTX's auto
# exposure, which would meter each render product on its own image, is off, and the tone mapping that turns the
# splat's emission into 8-bit pixels is set here, not left to Kit's defaults, so a Kit upgrade cannot change how
# bright a room renders. The values are Isaac Sim 6.2's defaults, at which the splat intensities were chosen.
TONEMAP_ACES = 6  # /rtx/post/tonemap/op: Clamp, Linear, Reinhard, Modified Reinhard, HejlHableAlu, HableUc2, ACES, Iray
FILM_ISO = 100.0
EXPOSURE_TIME_S = 0.02  # 1/50 s
F_NUMBER = 5.0  # with ISO 100 and 1/50 s: EV100 = log2(F^2 / t) = 10.3
RESPONSIVITY = 1.1026709079742432  # Kit's sensor responsivity, dimensionless
CM2_FACTOR = 1.0  # Kit's scale of the scene's radiance before tone mapping, dimensionless
WHITE_POINT = [1.0, 1.0, 1.0]  # linear RGB mapped to white
SRGB_LINEAR = 0  # /rtx/post/tonemap/colorMode: sRGB linear, ACEScg
AUTO_EXPOSURE = "/rtx/post/histogram/enabled"
EXPOSURE = {AUTO_EXPOSURE: False,
            "/rtx/post/tonemap/op": TONEMAP_ACES,
            "/rtx/post/tonemap/filmIso": FILM_ISO,
            "/rtx/post/tonemap/exposureTime": EXPOSURE_TIME_S,
            "/rtx/post/tonemap/cameraShutter": 1.0 / EXPOSURE_TIME_S,  # the same time as Kit's older 1/s setting
            "/rtx/post/tonemap/fNumber": F_NUMBER,
            "/rtx/post/tonemap/responsivity": RESPONSIVITY,
            "/rtx/post/tonemap/cm2Factor": CM2_FACTOR,
            "/rtx/post/tonemap/whitepoint": WHITE_POINT,
            "/rtx/post/tonemap/colorMode": SRGB_LINEAR,
            "/rtx/post/tonemap/enableSrgbToGamma": True}
# RTX's OpenCV lens models put pixel (u, v)'s centre at (u + 0.5, v + 0.5), OpenCV's calibration at (u, v): the
# principal point given to RTX is moved by half a pixel so the images match the calibration Controls publishes
# (measured against ray-cast masks in tests/test_rtx_sensors.py: half a pixel in both axes, both models).
RTX_PIXEL_CENTRE_PX = 0.5
LUMA_BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)
HIDDEN_FROM_CAMERAS = "primvars:hideForCamera"
CASTS_NO_SHADOWS = "primvars:doNotCastShadows"
SPLAT_PRIM = "/World/Splat"  # RTX: primary camera rays skip the prim; other rays still hit it


def start_kit(device: str, *, physics_cfg=None, window: bool = False) -> contextlib.ExitStack:
    """Start Kit (cameras on; headless, or with its window on ``$DISPLAY``) for the rest of the process; returns the
    stack that ends it. Use it in a ``with``: leaving it with an exception hands the exception to Isaac Lab's launch
    context, while ``close()`` leaves it cleanly, whatever failed (and Kit's close may end the process)."""
    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    physics_cfg = physics_cfg or NewtonCfg(solver_cfg=MJWarpSolverCfg())
    stack = contextlib.ExitStack()
    # Kit reads the process's command line as its own: a caller's arguments (pytest's, say) must not reach it.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        # Isaac Lab opens Kit's window for its Kit visualizer, whatever ``headless`` says.
        visualizer = {"visualizer": ["kit"]} if window else {}
        stack.enter_context(launch_simulation(sim_utils.SimulationCfg(device=device, physics=physics_cfg),
                                              {**visualizer, "headless": not window, "enable_cameras": True,
                                               "require_kit": True, "device": device}))
    finally:
        sys.argv = argv
    return stack


def entity_name(camera: CameraSpec, stream: Stream) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", f"{camera.camera_id}_{stream.name}")


def _distortion(lens: Lens):
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        OpenCvFisheyeDistortionCfg,
        OpenCvPinholeDistortionCfg,
    )

    common = {"fx": lens.fx_px, "fy": lens.fy_px, "cx": lens.cx_px + RTX_PIXEL_CENTRE_PX,
              "cy": lens.cy_px + RTX_PIXEL_CENTRE_PX, "image_size": (lens.width_px, lens.height_px)}
    if lens.distortion_model == "equidistant":
        k1, k2, k3, k4 = lens.distortion
        return OpenCvFisheyeDistortionCfg(**common, k1=k1, k2=k2, k3=k3, k4=k4)
    if lens.distortion_model == "plumb_bob":
        k1, k2, p1, p2, k3 = lens.distortion
        return OpenCvPinholeDistortionCfg(**common, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    return OpenCvPinholeDistortionCfg(**common)


def stream_offset(mount: Mount, stream: Stream) -> tuple[tuple, tuple]:
    """The stream's camera in its body: position, and orientation (x, y, z, w) in Isaac Lab's ``world`` camera
    convention (x forward, z up), which is the camera frame's; every stream of a camera looks along its x axis."""
    position = mount.R @ np.asarray(stream.translation_m) + np.asarray(mount.translation_m)
    return tuple(float(v) for v in position), quat_xyzw(mount.R)


def body_paths(usd_path, bodies) -> dict[str, str]:
    """Where each of ``bodies`` sits in the robot asset at ``usd_path``, relative to its default prim (the robot's
    root in the scene): the asset nests bodies along the kinematic tree (``Geometry/pelvis/.../torso_link``), and a
    sensor must be the body's child to ride on it."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    found = {}
    for prim in Usd.PrimRange(root):  # depth first: a body before any prim below it of the same name
        if prim.GetName() in bodies:
            found.setdefault(prim.GetName(), prim.GetPath().MakeRelativePath(root.GetPath()))
    missing = set(bodies) - set(found)
    if missing:
        raise KeyError(f"{usd_path} has no prim for {sorted(missing)}")
    return {name: str(path) for name, path in found.items()}


def camera_entities(cameras: list[tuple[CameraSpec, Mount]], bodies: dict[str, str],
                    robot_prim: str = ROBOT_PRIM) -> dict:
    """Isaac Lab camera configs, by :func:`entity_name`, for every stream of ``cameras`` (each with its mount), each
    a child of its body's prim (``bodies``: :func:`body_paths`)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    out = {}
    for cam, mount in cameras:
        body = bodies[mount.link]
        for s in cam.streams:
            lens = s.lens
            near, far = (s.range_m[0] * 0.5, s.range_m[1] * 2.0) if s.kind == "depth" else (0.05, 1000.0)
            spawn = sim_utils.PinholeCameraCfg(clipping_range=(near, far), distortion=_distortion(lens))
            pos, rot = stream_offset(mount, s)
            out[entity_name(cam, s)] = CameraCfg(
                prim_path=f"{robot_prim}/{body}/{entity_name(cam, s)}", update_period=0.0,
                height=lens.height_px, width=lens.width_px,
                data_types=["distance_to_image_plane"] if s.kind == "depth" else ["rgb"],
                spawn=spawn, offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
                renderer_cfg=IsaacRtxRendererCfg())
    return out


def set_splat_intensity(stage, intensity: float, prim_path: str = SPLAT_PRIM) -> None:
    """Scale the splat's emission (``ParticleFieldEmissive``'s ``emission_intensity``, 1 by default): RTX shows a
    splat's baked colours through its own exposure, which no camera's exposure control follows."""
    from pxr import Sdf, Usd, UsdShade

    shaders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)) if p.IsA(UsdShade.Shader)]
    if not shaders:
        raise ValueError(f"no splat material under {prim_path}")
    for shader in shaders:
        shader.CreateAttribute("inputs:emission_intensity", Sdf.ValueTypeNames.Float).Set(float(intensity))


def hide_from_cameras(stage, prim_path: str) -> int:
    """Hide every geometry prim under ``prim_path`` from camera rays (RTX's ``primvars:hideForCamera``) and from
    shadow rays (``primvars:doNotCastShadows``), leaving it to the physics and to the lidar; returns how many."""
    from pxr import Sdf, Usd, UsdGeom

    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if prim.IsA(UsdGeom.Gprim):
            prim.CreateAttribute(HIDDEN_FROM_CAMERAS, Sdf.ValueTypeNames.Bool).Set(True)
            # RTX dims a splat by the share of the stage's lights it cannot see, whatever their intensity: hidden walls
            # and the ground under the splat's floor would darken the room, more so with every light added.
            prim.CreateAttribute(CASTS_NO_SHADOWS, Sdf.ValueTypeNames.Bool).Set(True)
            n += 1
    return n


def hide_from_all_rays(stage, prim_path: str) -> None:
    """Make the prim at ``prim_path`` invisible to every ray, cameras' and lidars'; the physics keeps colliding
    with it once its model is built."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"no prim at {prim_path}")
    UsdGeom.Imageable(prim).MakeInvisible()


def splat_entity(usd_path: str, prim_path: str = SPLAT_PRIM):
    """A room's Gaussian splat (``humanoid-sim splat-usd``), already placed in the world frame."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path=prim_path, spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)))


def light_entity(intensity: float = DOME_INTENSITY, prim_path: str = "/World/DomeLight"):
    """A white dome light: the robot and the boxes are lit evenly; the splat is emissive and needs none."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path=prim_path, spawn=sim_utils.DomeLightCfg(intensity=intensity, color=(1.0, 1.0, 1.0)))


class KitRender:
    """Renders the stage at plant times. Kit's timeline stops advancing on its own and is set to each render's time:
    its plant time, or behind it where a lidar holds it back (``RtxLidar.render_time``). Every camera renders at the
    one fixed exposure :data:`EXPOSURE`.

    With ``color``, colour rendering is kept on: Isaac Lab's RTX renderer turns it off for the whole app
    (``/rtx/sdg/force/disableColorRender``) when it sets up a camera without a colour output, such as a depth stream,
    and every colour camera then renders black (Isaac Lab 3.0, ``IsaacRtxRenderer.create_render_data``)."""

    def __init__(self, sim, *, color: bool = True):
        import omni.timeline

        self.sim = sim
        self.color = color
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.set_auto_update(False)
        from isaaclab.app.settings_manager import get_settings_manager

        settings = get_settings_manager()
        for name, value in EXPOSURE.items():
            settings.set(name, value)
        if not self.timeline.is_playing():
            self.timeline.play()
        self.timeline.commit()

    def __call__(self, t_ns: int) -> None:
        from isaaclab.app.settings_manager import get_settings_manager
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        if self.color:
            get_settings_manager().set(DISABLE_COLOR_RENDER, False)
        self.timeline.set_current_time(t_ns / 1e9)
        self.timeline.commit()
        self.sim.render()
        ensure_isaac_rtx_render_update(force=True)

    @staticmethod
    def post_settings() -> dict[str, object]:
        """The active value of each of :data:`EXPOSURE`'s settings, as Kit's settings return it."""
        from isaaclab.app.settings_manager import get_settings_manager

        settings = get_settings_manager()
        return {name: settings.get(name) for name in EXPOSURE}


class RtxCameras:
    """The streams of the scene's cameras, read after a :class:`KitRender`: colour as ``[H, W, 3]`` uint8 RGB, grey
    as ``[H, W]`` uint8 (BT.601 luma of the colour render, full range), depth as metres along the optical axis."""

    def __init__(self, backend, cameras: list[CameraSpec]):
        self.backend = backend
        self.streams = [(cam, s, backend.scene[entity_name(cam, s)]) for cam in cameras for s in cam.streams]

    def update(self, dt_s: float) -> None:
        for _, _, camera in self.streams:
            camera.update(dt_s)

    def image(self, camera: CameraSpec, stream: Stream) -> np.ndarray:
        sensor = self.backend.scene[entity_name(camera, stream)]
        if stream.kind == "depth":
            return sensor.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        rgb = sensor.data.output["rgb"][0, ..., :3].cpu().numpy()
        if stream.kind == "mono":
            return np.round(rgb.astype(np.float32) @ LUMA_BT601).astype(np.uint8)
        return rgb
