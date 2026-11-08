"""What a camera's imagers make of the light RTX renders: exposure time and gain (set, or the device's auto exposure),
shot and read noise, a full well and an 8-bit output. Applied to the colour and grey streams when a session asks
for it (``SessionCfg.camera_effects``); without it every stream is the RTX render as it is.

RTX renders each colour camera into linear light (Isaac Lab's ``rgb_hdr``), already exposed by the fixed exposure
``rtx.EXPOSURE`` (1/50 s at ISO 100 and f/5): doubling the exposure time or the ISO doubles it. The 8-bit ``rgb`` it
hands over is that light tone mapped, per channel, by ``rtx.EXPOSURE``'s ACES curve (Narkowicz's fit) and encoded
as sRGB, :func:`tone_map` (measured on an RTX A5000: within one level at the 99th percentile, from the darkest
pixels to past white, on the plain ground, boxes and a splat). Asking Isaac Lab for both makes its ``rgb`` linear
instead, so a camera gets one or the other (``rtx.camera_entities``). The chain reads that light as each stream's
imager (:class:`~humanoid_sim.sensors.models.Imager`) would collect it:

* **Exposure.** An imager at exposure time ``t`` and gain ``g`` collects ``e = W · (x / X) · t / t_ref`` electrons:
  ``x`` the rendered light, ``W`` its full well, ``X`` the light the tone curve takes to white
  (:data:`WHITE_LIGHT`) and ``t_ref`` the render's exposure (:data:`REFERENCE_EXPOSURE_NS`). It reads out
  ``g · X · e / W`` and tone maps it as RTX does. The reference exposure (1/50 s, gain 1) without noise is the RTX
  render, to the rounding. A grey stream collects the luminance of the light (BT.601 weights) and tone maps that.
  The plain grey stream takes the luma of the tone-mapped colours instead: the same image on neutral surfaces, a few
  levels apart on strongly coloured ones.
* **Noise.** Each pixel's electrons are Poisson around ``e`` (shot noise), plus the imager's read noise (Gaussian,
  electrons). The gain scales both with the signal, so a dim scene read at high gain is noisy.
* **Saturation.** The well holds ``W`` electrons, which read white at unity gain; with more gain the output whites
  out sooner. Whatever is brighter clips.
* **Auto exposure.** With ``exposure`` among the effects, each stream's exposure is the camera instance's setting
  (``Camera.exposure``: ``"auto"`` by default, as Controls leaves every camera). An auto exposure runs the device's
  loop (:class:`~humanoid_sim.sensors.models.AutoExposure`):
  - it meters each frame's mean signal, linear (the sensor's statistics, before the tone curve), and moves the
    exposure towards the one whose signal tone maps to its target;
  - it lengthens the time first, up to the frame period and the imager's longest, then raises the gain;
  - so a camera entering a dark room is dark for a few frames, then noisy;
  - streams of one sync group (a stereo pair) share one exposure, as the devices expose them together.
  Without ``exposure``, every imager is at the reference exposure.

The noise has its own generator per camera, from the session's seed, so the depth errors' draws do not change with
the imaging. The chain runs in torch on the render's device.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

CAMERA_EFFECTS = ("exposure", "noise")
REFERENCE_EXPOSURE_NS = 20_000_000  # rtx.EXPOSURE_TIME_S: the exposure the RTX render is made at
FULL_SCALE = 255.0  # an 8-bit stream's levels
LUMA = (0.299, 0.587, 0.114)  # BT.601, as rtx.LUMA_BT601
DARKEST_METERED = 1e-4  # the signal an auto exposure reads a black frame as (about a hundredth of a level)
# rtx.EXPOSURE's tone map: ACES as Narkowicz fits it, A x (x + B) / (x (C x + D) + E) clipped at 1, then sRGB.
ACES = (2.51, 0.03, 2.43, 0.59, 0.14)
# The light ACES takes to white: A x^2 + B x = C x^2 + D x + E.
WHITE_LIGHT = ((ACES[3] - ACES[1]) + math.sqrt((ACES[3] - ACES[1]) ** 2 + 4 * (ACES[0] - ACES[2]) * ACES[4])) / (
    2 * (ACES[0] - ACES[2]))


def tone_level(x: float) -> float:
    """:func:`tone_map` of one value of light."""
    a, b, c, d, e = ACES
    x = max(x, 0.0)
    y = min(max(x * (a * x + b) / (x * (c * x + d) + e), 0.0), 1.0)
    return 12.92 * y if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055


def light_for(level: float) -> float:
    """The light :func:`tone_map` takes to ``level`` (a fraction of full scale)."""
    lo, hi = 0.0, WHITE_LIGHT
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tone_level(mid) < level else (lo, mid)
    return 0.5 * (lo + hi)


def tone_map(x):
    """RTX's 8-bit value, ``[0, 1]``, of linear light ``x`` (a torch tensor): ACES, then the sRGB encoding."""
    a, b, c, d, e = ACES
    x = x.clamp(min=0.0)
    y = (x * (a * x + b) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    return (12.92 * y).where(y <= 0.0031308, 1.055 * y.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def check_effects(effects: Iterable[str]) -> tuple[str, ...]:
    """The camera effects named, each once, in :data:`CAMERA_EFFECTS`' order; an unknown name raises."""
    effects = tuple(effects)
    unknown = sorted(set(effects) - set(CAMERA_EFFECTS))
    if unknown:
        raise ValueError(f"unknown camera effects {unknown}; known: {list(CAMERA_EFFECTS)}")
    return tuple(e for e in CAMERA_EFFECTS if e in effects)


def parse_effects(text: str) -> tuple[str, ...]:
    """``--camera-effects``: names separated by commas, or ``all``."""
    if text.strip() == "all":
        return CAMERA_EFFECTS
    return check_effects(e.strip() for e in text.split(",") if e.strip())


def camera_seed(seed: int, camera_id: str) -> int:
    """A camera's noise seed, from the session's."""
    return int(np.random.SeedSequence([seed, zlib.crc32(camera_id.encode())]).generate_state(1)[0])


@dataclass
class Exposure:
    """One sync group's exposure: its time (ns) and gain for the next frame, and the range of each over the frames
    taken so far."""

    exposure_ns: int
    gain: float
    auto: bool
    streams: list[str]
    last_ns: int | None = None
    frames: int = 0
    seen: dict = field(default_factory=dict)
    _metered: list[float] = field(default_factory=list)

    def note(self) -> None:
        for name, value in (("exposure_ns", self.exposure_ns), ("gain", self.gain)):
            lo, hi = self.seen.get(name, (value, value))
            self.seen[name] = (min(lo, value), max(hi, value))

    def describe(self) -> dict:
        return {"streams": self.streams, "mode": "auto" if self.auto else "set", "frames": self.frames,
                "next_exposure_ns": self.exposure_ns, "next_gain": self.gain,
                "exposure_ns_range": list(self.seen.get("exposure_ns", (self.exposure_ns,) * 2)),
                "gain_range": list(self.seen.get("gain", (self.gain,) * 2))}


class ImagingChain:
    """The imaging of one camera's colour and grey streams (``camera``: its model instance, ``spec``: its
    :class:`~humanoid_sim.sensors.geometry.CameraSpec`), with ``effects`` among :data:`CAMERA_EFFECTS`. Per frame:
    :meth:`image` for each stream, then :meth:`end_frame`."""

    def __init__(self, camera, spec, effects: Iterable[str], seed: int):
        self.effects = check_effects(effects)
        self.camera = camera
        self.period_ns = int(camera.frame_period_ns)
        where = f"camera {camera.name!r} ({camera.model})"
        images = [s for s in spec.streams if s.kind != "depth"]
        missing = [s.name for s in images if s.name not in camera.imagers]
        if missing:
            raise ValueError(f"{where}: no imager figures for {missing}: its model cannot be imaged")
        self.group = {s.name: s.sync_group or s.name for s in images}
        self.exposures: dict[str, Exposure] = {}
        for key in dict.fromkeys(self.group.values()):
            streams = [n for n, g in self.group.items() if g == key]
            imagers = {camera.imagers[n] for n in streams}
            settings = {camera.exposure[n] for n in streams}
            if len(imagers) > 1 or len(settings) > 1:
                raise ValueError(f"{where}: streams {streams} are exposed together but have different imagers or "
                                 f"exposure settings")
            setting = settings.pop() if "exposure" in self.effects else (REFERENCE_EXPOSURE_NS, 1.0)
            auto = setting == "auto"
            if auto and camera.auto_exposure is None:
                raise ValueError(f"{where}: exposure 'auto' but its model has no auto exposure")
            ns, gain = self._split(imagers.pop(), REFERENCE_EXPOSURE_NS) if auto else setting
            self.exposures[key] = Exposure(ns, gain, auto, streams)
        self.seed = seed
        self._generator = None

    def _longest_ns(self, imager) -> int:
        return min(imager.exposure_ns[1], self.period_ns)

    def _split(self, imager, product: float) -> tuple[int, float]:
        """An exposure of ``product`` ns times gain as the device sets it: time first, then gain, within range."""
        shortest, longest = imager.exposure_ns[0], self._longest_ns(imager)
        if product <= longest:
            return int(round(max(product, shortest))), float(imager.gain[0])
        return longest, float(min(max(product / longest, imager.gain[0]), imager.gain[1]))

    def exposure(self, stream: str) -> tuple[int, float]:
        """The exposure time (ns) and gain the stream's next frame is taken at."""
        e = self.exposures[self.group[stream]]
        return e.exposure_ns, e.gain

    def image(self, stream, hdr):
        """The stream's frame from the RTX render's linear light ``hdr`` (``[H, W, 3]``, torch): ``[H, W, 3]`` uint8
        RGB for colour, ``[H, W]`` uint8 for grey."""
        import torch

        imager = self.camera.imagers[stream.name]
        e = self.exposures[self.group[stream.name]]
        light = hdr[..., :3].float()
        if stream.kind == "mono":
            light = light @ torch.tensor(LUMA, dtype=light.dtype, device=light.device)
        electrons = light.clamp(min=0.0) * (imager.full_well_e / WHITE_LIGHT * e.exposure_ns / REFERENCE_EXPOSURE_NS)
        if "noise" in self.effects:
            generator = self._generator_on(light.device)
            electrons = torch.poisson(electrons, generator=generator)
            electrons = electrons + imager.read_noise_e * torch.randn(electrons.shape, generator=generator,
                                                                      device=light.device)
        electrons = electrons.clamp(0.0, imager.full_well_e)
        signal = (electrons * (e.gain * WHITE_LIGHT / imager.full_well_e)).clamp(max=WHITE_LIGHT)
        level = (tone_map(signal) * FULL_SCALE).round()
        e._metered.append(float(signal.mean()))
        return level.to(torch.uint8).cpu().numpy()

    def _generator_on(self, device):
        import torch

        if self._generator is None:
            self._generator = torch.Generator(device=device).manual_seed(self.seed)
        return self._generator

    def end_frame(self, now_ns: int) -> None:
        """Close the frame taken at plant time ``now_ns``: each auto exposure meters it and sets the next."""
        for e in self.exposures.values():
            metered, e._metered = e._metered, []
            e.frames += 1
            e.note()
            if e.auto and metered:
                dt = self.period_ns if e.last_ns is None else now_ns - e.last_ns
                ae = self.camera.auto_exposure
                imager = self.camera.imagers[e.streams[0]]
                signal = max(sum(metered) / len(metered), DARKEST_METERED)
                step = (1.0 - math.exp(-max(dt, 0) / ae.time_constant_ns)) * math.log(light_for(ae.target) / signal)
                e.exposure_ns, e.gain = self._split(imager, e.exposure_ns * e.gain * math.exp(step))
            e.last_ns = now_ns

    def describe(self) -> dict:
        return {key: e.describe() for key, e in self.exposures.items()}


__all__ = ["CAMERA_EFFECTS", "REFERENCE_EXPOSURE_NS", "WHITE_LIGHT", "ImagingChain", "camera_seed", "check_effects",
           "light_for", "parse_effects", "tone_level", "tone_map"]
