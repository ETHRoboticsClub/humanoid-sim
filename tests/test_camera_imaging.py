"""The imaging chain (sensors/imaging.py) on CPU: at the reference exposure it is the RTX render, exposure and gain
scale the signal, the noise has the variance of shot plus read noise, the well and the output clip, and the auto
exposure converges with its time constant, time before gain, one exposure per stereo pair."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from humanoid_sim.sensors import rigs
from humanoid_sim.sensors.geometry import camera_spec
from humanoid_sim.sensors.imaging import (
    CAMERA_EFFECTS,
    REFERENCE_EXPOSURE_NS,
    WHITE_LIGHT,
    ImagingChain,
    camera_seed,
    check_effects,
    light_for,
    parse_effects,
    tone_level,
)
from humanoid_sim.sensors.models import AutoExposure, Camera, Imager, OakDW

LUMA = torch.tensor([0.299, 0.587, 0.114])


def oak(**kw) -> OakDW:
    """The V1's OAK-D-W with ``kw`` replacing its instance arguments."""
    c = rigs.ethrc_v1().sensors["camera0"]
    args = dict(width=c.width, height=c.height, frame_period_ns=c.frame_period_ns, intrinsics=c.intrinsics,
                distortion=c.distortion, stream_offsets_m=c.stream_offsets_m,
                stream_rotations_xyzw=c.stream_rotations_xyzw, depth_focal_px=c.depth_focal_px) | kw
    camera = OakDW(c.frame, **args)
    camera.name = "camera0"
    return camera


def chain(effects, camera=None, seed=0) -> tuple[ImagingChain, object]:
    camera = camera or oak()
    spec = camera_spec(camera)
    return ImagingChain(camera, spec, effects, seed), spec


def flat(level: float, shape=(200, 200)) -> torch.Tensor:
    return torch.full((*shape, 3), level)


def test_effects_are_checked_and_ordered():
    assert check_effects(["noise", "exposure", "noise"]) == ("exposure", "noise")
    assert parse_effects("all") == CAMERA_EFFECTS
    assert parse_effects(" noise ,exposure") == ("exposure", "noise")
    assert parse_effects("") == ()
    with pytest.raises(ValueError, match="unknown camera effects"):
        parse_effects("exposure,bloom")


def rtx_curve(light: np.ndarray) -> np.ndarray:
    """RTX's 8-bit value of linear light before rounding, as measured against Isaac Lab's rgb_hdr: ACES (Narkowicz),
    then sRGB."""
    x = np.clip(light, 0.0, None)
    y = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    return 255 * np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)


def rtx_8bit(light: np.ndarray) -> np.ndarray:
    return np.round(rtx_curve(light))


def at(ns: int, gain: float) -> OakDW:
    return oak(exposure={s: (ns, gain) for s in ("rgb", "mono_left", "mono_right")})


def test_the_reference_exposure_is_the_rtx_render():
    """No noise, reference exposure: the 8-bit render, from the darkest light to past white. A grey imager tone maps
    the luminance: on neutral light that is the luma of the colour render."""
    c, spec = chain(("exposure",), at(REFERENCE_EXPOSURE_NS, 1.0))
    hdr = torch.rand(48, 64, 3, generator=torch.Generator().manual_seed(1)) ** 3 * 2 * WHITE_LIGHT
    rgb = c.image(spec.stream("rgb"), hdr)
    assert rgb.dtype == np.uint8 and rgb.shape == (48, 64, 3)
    assert np.abs(rgb - rtx_8bit(hdr.numpy())).max() <= 1
    assert (rgb[(hdr >= WHITE_LIGHT).numpy()] == 255).all()
    grey = c.image(spec.stream("mono_left"), hdr)
    assert grey.shape == (48, 64) and np.abs(grey - rtx_8bit(hdr.numpy() @ LUMA.numpy())).max() <= 1
    neutral = hdr[..., :1].expand(-1, -1, 3)
    assert np.abs(c.image(spec.stream("mono_left"), neutral) - np.round(
        rtx_8bit(neutral.numpy()) @ LUMA.numpy())).max() <= 1
    # without the exposure effect every imager is at the reference too, whatever the instance sets
    c, spec = chain((), oak(exposure={"rgb": (5_000_000, 1.0)}))
    assert c.exposure("rgb") == (REFERENCE_EXPOSURE_NS, 1.0)
    assert np.abs(c.image(spec.stream("rgb"), hdr) - rtx_8bit(hdr.numpy())).max() <= 1


def test_exposure_time_and_gain_scale_the_light():
    hdr = torch.linspace(0.0, 3.0, 64).reshape(8, 8, 1).expand(-1, -1, 3)
    c, spec = chain(("exposure",), at(REFERENCE_EXPOSURE_NS // 2, 1.0))
    assert np.abs(c.image(spec.stream("rgb"), hdr) - rtx_8bit(hdr.numpy() / 2)).max() <= 1
    c, spec = chain(("exposure",), at(REFERENCE_EXPOSURE_NS // 2, 3.0))
    assert np.abs(c.image(spec.stream("rgb"), hdr) - rtx_8bit(hdr.numpy() * 1.5)).max() <= 1


def test_the_output_whites_out_sooner_with_gain():
    c, spec = chain(("exposure",), at(REFERENCE_EXPOSURE_NS, 1.0))
    assert (c.image(spec.stream("rgb"), flat(WHITE_LIGHT * 1.01, (4, 4))) == 255).all()
    assert (c.image(spec.stream("rgb"), flat(WHITE_LIGHT * 0.5, (4, 4))) < 255).all()
    c, spec = chain(("exposure",), at(REFERENCE_EXPOSURE_NS, 2.0))
    assert (c.image(spec.stream("rgb"), flat(WHITE_LIGHT * 0.51, (4, 4))) == 255).all()


def test_the_noise_is_shot_noise_and_read_noise():
    """On a flat field each pixel collects Poisson(e) electrons plus Gaussian read noise r, read out at g X / W of
    light per electron and tone mapped: the 8-bit spread matches that model drawn here with numpy, and the same
    output read at 8 times the gain is noisier."""
    light, rng = 0.2, np.random.default_rng(0)
    stds = []
    for exposure_ns, gain in ((REFERENCE_EXPOSURE_NS, 1.0), (REFERENCE_EXPOSURE_NS // 8, 8.0)):
        camera = at(exposure_ns, gain)
        c, spec = chain(("exposure", "noise"), camera)
        imager = camera.imagers["mono_left"]
        image = c.image(spec.stream("mono_left"), flat(light, (400, 400))).astype(float)
        e = imager.full_well_e * light / WHITE_LIGHT * exposure_ns / REFERENCE_EXPOSURE_NS
        drawn = np.clip(rng.poisson(e, 400 * 400) + imager.read_noise_e * rng.standard_normal(400 * 400), 0,
                        imager.full_well_e)
        model = rtx_8bit(drawn * gain * WHITE_LIGHT / imager.full_well_e)
        assert image.mean() == pytest.approx(model.mean(), abs=0.05)
        assert image.std() == pytest.approx(model.std(), rel=0.02)
        stds.append(image.std())
    assert stds[1] > 2.5 * stds[0]


def test_the_noise_is_repeatable_per_seed_and_camera():
    hdr = flat(0.2, (32, 32))
    a, spec = chain(("noise",), seed=3)
    b, _ = chain(("noise",), seed=3)
    other, _ = chain(("noise",), seed=4)
    first = a.image(spec.stream("rgb"), hdr)
    assert (first == b.image(spec.stream("rgb"), hdr)).all()
    assert (first != other.image(spec.stream("rgb"), hdr)).any()
    assert (first != a.image(spec.stream("rgb"), hdr)).any()  # the next frame draws anew
    assert camera_seed(0, "sim-camera0") != camera_seed(0, "sim-zed") != camera_seed(1, "sim-zed")


def run_auto(c, spec, level: float, frames: int, streams=("rgb", "mono_left", "mono_right")):
    t = 0
    history = []
    for _ in range(frames):
        for name in streams:
            c.image(spec.stream(name), flat(level, (16, 16)))
        c.end_frame(t)
        history.append(c.exposure(streams[0]))
        t += c.period_ns
    return history


def test_auto_exposure_moves_by_its_time_constant():
    camera = oak()
    c, spec = chain(("exposure",), camera)
    ae = camera.auto_exposure
    assert c.exposure("rgb") == (REFERENCE_EXPOSURE_NS, 1.0)
    light = 0.05  # below the target at the reference exposure; the signal it meters
    (ns, gain), = run_auto(c, spec, light, 1, ("rgb",))
    target = light_for(ae.target)
    assert tone_level(target) == pytest.approx(ae.target, abs=1e-9)
    step = (1 - math.exp(-camera.frame_period_ns / ae.time_constant_ns)) * math.log(target / light)
    assert ns * gain == pytest.approx(REFERENCE_EXPOSURE_NS * math.exp(step), rel=1e-6)


def test_auto_exposure_lengthens_the_time_then_raises_the_gain():
    camera = oak()
    c, spec = chain(("exposure",), camera)
    history = run_auto(c, spec, 0.001, 200)
    longest = min(camera.imagers["rgb"].exposure_ns[1], camera.frame_period_ns)
    assert all(gain == 1.0 for ns, gain in history if ns < longest)
    assert history[-1] == (longest, camera.imagers["rgb"].gain[1])  # 0.001 needs more than the gain reaches
    frames_dark = sum(1 for ns, gain in history if ns * gain < 0.5 * longest * camera.imagers["rgb"].gain[1])
    assert 2 < frames_dark < 60  # it takes some frames: the lag the issue asks for
    # a bright room: shorter exposures at unity gain, settling on the target
    c, spec = chain(("exposure",), camera)
    history = run_auto(c, spec, 2.0, 200)
    ns, gain = history[-1]
    assert gain == 1.0 and ns < REFERENCE_EXPOSURE_NS
    assert 2.0 * ns / REFERENCE_EXPOSURE_NS == pytest.approx(light_for(camera.auto_exposure.target), rel=1e-3)
    assert rtx_8bit(np.array([2.0 * ns / REFERENCE_EXPOSURE_NS]))[0] == pytest.approx(255 * 0.46, abs=1)


def test_a_stereo_pair_shares_one_exposure():
    c, spec = chain(("exposure",))
    c.image(spec.stream("mono_left"), flat(0.05, (16, 16)))
    c.image(spec.stream("mono_right"), flat(0.25, (16, 16)))
    c.image(spec.stream("rgb"), flat(0.3, (16, 16)))
    c.end_frame(0)
    assert c.exposure("mono_left") == c.exposure("mono_right") != c.exposure("rgb")
    assert c.describe()["stereo"]["streams"] == ["mono_left", "mono_right"]
    assert c.describe()["stereo"]["mode"] == "auto" and c.describe()["stereo"]["frames"] == 1


def test_exposure_settings_are_checked():
    with pytest.raises(ValueError, match="outside"):
        oak(exposure={"rgb": (40_000_000, 1.0)})
    with pytest.raises(ValueError, match="outside"):
        oak(exposure={"rgb": (10_000_000, 0.5)})
    with pytest.raises(ValueError, match="without an imager"):
        oak(exposure={"depth": "auto"})
    with pytest.raises(ValueError, match="expected 'auto'"):
        oak(exposure={"rgb": "manual"})
    with pytest.raises(ValueError, match="exposed together"):
        chain(("exposure",), oak(exposure={"mono_left": (10_000_000, 1.0)}))
    assert oak(exposure={"rgb": (10_000_000, 2)}).exposure["rgb"] == (10_000_000, 2.0)


def test_imager_figures_are_checked():
    with pytest.raises(ValueError):
        Imager("x", 0.0, 1.0, (1, 2), (1.0, 2.0), "test")
    with pytest.raises(ValueError):
        Imager("x", 100.0, 1.0, (2, 1), (1.0, 2.0), "test")
    with pytest.raises(ValueError):
        Imager("x", 100.0, 1.0, (1, 2), (0.5, 2.0), "test")
    with pytest.raises(ValueError):
        AutoExposure(1.2, 1, "test")

    with pytest.raises(ValueError, match="imagers"):
        type("BadCamera", (Camera,), {"model": "bad", "product": "bad", "source": "test", "streams": ("depth",),
                                      "stream_kinds": {"depth": "depth"},
                                      "imagers": {"depth": Imager("x", 100.0, 1.0, (1, 2), (1.0, 2.0), "t")}})


def test_a_camera_without_imager_figures_is_not_imaged():
    class Plain(Camera):
        model = "plain"
        product = "plain"
        source = "test"
        streams = ("rgb",)
        stream_kinds = {"rgb": "color"}
        nominal_offsets_m = {"rgb": (0.0, 0.0, 0.0)}

    camera = Plain("cam", width=8, height=8, frame_period_ns=33_333_333, intrinsics={"rgb": (8.0, 8.0, 4.0, 4.0)})
    camera.name = "plain"
    with pytest.raises(ValueError, match="no imager figures"):
        ImagingChain(camera, camera_spec(camera), ("noise",), 0)


@pytest.mark.parametrize("make", [rigs.ethrc_v1, rigs.g1_cameras_d435])
def test_every_rig_camera_can_be_imaged(make):
    for camera in make().sensors.values():
        if camera.kind == "camera":
            spec = camera_spec(camera)
            c = ImagingChain(camera, spec, CAMERA_EFFECTS, 0)
            assert set(c.group) == {s.name for s in spec.streams if s.kind != "depth"}
