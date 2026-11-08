"""The imaging chain (``sensors/imaging.py``) on the RTX cameras: the G1 welded at the origin with the ``g1-cameras-d435``
rig under the default dome, every colour and grey stream imaged. At the reference exposure the chain gives the plain
render back. Exposure time and gain scale the light it tone maps. The noise of a short exposure read at high gain
has the spread of shot and read noise. Auto exposure brings each stream's signal to its target, sooner or later,
under the default dome and under a dome ten times brighter, one exposure per stereo pair.

Each case is rendered by ``tests/rtx_imaging_render.py`` in a process of its own (Kit starts once a process); this
process starts neither Kit nor USD. Needs the Isaac Sim image (``docker/Dockerfile.kit``) on an RTX GPU:

    pytest -m rtx tests/test_rtx_camera_imaging.py
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

from humanoid_sim.sensors import rigs  # noqa: E402
from humanoid_sim.sensors.imaging import REFERENCE_EXPOSURE_NS, WHITE_LIGHT, light_for  # noqa: E402

pytestmark = pytest.mark.rtx

REPO = Path(__file__).resolve().parents[1]
STREAMS = ("sim-zed/left", "sim-zed/right", "sim-d435/rgb", "sim-d435/mono_left", "sim-d435/mono_right")
REF = REFERENCE_EXPOSURE_NS
# (camera effects, a set exposure for every imaged stream or None for the instances' auto exposure, the dome's nits)
CASES = {"plain": ("", None, None),
         "ref": ("exposure", (REF, 1), None),
         "half": ("exposure", (REF // 2, 1), None),
         "gain": ("exposure", (REF // 2, 2), None),
         "noisy": ("exposure,noise", (REF // 8, 8), None),
         "auto": ("exposure", None, None),
         "auto_bright": ("exposure", None, 6000.0)}

# Tolerances, in 8-bit levels. Two processes render a frame a few levels apart on a few pixels (RTX's sampling).
SAME_P99 = 1.0
SAME_MEAN = 0.5
NOISE_SHARE = 0.1  # the spread of the noise against the model's, drawn here from the reference image
BRIGHTER = 5.0  # a dome ten times brighter shortens the auto exposure by more than this (the ambient light stays)
DARKEST_LEVEL = 30  # pixels darker than this in the reference are dropped from the noise check (a flat curve)


def curve(light):
    """RTX's 8-bit value of linear light before rounding (ACES as Narkowicz fits it, then sRGB)."""
    x = np.clip(light, 0.0, None)
    y = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    return 255 * np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)


GRID = np.linspace(0.0, WHITE_LIGHT, 400001)
TABLE = curve(GRID)


def light_of(levels):
    """The light the reference render's levels stand for."""
    return np.interp(levels, TABLE, GRID)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("imaging")
    out = {}
    for name, (effects, exposure, light) in CASES.items():
        args = ["--effects", effects]
        if exposure:
            args += ["--exposure", *map(str, exposure)]
        if light:
            args += ["--light", str(light)]
        path = d / f"{name}.npz"
        done = subprocess.run([sys.executable, str(REPO / "tests" / "rtx_imaging_render.py"), "--out", str(path), *args],
                              cwd=REPO, capture_output=True, text=True,
                              env={**__import__("os").environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'tests'}"})
        if not path.exists():
            pytest.fail(f"{name} did not render:\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
        out[name] = {**dict(np.load(path)), "summary": json.loads(path.with_suffix(".json").read_text())}
    return out


def levels(run, stream):
    return run[stream].astype(float)


def test_the_reference_exposure_is_the_plain_render(runs):
    for stream in STREAMS:
        d = np.abs(levels(runs["ref"], stream) - levels(runs["plain"], stream))
        assert np.percentile(d, 99) <= SAME_P99 and d.mean() <= SAME_MEAN, stream
    assert runs["plain"]["summary"] == {"camera_effects": [], "imaging": {}}
    assert runs["ref"]["summary"]["camera_effects"] == ["exposure"]


def test_exposure_time_and_gain_scale_the_light(runs):
    for stream in STREAMS:
        ref = levels(runs["ref"], stream)
        assert (levels(runs["gain"], stream) == ref).all(), stream  # half the time at twice the gain
        unsaturated = ref < 250
        half = np.round(curve(light_of(ref) / 2))
        d = np.abs(levels(runs["half"], stream) - half)[unsaturated]
        assert np.percentile(d, 99) <= SAME_P99 and d.mean() <= SAME_MEAN, stream


def test_a_short_exposure_at_high_gain_has_shot_and_read_noise(runs):
    rig = rigs.g1_cameras_d435()
    imagers = {"sim-zed": rig.sensors["zed"].imagers, "sim-d435": rig.sensors["d435"].imagers}
    rng = np.random.default_rng(0)
    exposure_ns, gain = CASES["noisy"][1]
    for stream in STREAMS:
        camera, name = stream.split("/")
        imager = imagers[camera][name]
        ref = levels(runs["ref"], stream)
        keep = (ref > DARKEST_LEVEL) & (ref < 250)
        if ref.ndim == 3:
            keep = keep.all(axis=-1)
        e = imager.full_well_e * light_of(ref[keep]) / WHITE_LIGHT * exposure_ns / REF
        drawn = np.clip(rng.poisson(e) + imager.read_noise_e * rng.standard_normal(e.shape), 0, imager.full_well_e)
        model = np.round(curve(drawn * gain * WHITE_LIGHT / imager.full_well_e)) - ref[keep]
        measured = levels(runs["noisy"], stream)[keep] - ref[keep]
        assert measured.std() == pytest.approx(model.std(), rel=NOISE_SHARE), stream
        assert abs(measured.mean() - model.mean()) < 0.5, stream


def test_auto_exposure_brings_each_signal_to_its_target(runs):
    target = light_for(rigs.g1_cameras_d435().sensors["d435"].auto_exposure.target)
    for name in ("auto", "auto_bright"):
        summary = runs[name]["summary"]["imaging"]
        for camera, groups in summary.items():
            for key, group in groups.items():
                assert group["mode"] == "auto" and group["frames"] > 30
                signal = np.mean([light_of(levels(runs[name], f"{camera}/{s}")).mean() for s in group["streams"]])
                assert signal == pytest.approx(target, rel=0.1), (name, camera, key)
    for camera in ("zed", "d435"):
        default = runs["auto"][f"exposure/{camera}/stereo"]
        bright = runs["auto_bright"][f"exposure/{camera}/stereo"]
        # ten times the dome (RTX's ambient light stays as it is): a fraction of the exposure, at unity gain, reached
        # over frames, not at once
        assert bright[-1, 1] < default[-1, 1] / BRIGHTER
        assert (bright[:, 2] == 1.0).all() and (np.diff(bright[:, 1]) <= 0).all()
        assert bright[2, 1] > 3 * bright[-1, 1]
