"""The RTX lidar's emitter states and prim attributes (``sensors/lidar.py``) against the catalog's Mid-360 and the
scan pattern: no Kit needed."""

import numpy as np
import pytest

from humanoid_sim.sensors import geometry
from humanoid_sim.sensors.models import load_catalog


def test_the_rtx_emitter_states_fire_the_pattern():
    from humanoid_sim.sensors import lidar

    model = load_catalog()["livox-mid360"]
    az, el, fire_ns = lidar.pattern_states(model, sweeps=2)
    # One state per sweep of 20,000 rays, fired evenly through the sweep's 100 ms.
    assert az.shape == el.shape == (2, 20_000) and fire_ns.shape == (20_000,)
    d = np.concatenate([geometry.scan_pattern(model, 0)[0], geometry.scan_pattern(model, 1)[0]])
    # RTX takes the emitter azimuth clockwise: the state holds it negated.
    a, e = np.radians(-az.reshape(-1)), np.radians(el.reshape(-1))
    np.testing.assert_allclose(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=1), d,
                               atol=1e-6)
    assert fire_ns[0] == 2500 and np.all(np.diff(fire_ns) == 5000) and fire_ns[-1] < 100_000_000
    exact, noisy = lidar.lidar_attributes(model, noise=False), lidar.lidar_attributes(model, noise=True)
    assert exact[lidar.CORE + "numberOfEmitters"] == 20_000 == sum(exact[lidar.CORE + "numRaysPerLine"])
    assert exact[lidar.CORE + "rangeAccuracyM"] == 0.0 == exact[lidar.CORE + "azimuthErrorStd"]
    # The Livox manual's 2 cm range and 0.15 deg angular precision (1 sigma).
    assert noisy[lidar.CORE + "rangeAccuracyM"] == 0.02
    assert noisy[lidar.CORE + "azimuthErrorStd"] == pytest.approx(0.15) == noisy[lidar.CORE + "elevationErrorStd"]
