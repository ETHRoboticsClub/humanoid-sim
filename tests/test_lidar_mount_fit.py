"""A profile's lidar mount fit, given on the head link, replaces the description's and rides with the head."""
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.sensors.geometry import Mount, rpy_matrix
from humanoid_sim.sensors.session import mount_on

IN_HEAD = ([-0.003680, 0.00003, 0.457434], [np.pi, 0.0511, 0.0])
HEAD = Mount("head_link", "torso_link", (0.0039635, 0.0, -0.044), tuple(map(tuple, np.eye(3))))


def test_with_the_head_untilted_the_fit_is_the_factory_mount_15_mm_lower():
    m = mount_on(HEAD, *IN_HEAD)
    np.testing.assert_allclose(m.translation_m, [0.0002835, 0.00003, 0.428434 - 0.015], atol=1e-9)
    np.testing.assert_allclose(m.R, rpy_matrix(np.pi, 0.0511, 0), atol=1e-12)
    assert m.link == "torso_link"


def test_a_tilted_head_carries_the_sensor_rigidly():
    tilt = rpy_matrix(0, np.radians(2), 0)
    tilted = Mount("head_link", "torso_link", HEAD.translation_m, tuple(map(tuple, tilt)))
    a, b = mount_on(HEAD, *IN_HEAD), mount_on(tilted, *IN_HEAD)
    np.testing.assert_allclose(b.R, tilt @ a.R, atol=1e-12)  # orientation turns with the head
    head_o = np.array(HEAD.translation_m)
    np.testing.assert_allclose(np.array(b.translation_m) - head_o, tilt @ (np.array(a.translation_m) - head_o), atol=1e-12)
    assert np.linalg.norm(np.subtract(b.translation_m, a.translation_m)) > 0.01  # the ray origin moved
    # in the head frame nothing changed: the sensor stays where it was relative to the shell
    np.testing.assert_allclose(tilt.T @ (np.array(b.translation_m) - head_o), np.array(a.translation_m) - head_o, atol=1e-12)


def test_a_malformed_fit_is_refused():
    with pytest.raises(ValueError, match="3 numbers"):
        mount_on(HEAD, [0.0, 0.0], [np.pi, 0.0, 0.0])
