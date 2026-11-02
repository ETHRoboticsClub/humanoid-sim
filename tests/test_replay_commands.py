"""Resolved command stream and timestamp contracts, without running a solver."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.description import Description
from humanoid_sim.replay.exc import DriveTrace
from humanoid_sim.replay.replay import DriveCommands, _prepare


@pytest.fixture
def inputs():
    desc = Description.load()
    plant = PlantCore(desc, list(desc.joints), 4, PlantCfg(dtype=torch.float64))
    sim = SimpleNamespace(plant=plant, period_ns=4_000_000)
    stamps = np.arange(20, dtype=np.int64) * 5_000_000

    class Capture:
        def pdo(self):
            return {
                d: DriveTrace(
                    d,
                    stamps,
                    {
                        key: np.zeros(len(stamps))
                        for key in [
                            "native_position_rad",
                            "velocity_rad_s",
                            "torque_nm",
                            "position_demand_rad",
                        ]
                    },
                )
                for d in plant.map.drive_names
            }

        def time_span_ns(self):
            return int(stamps[0]), int(stamps[-1])

    stream = DriveCommands(
        tuple(plant.map.drive_names),
        np.array([0, 20_000_000, 40_000_000], dtype=np.int64),
        np.zeros((3, len(plant.map.drive_names))),
        "applied",
    )
    return sim, Capture(), stream


def test_resolved_stream_and_fractional_latency_are_preserved(inputs):
    sim, cap, stream = inputs
    latency = np.array([0, 1_500_000, 4_000_000, 7_500_000], dtype=np.int64)
    run = _prepare(cap, sim, None, None, source="command", latency_ns=latency, commands=stream)
    np.testing.assert_array_equal(run.command_t, stream.t_ns)
    np.testing.assert_array_equal(run.demand, stream.position_rad)
    np.testing.assert_array_equal(run.latency_ns.numpy(), latency)
    again = _prepare(cap, sim, None, None, source="command", latency_ns=latency, commands=stream)
    np.testing.assert_array_equal(run.t_grid, again.t_grid)


def test_command_reference_and_order_must_be_explicit(inputs):
    sim, cap, stream = inputs
    with pytest.raises(ValueError, match="resolved"):
        _prepare(cap, sim, None, None, source="command")
    wrong = DriveCommands(tuple(reversed(stream.drives)), stream.t_ns, stream.position_rad, "frame")
    with pytest.raises(ValueError, match="drive order"):
        _prepare(cap, sim, None, None, source="command", commands=wrong)
    with pytest.raises(ValueError, match="delays >= 0"):
        _prepare(
            cap, sim, None, None, source="command", commands=stream, latency_ns=-np.ones(4, dtype=np.int64)
        )
    # A negative PDO stamp-alignment shift is meaningful and remains supported.
    _prepare(cap, sim, None, None, delays_ns=np.array([-4_000_000, 0, 1_000_000, 8_000_000]))


@pytest.mark.parametrize(
    "stamps,positions,reference",
    [
        (np.array([0.0, 1.0]), np.zeros((2, 1)), "frame"),
        (np.array([1, 0], dtype=np.int64), np.zeros((2, 1)), "frame"),
        (np.array([0, 1], dtype=np.int64), np.full((2, 1), np.nan), "frame"),
        (np.array([0, 1], dtype=np.int64), np.zeros((2, 1)), "unknown"),
    ],
)
def test_invalid_stream_is_rejected(stamps, positions, reference):
    with pytest.raises(ValueError):
        DriveCommands(("drive",), stamps, positions, reference)
