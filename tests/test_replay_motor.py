"""The replay's motor encoder on both sides and the replay command's physics overrides, without running a solver."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.cli import physics_overrides
from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.description import Description
from humanoid_sim.replay.exc import DriveTrace
from humanoid_sim.replay.replay import _prepare, motor_side

FIELDS = ["native_position_rad", "velocity_rad_s", "torque_nm", "position_demand_rad"]


@pytest.fixture
def plant():
    desc = Description.load()
    return PlantCore(desc, list(desc.joints), 2, PlantCfg(dtype=torch.float64))


def _capture(plant, fields):
    stamps = np.arange(20, dtype=np.int64) * 5_000_000

    class Capture:
        def pdo(self):
            return {d: DriveTrace(d, stamps, {k: np.full(len(stamps), 0.25 * i) for k in fields})
                    for i, d in enumerate(plant.map.drive_names)}

        def time_span_ns(self):
            return int(stamps[0]), int(stamps[-1])

    return Capture()


def test_the_capture_motor_encoder_is_kept_when_the_capture_has_one(plant):
    sim = SimpleNamespace(plant=plant, period_ns=4_000_000)
    run = _prepare(_capture(plant, FIELDS + ["motor_position_rad"]), sim, None, None)
    expected = 0.25 * np.arange(plant.map.num_drives)
    np.testing.assert_allclose(run.real["motor_position"], np.broadcast_to(expected, run.real["position"].shape))
    assert "motor_position" not in _prepare(_capture(plant, FIELDS), sim, None, None).real


def test_the_sim_motor_side_is_the_reading_less_each_crank_play(plant):
    reading = torch.linspace(-1.0, 1.0, plant.map.num_drives, dtype=torch.float64)
    assert torch.equal(motor_side(plant, reading), reading)  # no backlash: one encoder
    plant.use_backlash()
    play = torch.tensor([[0.001, -0.002, 0.003, -0.004], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    plant.write_backlash(play[:, :len(plant.crank_drives)] * plant._crank_sign)  # joint frame -> drive frame
    motor = motor_side(plant, reading)
    cranks = plant.crank_drives.tolist()
    others = [i for i in range(plant.map.num_drives) if i not in cranks]
    torch.testing.assert_close(motor[cranks], reading[cranks] - play[0, :len(cranks)])
    assert torch.equal(motor[others], reading[others])
    assert torch.equal(motor_side(plant, reading, env=1), reading)


def test_physics_overrides_come_inline_or_from_a_file(tmp_path):
    assert physics_overrides(None) is None
    assert physics_overrides("{ankle_backlash: {width_rad: 0.008}}") == {"ankle_backlash": {"width_rad": 0.008}}
    f = tmp_path / "o.yaml"
    f.write_text("ankle_backlash:\n  width_rad: {A: 0.007}\n")
    assert physics_overrides(str(f)) == {"ankle_backlash": {"width_rad": {"A": 0.007}}}
    with pytest.raises(SystemExit, match="mapping"):
        physics_overrides("0.008")
