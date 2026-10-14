import hashlib

import pytest
import yaml

from humanoid_sim.config import TimingConfig
from humanoid_sim.robot import load_v1


def test_description_inventory_preserves_coupled_identity(tmp_path):
    data = {
        "schema": "ethrc-actuators/v1",
        "families": {"HEJ50-48-30": {"controller_rate_hz": 2500}},
        "baseline_position_gains": {"motor_a": {"stiffness": 480, "damping": 8}},
        "joints": {"pitch": {"device": None}, "roll": {"device": None}},
        "drives": {
            "motor_a": {"family": "HEJ50-48-30", "joints": ["pitch", "roll"], "crank": "upper", "effort": 30}
        },
    }
    (tmp_path / "actuators.yaml").write_text(yaml.safe_dump(data))
    (tmp_path / "robot.urdf").write_text('<robot name="fixture"/>')
    profile = {
        "schema": "humanoid-sim-robot/v1",
        "robot": "test",
        "source": {"revision": "fixture"},
        "files": {
            key: {"path": name, "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for key, name in [("actuators", "actuators.yaml"), ("urdf", "robot.urdf")]
        },
        "expected_drive_count": 1,
        "modeled_firmware": {"watchdog_ns": 100000000, "command_delay_ns": 0},
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    robot = load_v1(tmp_path, profile_path)
    assert robot.firmware_period_ns == 400000
    (drive,) = robot.drives
    assert drive.joints == ("pitch", "roll")
    assert drive.sign is None and drive.crank == "upper"
    assert drive.config.torque_limit == 30
    (tmp_path / "actuators.yaml").write_text("edited")
    with pytest.raises(ValueError, match="differs from pinned"):
        load_v1(tmp_path, profile_path)


def test_timing_refuses_unresolved_firmware_events():
    with pytest.raises(ValueError, match="resolve"):
        TimingConfig(physics_period_ns=1000000, firmware_period_ns=400000)
    assert TimingConfig().bus_period_ns == 2000000
