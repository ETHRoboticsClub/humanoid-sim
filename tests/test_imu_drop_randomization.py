from types import SimpleNamespace

import pytest
import torch

from humanoid_sim.randomization.spec import PARAMS, ParamSpec
from humanoid_sim.sensors.imu import ImuDelivery
from humanoid_sim.train.plant import RobotPlant
from test_imu_delivery import _every_sample


def _run(imu, n, e):
    imu.reset(torch.arange(e), 0, torch.zeros(e, 6, dtype=torch.float64))
    t = 0
    for _ in range(n):
        t += imu.cfg.sample_period_ns
        imu.step(t, torch.zeros(e, 3, dtype=torch.float64), torch.zeros(e, 3, dtype=torch.float64))
    return imu


def test_a_rate_written_after_construction_drops_in_those_envs_only():
    p, n, e = 0.25, 4000, 3
    cfg = _every_sample(drop_probability=0.0, history=n, pending=16)
    assert cfg.drop_probability == 0.0
    imu = ImuDelivery(cfg, e, dtype=torch.float64, seed=5)
    imu.drop_probability[1] = p
    _run(imu, n + 8, e)
    rate = 1.0 - imu.samples.count.double() / imu.seq.double()
    assert abs(rate[1].item() - p) < 0.03
    # the untouched envs lose nothing (a sample or two may still be pending at the end)
    assert (imu.seq - imu.samples.count)[[0, 2]].max() <= 2
    assert (imu.seq - imu.samples.count)[1] > 500


def test_envs_at_zero_match_a_model_without_drops_bit_for_bit():
    n, e = 400, 2
    cfg = _every_sample(drop_probability=0.0, history=n, pending=16, gyro_noise_density=1e-3)
    plain = _run(ImuDelivery(cfg, e, dtype=torch.float64, seed=7), n, e)
    mixed = ImuDelivery(cfg, e, dtype=torch.float64, seed=7)
    mixed.drop_probability[1] = 0.5
    _run(mixed, n, e)
    # env 0 sees the same noise stream and every sample; the drops of env 1 do not move it
    m = int(plain.samples.count[0])
    assert int(mixed.samples.count[0]) == m
    assert torch.equal(plain.samples.gyro[0], mixed.samples.gyro[0])
    assert torch.equal(plain.samples.seq[0], mixed.samples.seq[0])


def test_the_parameter_is_a_per_env_device_parameter_validated_to_its_range():
    assert PARAMS["imu_drop_probability"].kind == "device" and PARAMS["imu_drop_probability"].target == "env"
    ParamSpec("imu_drop_probability", "uniform", (0.0, 0.05), "set")
    ParamSpec("imu_drop_probability", "uniform", (0.0, 0.05), "set", scope="startup")


def _plant(imu, values):
    plant = RobotPlant.__new__(RobotPlant)
    bank = SimpleNamespace(p_scale=None, d_scale=None, torque_scale=None, position_offset=None)
    plant.core, plant.imu = SimpleNamespace(bank=bank), imu
    plant.randomizer = SimpleNamespace(values=values)
    return plant


def test_the_plant_writes_the_drawn_rate_for_the_envs_being_reset():
    imu = ImuDelivery(_every_sample(), 3, dtype=torch.float64)
    drawn = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64)
    plant = _plant(imu, {"imu_drop_probability": drawn})
    # the unused drive targets are skipped: only the IMU parameter is in the values
    plant._write_device(torch.tensor([0, 2]))
    assert torch.equal(imu.drop_probability, torch.tensor([0.1, 0.0, 0.3], dtype=torch.float64))


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_plant_rejects_a_rate_outside_zero_to_one(bad):
    imu = ImuDelivery(_every_sample(), 2, dtype=torch.float64)
    plant = _plant(imu, {"imu_drop_probability": torch.tensor([[0.1], [bad]], dtype=torch.float64)})
    with pytest.raises(ValueError, match="imu_drop_probability"):
        plant._write_device(torch.arange(2))
