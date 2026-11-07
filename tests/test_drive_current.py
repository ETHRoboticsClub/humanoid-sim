"""The current limits wired from the profile and the description into the plant.

The winding time constant is not in the description yet: tests give a loaded description a fixture value in memory,
so they hold whether or not humanoid-urdf records one.
"""

import math

import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.devices.winding import CURRENT_KEYS, current_limits, profile_current
from humanoid_sim.model.description import Description

D64 = torch.float64
TAU_S = 0.5  # fixture winding time constant


def _desc(tau_s=TAU_S):
    """A fresh V1 description whose every family has the fixture winding time constant (None: removed)."""
    desc = Description.load("ethrc_v1")
    for family in desc.table["families"].values():
        family.pop("winding_time_constant_s", None)
        if tau_s is not None:
            family["winding_time_constant_s"] = tau_s
    return desc


def test_the_profile_block_defaults_off_and_names_the_plant_fields():
    assert profile_current({}) == {"output_current_limit": False, "winding_i2t": False}
    assert profile_current({"devices": {"current": {"winding_i2t": True}}}) == {"output_current_limit": False,
                                                                                "winding_i2t": True}
    assert set(CURRENT_KEYS) <= set(PlantCfg.__dataclass_fields__)


@pytest.mark.parametrize("block", [{"i2t": True}, {"winding_i2t": "yes"}, {"output_current_limit": 1}])
def test_the_profile_block_refuses_unknown_keys_and_non_booleans(block):
    with pytest.raises(ValueError):
        profile_current({"devices": {"current": block}})


def test_the_description_gives_the_winding_time_constant_in_ns():
    desc = _desc()
    assert {desc.winding_time_constant_ns(n) for n in desc.drives} == {round(TAU_S * 1e9)}
    assert all(_desc(None).winding_time_constant_ns(n) is None for n in desc.drives)
    bad = _desc(-1.0)
    with pytest.raises(ValueError, match="winding_time_constant_s"):
        bad.winding_time_constant_ns(next(iter(bad.drives)))


def test_current_limits_from_the_description():
    desc = _desc()
    names = list(desc.drives)
    limit, winding = current_limits(desc, names, output_current_limit=True, winding_i2t=True, dtype=D64)
    assert limit.tolist() == pytest.approx([desc.current_limit_torque_nm(n) for n in names])
    assert winding.nominal_nm.tolist() == pytest.approx([desc.drives[n].effort_nominal for n in names])
    assert winding.time_constant_ns.tolist() == [round(TAU_S * 1e9)] * len(names)
    assert current_limits(desc, names, output_current_limit=False, winding_i2t=False) == (None, None)


def test_a_limit_asked_for_without_its_figures_raises():
    desc = _desc(None)
    with pytest.raises(ValueError, match="winding_time_constant_s"):
        current_limits(desc, list(desc.drives), output_current_limit=False, winding_i2t=True)
    desc = _desc()
    for family in desc.table["families"].values():
        family.pop("max_current_a", None)
    with pytest.raises(ValueError, match="max_current_a"):
        current_limits(desc, list(desc.drives), output_current_limit=True, winding_i2t=False)


def test_the_default_plant_has_no_current_limit():
    desc = _desc()
    plant = PlantCore(desc, list(desc.joints), 1, PlantCfg(dtype=D64))
    assert plant.bank.current_limit_nm is None and plant.bank.winding is None


def test_the_plant_limits_a_drive_held_above_nominal_after_the_i2t_time():
    desc = _desc()
    joints = list(desc.joints)
    plant = PlantCore(desc, joints, 1, PlantCfg(output_current_limit=True, winding_i2t=True, dtype=D64))
    names = plant.map.drive_names
    i = next(k for k, n in enumerate(names) if desc.drives[n].crank is None)  # a 1:1 drive
    drive = desc.drives[names[i]]
    nominal, peak = drive.effort_nominal, plant.envelope.peak[i].item()
    held = (nominal + peak) / 2
    ff = torch.zeros(1, len(names), dtype=D64)
    ff[0, i] = held
    standing = torch.tensor([desc.joints[n].standing_offset_rad for n in joints], dtype=D64).unsqueeze(0)
    qdrv, _, _ = plant.map.drive_state(standing, torch.zeros_like(standing))
    plant.submit(DriveDemand.pd(qdrv.clone(), 0.0, 0.0, feedforward=ff), torch.zeros(1, dtype=torch.long))
    period = plant.cfg.physics_period_ns
    first = math.ceil(round(TAU_S * 1e9) * math.log(held**2 / (held**2 - nominal**2)) / period)
    applied = []
    for k in range(first + 1):
        plant.pre_physics(k * period, standing, torch.zeros_like(standing))
        applied.append(plant.last.applied[0, i].item())
    assert applied[first - 1] == pytest.approx(held) and applied[first] == pytest.approx(nominal)
