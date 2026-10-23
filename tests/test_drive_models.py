"""Drive models: every description drive has its product's model, with the timing the description records."""

import pytest

from humanoid_sim.devices.models import MODELS, drive_models
from humanoid_sim.model.description import Description, robots


@pytest.mark.parametrize("robot", robots())
def test_every_drive_has_its_family_model(robot):
    desc = Description.load(robot)
    models = drive_models(desc)
    assert list(models) == list(desc.drives)
    for name, model in models.items():
        assert model.family == desc.drives[name].family and MODELS[model.family] is type(model)


@pytest.mark.parametrize("robot", robots())
def test_drives_take_their_delay_from_the_description(robot):
    desc = Description.load(robot)
    models = drive_models(desc)
    for name, model in models.items():
        recorded = desc.family_value(model.family, "drive_delay_s")
        assert model.delay_measured == (recorded is not None)
        assert model.drive_delay_ns == (0 if recorded is None else round(float(recorded) * 1e9))
        assert isinstance(model.drive_delay_ns, int)
        # One instance per family.
        assert all(models[other] is model for other, d in desc.drives.items() if d.family == model.family)


def test_a_family_without_a_recorded_delay_has_none_modeled():
    desc = Description.load("ethrc_v1")
    for family in desc.table["families"].values():
        family.pop("drive_delay_s", None)
    assert all(m.drive_delay_ns == 0 and not m.delay_measured for m in drive_models(desc).values())


def test_a_bad_delay_or_an_unknown_family_is_refused():
    desc = Description.load("ethrc_v1")
    name, drive = next(iter(desc.drives.items()))
    desc.table["families"][drive.family]["drive_delay_s"] = -0.001
    with pytest.raises(ValueError, match="non-negative"):
        drive_models(desc)
    desc = Description.load("ethrc_v1")
    desc.drives[name] = drive.__class__(**{**drive.__dict__, "family": "HEJ120"})
    with pytest.raises(ValueError, match="no drive model"):
        drive_models(desc)
