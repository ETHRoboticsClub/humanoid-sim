"""Drive models: every description drive has its product's model, with the timing the description records."""

import pytest

from humanoid_sim.devices.models import MODELS, Hej50, Hej90, UnitreeMotor, drive_models
from humanoid_sim.model.description import Description, robots


@pytest.mark.parametrize("robot", robots())
def test_every_drive_has_its_family_model(robot):
    desc = Description.load(robot)
    models = drive_models(desc)
    assert list(models) == list(desc.drives)
    for name, model in models.items():
        assert model.family == desc.drives[name].family and MODELS[model.family] is type(model)


def test_the_v1_drives_take_their_delay_from_the_description():
    desc = Description.load("ethrc_v1")
    models = drive_models(desc)
    assert isinstance(models["maxonHEJ90L4"], Hej90) and isinstance(models["maxonHEJ50L_ANK1"], Hej50)
    for model in models.values():
        assert model.delay_measured
        assert model.drive_delay_ns == round(float(desc.family(model.family)["drive_delay_s"]) * 1e9)
        assert isinstance(model.drive_delay_ns, int) and model.delay_source
    # One instance per family.
    assert models["maxonHEJ90L4"] is models["maxonHEJ90R4"]


def test_a_family_without_a_recorded_delay_has_none_modeled():
    models = drive_models(Description.load("g1_29dof_rev_1_0"))
    assert all(isinstance(m, UnitreeMotor) and m.drive_delay_ns == 0 and not m.delay_measured
               for m in models.values())


def test_a_bad_delay_or_an_unknown_family_is_refused():
    desc = Description.load("ethrc_v1")
    desc.table["families"]["HEJ90-48-140"]["drive_delay_s"] = -0.001
    with pytest.raises(ValueError, match="non-negative"):
        drive_models(desc)
    desc = Description.load("ethrc_v1")
    desc.drives["maxonHEJ90L4"] = desc.drives["maxonHEJ90L4"].__class__(
        **{**desc.drives["maxonHEJ90L4"].__dict__, "family": "HEJ120"})
    with pytest.raises(ValueError, match="no drive model"):
        drive_models(desc)
