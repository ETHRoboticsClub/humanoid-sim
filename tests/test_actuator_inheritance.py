"""Sidecar defaults preserve actuator overrides and omitted nominal ratings."""
import yaml

from humanoid_sim.model.description import Description


def test_family_torque_ratings_and_optional_standing_offset(tmp_path):
    package = tmp_path / "fixture" / "fixture_description"
    package.mkdir(parents=True)
    (package / "robot.urdf").write_text('<robot name="fixture"/>')
    entry = dict(family="motor", region="body", sign=1, velocity=4,
                 baseline_stiffness=10, baseline_damping=1)
    document = dict(
        schema="ethrc-actuators/v1", urdf="robot.urdf",
        families={"motor": dict(effort=23, effort_nominal=7, armature=.01, damping=.1, friction=.2)},
        joints={"inherited": entry, "override": dict(entry, effort=13, effort_nominal=5, standing_offset_rad=.3)},
        drives={"crank": dict(entry, crank="upper", joints=["inherited", "override"], position_limit=[-.4, .5])},
    )
    path = package.parent / "actuators.yaml"
    path.write_text(yaml.safe_dump(document))
    desc = Description.load("fixture", tmp_path)
    assert desc.joints["inherited"].effort_peak == desc.drives["inherited"].effort_peak == 23
    assert desc.drives["inherited"].effort_nominal == desc.drives["crank"].effort_nominal == 7
    assert desc.drives["crank"].effort_peak == 23
    assert desc.joints["inherited"].standing_offset_rad == 0
    assert desc.joints["override"].effort_peak == desc.drives["override"].effort_peak == 13
    assert desc.drives["override"].effort_nominal == 5
    assert desc.joints["override"].standing_offset_rad == .3
    assert "effort" not in desc.table["joints"]["inherited"]
    del document["families"]["motor"]["effort_nominal"]
    path.write_text(yaml.safe_dump(document))
    assert Description.load("fixture", tmp_path).drives["inherited"].effort_nominal is None
