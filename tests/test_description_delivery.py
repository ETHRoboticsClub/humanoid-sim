import pytest
import torch

from humanoid_sim.model.description import Description


def test_family_ratings_are_resolved_for_every_drive():
    desc = Description.load()
    for name, drive in desc.drives.items():
        entry = desc.table.get("drives", {}).get(name)
        if entry is None:
            entry = desc.table["joints"][drive.joints[0]]
        ratings = desc.family(drive.family) | entry
        assert drive.effort_peak == ratings["effort"]
        assert drive.effort_nominal == ratings["effort_nominal"]


@pytest.mark.parametrize("side", ["l", "r"])
def test_transmission_matches_usd_geometry_over_reachable_workspace(side):
    desc = Description.load()
    model = desc.ankle_model
    transmission = desc.ankle_transmission(side, dtype=torch.float64)
    bounds = model.bounds(side)
    axes = [torch.linspace(*bound, 65, dtype=torch.float64) for bound in bounds]
    poses = torch.cartesian_prod(*axes)
    poses = poses[torch.from_numpy(model.feasible(poses.numpy(), side))]
    assert len(poses) > 0
    reference, reach = model.cranks(poses.numpy(), side)
    assert (reach < 1).all()
    torch.testing.assert_close(transmission.cranks(poses), torch.from_numpy(reference), atol=1e-12, rtol=0)
    shafts, _ = desc.ankle_model_module()._motor_angles_numpy(poses.numpy(), model.geometries[side])
    torch.testing.assert_close(transmission.shafts(poses), torch.from_numpy(shafts), atol=1e-12, rtol=0)
    directions = torch.eye(2, dtype=torch.float64)
    derivative = torch.stack([(transmission.cranks(poses + 1e-6 * direction)
                               - transmission.cranks(poses - 1e-6 * direction)) / 2e-6
                              for direction in directions], dim=-1)
    torch.testing.assert_close(transmission.jacobian(poses), derivative, atol=1e-7, rtol=0)
