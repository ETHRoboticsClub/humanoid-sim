"""Reset coordinates must agree with whole-tree kinematics and its derivatives."""

import pytest
import torch

from humanoid_sim.model.description import Description
from humanoid_sim.model.native_state import NativeState


class WholeTreeState(NativeState):
    """Reference: close rods in the root frame, differentiating the entire tree."""

    def coordinates(self, q):
        values, poses = self.forward(q)
        balls = {}
        for name, parent, foot, f0, rod_vector, foot_point in self.rods:
            joint = poses[parent] @ f0
            target = poses[foot][:, :3, :3] @ foot_point + poses[foot][:, :3, 3]
            wanted = (joint[:, :3, :3].transpose(-1, -2)
                      @ (target - joint[:, :3, 3]).unsqueeze(-1)).squeeze(-1)
            balls[name] = self.closing(name, rod_vector, wanted)
        return values, balls

    def dofs(self, q, qd):
        q, qd = q.to(device=self.device, dtype=self.dtype), qd.to(device=self.device, dtype=self.dtype)
        values, balls = self.coordinates(q)
        rates = {name: qd[:, i] for i, name in enumerate(self.names)}
        for cols, names, transmission in self.ankles:
            velocity = torch.func.jvp(transmission.shafts, (q[:, cols],), (qd[:, cols],))[1]
            rates.update({name: velocity[:, i] for i, name in enumerate(names)})
        _, derivatives = torch.func.jvp(lambda pose: self.coordinates(pose)[1], (q,), (qd,))
        for name, coordinates in balls.items():
            for axis, (position, velocity) in enumerate(self.closing_dofs(name, coordinates, derivatives[name])):
                values[f"{name}:{axis}"] = position
                rates[f"{name}:{axis}"] = velocity
        return values, rates


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("rigid_frames", [False, True])
def test_reset_coordinates_and_rates_match_whole_tree(device, rigid_frames):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    desc = Description.load("ethrc_v1")
    generator = torch.Generator().manual_seed(517)
    q = torch.randn(32, len(desc.joints), generator=generator, dtype=torch.float64) * .3
    qd = torch.randn(q.shape, generator=generator, dtype=q.dtype)
    # Exercise independent upstream body motions as well as both ankle branches.
    for linkage in desc.linkages:
        neutral = torch.as_tensor(desc.ankle_model.neutral(linkage.side), dtype=q.dtype)
        candidates = neutral + (torch.rand(len(q), 2, generator=generator, dtype=q.dtype) - .5) * .2
        projected = torch.as_tensor(desc.ankle_model.project(candidates.numpy(), linkage.side))
        accepted = neutral + .8 * (projected - neutral)
        assert desc.ankle_model.feasible(accepted.numpy(), linkage.side).all()
        q[:, [list(desc.joints).index(name) for name in linkage.joints]] = accepted
    q, qd = q.to(device), qd.to(device)
    state, whole = NativeState(desc, device=device), WholeTreeState(desc, device=device)
    if rigid_frames:
        # USD's float quaternion frames can have ~1e-7 orthogonality error.
        # Also check the algebra at double precision using truly rigid frames.
        for model in (state, whole):
            for _, _, _, f0, inv_f1, _ in model.tree:
                for frame in (f0, inv_f1):
                    u, _, vh = torch.linalg.svd(frame[:3, :3])
                    frame[:3, :3] = u @ vh
    actual, expected = state.dofs(q, qd), whole.dofs(q, qd)
    for tolerance, found, reference in zip((5e-7, 3e-6), actual, expected, strict=True):
        assert found.keys() == reference.keys()
        for name in reference:
            assert torch.isfinite(found[name]).all()
            torch.testing.assert_close(found[name], reference[name],
                                       atol=1e-9 if rigid_frames else tolerance, rtol=1e-9)
