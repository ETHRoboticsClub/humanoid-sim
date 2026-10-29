"""Compiled native reset coordinates and velocities keep fixed graph dimensions."""

import pytest
import torch


def test_compiled_native_reset_handles_varying_subsets_without_recompiling():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from humanoid_sim.model.description import Description
    from humanoid_sim.model.native_state import NativeState

    desc = Description.load("ethrc_v1")
    names = list(desc.joints)
    generator = torch.Generator(device="cuda:0").manual_seed(198)
    q = torch.randn(65, len(names), generator=generator, device="cuda:0") * .1
    qd = torch.randn(q.shape, generator=generator, device="cuda:0") * .1
    for linkage in desc.linkages:
        cols = [names.index(n) for n in linkage.joints]
        neutral = torch.tensor(desc.ankle_model.neutral(linkage.side), device="cuda:0", dtype=q.dtype)
        q[:, cols] = neutral + torch.randn(65, 2, generator=generator, device="cuda:0") * .001
    eager = NativeState(desc, device="cuda:0")
    compiled = NativeState(desc, device="cuda:0", compile_batch_size=65)
    compiled.dofs(q, qd)
    with torch.compiler.set_stance("fail_on_recompile"):
        for count in (17, 1, 0, 2, 33, 65):
            actual, expected = compiled.dofs(q[:count], qd[:count]), eager.dofs(q[:count], qd[:count])
            for got, want in zip(actual, expected, strict=True):
                assert got.keys() == want.keys()
                for name in want:
                    torch.testing.assert_close(got[name], want[name], atol=1e-9, rtol=1e-9)
    with pytest.raises(ValueError, match="exceeds"):
        compiled.dofs(q.repeat(2, 1), qd.repeat(2, 1))
