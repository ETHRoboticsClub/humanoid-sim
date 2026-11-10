"""A paused reset must publish the new pose before any controller can re-arm."""

import numpy as np
import pytest
from conftest import PARTS, PHYSICS_ROBOTS, built_usd

from humanoid_sim.runtime.abi import plant_abi as abi

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
def test_paused_reset_replaces_cached_drive_feedback(tmp_path, robot, device):
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.runtime.host import RuntimeHost

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    host = RuntimeHost(tmp_path / "plant", built_usd(robot), device=device, robot=robot, compile=False)
    try:
        n = len(host.drives)
        _, initial = host.seg.regions["feedback"].read()
        expected = initial["drive"]["position_rad"][:n].copy()
        q, _ = host.sim.joint_state()
        joint = PARTS[robot]["elbow"]
        home = float(q[0, host.joints.index(joint)])
        limits = host.desc.calibration_travel(joint)
        far = max(limits, key=lambda bound: abs(bound-home))
        displaced = home + 0.25 * (far-home)

        def command(code, args):
            seq = host.seg.send(code, args)
            host.step()
            assert host.seg.wait_ack(seq)[0] == abi.ACK_OK

        command(abi.CMD_RESET, {"joint_pos": {joint: displaced}})
        _, before = host.seg.regions["feedback"].read()
        assert not np.allclose(before["drive"]["position_rad"][:n], expected, atol=1e-4)
        command(abi.CMD_PAUSE, {})
        paused_time_ns = host.sim.t_ns
        command(abi.CMD_RESET, {"pose": host.start_pose})
        _, after = host.seg.regions["feedback"].read()
        assert after["paused"] == 1
        assert host.sim.t_ns == paused_time_ns
        np.testing.assert_allclose(after["drive"]["position_rad"][:n], expected, atol=1e-6)
        np.testing.assert_allclose(after["drive"]["velocity_rad_s"][:n], 0, atol=1e-8)
    finally:
        host.close()
