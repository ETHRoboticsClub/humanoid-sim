import torch

from humanoid_sim.devices.envelope import TorqueSpeedEnvelope

HEJ90 = [(0.0, 140.0), (5.0, 140.0), (6.0, 135.0), (8.0, 100.0), (10.0, 25.0), (10.4, 0.0)]
HEJ50 = [(0.0, 30.0), (10.0, 25.0), (15.0, 20.0), (20.0, 5.0), (21.0, 0.0)]


def test_interpolation_and_quadrants():
    env = TorqueSpeedEnvelope([HEJ90, HEJ50], dtype=torch.float64)
    v = torch.tensor([[7.0, 12.5]])
    assert torch.allclose(env.motoring_limit(v), torch.tensor([[117.5, 22.5]], dtype=torch.float64))
    lo, hi = env.limits(v)
    assert torch.allclose(hi, torch.tensor([[117.5, 22.5]], dtype=torch.float64))
    assert torch.allclose(lo, torch.tensor([[-140.0, -30.0]], dtype=torch.float64))  # braking: peak
    lo, hi = env.limits(-v)
    assert torch.allclose(lo, torch.tensor([[-117.5, -22.5]], dtype=torch.float64))
    assert torch.allclose(hi, torch.tensor([[140.0, 30.0]], dtype=torch.float64))
    assert torch.all(env.motoring_limit(torch.tensor([[11.0, 25.0]], dtype=torch.float64)) == 0)
    assert torch.allclose(env.clamp(torch.tensor([[500.0, -500.0]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64)),
                          torch.tensor([[140.0, -30.0]], dtype=torch.float64))
