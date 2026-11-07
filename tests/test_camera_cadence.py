"""Extra viewport renders must not accelerate camera publications or replay a backlog."""

from types import SimpleNamespace

import numpy as np

from humanoid_sim.sensors.session import SensorSession


def test_each_camera_stream_keeps_its_rate_across_fast_renders_stalls_and_resets():
    session = SensorSession.__new__(SensorSession)
    session.cfg = SimpleNamespace(depth_noise=False, images=None)
    session.clock = None
    session.sim = SimpleNamespace(t_ns=0)
    session.recorder = None
    session.camera_specs = [SimpleNamespace(camera_id="camera", streams=[
        SimpleNamespace(name="fast", kind="mono", rate_hz=30),
        SimpleNamespace(name="slow", kind="mono", rate_hz=10),
    ])]
    reads, published = [], []

    def image(_cam, stream):
        reads.append(stream.name)
        return np.zeros((2, 2), dtype=np.uint8)

    session.cameras = SimpleNamespace(image=image)
    session.feeds = {"camera": SimpleNamespace(publish=lambda name, stamp, frame: published.append(name))}
    session.counts = {"camera_frames": 0}
    session.ms = {"camera_read": [], "publish": []}
    session.last_images = {}
    session._saved = {}
    session.reset_capture_schedule()
    for t in range(0, 1_000_000_000, 4_000_000):
        session._cameras(t)
    assert published.count("fast") == 30
    assert published.count("slow") == 10
    assert reads == published  # no CPU/GPU image copy for a stream that is not due
    session._cameras(2_000_000_000)
    assert len(published) == 42  # one fresh frame per stream, no stale catch-up
    session._cameras(2_000_000_000)
    assert len(published) == 42
    session.reset_capture_schedule()
    session._cameras(0)
    assert len(published) == 44
