"""The read-only viewer's publication period includes its work and skips missed polls."""

from types import SimpleNamespace

import pytest

from humanoid_sim.runtime import view


@pytest.fixture
def viewer(monkeypatch):
    clock = [0]
    starts, sleeps, closed = [], [], []
    work_ns = [4_000_000]

    def sleep(seconds):
        elapsed = round(seconds * 1e9)
        sleeps.append(elapsed)
        clock[0] += elapsed

    class RuntimeView:
        def __init__(self, *_args, **_kwargs):
            self.drives, self.draw, self.frames = [], "boxes", 0
            self.seg = SimpleNamespace(buf=True)

        def log_static(self):
            pass

        def frame(self):
            starts.append(clock[0])
            clock[0] += work_ns[min(self.frames, len(work_ns) - 1)]
            self.frames += 1
            return True

        def close(self):
            closed.append("view")

    rr = SimpleNamespace(connect_grpc=lambda *_args, **_kwargs: None,
                         disconnect=lambda: closed.append("stream"))
    monkeypatch.setattr(view, "RuntimeView", RuntimeView)
    monkeypatch.setattr(view, "SignalLog", lambda *_args, **_kwargs: SimpleNamespace(rr=rr))
    monkeypatch.setattr(view, "blueprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(view.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(view.time, "sleep", sleep)
    return clock, starts, sleeps, closed, work_ns


def test_processing_time_is_part_of_the_period_and_the_duration_bounds_the_last_wait(viewer):
    clock, starts, sleeps, closed, _work = viewer
    frames = view.view_runtime("unused", "unused", connect="unused", fps=100, seconds=0.035)
    assert frames == 4
    assert starts == [0, 10_000_000, 20_000_000, 30_000_000]
    assert sleeps == [6_000_000, 6_000_000, 6_000_000, 1_000_000]
    assert clock[0] == 35_000_000
    assert closed == ["view", "stream"]


def test_a_slow_frame_skips_missed_polls_without_a_catch_up_burst(viewer):
    _clock, starts, _sleeps, closed, work = viewer
    work[:] = [4_000_000, 25_000_000, 4_000_000]
    frames = view.view_runtime("unused", "unused", connect="unused", fps=100, seconds=0.055)
    assert frames == 4
    assert starts == [0, 10_000_000, 40_000_000, 50_000_000]
    assert closed == ["view", "stream"]


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), 2e9])
def test_invalid_rate_does_not_initialize_a_recording_or_open_a_segment(monkeypatch, fps):
    monkeypatch.setattr(view, "SignalLog", lambda *_args, **_kwargs: pytest.fail("opened recording"))
    with pytest.raises(ValueError, match="fps"):
        view.view_runtime("unused", "unused", fps=fps)
