"""The render runtime's clock and scene, without Kit: plant time now from the host's feedback, the host's scene
from its descriptor."""

import signal
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.runtime.render import RuntimeRender, host_scene, plant_now_ns, render_runtime


def test_plant_time_now_follows_the_hosts_mapping():
    fb = np.zeros((), dtype=abi.FEEDBACK)
    fb["sim_time_ns"], fb["written_mono_ns"] = 4_000_000_000, 10_000_000_000
    assert plant_now_ns(fb, 10_000_000_000) == 4_000_000_000
    assert plant_now_ns(fb, 10_003_000_000) == 4_003_000_000  # 3 ms after the write


def test_the_hosts_scene_is_taken_only_when_readable_here(tmp_path: Path):
    assert host_scene(None) == (None, (0.0, 0.0), None)
    # The boxes are not here; the spawn still places what the caller brings (lights, a scan).
    assert host_scene({"file": str(tmp_path / "missing.npz"), "spawn_m": [1.0, 2.0]}) == (None, (1.0, 2.0), None)
    boxes, mesh = tmp_path / "boxes.npz", tmp_path / "scan.obj"
    boxes.write_bytes(b"x")
    record = {"file": str(boxes), "spawn_m": [-4.42, 3.48], "scan": {"mesh": {"file": str(mesh)}}}
    assert host_scene(record) == (str(boxes), (-4.42, 3.48), None)  # the scan is not here
    mesh.write_text("o scan\n")
    assert host_scene(record) == (str(boxes), (-4.42, 3.48), str(mesh))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from humanoid_sim.model.description import Description
    from humanoid_sim.sensors import session

    clock = [10_000_000_000]
    monkeypatch.setattr(abi, "mono_ns", lambda: clock[0])
    identity = {"robot_id": "fixture", "description_digest": "d" * 64}
    descriptor = {"drives": ["drive"], "joints": ["joint"]}
    path = tmp_path / "plant"

    def create():
        return abi.Segment.create(path, **identity, sim_revision="test", num_drives=1, num_joints=1,
                                  physics_period_ns=4_000_000, bus_period_ns=2_000_000, descriptor=descriptor)

    host = create()
    monkeypatch.setattr(Description, "load", lambda _: SimpleNamespace(robot="fixture", digest="d" * 64,
                                                                       joints={"joint": None}))

    class Sensors:
        def __init__(self, cfg):
            self.render_ns = 33_333_333
            self.sim = SimpleNamespace(plant=SimpleNamespace(cfg=SimpleNamespace(device="cpu")))
            self.renders = 0
            self.viewport_times = []
            self.poses = []
            self.sensor_intervals = []

        def pose(self, t_ns, q, base):
            self.poses.append((t_ns, q.clone(), base.clone()))

        def render(self, t_ns):
            self.viewport_times.append(t_ns)

        def render_at(self, t_ns, dt_ns):
            self.renders += 1
            self.sensor_intervals.append(dt_ns)
            self.render(t_ns)

        def close(self):
            pass

    monkeypatch.setattr(session, "SensorSession", Sensors)
    cfg = session.SessionCfg(robot="fixture")
    truth, fb = np.zeros((), abi.TRUTH), np.zeros((), abi.FEEDBACK)
    truth["sim_time_ns"] = fb["sim_time_ns"] = 1_000_000_000
    fb["written_mono_ns"] = clock[0]
    host.regions["truth"].write(truth)
    host.regions["feedback"].write(fb)
    renderer = RuntimeRender(path, cfg, window=True)
    try:
        yield host, renderer, clock, create, cfg
    finally:
        renderer.close()
        host.close()


def test_a_paused_host_stays_alive_but_a_stopped_heartbeat_ends_rendering(runtime):
    host, renderer, clock, _, _ = runtime
    assert renderer.frame()
    assert renderer.session.renders == 1
    clock[0] += 600_000_000
    host.heartbeat()  # paused plant time, live host
    assert renderer.frame()
    assert renderer.session.renders == 1
    clock[0] += 500_000_001
    assert not renderer.frame()


def test_a_future_dated_heartbeat_ends_rendering(runtime):
    host, renderer, clock, _, _ = runtime
    host.header["heartbeat_mono_ns"] = clock[0] + 1
    assert not renderer.frame()


def test_paused_reset_updates_the_viewport_without_publishing_sensor_frames(runtime):
    host, renderer, clock, _, _ = runtime
    assert renderer.frame()
    _, truth = host.regions["truth"].read()
    _, fb = host.regions["feedback"].read()
    truth["joint_pos_rad"][0] = 0.37
    truth["base_position_m"] = [1.0, -2.0, 1.4]
    truth["base_quat_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    fb["paused"] = 1
    host.regions["truth"].write(truth)
    host.regions["feedback"].write(fb)
    clock[0] += 100_000_000
    host.heartbeat()

    assert renderer.frame()
    assert renderer.session.viewport_times == [1_000_000_000, 1_000_000_000]
    assert renderer.session.renders == 1
    _, q, base = renderer.session.poses[-1]
    np.testing.assert_allclose(q.numpy(), [[0.37]])
    np.testing.assert_allclose(base.numpy(), [[1.0, -2.0, 1.4, 0.0, 0.0, 0.0, 1.0]])

    # Frozen plant time still allows camera/window interaction at a bounded wall-clock rate.
    for _ in range(5):
        clock[0] += 1_000_000
        assert renderer.frame()
    assert len(renderer.session.viewport_times) == 2
    clock[0] += 100_000_000
    host.heartbeat()
    assert renderer.frame()
    assert len(renderer.session.viewport_times) == 3
    assert renderer.session.renders == 1

    truth["sim_time_ns"] = fb["sim_time_ns"] = 1_100_000_000
    fb["paused"] = 0
    host.regions["truth"].write(truth)
    host.regions["feedback"].write(fb)
    assert renderer.frame()
    assert renderer.session.renders == 2
    assert renderer.session.sensor_intervals[-1] == 100_000_000


def test_headless_pause_does_not_render_or_publish(runtime):
    host, _, _, _, cfg = runtime
    _, fb = host.regions["feedback"].read()
    fb["paused"] = 1
    host.regions["feedback"].write(fb)
    renderer = RuntimeRender(host.path, cfg)
    try:
        assert renderer.frame()
        assert renderer.session.viewport_times == []
        assert renderer.session.renders == 0
        assert renderer.session.poses == []
    finally:
        renderer.close()


def test_reopen_refuses_the_dead_file_and_accepts_a_replacement_session(runtime):
    _, renderer, clock, create, _ = runtime
    clock[0] += 500_000_001
    with pytest.raises(abi.AbiError, match="stale heartbeat"):
        renderer.reopen(wait_s=0)
    old_session = renderer.session_id
    replacement = create()
    try:
        renderer.reopen(wait_s=0)
        assert renderer.session_id == replacement.session_id != old_session
        assert renderer._last_t is None
    finally:
        replacement.close()


def test_kit_is_closed_when_startup_refuses_a_dead_host(runtime, monkeypatch):
    from humanoid_sim.sensors import rtx

    host, _, clock, _, cfg = runtime
    exits = []

    @contextmanager
    def kit_runtime():
        # Isaac Lab chooses the process exit status from the exception passed into
        # its context. ExitStack.close() would hide it and exit successfully.
        try:
            yield
        except BaseException as exc:
            exits.append(type(exc))
            raise
        else:
            exits.append(None)

    def start_kit(*_args, **_kwargs):
        stack = ExitStack()
        stack.enter_context(kit_runtime())
        return stack

    monkeypatch.setattr(rtx, "start_kit", start_kit)
    clock[0] += 500_000_001
    with pytest.raises(abi.AbiError, match="stale heartbeat"):
        render_runtime(host.path, cfg, wait_s=0)
    assert exits == [abi.AbiError]


def test_ctrl_c_stops_even_when_a_kit_callback_swallows_the_interrupt(monkeypatch):
    from humanoid_sim.runtime import render as module
    from humanoid_sim.sensors import rtx

    frames, closed = [], []

    class Renderer:
        def __init__(self, *_args, **_kwargs):
            pass

        def frame(self):
            frames.append(True)
            try:
                signal.raise_signal(signal.SIGINT)
            except KeyboardInterrupt:
                pass  # Kit's async callback can consume the interrupt.
            return True

        def summary(self, _elapsed):
            return {"frames": len(frames)}

        def close(self):
            closed.append(True)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(rtx, "start_kit", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(module, "RuntimeRender", Renderer)
    assert render_runtime("unused", SimpleNamespace(device="cpu"), seconds=0.02) == {"frames": 1}
    assert closed == [True]
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
