"""The render runtime's clock and scene, without Kit: plant time now from the host's feedback, the host's scene
from its descriptor."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.runtime.render import RuntimeRender, host_scene, plant_now_ns, render_runtime
from humanoid_sim.sensors.session import SensorSession


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

        def pose(self, *_args):
            pass

        def render_at(self, *_args):
            self.renders += 1

        def close(self):
            pass

    monkeypatch.setattr(session, "SensorSession", Sensors)
    cfg = session.SessionCfg(robot="fixture")
    truth, fb = np.zeros((), abi.TRUTH), np.zeros((), abi.FEEDBACK)
    truth["sim_time_ns"] = fb["sim_time_ns"] = 1_000_000_000
    fb["written_mono_ns"] = clock[0]
    host.regions["truth"].write(truth)
    host.regions["feedback"].write(fb)
    renderer = RuntimeRender(path, cfg)
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


def test_render_latency_does_not_move_camera_or_lidar_capture_timestamps(runtime, monkeypatch):
    from humanoid_sim.sensors import feeds, unitree_lidar
    from humanoid_sim.sensors.lidar import Sweep

    _, renderer, clock, _, _ = runtime
    monkeypatch.setattr(feeds.time, "monotonic_ns", lambda: clock[0])
    monkeypatch.setattr(feeds.time, "time", lambda: 90.0 + clock[0] / 1e9)
    camera_stamps, clouds = [], []

    class DelayedSensors(SensorSession):
        def __init__(self):
            self.render_ns = 33_333_333
            self.sim = SimpleNamespace(t_ns=1_000_000_000,
                                       plant=SimpleNamespace(cfg=SimpleNamespace(device="cpu")))
            self.cfg = SimpleNamespace(images=None)
            self.recorder = None
            self.camera_specs = [SimpleNamespace(camera_id="stereo", streams=[
                SimpleNamespace(name=name, kind="mono") for name in ("left", "right")])]
            self.cameras = SimpleNamespace(update=lambda _dt: None,
                                           image=lambda *_args: np.zeros((2, 2), dtype=np.uint8))
            self.feeds = {"stereo": SimpleNamespace(publish=lambda _name, stamp, _image:
                                                    camera_stamps.append(stamp))}
            sweep = Sweep(start_ns=900_000_000, points_m=np.ones((1, 3)), hit=np.array([True]),
                          time_s=np.array([0.0]), origin_w=np.zeros(3), rotation_w=np.eye(3))
            self.lidars = [SimpleNamespace(sensor=SimpleNamespace(name="lidar", model=unitree_lidar.UNITREE_MODEL),
                                           render_time=lambda now: now, read=lambda *_args: [sweep])]
            self.publisher = SimpleNamespace(publish=clouds.append)
            self.counts = {key: 0 for key in ("renders", "camera_frames", "lidar_sweeps", "lidar_points")}
            self.ms = {key: [] for key in ("render", "camera_read", "lidar_read", "publish")}
            self.last_images, self._last_sweep, self._saved = {}, {}, {}
            self.sweeps = None

        def pose(self, *_args):
            pass

        def render(self, _timeline_ns):
            clock[0] += 2_000_000_000

        def close(self):
            pass

    renderer.session = DelayedSensors()
    assert renderer.frame()
    assert camera_stamps == [100.0, 100.0]
    assert clouds[0].stamp_ns == 99_900_000_000
