"""The render runtime's clock and scene, without Kit: plant time now from the host's feedback, the host's scene
from its descriptor."""

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
    closed = []
    monkeypatch.setattr(rtx, "start_kit", lambda *_args, **_kwargs: SimpleNamespace(close=lambda: closed.append(True)))
    clock[0] += 500_000_001
    with pytest.raises(abi.AbiError, match="stale heartbeat"):
        render_runtime(host.path, cfg, wait_s=0)
    assert closed == [True]
