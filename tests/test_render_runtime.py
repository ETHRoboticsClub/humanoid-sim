"""The render runtime's clock and scene, without Kit: plant time now from the host's feedback, the host's scene
from its descriptor."""

from pathlib import Path

import numpy as np

from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.runtime.render import host_scene, plant_now_ns


def test_plant_time_now_follows_the_hosts_mapping():
    fb = np.zeros((), dtype=abi.FEEDBACK)
    fb["sim_time_ns"], fb["written_mono_ns"] = 4_000_000_000, 10_000_000_000
    assert plant_now_ns(fb, 10_000_000_000) == 4_000_000_000
    assert plant_now_ns(fb, 10_003_000_000) == 4_003_000_000  # 3 ms after the write


def test_the_hosts_scene_is_taken_only_when_readable_here(tmp_path: Path):
    assert host_scene(None) == (None, (0.0, 0.0), None)
    assert host_scene({"file": str(tmp_path / "missing.npz"), "spawn_m": [1.0, 2.0]})[0] is None
    boxes, mesh = tmp_path / "boxes.npz", tmp_path / "scan.obj"
    boxes.write_bytes(b"x")
    record = {"file": str(boxes), "spawn_m": [-4.42, 3.48], "scan": {"mesh": {"file": str(mesh)}}}
    assert host_scene(record) == (str(boxes), (-4.42, 3.48), None)  # the scan is not here
    mesh.write_text("o scan\n")
    assert host_scene(record) == (str(boxes), (-4.42, 3.48), str(mesh))
