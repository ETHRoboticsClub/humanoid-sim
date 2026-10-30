"""Explicit capture-frame translation, preserving clocks, raw signals and encoder bias."""

import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.model.description import Description
from humanoid_sim.replay.coordinates import SCHEMA, CaptureCoordinates, validate_capture_coordinates
from humanoid_sim.replay.exc import PDO_FIELDS, ExcitationCapture
from humanoid_sim.replay.replay import DriveCommands, initial_joint_state, replay, replay_position_rms


@pytest.fixture
def fixture(tmp_path):
    desc = Description.load()
    plant = PlantCore(desc, list(desc.joints), 4, PlantCfg(dtype=torch.float64))
    source = dict(desc.table["standing_reference"], model_revision="synthetic historical model")
    doc = {
        "schema": SCHEMA,
        "source_identity": source,
        "target_description_digest": desc.digest,
        "drive_offsets_rad": {
            n: 0.0 if d.crank else (i + 1) * 0.03 for i, (n, d) in enumerate(desc.drives.items())
        },
        "command_offsets_rad": {n: (i + 1) * 0.01 for i, n in enumerate(desc.joints)},
        "provenance": {"kind": "synthetic fixture; not robot calibration"},
    }
    adapter = CaptureCoordinates.from_document(doc, desc)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"format": "ethrc-excitation/v8", "complete": True, "identity": source, "dataset_id": "fixture"}
        )
    )
    q = torch.full((1, len(desc.joints)), 0.02, dtype=torch.float64)
    for _, _, cols, tr in plant.map.ankles:
        q[:, cols] = tr.neutral
    native = plant.map.drive_state(q, torch.zeros_like(q))[0][0].numpy()
    stamps = np.arange(21, dtype=np.int64) * 5_000_000 + 1_700_000_000_000_000_000
    with (tmp_path / "pdo_feedback.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["device", "t_source_ros_ns", *PDO_FIELDS])
        writer.writeheader()
        for i, n in enumerate(plant.map.drive_names):
            for t in stamps:
                row = dict.fromkeys(PDO_FIELDS, 0.0)
                row.update(
                    native_position_rad=native[i] - doc["drive_offsets_rad"][n],
                    position_demand_rad=native[i] - doc["drive_offsets_rad"][n],
                    current_a=1.7,
                    current_demand_a=2.1,
                    motor_position_rad=0.4,
                )
                writer.writerow(dict(row, device=n, t_source_ros_ns=int(t)))
    gains = {
        n: {
            "stiffness_nm_per_rad": d.baseline_stiffness,
            "damping_nm_s_per_rad": d.baseline_damping,
            "raw_p_gain": 0,
            "raw_d_gain": 0,
        }
        for n, d in desc.drives.items()
    }
    (tmp_path / "gains.json").write_text(
        json.dumps({"start": {"actuators": gains}, "end": {"actuators": gains}})
    )
    with (tmp_path / "commands.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "t_frame_ros_ns", "joint_id", "q_nominal_rad"])
        w.writeheader()
        for i, t in enumerate(stamps[::4]):
            for k, n in enumerate(plant.map.joint_names):
                value = float(q[0, k]) - desc.joints[n].standing_offset_rad - doc["command_offsets_rad"][n]
                w.writerow({"seq": i, "t_frame_ros_ns": int(t), "joint_id": n, "q_nominal_rad": value})
    with (tmp_path / "acks.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["receiver_id", "state", "seq", "t_ack_ros_ns"])
        w.writeheader()
        for i, t in enumerate(stamps[::4]):
            w.writerow(dict(receiver_id="controller", state="applied", seq=i, t_ack_ros_ns=int(t)))
    return desc, plant, doc, adapter, ExcitationCapture(tmp_path, coordinates=adapter), q, native


def test_normalization_preserves_raw_and_nonposition_signals(fixture):
    desc, plant, doc, adapter, cap, q, native = fixture
    validate_capture_coordinates(cap, desc)
    for i, n in enumerate(plant.map.drive_names):
        raw, model = cap.capture_pdo()[n], cap.pdo()[n]
        np.testing.assert_allclose(model["native_position_rad"], native[i], atol=1e-15)
        np.testing.assert_allclose(model["position_demand_rad"], native[i], atol=1e-15)
        np.testing.assert_array_equal(raw.t_ns, model.t_ns)
        for field in set(PDO_FIELDS) - {"native_position_rad", "position_demand_rad"}:
            np.testing.assert_array_equal(model[field], raw[field])
    for commands in [cap.commands(), cap.applied_commands()]:
        offsets = np.array([desc.joints[n].standing_offset_rad for n in commands.joints])
        np.testing.assert_allclose(
            commands.positions_rad[0] + offsets,
            q[0, [plant.map.joint_names.index(n) for n in commands.joints]],
            atol=1e-15,
        )
    # Cached reads must never accumulate translations.
    np.testing.assert_array_equal(
        cap.pdo()[plant.map.drive_names[0]]["native_position_rad"],
        cap.pdo()[plant.map.drive_names[0]]["native_position_rad"],
    )
    with pytest.raises(ValueError, match="explicit"):
        validate_capture_coordinates(ExcitationCapture(cap.root), desc)


@pytest.mark.parametrize(
    "change,error",
    [
        ("target", "digest"),
        ("source", "source identity"),
        ("missing_drive", "inventory"),
        ("nan", "finite"),
        ("crank", "crank"),
        ("provenance", "provenance"),
    ],
)
def test_reject_incompatible_or_incomplete_artifacts(fixture, change, error):
    desc, _, doc, _, cap, _, _ = fixture
    doc = copy.deepcopy(doc)
    if change == "target":
        doc["target_description_digest"] = "wrong"
    if change == "source":
        doc["source_identity"]["calibration_revision"] = "wrong"
        adapter = CaptureCoordinates.from_document(doc, desc)
        with pytest.raises(ValueError, match=error):
            ExcitationCapture(cap.root, coordinates=adapter)
        return
    if change == "missing_drive":
        doc["drive_offsets_rad"].pop(next(iter(desc.drives)))
    if change == "nan":
        doc["command_offsets_rad"][next(iter(desc.joints))] = float("nan")
    if change == "crank":
        doc["drive_offsets_rad"][next(n for n, d in desc.drives.items() if d.crank)] = 0.1
    if change == "provenance":
        doc["provenance"] = {}
    with pytest.raises(ValueError, match=error):
        CaptureCoordinates.from_document(doc, desc)


@pytest.mark.parametrize("source", ["drive", "command"])
def test_replay_frames_and_encoder_bias_compose_in_both_scoring_paths(fixture, source):
    desc, plant, doc, adapter, cap, q, native = fixture

    # Clearly labeled test double: constant-state plumbing, no dynamics validation.
    class SimDouble:
        period_ns = 4_000_000
        desc = plant.desc
        backend = SimpleNamespace(contacts=None)

        def __init__(self):
            self.plant = plant
            self.t_ns = 0

        def set_base_acceleration(self, value):
            pass

        def write_joint_state(self, q, v):
            self.q, self.v = q, v

        def joint_state(self):
            return self.q, self.v

        def step(self):
            self.t_ns += self.period_ns
            plant.bank.queue.advance(self.t_ns)
            p, v, _ = plant.map.drive_state(self.q, self.v)
            return SimpleNamespace(
                position_rad=p, velocity_rad_s=v, applied_torque_nm=p * 0, demanded_torque_nm=p * 0
            )

    bias = torch.tensor([0.0, 0.001, -0.002, 0.003], dtype=torch.float64)[:, None]
    # Use direct drives for nonzero bias; keep crank bias zero for a strict FK round trip.
    plant.bank.position_offset[:] = bias
    for _, ds, _, _ in plant.map.ankles:
        plant.bank.position_offset[:, ds] = 0
    stamps = cap.commands().t_ns
    stream = DriveCommands(tuple(plant.map.drive_names), stamps, np.tile(native, (len(stamps), 1)), "frame")
    options = {"source": source, "commands": stream if source == "command" else None}
    sim = SimDouble()
    result = replay(cap, sim, **options)
    np.testing.assert_allclose(result.sim["position"], result.real["position"], atol=1e-7)
    np.testing.assert_allclose(
        result.real["capture_position"][0],
        native - np.array([doc["drive_offsets_rad"][n] for n in plant.map.drive_names]),
        atol=1e-14,
    )
    expected = initial_joint_state(plant, torch.tensor(native)[None] - plant.bank.position_offset)
    torch.testing.assert_close(sim.q, expected)
    assert result.meta["capture_coordinates"]["sha256"] == adapter.sha256
    rms = replay_position_rms(cap, sim, skip_s=0, **options)
    np.testing.assert_allclose(rms, 0, atol=1e-7)
