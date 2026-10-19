"""The real-time host end to end: `humanoid-sim serve` on Warp's CPU device, this test as the client.

On the CPU the host runs far below real time; the protocol does not depend on it. Runs in the Isaac Lab image.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.runtime.abi import plant_abi as abi

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.physics


def _feedback(seg):
    got = seg.regions["feedback"].read()
    assert got is not None
    return got[1]


def _wait(cond, timeout_s, what):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def test_serve_round_trip(tmp_path):
    pytest.importorskip("isaaclab_newton")
    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built; run `humanoid-sim build-asset` in the Isaac Lab image")
    path, log = tmp_path / "plant", tmp_path / "serve.log"
    proc = subprocess.Popen([sys.executable, "-m", "humanoid_sim", "serve", "--segment", str(path), "--device", "cpu",
                             "--support", "suspended", "--client-timeout-ms", "1500"],
                            cwd=REPO, stdout=log.open("w"), stderr=subprocess.STDOUT, text=True)
    try:
        seg = abi.Segment.open(path, robot_id="ethrc_v1", description_digest=Description.load().digest, wait_s=180)
        desc = seg.descriptor()
        drives = desc["drives"]
        assert len(drives) == seg.num_drives == 22 and desc["physics"]["engine"] == "isaaclab/mjwarp"

        # Hold the pose it starts in at the baseline gains.
        hold = _feedback(seg)["drive"]["position_rad"][:22].copy()
        d = Description.load()
        gains = d.table["baseline_position_gains"]
        n = len(drives)
        demand = {"torque_enabled": np.ones(n), "mode": np.full(n, -64), "homing_hold": np.zeros(n),
                  "target_position_rad": hold, "target_velocity_rad_s": np.zeros(n),
                  "feedforward_torque_nm": np.zeros(n), "i_gain_raw": np.zeros(n), "max_torque_mnm": np.zeros(n),
                  "p_gain_raw": np.array([round(gains[x]["stiffness"] * 1000) for x in drives]),
                  "d_gain_raw": np.array([round(gains[x]["damping"] * 1000) for x in drives])}
        cycle = 0

        def send_demands():
            nonlocal cycle
            cycle += 1
            seg.regions["demands"].write(abi.demands_record(demand, cycle=cycle, session_id=seg.session_id))

        def serve_for(seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                send_demands()
                time.sleep(0.005)

        deadline = time.monotonic() + 300  # the first CPU steps compile kernels
        while int(_feedback(seg)["step"]) < 2 and time.monotonic() < deadline:
            serve_for(0.1)
        step0 = int(_feedback(seg)["step"])
        serve_for(4.0)
        fb = _feedback(seg)
        assert int(fb["step"]) > step0 + 10
        pos = fb["drive"]["position_rad"][:n]
        assert np.abs(pos - hold).max() < 0.03  # held at baseline gains
        assert np.abs(fb["drive"]["applied_torque_nm"][:n]).max() > 0.1  # gravity on the limbs
        assert int(fb["imu_total"]) > 0 and int(fb["imu_count"]) > 0
        imu = fb["imu"][: int(fb["imu_count"])]
        assert (np.diff(imu["seq"].astype(np.int64)) == -1).all()  # newest first, consecutive
        assert abs(np.linalg.norm(imu["specific_force_m_s2"][0]) - 9.81) < 0.3
        _, truth = seg.regions["truth"].read()
        assert int(truth["num_joints"]) == 22 and float(truth["mass_kg"]) == pytest.approx(60.15, abs=0.05)

        def command(code, args):
            seq = seg.send(code, args)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:  # keep the client alive while the host works
                send_demands()
                try:
                    return seg.wait_ack(seq, timeout_s=0.05)
                except TimeoutError:
                    pass
            raise AssertionError(f"no ack for command {code}")

        assert command(abi.CMD_PAUSE, {})[0] == abi.ACK_OK
        serve_for(0.5)
        paused_step = int(_feedback(seg)["step"])
        serve_for(1.0)
        assert int(_feedback(seg)["step"]) == paused_step and int(_feedback(seg)["paused"]) == 1
        assert command(abi.CMD_RESUME, {})[0] == abi.ACK_OK
        assert command(abi.CMD_SET_SUPPORT, {"support": "gantry", "tension_breakpoints_s_n": [[0, 100]]})[0] \
            == abi.ACK_REJECTED  # the base is welded in a suspended session
        assert command(99, {})[0] == abi.ACK_UNKNOWN
        drive = drives[0]
        assert command(abi.CMD_INJECT_FAULT, {"fault": "encoder_dropout", "drive": drive})[0] == abi.ACK_REJECTED
        assert command(abi.CMD_INJECT_FAULT, {"fault": "encoder_freeze", "drive": drive, "value": 1})[0] == abi.ACK_OK
        before = _feedback(seg)["drive"]["position_rad"][:n].copy()
        demand["target_position_rad"] = hold + 0.1  # move every drive
        serve_for(2.0)
        moved = _feedback(seg)["drive"]["position_rad"][:n] - before
        assert moved[0] == 0.0 and np.abs(moved[1:]).max() > 0.02  # drive 0 reports its frozen reading
        assert command(abi.CMD_RESET, {"pose": "standing"})[0] == abi.ACK_OK

        # Client goes away: the host ends the session.
        _wait(lambda: proc.poll() is not None, 120, "the host to end the session")
        assert proc.returncode == 3, log.read_text()[-3000:]
        assert seg.state == abi.STATE_ENDED
        last = json.loads(log.read_text().strip().splitlines()[-1])
        assert last["exit"] == 3 and last["attached"]
    finally:
        if proc.poll() is None:
            proc.kill()


def test_prescribed_pose_with_floor_clearance_and_sole_truth(tmp_path):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built; run `humanoid-sim build-asset` in the Isaac Lab image")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    host = RuntimeHost(tmp_path / "plant", usd, device="cpu", support="free", compile=False)
    try:
        tilt = [0.0, 0.0871557, 0.0, 0.9961947]  # 10 degrees about y
        result = host._prescribe({"joint_pos": {}, "base_pose": [0.0, 0.0, 1.5, *tilt], "floor_clearance_m": 0.002})
        assert result == "prescribed"
        host.step()  # the pose is held: written again before the step
        host._publish_truth()
        _, truth = host.seg.regions["truth"].read()
        lowest = truth["sole"]["lowest_z_m"].min()
        assert lowest == pytest.approx(0.002, abs=0.003)  # one step of motion under the hold
        normals = truth["sole"]["plate_normal_w"]
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
        assert (truth["sole"]["plate_face_height_span_m"] > 0.01).all()  # tilted plates
        face = truth["sole"]["face_xy_m"]
        assert np.allclose(face.mean(axis=1), truth["sole"]["position_m"][:, :2], atol=0.15)
        with pytest.raises(ValueError, match="floor_clearance_m"):
            host._prescribe({"floor_clearance_m": 0.002})
        host._prescribe({"release": True})
        assert host._reset({"pose": "home"}) is not None  # both soles level, on the ground
        host._publish_truth()
        _, truth = host.seg.regions["truth"].read()
        assert (truth["sole"]["plate_face_height_span_m"] < 0.002).all()
        assert truth["sole"]["lowest_z_m"] == pytest.approx([0.001, 0.001], abs=0.001)
    finally:
        host.close()


def test_detach_and_next_session(tmp_path):
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.runtime.host import EXIT_DETACHED, RuntimeHost

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built; run `humanoid-sim build-asset` in the Isaac Lab image")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    path = tmp_path / "plant"
    host = RuntimeHost(path, usd, device="cpu", support="free", compile=False)
    try:
        first = abi.Segment.open(path, wait_s=1)
        n = first.num_drives
        demand = {"torque_enabled": np.zeros(n), "mode": np.zeros(n), "homing_hold": np.zeros(n),
                  "target_position_rad": np.zeros(n), "target_velocity_rad_s": np.zeros(n),
                  "feedforward_torque_nm": np.zeros(n), "i_gain_raw": np.zeros(n), "max_torque_mnm": np.zeros(n),
                  "p_gain_raw": np.zeros(n), "d_gain_raw": np.zeros(n)}
        first.regions["demands"].write(abi.demands_record(demand, cycle=1, session_id=first.session_id))
        host.step()
        assert host.attached
        assert host._set_support({"support": "gantry", "tension_breakpoints_s_n": [[0, 300]],
                                  "attachment_local_m": [[0.0, 0.0, 0.3]]}) == "gantry"
        assert host._inject({"fault": "power_sag", "value": 0.5}) == "power_sag active"
        assert host._set_fidelity({"imu": "ideal"}) == "imu ideal"
        assert float(host.sim.imu._sigma.abs().max()) == 0.0
        with pytest.raises(ValueError, match="ideal or real"):
            host._set_fidelity({"imu": "minimal"})
        for _ in range(20):
            host.step()

        seq = first.send(abi.CMD_DETACH)
        assert host.run(max_steps=5) == EXIT_DETACHED
        assert first.wait_ack(seq, timeout_s=1) == (abi.ACK_OK, "detached")
        assert first.state == abi.STATE_ENDED

        host.new_session()
        assert first.state == abi.STATE_ENDED  # the old client keeps the ended file
        second = abi.Segment.open(path, wait_s=1)
        assert second.state == abi.STATE_READY and second.session_id not in (0, first.session_id)
        assert int(second.regions["feedback"].read()[1]["step"]) == 0
        assert not host._faults and float(host.sim.plant.bank.torque_scale.min()) == 1.0
        assert torch.equal(host.sim.imu._sigma, host.sim.imu._nominal_sigma)  # every session starts real
        assert int(host.sim.gantry.mode[0]) == 0
        _, truth = second.regions["truth"].read()
        home = [host.desc.home_pose[j] for j in host.joints]  # the start pose
        assert np.allclose(truth["joint_pos_rad"][: len(home)], home, atol=1e-4)
        assert float(truth["base_position_m"][2]) == pytest.approx(host._poses["home"][1] + 0.001, abs=1e-3)

        first.regions["demands"].write(abi.demands_record(demand, cycle=2, session_id=first.session_id))
        host.step()
        assert not host.attached  # a client of the ended session does not attach
        second.regions["demands"].write(abi.demands_record(demand, cycle=1, session_id=second.session_id))
        host.step()
        assert host.attached
        first.close()
        second.close()
    finally:
        host.close()


def test_demand_intake_queues_what_the_record_says(tmp_path):
    """The packed, graphed intake queues exactly the demand built field by field from the record."""
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.runtime.host import _DemandIntake
    from humanoid_sim.sim import SimCore

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built; run `humanoid-sim build-asset` in the Isaac Lab image")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimCore(usd, device=device, fixed_base=True)
    try:
        p, n = sim.plant, len(sim.plant.map.drive_names)
        rng = np.random.default_rng(0)
        rec = np.zeros(n, abi.DRIVE_DEMAND)
        rec["torque_enabled"], rec["mode"], rec["homing_hold"] = rng.integers(0, 2, n), -64, rng.integers(0, 2, n)
        rec["mode"][:3] = [6, -62, 0]
        for name in ("target_position_rad", "target_velocity_rad_s", "feedforward_torque_nm"):
            rec[name] = rng.normal(size=n)
        for name in ("p_gain_raw", "d_gain_raw", "max_torque_mnm"):
            rec[name] = rng.integers(0, 2**32 - 1, n)
        env = torch.tensor([0], device=device)

        def queued():
            q = p.bank.queue
            return {k: v.clone() for k, v in q.pending.items()} | {"arrival": q.arrival.clone(),
                                                                   "count": q.count.clone()}

        def t(name, dtype):
            return torch.as_tensor(rec[name].astype(np.float64), device=device).to(dtype).view(1, n)

        dt = p.cfg.dtype
        p.reset(env)
        p.submit(DriveDemand(torque_enabled=t("torque_enabled", torch.bool), mode=t("mode", torch.int8),
                             target_position_rad=t("target_position_rad", dt),
                             target_velocity_rad_s=t("target_velocity_rad_s", dt),
                             feedforward_torque_nm=t("feedforward_torque_nm", dt),
                             p_gain_raw=t("p_gain_raw", torch.int64), i_gain_raw=t("i_gain_raw", torch.int64),
                             d_gain_raw=t("d_gain_raw", torch.int64), max_torque_mnm=t("max_torque_mnm", torch.int64),
                             homing_hold=t("homing_hold", torch.bool)),
                 torch.tensor([123_456_789], device=device))
        expected = queued()
        cuda = device.startswith("cuda")
        for graph in ((False, True) if cuda else (False,)):
            intake = _DemandIntake(p, n, torch.device(device), cuda_graph=graph, compile=False)
            p.reset(env)
            intake(rec, 123_456_789)
            got = queued()
            for k, v in expected.items():
                assert torch.equal(got[k], v), (graph, k)
    finally:
        sim.close()


def test_joint_disturbance_deflects_a_held_joint_and_clears(tmp_path):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built; run `humanoid-sim build-asset` in the Isaac Lab image")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    path = tmp_path / "plant"
    host = RuntimeHost(path, usd, device="cpu", support="suspended", compile=False)
    try:
        client = abi.Segment.open(path, wait_s=1)
        drives = client.descriptor()["drives"]
        n = len(drives)
        gains = Description.load().table["baseline_position_gains"]
        hold = client.regions["feedback"].read()[1]["drive"]["position_rad"][:n].copy()
        demand = {name: np.zeros(n) for name in abi.DRIVE_DEMAND.names if not name.startswith("_")}
        demand |= {"torque_enabled": np.ones(n), "mode": np.full(n, -64), "target_position_rad": hold,
                   "p_gain_raw": np.array([round(gains[d]["stiffness"] * 1000) for d in drives]),
                   "d_gain_raw": np.array([round(gains[d]["damping"] * 1000) for d in drives])}
        client.regions["demands"].write(abi.demands_record(demand, cycle=1, session_id=client.session_id))
        j = host.joints.index("elbow_flexion_r")

        def settle(steps=60):
            for _ in range(steps):
                host.step()
            return float(host.sim.joint_state()[0][0, j])

        reference = settle()
        assert host._disturb({"drive": "maxonHEJ50AR4", "torque_nm": 10.0, "duration_s": None}) \
            == "elbow_flexion_r until cleared"
        deflected = settle()
        assert float(host.sim.plant.external_torque[0, j]) == 10.0
        assert deflected - reference == pytest.approx(10.0 / 50.0, rel=0.35)  # against the drive's 50 Nm/rad
        assert host._disturb({"joint": "elbow_flexion_r", "torque_nm": 0.0}) == "cleared"
        assert abs(settle() - reference) < 0.02
        assert float(host.sim.plant.external_torque.abs().max()) == 0.0
        host._disturb({"joint": "elbow_flexion_r", "torque_nm": 4.0, "frequency_hz": 5.0, "duration_s": 0.04})
        host.step()
        t = host.sim.t_ns - host.period_ns  # the step just taken started here
        assert float(host.sim.plant.external_torque[0, j]) == pytest.approx(4.0 * np.sin(2 * np.pi * 5.0 * t / 1e9),
                                                                         abs=1e-5)
        settle(12)
        assert float(host.sim.plant.external_torque.abs().max()) == 0.0  # expired
        with pytest.raises(ValueError, match="1:1 drive"):
            host._disturb({"drive": "maxonHEJ50L_ANK1", "torque_nm": 1.0})
        client.close()
    finally:
        host.close()


def test_a_disabled_drive_is_back_driven_through_its_gearbox(tmp_path):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    out = asset_dir(Description.load(), load_profile("ethrc_v1"), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip("robot asset not built; run `humanoid-sim build-asset` in the Isaac Lab image")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    path = tmp_path / "plant"
    host = RuntimeHost(path, usd, device="cpu", support="suspended", compile=False)
    try:
        desc = Description.load()
        knee = host.robot.joint_names.index("knee_r")
        ankle = host.robot.joint_names.index("foot_pitch_r")

        def friction(j):
            return float(host.robot.data.joint_friction_coeff.torch[0, j])

        backdrive = float(desc.family("HEJ90-48-140")["backdrive_torque"])
        assert friction(knee) == pytest.approx(backdrive)  # no client: every drive off
        assert friction(ankle) == pytest.approx(desc.joints["foot_pitch_r"].friction)  # cranks keep theirs
        client = abi.Segment.open(path, wait_s=1)
        n = client.num_drives
        demand = {name: np.zeros(n) for name in abi.DRIVE_DEMAND.names if not name.startswith("_")}
        demand |= {"torque_enabled": np.ones(n), "mode": np.full(n, -64),
                   "target_position_rad": client.regions["feedback"].read()[1]["drive"]["position_rad"][:n].copy(),
                   "p_gain_raw": np.full(n, 100_000), "d_gain_raw": np.full(n, 5_000)}
        for cycle, enabled in enumerate((1, 0), start=1):
            demand["torque_enabled"][:] = enabled
            client.regions["demands"].write(abi.demands_record(demand, cycle=cycle, session_id=client.session_id))
            host.step()
            expected = desc.joints["knee_r"].friction if enabled else backdrive
            assert friction(knee) == pytest.approx(expected)
        client.close()
    finally:
        host.close()
