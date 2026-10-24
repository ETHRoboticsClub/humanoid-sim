"""The real-time host end to end: `humanoid-sim serve` on Warp's CPU device, this test as the client.

On the CPU the host runs far below real time; the protocol does not depend on it. Runs in the Isaac Lab image.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import PARTS, PHYSICS_ROBOTS, built_usd, gravity_torque, joints_below, urdf_mass_kg

from humanoid_sim.model.asset import asset_dir, load_profile
from humanoid_sim.model.description import Description
from humanoid_sim.model.poses import pose_joint_positions, start_pose
from humanoid_sim.model.urdf import PreparedUrdf
from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.sensors import rigs

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


@pytest.mark.parametrize(
    ("robot", "model"), [("ethrc_v1", "mti-610"), ("g1_29dof_with_hand_rev_1_0", "unitree-g1-imu")]
)
def test_the_descriptor_names_the_rigs_imu(robot, model):
    """A client tells from the descriptor which device its IMU samples come from: the rig's model, where it
    sits, and the sample period and delivery the rig gives it."""
    from humanoid_sim.runtime.host import imu_descriptor

    desc, rig_imu = Description.load(robot), rigs.rig(robot).imu
    imu = json.loads(json.dumps(imu_descriptor(desc)))
    assert imu["model"] == rig_imu.model == model
    assert imu["parent"] == desc.imu_mount()[0] and imu["frame"] == rig_imu.frame
    assert imu["product"]
    assert imu["sample_period_ns"] == rig_imu.sample_period_ns
    assert imu["burst_period_ns"] == list(rig_imu.burst_period_ns)
    assert imu["in_burst_spacing_ns"] == rig_imu.in_burst_spacing_ns


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_serve_round_trip(tmp_path, robot):
    pytest.importorskip("isaaclab_newton")
    import torch

    usd = built_usd(robot)
    d = Description.load(robot)
    path, log = tmp_path / "plant", tmp_path / "serve.log"
    proc = subprocess.Popen([sys.executable, "-m", "humanoid_sim", "--robot", robot, "serve", "--segment", str(path),
                             "--device", "cpu", "--support", "suspended", "--client-timeout-ms", "1500"],
                            cwd=REPO, stdout=log.open("w"), stderr=subprocess.STDOUT, text=True)
    try:
        seg = abi.Segment.open(path, robot_id=robot, description_digest=d.digest, wait_s=180)
        desc = seg.descriptor()
        drives = desc["drives"]
        n = len(d.drives)
        assert len(drives) == seg.num_drives == n and desc["physics"]["engine"] == "isaaclab/mjwarp"
        assert (desc["imu"]["model"], desc["imu"]["frame"]) == (rigs.rig(robot).imu.model, "imu_link")

        # Hold the pose it starts in at the baseline gains.
        hold = _feedback(seg)["drive"]["position_rad"][:n].copy()
        demand = abi.pd_demand(hold, [d.drives[x].baseline_stiffness for x in drives],
                               [d.drives[x].baseline_damping for x in drives])
        # How far each joint the harness body carries (arms, head) sags at those gains: its gravity torque over its
        # stiffness, as with the base welded upright, since the gantry holds that body; the body's few degrees of
        # tilt on the cable stay within the 1 cm slack. The waist and legs hang off it and are not checked here.
        joints = list(d.joints)
        start = pose_joint_positions(d, load_profile(robot), start_pose(load_profile(robot)))
        tau = gravity_torque(usd, joints, torch.tensor([start[j] for j in joints], dtype=torch.float64), 1.0)
        carried = joints_below(robot, load_profile(robot)["support"]["harness"]["body"])
        checked = np.array([d.drives[x].crank is None and d.drives[x].joints[0] in carried for x in drives])
        assert checked.any()
        sag = {x: abs(tau[joints.index(d.drives[x].joints[0])]) / d.drives[x].baseline_stiffness
               for x, c in zip(drives, checked, strict=True) if c}
        allowed = np.array([1.2 * sag[x] + 0.01 if x in sag else np.inf for x in drives])
        cycle = 0

        def send_demands():
            nonlocal cycle
            cycle += 1
            seg.write_demands(demand, cycle=cycle)

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
        assert (np.abs(pos - hold) < allowed).all()  # held at baseline gains, sagging by gravity
        assert np.abs(fb["drive"]["applied_torque_nm"][:n]).max() > 0.1  # gravity on the limbs
        assert int(fb["imu_total"]) > 0 and int(fb["imu_count"]) > 0
        imu = fb["imu"][: int(fb["imu_count"])]
        assert (np.diff(imu["seq"].astype(np.int64)) == -1).all()  # newest first, consecutive
        assert abs(np.linalg.norm(imu["specific_force_m_s2"][0]) - 9.81) < 0.3
        _, truth = seg.regions["truth"].read()
        assert int(truth["num_joints"]) == len(joints)
        assert float(truth["mass_kg"]) == pytest.approx(urdf_mass_kg(robot), abs=0.05)

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
        assert command(abi.CMD_RESET, {"joint_pos": {}, "base_pose": [0, 0, 1, 0, 0, 0, 1]})[0] \
            == abi.ACK_REJECTED  # a suspended session hangs at its gantry
        assert command(99, {})[0] == abi.ACK_UNKNOWN
        drive = drives[0]
        assert command(abi.CMD_INJECT_FAULT, {"fault": "encoder_dropout", "drive": drive})[0] == abi.ACK_REJECTED
        assert command(abi.CMD_INJECT_FAULT, {"fault": "encoder_freeze", "drive": drive, "value": 1})[0] == abi.ACK_OK
        before = _feedback(seg)["drive"]["position_rad"][:n].copy()
        demand["target_position_rad"] = hold + 0.1  # move every drive
        serve_for(2.0)
        moved = _feedback(seg)["drive"]["position_rad"][:n] - before
        assert moved[0] == 0.0 and np.abs(moved[1:]).max() > 0.02  # drive 0 reports its frozen reading
        assert command(abi.CMD_RESET, {"pose": start_pose(load_profile(robot))})[0] == abi.ACK_OK

        # Client goes away: the host ends the session.
        _wait(lambda: proc.poll() is not None, 120, "the host to end the session")
        assert proc.returncode == 3, log.read_text()[-3000:]
        assert seg.state == abi.STATE_ENDED
        last = json.loads(log.read_text().strip().splitlines()[-1])
        assert last["exit"] == 3 and last["attached"]
    finally:
        if proc.poll() is None:
            proc.kill()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_prescribed_pose_with_floor_clearance_and_sole_truth(tmp_path, robot):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    host = RuntimeHost(tmp_path / "plant", built_usd(robot), device="cpu", robot=robot, support="free", compile=False)
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


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_detach_and_next_session(tmp_path, robot):
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.runtime.host import EXIT_DETACHED, RuntimeHost

    path = tmp_path / "plant"
    host = RuntimeHost(path, built_usd(robot), device="cpu", robot=robot, support="free", compile=False)
    try:
        first = abi.Segment.open(path, wait_s=1)
        n = first.num_drives
        demand = abi.pd_demand(np.zeros(n), 0.0, 0.0, law=abi.LAW_DISABLED)
        first.write_demands(demand, cycle=1)
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
        profile = load_profile(robot)
        assert host.start_pose == start_pose(profile)  # the profile's poses.start
        start_q = pose_joint_positions(host.desc, profile, host.start_pose)
        start = [start_q[j] for j in host.joints]
        assert np.allclose(truth["joint_pos_rad"][: len(start)], start, atol=1e-4)
        assert float(truth["base_position_m"][2]) == pytest.approx(host._poses[host.start_pose][1] + 0.001, abs=1e-3)

        first.write_demands(demand, cycle=2)
        host.step()
        assert not host.attached  # a client of the ended session does not attach
        second.write_demands(demand, cycle=1)
        host.step()
        assert host.attached
        first.close()
        second.close()
    finally:
        host.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_the_crane_lifts_and_carries_a_hoisted_robot(tmp_path, robot):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost
    from humanoid_sim.world.trolley import GantryRig

    rig = GantryRig(center_m=(0.0, 0.0), yaw_rad=0.0, anchor_z_m=4.0, travel_x_m=(-1.0, 1.0), travel_y_m=(-1.0, 1.0),
                    hook_z_m=(0.5, 3.0), max_speed_m_s=(0.5, 0.5, 0.5), max_accel_m_s2=(2.0, 2.0, 2.0),
                    home_m=(0.0, 0.0, 0.5))
    path = tmp_path / "plant"
    host = RuntimeHost(path, built_usd(robot), device="cpu", robot=robot, support="free", compile=False, gantry=rig)
    try:
        seg = abi.Segment.open(path, wait_s=1)
        assert seg.descriptor()["gantry"]["anchor_z_m"] == 4.0
        n = seg.num_drives
        seg.write_demands(abi.pd_demand(np.zeros(n), 0.0, 0.0, law=abi.LAW_DISABLED), cycle=1)
        host.step()

        def truth():
            host._publish_truth()
            return seg.regions["truth"].read()[1]

        with pytest.raises(ValueError, match="outside the gantry's travel"):
            host._move_gantry({"position_m": [2.0, 0.0, 1.0]})
        hoist = {"stiffness_n_m": 20000.0, "damping_n_s_m": 500.0, "swing_damping_n_s_m": 50.0}
        assert host._set_support({"support": "gantry", "hoist": hoist, "attachment_local_m": [0.0, 0.0, 0.3]}) \
            == "gantry"
        t0 = truth()
        harness_z = float(t0["base_position_m"][2]) + 0.3
        weight = float(t0["mass_kg"]) * 9.81
        # Hook 10 cm above the harness: the robot comes up by that less the rope's stretch under its weight.
        host._move_gantry({"position_m": [0.0, 0.0, harness_z + 0.1]})
        for _ in range(250):
            host.step()
        t1 = truth()
        assert t1["gantry_command_m"].tolist() == pytest.approx([0.0, 0.0, harness_z + 0.1], abs=1e-6)
        assert t1["gantry_position_m"].tolist() == pytest.approx(t1["gantry_command_m"].tolist(), abs=1e-3)
        assert float(t1["base_position_m"][2]) > float(t0["base_position_m"][2]) + 0.03
        assert float(t1["gantry_tension_n"]) > 0.5 * weight
        # The trolley 30 cm along x carries the hanging robot with it.
        assert host._move_gantry({"position_m": [0.3, 0.0, harness_z + 0.1]}).startswith("moving to 0.300")
        for _ in range(500):
            host.step()
        t2 = truth()
        assert t2["gantry_top_m"].tolist() == pytest.approx([0.3, 0.0, 4.0], abs=1e-3)
        assert float(t2["base_position_m"][0]) > 0.15
        seg.close()
    finally:
        host.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_demand_intake_queues_what_the_record_says(tmp_path, robot):
    """The packed, graphed intake queues exactly the demand built field by field from the record."""
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.devices.demand import DriveDemand
    from humanoid_sim.runtime.host import _DemandIntake
    from humanoid_sim.sim import SimCore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimCore(built_usd(robot), device=device, robot=robot, fixed_base=True)
    try:
        p, n = sim.plant, len(sim.plant.map.drive_names)
        rng = np.random.default_rng(0)
        rec = np.zeros(n, abi.DRIVE_DEMAND)
        rec["law"] = rng.integers(0, 3, n)
        rec["law"][:2] = [7, 255]  # unknown laws: DISABLED
        for name in ("target_position_rad", "target_velocity_rad_s", "feedforward_torque_nm"):
            rec[name] = rng.normal(size=n)
        for name in ("stiffness_nm_per_rad", "damping_nm_s_per_rad"):
            rec[name] = rng.uniform(0, 500, n)
        rec["torque_limit_nm"] = np.where(rng.integers(0, 2, n) == 1, np.inf, rng.uniform(0, 100, n))
        env = torch.tensor([0], device=device)

        def queued():
            q = p.bank.queue
            return {k: v.clone() for k, v in q.pending.items()} | {"arrival": q.arrival.clone(),
                                                                   "count": q.count.clone()}

        def t(name, dtype):
            return torch.as_tensor(rec[name].astype(np.float64), device=device).to(dtype).view(1, n)

        dt = p.cfg.dtype
        p.reset(env)
        law = t("law", torch.int16)
        p.submit(DriveDemand(law=torch.where(law <= 2, law, torch.zeros_like(law)).to(torch.int8),
                             target_position_rad=t("target_position_rad", dt),
                             target_velocity_rad_s=t("target_velocity_rad_s", dt),
                             feedforward_torque_nm=t("feedforward_torque_nm", dt),
                             stiffness_nm_per_rad=t("stiffness_nm_per_rad", dt),
                             damping_nm_s_per_rad=t("damping_nm_s_per_rad", dt),
                             torque_limit_nm=t("torque_limit_nm", dt)),
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


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_joint_disturbance_deflects_a_held_joint_and_clears(tmp_path, robot):
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    path = tmp_path / "plant"
    host = RuntimeHost(path, built_usd(robot), device="cpu", robot=robot, support="suspended", compile=False)
    try:
        client = abi.Segment.open(path, wait_s=1)
        drives = client.descriptor()["drives"]
        n = len(drives)
        d = Description.load(robot)
        hold = client.regions["feedback"].read()[1]["drive"]["position_rad"][:n].copy()
        demand = abi.pd_demand(hold, [d.drives[x].baseline_stiffness for x in drives],
                               [d.drives[x].baseline_damping for x in drives])
        client.write_demands(demand, cycle=1)
        elbow = PARTS[robot]["elbow"]
        drive = next(x for x, dr in d.drives.items() if dr.joints == (elbow,))
        j = host.joints.index(elbow)

        def settle(steps=60):
            for _ in range(steps):
                host.step()
            return float(host.sim.joint_state()[0][0, j])

        reference = settle()
        assert host._disturb({"drive": drive, "torque_nm": 10.0, "duration_s": None}) == f"{elbow} until cleared"
        deflected = settle()
        assert float(host.sim.plant.external_torque[0, j]) == 10.0
        # Against the drive's stiffness (the elbow's drive and joint frames agree in sign).
        assert deflected - reference == pytest.approx(10.0 / d.drives[drive].baseline_stiffness, rel=0.35)
        assert host._disturb({"joint": elbow, "torque_nm": 0.0}) == "cleared"
        # Back to where it was: friction holds the joint anywhere within friction / stiffness of where the drive
        # and gravity balance, on either side, before and after.
        band = d.joints[elbow].friction / d.drives[drive].baseline_stiffness
        assert abs(settle() - reference) < 2 * band + 0.005
        assert float(host.sim.plant.external_torque.abs().max()) == 0.0
        host._disturb({"joint": elbow, "torque_nm": 4.0, "frequency_hz": 5.0, "duration_s": 0.04})
        host.step()
        t = host.sim.t_ns - host.period_ns  # the step just taken started here
        assert float(host.sim.plant.external_torque[0, j]) == pytest.approx(4.0 * np.sin(2 * np.pi * 5.0 * t / 1e9),
                                                                         abs=1e-5)
        settle(12)
        assert float(host.sim.plant.external_torque.abs().max()) == 0.0  # expired
        cranks = [x for x, dr in d.drives.items() if dr.crank is not None]
        if cranks:
            with pytest.raises(ValueError, match="1:1 drive"):
                host._disturb({"drive": cranks[0], "torque_nm": 1.0})
        client.close()
    finally:
        host.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_a_disabled_drive_is_back_driven_through_its_gearbox(tmp_path, robot):
    """A disabled 1:1 drive's joint takes its family's backdrive torque as friction (the joint's own friction when
    the description records none, as for the G1's motors); crank-driven joints keep theirs."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    path = tmp_path / "plant"
    host = RuntimeHost(path, built_usd(robot), device="cpu", robot=robot, support="suspended", compile=False)
    try:
        desc = Description.load(robot)
        knee_drive = next(x for x, dr in desc.drives.items() if dr.crank is None and "knee" in dr.joints[0])
        knee_joint = desc.drives[knee_drive].joints[0]
        knee = host.robot.joint_names.index(knee_joint)

        def friction(j):
            return float(host.robot.data.joint_friction_coeff.torch[0, j])

        recorded = desc.family_value(desc.drives[knee_drive].family, "backdrive_torque")
        backdrive = desc.joints[knee_joint].friction if recorded is None else float(recorded)
        assert friction(knee) == pytest.approx(backdrive)  # no client: every drive off
        for lk in desc.linkages:  # cranks keep theirs
            for name in lk.joints:
                assert friction(host.robot.joint_names.index(name)) == pytest.approx(desc.joints[name].friction)
        client = abi.Segment.open(path, wait_s=1)
        n = client.num_drives
        demand = abi.pd_demand(client.regions["feedback"].read()[1]["drive"]["position_rad"][:n].copy(), 100.0, 5.0)
        for cycle, enabled in enumerate((1, 0), start=1):
            demand["law"][:] = abi.LAW_PD if enabled else abi.LAW_DISABLED
            client.write_demands(demand, cycle=cycle)
            host.step()
            expected = desc.joints[knee_joint].friction if enabled else backdrive
            assert friction(knee) == pytest.approx(expected)
        client.close()
    finally:
        host.close()


def test_the_host_serves_the_g1(tmp_path):
    """Another robot in humanoid-urdf serves the same way: the G1's motors take SI demands and the truth reports
    its soles (the boxes bounding its contact spheres)."""
    pytest.importorskip("isaaclab_newton")
    import torch

    from humanoid_sim.runtime.host import RuntimeHost

    robot = "g1_29dof_with_hand_rev_1_0"
    desc = Description.load(robot)
    out = asset_dir(desc, load_profile(robot), REPO / "outputs" / "assets")
    if not (out / "asset.json").exists():
        pytest.skip(f"robot asset not built; run `humanoid-sim --robot {robot} build-asset` in the Isaac Lab image")
    usd = out / json.loads((out / "asset.json").read_text())["usd"]
    path = tmp_path / "plant"
    host = RuntimeHost(path, usd, device="cpu", robot=robot, support="suspended", compile=False, truth_every=1)
    try:
        client = abi.Segment.open(path, robot_id=robot, description_digest=desc.digest, wait_s=1)
        info = client.descriptor()
        drives = info["drives"]
        n = len(desc.drives)
        assert len(drives) == client.num_drives == n
        soles = PreparedUrdf(desc, load_profile(robot)).build().soles
        assert info["soles"] == [s.body for s in sorted(soles, key=lambda s: s.side)]
        q, _ = host._named_pose("home")
        hold = host.sim.plant.map.drive_state(q.to(host.sim.plant.cfg.dtype), torch.zeros_like(q))[0][0].numpy()
        demand = abi.pd_demand(hold, [desc.drives[d].baseline_stiffness for d in drives],
                               [desc.drives[d].baseline_damping for d in drives])
        for cycle in range(1, 51):
            client.write_demands(demand, cycle=cycle)
            host.step()
        assert host.attached
        fb = _feedback(client)["drive"]
        assert np.abs(fb["position_rad"][:n] - hold).max() < 0.1  # hanging, at the deploy gains
        assert np.abs(fb["applied_torque_nm"][:n]).max() > 0.1
        _, truth = client.regions["truth"].read()
        soles = truth["sole"]
        assert (soles["lowest_z_m"] > 0.2).all() and (soles["normal_force_n"] == 0).all()  # hung 0.3 m up
        assert soles["position_m"][0][1] > soles["position_m"][1][1]  # left at +y: the G1 faces +x
        demand["law"][:] = abi.LAW_DISABLED
        client.write_demands(demand, cycle=51)
        # Sent after the last step ended, so in force from just after the next one starts: off for all of the one
        # after.
        host.step()
        host.step()
        assert np.abs(_feedback(client)["drive"]["demanded_torque_nm"][:n]).max() == 0.0
        client.close()
    finally:
        host.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_a_suspended_robot_hangs_by_its_torso_with_the_drives_off(tmp_path, robot):
    """The gantry holds the torso: with idle drives the legs hang off the waist and the torso stays as it was hung,
    the cable carrying the weight. Held by the pelvis instead, the G1's torso folds about 30 deg onto its waist
    stops (Controls' MuJoCo model of the G1, its pelvis welded, damped drives)."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost
    from humanoid_sim.world.supports import _quat_rotate

    host = RuntimeHost(tmp_path / "plant", built_usd(robot), device="cpu", robot=robot, support="suspended",
                       compile=False)
    try:
        host._reset({"pose": host.start_pose})  # no client: the drives stay off and nothing holds the pose
        host.step()  # hangs the harness
        torso = host.robot.body_names.index(load_profile(robot)["support"]["harness"]["body"])
        hung = host.robot.data.body_link_pose_w.torch[0, torso, 3:].clone()
        up = torch.tensor([0.0, 0.0, 1.0])
        up_in_torso = _quat_rotate(torch.cat([-hung[:3], hung[3:]]), up)  # world up, in the hung torso's frame
        tension = []
        for i in range(500):  # 2 s
            host.step()
            if i >= 400:
                tension.append(float(host.sim.gantry.tension_n[0]))
        now = host.robot.data.body_link_pose_w.torch[0, torso, 3:]
        tilt = math.degrees(math.acos(min(1.0, float(_quat_rotate(now, up_in_torso) @ up))))
        assert tilt < 15.0
        assert sum(tension) / len(tension) == pytest.approx(urdf_mass_kg(robot) * 9.81, rel=0.1)
    finally:
        host.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_a_client_cable_hangs_from_the_harness_hook(tmp_path, robot):
    """SET_SUPPORT's cable with ``attachment_bodies`` hangs from the midpoint of those bodies on the harness body, as
    the lab holds the robot, with its anchor ``overhead_distance_m`` straight above that point; the descriptor
    names the harness a client needs for it. The oracle is the bodies' own world positions at hang time."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import RuntimeHost

    harness = load_profile(robot)["support"]["harness"]
    host = RuntimeHost(tmp_path / "plant", built_usd(robot), device="cpu", robot=robot, support="suspended",
                       compile=False)
    try:
        assert host._descriptor("suspended")["harness"] == harness
        host._reset({"pose": host.start_pose})
        host.step()
        names = host.robot.body_names
        ids = [names.index(b) for b in harness["shoulders"]]
        hook = host.robot.data.body_link_pose_w.torch[0, ids, :3].mean(0).double()
        overhead = 1.0
        ack = host._set_support({"support": "gantry", "attachment_bodies": harness["shoulders"],
                                 "cable": {"overhead_distance_m": overhead, "length_breakpoints_s_m": [[0.0, 0.99]],
                                           "stiffness_n_m": 20000.0, "damping_n_s_m": 500.0}})
        assert ack == "gantry"
        for _ in range(2):  # a rope asked for between steps hangs after the next one
            host.step()
        top, point = host.sim.gantry.top_w[0].double(), host.sim.gantry.harness_w[0].double()
        assert torch.allclose(top, hook + torch.tensor([0.0, 0.0, overhead], dtype=torch.float64), atol=0.005)
        assert torch.allclose(point, hook, atol=0.005)
        # The truth carries one strap per shoulder, from its end of the bar (straight over it at hang time) to it.
        host._publish_truth()
        _, truth = host.seg.regions["truth"].read()
        shoulders = host.robot.data.body_link_pose_w.torch[0, ids, :3].double().numpy()
        assert np.allclose(truth["gantry_strap_bottom_m"], shoulders, atol=0.005)
        assert np.allclose(truth["gantry_strap_top_m"][:, :2], shoulders[:, :2], atol=0.005)
        assert np.allclose(truth["gantry_strap_top_m"][:, 2], float(hook[2]) + overhead, atol=0.005)
    finally:
        host.close()


@pytest.mark.parametrize("robot", PHYSICS_ROBOTS)
def test_set_support_sets_the_robot_down_and_back_on_the_harness(tmp_path, robot):
    """``free`` lets the robot down where it hangs: the start pose upright at standing height, its x, y and heading
    kept; a robot no rope holds stays as it is. ``suspended`` puts it back where a suspended session starts: the
    start pose at the spawn, lifted, the harness carrying its weight."""
    pytest.importorskip("isaaclab_newton")
    from humanoid_sim.runtime.host import SUSPENDED_LIFT_M, RuntimeHost
    from humanoid_sim.world.supports import NONE

    host = RuntimeHost(tmp_path / "plant", built_usd(robot), device="cpu", robot=robot, support="suspended",
                       compile=False)
    try:
        host.step()  # hangs the harness
        host._prescribe({"release": True})  # no client: else the start pose is held where it was
        yaw = 0.5
        moved = torch.tensor([[0.4, -0.2, 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
        moved[0, 2] = host.robot.data.root_link_pose_w.torch[0, 2]
        host.robot.write_root_pose_to_sim_index(root_pose=moved)
        host.step()
        standing = host._named_pose(host.start_pose, lift_m=0.0)[1][0, 2]
        assert host._set_support({"support": "free"}) == "free"
        assert int(host.sim.gantry.mode[0]) == NONE
        host.step()
        base = host.robot.data.root_link_pose_w.torch[0]
        assert torch.allclose(base[:2], moved[0, :2], atol=0.01)
        assert float(base[2]) == pytest.approx(float(standing), abs=0.01)
        assert torch.allclose(base[3:].abs(), moved[0, 3:].abs(), atol=0.01)

        for _ in range(5):
            host.step()
        before = host.robot.data.root_link_pose_w.torch[0, :3].clone()
        assert host._set_support({"support": "free"}) == "free"  # no rope: not moved
        assert torch.equal(host.robot.data.root_link_pose_w.torch[0, :3], before)

        assert host._set_support({"support": "suspended"}) == "suspended"
        host.step()  # the harness hangs after the step that follows the pose write
        base = host.robot.data.root_link_pose_w.torch[0]
        assert torch.allclose(base[:2], torch.zeros(2), atol=0.05)
        assert float(base[2]) == pytest.approx(float(standing) + SUSPENDED_LIFT_M, abs=0.05)
        tension = []
        for i in range(500):  # 2 s
            host.step()
            if i >= 400:
                tension.append(float(host.sim.gantry.tension_n[0]))
        assert sum(tension) / len(tension) == pytest.approx(urdf_mass_kg(robot) * 9.81, rel=0.1)
    finally:
        host.close()
