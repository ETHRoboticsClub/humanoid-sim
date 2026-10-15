"""The real-time host: one robot stepped against the wall clock, serving the runtime segment.

Every physics period (4 ms), in order:

1. handle a new mailbox command, if any, and acknowledge it;
2. take the client's latest demands (if they changed) into the drive bank, in force from this step on;
3. step the physics (unless paused), with the active gantry, disturbances and prescribed pose;
4. publish feedback (physical drive feedback, the latest raw IMU samples, lateness); every ``truth_every`` steps
   (and after a command) the truth; heartbeat.

Pacing is to ``CLOCK_MONOTONIC``: the host sleeps until each step's deadline. A step that starts more than a
period late counts as late and the schedule restarts from now (plant time never jumps). ``keeping_up`` is
true while no step in the last second was late.

Session: the client attaches by writing demands that carry the segment's ``session_id``. Once attached, demands
older than ``client_timeout_ns`` end the session (the segment's state becomes ended and :meth:`RuntimeHost.run`
returns :data:`EXIT_CLIENT_LOST`); Controls' bank ends its side when this host's heartbeat stops.

IMU sample times (``sample_ns``, ``stamp_ns``) are plant time; the feedback region's ``sim_time_ns`` and
``written_mono_ns`` map them to the host clock.
"""

from __future__ import annotations

import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.timing.clock import NS_PER_S
from humanoid_sim.world.supports import _quat_rotate

EXIT_DONE = 0  # max steps or a stop request
EXIT_CLIENT_LOST = 3

KEEPING_UP_WINDOW_NS = NS_PER_S
SPIN_NS = 300_000  # sleep until this close to a deadline, then spin

# What the runtime does not model (published in the descriptor).
FIDELITY_GAPS = [
    "drive current, thermal and backlash models absent; applied torque is the envelope-clipped control law",
    "upper-body passive joint parameters (waist, shoulders, elbows) are description values, not identified",
    "ankle as pitch and roll hinges driven through the crank transmission (no closed chain)",
]


@dataclass
class _Disturbance:
    body_id: int
    force: torch.Tensor  # [1, 1, 3] world
    torque: torch.Tensor
    end_ns: int


class RuntimeHost:
    def __init__(self, segment: str | Path, usd: Path, *, device: str = "cuda:0", robot: str = "ethrc_v1",
                 support: str = "suspended", bus_period_ns: int = 1_500_000, client_timeout_ns: int = 500_000_000,
                 seed: int = 0, sim_revision: str = "unknown", truth_every: int = 5, compile: bool = True):
        from humanoid_sim.model.asset import asset_meta
        from humanoid_sim.model.kinematics import standing_base_height
        from humanoid_sim.sim import SimCore
        from humanoid_sim.train.scene import STANDING_CLEARANCE_M

        if support not in ("suspended", "free"):
            raise ValueError(f"support {support!r}: start suspended or free (hang a gantry by command)")
        self.support = support
        from humanoid_sim.model.description import Description

        desc = Description.load(robot)
        meta = asset_meta(usd)
        self.standing_height_m = standing_base_height(usd, desc, meta["sole_boxes"]) + STANDING_CLEARANCE_M
        height = self.standing_height_m + (0.3 if support == "suspended" else 0.0)
        cuda = str(device).startswith("cuda")
        self.sim = SimCore(usd, device=device, robot=robot, fixed_base=support == "suspended",
                           base_height_m=height, imu=True, contacts=True, seed=seed, cuda_graph=cuda,
                           compile=compile and cuda)
        self.device = torch.device(device)
        self.desc = self.sim.desc
        self.robot = self.sim.backend.robot
        self.period_ns = self.sim.period_ns
        self.drives = self.sim.plant.map.drive_names
        self.joints = list(self.desc.joints)
        self.client_timeout_ns = client_timeout_ns
        self.truth_every = truth_every
        self._truth_due = True
        self._soles = self._sole_geometry(meta["sole_boxes"])
        contacts = self.sim.backend.contacts
        names = [n.rsplit("/", 1)[-1] for n in contacts.sensor_names]
        self._sole_contacts = torch.tensor([names.index(s["body"]) for s in self._soles], device=self.device)
        self._mass = self.robot.data.body_mass.torch[0]
        self.seg = abi.Segment.create(
            segment, robot_id=robot, description_digest=self.desc.digest, sim_revision=sim_revision,
            num_drives=len(self.drives), num_joints=len(self.joints), physics_period_ns=self.period_ns,
            bus_period_ns=bus_period_ns, descriptor=self._descriptor(support), ready=False)
        self.session_id = self.seg.session_id
        # State.
        self.step_count = 0
        self.late_steps = 0
        self._last_late_ns = -KEEPING_UP_WINDOW_NS
        self.paused = False
        self.attached = False
        self._client_mono_ns = 0
        self._demand_seq = 0
        self._command_seq = 0
        self._disturbances: list[_Disturbance] = []
        self._prescribed: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._faults: dict[tuple[str, int | None], tuple[int | None, object]] = {}
        self._stop = False
        self._warm_up()
        self._publish()
        self.seg.mark_ready()

    def _warm_up(self, steps: int = 3) -> None:
        """Step a few times and put the state back: the first steps compile kernels (seconds on a CPU), which must
        not happen while a client watches the heartbeat."""
        r = self.robot
        q, qd = self.sim.joint_state()
        q, qd = q.clone(), qd.clone()
        root, vel = r.data.root_link_pose_w.torch.clone(), r.data.root_link_vel_w.torch.clone()
        for _ in range(steps):
            self.sim.step()
        if self.support != "suspended":
            r.write_root_pose_to_sim_index(root_pose=root)
            r.write_root_velocity_to_sim_index(root_velocity=vel)
        self.sim.write_joint_state(q, qd)
        env = torch.tensor([0], device=self.device)
        self.sim.plant.reset(env)  # plant time keeps running: the IMU's sample clock depends on it
        self.sim.feedback = None
        self.sim.imu.reset(env, self.sim.t_ns)

    # -- descriptor and geometry ------------------------------------------------------------------

    def _descriptor(self, support: str) -> dict:
        phys = self.sim.profile["physics"]
        return {
            "robot": self.desc.robot, "drives": self.drives, "joints": self.joints, "support": support,
            "bodies": list(self.robot.body_names), "description": self.desc.identity(),
            "physics": {"engine": "isaaclab/mjwarp", "period_ns": self.period_ns, "integrator": phys["integrator"]},
            "units": {"drive": "physical drive-output frame (crank angle for ankle cranks), SI",
                      "joint": "URDF joint coordinates", "imu": "sensor frame, rad/s and m/s^2 (specific force)",
                      "time": "sim_time_ns and IMU stamps are plant time; *_mono_ns is CLOCK_MONOTONIC"},
            "imu": {"parent": self.desc.imu_mount()[0]},
            "soles": [s["body"] for s in self._soles],
            "commands": {str(k): v for k, v in abi.COMMANDS.items()}, "faults": list(abi.FAULTS),
            "fidelity_gaps": FIDELITY_GAPS,
        }

    def _sole_geometry(self, boxes: list[dict]) -> list[dict]:
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        out = []
        for side in ("l", "r"):
            box = next(b for b in boxes if b["side"] == side)
            corners = (torch.tensor(box["center"]) + (signs * torch.tensor(box["half_extents"]))
                       @ torch.tensor(box["rotation"]).T).to(torch.float32)
            out.append({"body": box["body"], "id": self.robot.body_names.index(box["body"]),
                        "corners": corners.to(self.device)})
        return out

    # -- commands ---------------------------------------------------------------------------------

    def _handle_command(self) -> None:
        got = self.seg.poll_command(self._command_seq)
        if got is None:
            return
        seq, command, args = got
        self._command_seq = seq
        if args is None or not isinstance(args, dict):
            self.seg.ack(seq, abi.ACK_REJECTED, "arguments are not a JSON object")
            return
        handler = {abi.CMD_RESET: self._reset, abi.CMD_PAUSE: self._pause, abi.CMD_RESUME: self._resume,
                   abi.CMD_SET_SUPPORT: self._set_support, abi.CMD_APPLY_DISTURBANCE: self._disturb,
                   abi.CMD_INJECT_FAULT: self._inject, abi.CMD_PRESCRIBE_POSE: self._prescribe}.get(command)
        if handler is None:
            self.seg.ack(seq, abi.ACK_UNKNOWN, f"unknown command {command}")
            return
        try:
            message = handler(args) or ""
        except (KeyError, TypeError, ValueError) as e:
            self.seg.ack(seq, abi.ACK_REJECTED, f"{type(e).__name__}: {e}")
            return
        except RuntimeError as e:
            self.seg.ack(seq, abi.ACK_FAILED, str(e))
            return
        self.seg.ack(seq, abi.ACK_OK, message)
        self._truth_due = True

    def _joint_vector(self, joint_pos: dict) -> torch.Tensor:
        q, _ = self.sim.joint_state()
        q = q.clone()
        for name, value in joint_pos.items():
            if name not in self.joints:
                raise ValueError(f"unknown joint {name!r}")
            lo, hi = self.desc.calibration_travel(name)
            if not (math.isfinite(float(value)) and lo <= float(value) <= hi):
                raise ValueError(f"{name}: {value} outside its travel [{lo:.3f}, {hi:.3f}]")
            q[0, self.joints.index(name)] = float(value)
        return q

    def _base_pose(self, pose) -> torch.Tensor:
        if self.support == "suspended":
            raise ValueError("the base is welded to the world in a suspended session")
        p = torch.tensor([pose], dtype=torch.float32, device=self.device)
        if p.shape != (1, 7) or not torch.isfinite(p).all() or abs(float(p[0, 3:].norm()) - 1) > 1e-4:
            raise ValueError("base_pose is [x, y, z, qx, qy, qz, qw] with a unit quaternion")
        return p

    def _standing(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = torch.tensor([[self.desc.joints[n].standing_offset_rad for n in self.joints]], device=self.device)
        base = None
        if self.support != "suspended":
            base = torch.tensor([[0.0, 0.0, self.standing_height_m, 0.0, 0.0, 0.0, 1.0]], device=self.device)
        return q, base

    def _reset(self, args: dict) -> str:
        if "joint_pos" in args:
            q, base = self._joint_vector(args["joint_pos"]), None
            if args.get("base_pose") is not None:
                base = self._base_pose(args["base_pose"])
        elif args.get("pose", "standing") == "standing":
            q, base = self._standing()
        elif args["pose"] == "current":
            q, base = self.sim.joint_state()[0], None
        else:
            raise ValueError(f"unknown pose {args['pose']!r}")
        self._write_pose(q, base)
        env = torch.tensor([0], device=self.device)
        self.sim.plant.reset(env)
        self.sim.imu.reset(env, self.sim.t_ns)
        self._prescribed = None
        self._disturbances.clear()
        self._demand_seq = 0  # the client's next demands take effect again
        return "reset"

    def _write_pose(self, q: torch.Tensor, base: torch.Tensor | None) -> None:
        if base is not None:
            self.robot.write_root_pose_to_sim_index(root_pose=base)
            self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device))
        self.sim.write_joint_state(q, torch.zeros_like(q))

    def _pause(self, args: dict) -> str:
        self.paused = True
        return "paused"

    def _resume(self, args: dict) -> str:
        self.paused = False
        return "resumed"

    def _set_support(self, args: dict) -> str:
        kind = args["support"]
        env = torch.tensor([0], device=self.device)
        if kind == "free":
            if self.support == "suspended":
                raise ValueError("a suspended session cannot release its base; restart the host with free support")
            self.sim.gantry.release(env)
            return "free"
        if kind == "suspended":
            if self.support != "suspended":
                raise ValueError("suspension welds the base at startup; restart the host with suspended support")
            return "suspended"
        if kind != "gantry":
            raise ValueError(f"unknown support {kind!r}")
        if self.support == "suspended":
            raise ValueError("the base is welded to the world in a suspended session")
        if "cable" in args:
            c = args["cable"]
            self.sim.gantry.hang_cable(env, self.sim.t_ns, attachment_b=args["attachment_local_m"],
                                       overhead_m=float(c["overhead_distance_m"]),
                                       length_breakpoints_s_m=c["length_breakpoints_s_m"],
                                       stiffness_n_m=float(c["stiffness_n_m"]), damping_n_s_m=float(c["damping_n_s_m"]))
        else:
            self.sim.gantry.hang_tension(env, self.sim.t_ns, args["tension_breakpoints_s_n"],
                                         attachment_b=args.get("attachment_local_m"),
                                         attachment_bodies=args.get("attachment_bodies"))
        return "gantry"

    def _disturb(self, args: dict) -> str:
        body = self.robot.body_names.index(args["body"])
        force = torch.tensor(args.get("force_n", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        torque = torch.tensor(args.get("torque_nm", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        duration = float(args["duration_s"])
        if force.shape != (3,) or torque.shape != (3,) or not (duration > 0):
            raise ValueError("force_n and torque_nm are 3-vectors, duration_s positive")
        end = self.sim.t_ns + int(round(duration * NS_PER_S))
        self._disturbances.append(_Disturbance(body, force.view(1, 1, 3), torque.view(1, 1, 3), end))
        return f"until {end} ns"

    def _inject(self, args: dict) -> str:
        fault = args["fault"]
        if fault not in abi.FAULTS:
            raise ValueError(f"unknown fault {fault!r}; known: {list(abi.FAULTS)}")
        value = float(args.get("value", 1.0))
        drive = args.get("drive")
        d = None
        if fault == "encoder_freeze":
            if drive not in self.drives:
                raise ValueError(f"encoder_freeze needs a drive; known: {self.drives}")
            d = self.drives.index(drive)
        duration = args.get("duration_s")
        end = None if duration is None else self.sim.t_ns + int(round(float(duration) * NS_PER_S))
        key = (fault, d)
        self._clear_fault(key)
        if value == (1.0 if fault == "power_sag" else 0.0):  # the value that means no fault
            return "cleared"
        if fault == "encoder_freeze":
            fb = self.sim.feedback
            held = (fb.position_rad[0, d].item(), fb.velocity_rad_s[0, d].item()) if fb is not None else (0.0, 0.0)
            self._faults[key] = (end, held)
        elif fault == "imu_bias":
            self.sim.imu.gyro_bias += value
            self._faults[key] = (end, value)
        elif fault == "imu_spike":
            self.sim.imu.accel_bias += value
            self._faults[key] = (end if end is not None else self.sim.t_ns + self.period_ns, value)
        elif fault == "power_sag":
            if not 0.0 <= value <= 1.0:
                raise ValueError("power_sag value is the torque scale, in [0, 1]")
            self.sim.plant.bank.torque_scale.fill_(value)
            self._faults[key] = (end, value)
        return f"{fault} active"

    def _clear_fault(self, key) -> None:
        got = self._faults.pop(key, None)
        if got is None:
            return
        fault, value = key[0], got[1]
        if fault == "imu_bias":
            self.sim.imu.gyro_bias -= value
        elif fault == "imu_spike":
            self.sim.imu.accel_bias -= value
        elif fault == "power_sag":
            self.sim.plant.bank.torque_scale.fill_(1.0)

    def _prescribe(self, args: dict) -> str:
        if args.get("release"):
            self._prescribed = None
            return "released"
        q = self._joint_vector(args.get("joint_pos", {}))
        base = self._base_pose(args["base_pose"]) if args.get("base_pose") is not None else None
        self._prescribed = (q, base)
        self._write_pose(q, base)
        return "prescribed"

    # -- demands ----------------------------------------------------------------------------------

    def _take_demands(self) -> None:
        got = self.seg.regions["demands"].read()
        if got is None or got[0] == self._demand_seq:
            return
        seq, rec = got
        if int(rec["session_id"]) != self.session_id:
            return  # a client of an earlier session
        self._demand_seq = seq
        self.attached = True
        self._client_mono_ns = int(rec["written_mono_ns"])
        n = len(self.drives)
        d = rec["drive"][:n]
        dev, dt = self.device, self.sim.plant.cfg.dtype

        def t(name, dtype):
            a = np.ascontiguousarray(d[name])
            a = a.astype(np.int64) if a.dtype.kind in "iu" else a.astype(np.float64)
            return torch.from_numpy(a).to(device=dev, dtype=dtype).view(1, n)

        demand = DriveDemand(torque_enabled=t("torque_enabled", torch.bool), mode=t("mode", torch.int8),
                             target_position_rad=t("target_position_rad", dt),
                             target_velocity_rad_s=t("target_velocity_rad_s", dt),
                             feedforward_torque_nm=t("feedforward_torque_nm", dt),
                             p_gain_raw=t("p_gain_raw", torch.int64), i_gain_raw=t("i_gain_raw", torch.int64),
                             d_gain_raw=t("d_gain_raw", torch.int64), max_torque_mnm=t("max_torque_mnm", torch.int64),
                             homing_hold=t("homing_hold", torch.bool))
        self.sim.plant.submit(demand, torch.full((1,), self.sim.t_ns, dtype=torch.int64, device=dev))

    def _client_lost(self) -> bool:
        return self.attached and abi.mono_ns() - self._client_mono_ns > self.client_timeout_ns

    # -- stepping and publishing ------------------------------------------------------------------

    def _before_physics(self) -> None:
        if self._prescribed is not None:
            self._write_pose(*self._prescribed)
        t = self.sim.t_ns
        self._disturbances = [x for x in self._disturbances if x.end_ns > t]
        composer = self.robot.instantaneous_wrench_composer
        for x in self._disturbances:
            composer.add_forces_and_torques_index(forces=x.force, torques=x.torque, body_ids=[x.body_id],
                                                  env_ids=torch.tensor([0], device=self.device), is_global=True)
        for key, (end, _) in list(self._faults.items()):
            if end is not None and t >= end:
                self._clear_fault(key)

    def step(self) -> None:
        """One period: command, demands, physics, publish (no pacing)."""
        self._handle_command()
        self._take_demands()
        if not self.paused:
            self._before_physics()
            self.sim.step()
            self.step_count += 1
        self._publish()

    def _publish(self) -> None:
        """Feedback every step; truth every ``truth_every`` steps (its ``sim_time_ns`` says when)."""
        self._publish_feedback()
        if self.step_count % self.truth_every == 0 or self._truth_due:
            self._publish_truth()
            self._truth_due = False
        self.seg.heartbeat()

    def _publish_feedback(self) -> None:
        sim, n = self.sim, len(self.drives)
        fb = sim.feedback if sim.feedback is not None else sim.plant.feedback(*sim.joint_state())
        imu = sim.imu.samples
        h = imu.gyro.shape[1]
        # One transfer: the integer fields travel bit-for-bit as float64.
        ints = torch.cat([imu.seq[0], imu.sample_ns[0], imu.stamp_ns[0], imu.count[:1]]).view(torch.float64)
        flat = torch.cat([fb.position_rad[0], fb.velocity_rad_s[0], fb.applied_torque_nm[0],
                          fb.demanded_torque_nm[0], imu.gyro[0].reshape(-1), imu.specific_force[0].reshape(-1),
                          ]).to(torch.float64)
        host = torch.cat([flat, ints]).cpu().numpy()
        drive = host[: 4 * n].reshape(4, n)
        for (position, velocity), d in self._frozen_encoders():
            drive[0, d], drive[1, d] = position, velocity
        gyro = host[4 * n: 4 * n + 3 * h].reshape(h, 3)
        accel = host[4 * n + 3 * h: 4 * n + 6 * h].reshape(h, 3)
        ints = host[4 * n + 6 * h:].view(np.int64)
        seqs, sample_ns, stamp_ns, total = ints[:h], ints[h:2 * h], ints[2 * h:3 * h], int(ints[3 * h])

        now = abi.mono_ns()
        rec = np.zeros((), abi.FEEDBACK)
        rec["step"], rec["sim_time_ns"], rec["written_mono_ns"] = self.step_count, sim.t_ns, now
        rec["late_steps"], rec["paused"], rec["num_drives"] = self.late_steps, self.paused, n
        rec["keeping_up"] = self._keeping_up(now)
        rec["imu_total"] = total
        valid = min(total, h, abi.IMU_SAMPLES)
        rec["imu_count"] = valid
        for k, name in enumerate(abi.DRIVE_FEEDBACK.names):
            rec["drive"][name][:n] = drive[k]
        imu_rec = rec["imu"]
        imu_rec["seq"][:valid], imu_rec["sample_ns"][:valid], imu_rec["stamp_ns"][:valid] = \
            seqs[:valid], sample_ns[:valid], stamp_ns[:valid]
        imu_rec["gyro_rad_s"][:valid], imu_rec["specific_force_m_s2"][:valid] = gyro[:valid], accel[:valid]
        self.seg.regions["feedback"].write(rec)

    def _publish_truth(self) -> None:
        sim, nj = self.sim, len(self.joints)
        q, qd = sim.joint_state()
        data = self.robot.data
        root = data.root_link_pose_w.torch[0]
        vel = data.root_link_vel_w.torch[0]
        com = (self._mass[:, None] * data.body_com_pos_w.torch[0]).sum(0) / self._mass.sum()
        soles = []
        normal = self.sim.backend.contacts.data.net_forces_w.torch[0, self._sole_contacts, 2]
        for k, s in enumerate(self._soles):
            pose = data.body_link_pose_w.torch[0, s["id"]]
            corners = pose[:3] + _quat_rotate(pose[3:].expand(8, 4), s["corners"])
            soles.append(torch.cat([pose, corners[:, 2].min().view(1), normal[k].view(1)]))
        flat = torch.cat([root, vel, com, q[0], qd[0], sim.gantry.tension_n[:1], self._mass.sum().view(1),
                          *soles]).to(torch.float64).cpu().numpy()
        i = 0

        def take(k):
            nonlocal i
            out = flat[i:i + k]
            i += k
            return out

        root_np, vel_np, com_np, q_np, qd_np = take(7), take(6), take(3), take(nj), take(nj)
        tension, mass = take(1)[0], take(1)[0]
        t = np.zeros((), abi.TRUTH)
        t["sim_time_ns"] = sim.t_ns
        t["base_position_m"], t["base_quat_xyzw"] = root_np[:3], root_np[3:]
        t["base_lin_vel_m_s"], t["base_ang_vel_rad_s"], t["com_m"] = vel_np[:3], vel_np[3:], com_np
        t["gantry_tension_n"], t["mass_kg"], t["num_joints"] = tension, mass, nj
        t["joint_pos_rad"][:nj], t["joint_vel_rad_s"][:nj] = q_np, qd_np
        for k in range(len(soles)):
            s = take(9)
            t["sole"]["position_m"][k], t["sole"]["quat_xyzw"][k] = s[:3], s[3:7]
            t["sole"]["lowest_z_m"][k], t["sole"]["normal_force_n"][k] = s[7], s[8]
        self.seg.regions["truth"].write(t)

    def _frozen_encoders(self):
        for (fault, d), (_, held) in self._faults.items():
            if fault == "encoder_freeze":
                yield held, d

    def _keeping_up(self, now: int) -> bool:
        return now - self._last_late_ns > KEEPING_UP_WINDOW_NS

    # -- real-time loop ---------------------------------------------------------------------------

    def stop(self, *_) -> None:
        self._stop = True

    def run(self, max_steps: int | None = None) -> int:
        """Serve in real time until ``max_steps`` periods, a stop request, or the client is lost."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.stop)
            except ValueError:
                pass  # not the main thread
        code = EXIT_DONE
        deadline = abi.mono_ns()
        periods = 0
        try:
            while not self._stop and (max_steps is None or periods < max_steps):
                now = abi.mono_ns()
                if now < deadline:
                    if deadline - now > SPIN_NS:
                        time.sleep((deadline - now - SPIN_NS) / NS_PER_S)
                    while abi.mono_ns() < deadline:
                        pass
                elif now - deadline > self.period_ns:
                    self.late_steps += 1
                    self._last_late_ns = now
                    deadline = now
                self.step()
                periods += 1
                if self._client_lost():
                    code = EXIT_CLIENT_LOST
                    break
                deadline += self.period_ns
        finally:
            self.seg.end()
        return code

    def close(self) -> None:
        self.seg.end()
        self.seg.close()
        self.sim.close()

    def status(self) -> dict:
        return {"session_id": self.session_id, "steps": self.step_count, "late_steps": self.late_steps,
                "sim_time_s": self.sim.t_ns / NS_PER_S, "attached": self.attached}

