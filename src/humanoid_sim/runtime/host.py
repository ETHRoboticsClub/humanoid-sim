"""The real-time host: one robot stepped against the wall clock, serving the runtime segment.

Every physics period (4 ms), in order:

1. handle a new mailbox command, if any, and acknowledge it;
2. take every demand record the client appended since the last step into the drive bank, each sent at its own
   time (its ``sent_mono_ns`` in plant time);
3. step the physics (unless paused), with the active gantry, disturbances and prescribed pose;
4. publish feedback (physical drive feedback, the latest raw IMU samples, lateness); every ``truth_every`` steps
   (and after a command) the truth; heartbeat.

Pacing is to ``CLOCK_MONOTONIC``: the host sleeps until each step's deadline. A step that starts more than a
period late counts as late and the schedule restarts from now (plant time never jumps). ``keeping_up`` is
true while no step in the last second was late.

Session: the client attaches by writing demands that carry the segment's ``session_id``. A DETACH command ends
the session (:meth:`RuntimeHost.run` returns :data:`EXIT_DETACHED`), as do demands older than
``client_timeout_ns`` once attached (:data:`EXIT_CLIENT_LOST`); the segment's state becomes ended, and Controls'
bank ends its side when this host's heartbeat stops. :meth:`RuntimeHost.new_session` then serves the next client
on a new segment at the same path, the robot back in its start state, without recompiling the step. Until its
client attaches, a session holds the robot in its start pose (the profile's ``poses.start`` unless ``pose`` names
another), so the session starts when the client does, as a plant started with its client would: a robot left
hanging with idle drives between clients would sag to its rest pose first.

A suspended session hangs the robot from the lab gantry, its soles ``SUSPENDED_LIFT_M`` above the ground: one cable
from the harness hook on the torso (the profile's ``support``) to an anchor overhead, carrying the robot's weight at
its full-weight stretch, so the waist carries the hanging legs as on the lab's gantry. A reset to a named pose puts
the robot back at that height; every reset rehangs the cable where the robot then is. A rope requested after a pose
write is checked at once and hung after the next step, since body poses update only in a step. The descriptor's
``support`` is the support the session started in.

A disabled drive is back-driven through its gearbox: while a 1:1 drive's law is DISABLED, its joint's Coulomb
friction is the family's ``backdrive_torque`` (the joint's own friction when the family has none) instead of the
enabled-drive friction, as in Controls' MuJoCo plant (the ankle joints keep theirs: the cranks have no joint-space
figure). The friction is written when a drive turns on or off, not every step.

With ``scene`` (:class:`~humanoid_sim.world.boxes.StaticBoxes`) the robot stands among static boxes, a room's walls
and furniture, starting at the scene's spawn point (the world origin); the descriptor's ``scene`` names the file,
its digest and the spawn point, so the client can map the truth's world frame to the scene's. A spawn point inside
a box is refused.

IMU sample times (``sample_ns``, ``stamp_ns``) are plant time; the feedback region's ``sim_time_ns`` and
``written_mono_ns`` map them to the host clock.
"""

from __future__ import annotations

import math
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from humanoid_sim.devices.demand import LAW_FREEZE, DriveDemand
from humanoid_sim.runtime.abi import plant_abi as abi
from humanoid_sim.timing.clock import NS_PER_S
from humanoid_sim.world.boxes import StaticBoxes
from humanoid_sim.world.supports import NONE, _quat_rotate

EXIT_DONE = 0  # max steps or a stop request
EXIT_CLIENT_LOST = 3
EXIT_DETACHED = 4


KEEPING_UP_WINDOW_NS = NS_PER_S
SPIN_NS = 100_000  # sleep until this close to a deadline, then spin
SUSPENDED_LIFT_M = 0.3  # a suspended robot's soles above the ground
GRAVITY_M_S2 = 9.81  # Isaac Lab's SimulationCfg default; humanoid-sim never sets another

# What the runtime does not model (published in the descriptor).
FIDELITY_GAPS = [
    "drive current, thermal and backlash models absent; applied torque is the envelope-clipped control law",
    "upper-body passive joint parameters (waist, shoulders, elbows) are description values, not identified",
    "ankle as pitch and roll hinges driven through the crank transmission (no closed chain)",
    "IMU errors from the rig's model: white noise, turn-on bias, Gauss-Markov drift (SET_FIDELITY ideal turns them "
    "off); the turn-on bias is redrawn at every IMU reset, not only at power-on",
    "drive delay is each family's fixed figure (actuators.yaml drive_delay_s), without per-cycle jitter",
]


def fidelity_gaps(models: dict) -> list[str]:
    """The runtime's gaps for drives with these models: :data:`FIDELITY_GAPS` and the families whose delay the
    description does not record."""
    unmeasured = sorted({m.family for m in models.values() if not m.delay_measured})
    if not unmeasured:
        return list(FIDELITY_GAPS)
    return FIDELITY_GAPS + [f"drive delay unmeasured for {', '.join(unmeasured)}: their demands act when the bus "
                            "delivers them"]


class _DemandIntake:
    """The client's demands into the drive bank. A record's fields are packed into one pinned host buffer and
    copied to the device in one transfer; unpacking and queueing run there, as a CUDA graph on CUDA (otherwise a
    new demand every bus cycle costs more than the physics). Each drive's arrival time travels in the same
    buffer."""

    FIELDS = ("law", "target_position_rad", "target_velocity_rad_s", "feedforward_torque_nm", "stiffness_nm_per_rad",
              "damping_nm_s_per_rad", "torque_limit_nm")

    def __init__(self, plant, num_drives: int, device: torch.device, *, cuda_graph: bool, compile: bool,
                 warmup: int = 3):
        self.plant, self.n = plant, num_drives
        cuda = device.type == "cuda"
        # One host buffer per record a step can take: a buffer is written again only after the step's feedback
        # came back, and by then its copy, queued before, is done.
        self.hosts = torch.zeros(abi.DEMAND_RING, (len(self.FIELDS) + 1) * num_drives, dtype=torch.float64,
                                 pin_memory=cuda)
        self.hosts_np = self.hosts.numpy()
        self._next = 0
        self.dev = torch.zeros_like(self.hosts[0], device=device)
        self._submit = torch.compile(self._unpack_and_submit, dynamic=False) if compile else self._unpack_and_submit
        self.graph = None
        if cuda_graph:  # the warm-up queues disabled demands: reset the plant afterwards
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    self._submit()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._submit()

    def _unpack_and_submit(self) -> None:
        f = self.dev[:-self.n].view(len(self.FIELDS), 1, self.n)
        dt = self.plant.cfg.dtype
        law = f[0].to(torch.int16)  # u1 on the wire: 0 to 255
        law = torch.where(law <= LAW_FREEZE, law, torch.zeros_like(law)).to(torch.int8)  # unknown: DISABLED
        demand = DriveDemand(law=law, target_position_rad=f[1].to(dt), target_velocity_rad_s=f[2].to(dt),
                             feedforward_torque_nm=f[3].to(dt), stiffness_nm_per_rad=f[4].to(dt),
                             damping_nm_s_per_rad=f[5].to(dt), torque_limit_nm=f[6].to(dt))
        self.plant.submit(demand, self.dev[-self.n:].view(torch.int64).view(1, self.n))

    def __call__(self, drive: np.ndarray, arrival_ns: np.ndarray) -> None:
        """Queue the record's per-drive demands (``drive``: the first ``num_drives`` entries), drive ``d``'s from
        ``arrival_ns[d]``."""
        n, slot = self.n, self._next
        self._next = (slot + 1) % abi.DEMAND_RING
        host = self.hosts_np[slot]
        for k, name in enumerate(self.FIELDS):
            host[k * n:(k + 1) * n] = drive[name]
        host[-n:].view(np.int64)[:] = arrival_ns
        self.dev.copy_(self.hosts[slot], non_blocking=True)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._submit()


def _queue_capacity(max_delay_ns: int, bus_period_ns: int, period_ns: int) -> int:
    """Pending demand records per drive bank: every bus cycle's record stays queued until it has arrived on every
    drive, up to the longest delay after a send that can fall a step late."""
    return math.ceil((max_delay_ns + 2 * period_ns) / bus_period_ns) + 1


@dataclass
class _Disturbance:
    body_id: int
    force: torch.Tensor  # [1, 1, 3] world
    torque: torch.Tensor
    end_ns: int


def imu_descriptor(desc) -> dict:
    """The descriptor's ``imu``: the link ``imu_joint`` mounts it on, and the rig's IMU (its model and product, the
    link it sits at, its sample period and delivery to the host), so a client can tell which device it reads."""
    from humanoid_sim.sensors import rigs

    imu = rigs.imu(desc)
    return {"parent": desc.imu_mount()[0], "model": imu.model, "product": imu.product, "frame": imu.frame,
            "sample_period_ns": imu.sample_period_ns, "burst_period_ns": list(imu.burst_period_ns),
            "in_burst_spacing_ns": imu.in_burst_spacing_ns}


class RuntimeHost:
    def __init__(self, segment: str | Path, usd: Path, *, device: str = "cuda:0", robot: str = "ethrc_v1",
                 solver: str | None = None, support: str = "suspended", pose: str | None = None,
                 bus_period_ns: int = 1_500_000,
                 client_timeout_ns: int = 500_000_000,
                 seed: int = 0, sim_revision: str = "unknown", truth_every: int = 5, compile: bool = True,
                 scene: StaticBoxes | None = None):
        from humanoid_sim.devices.models import drive_models
        from humanoid_sim.model.asset import asset_meta, load_profile
        from humanoid_sim.model.description import Description
        from humanoid_sim.model.poses import POSES, STANDING_CLEARANCE_M, initial_pose, start_pose
        from humanoid_sim.sim import SimCore

        if support not in ("suspended", "free"):
            raise ValueError(f"support {support!r}: start suspended or free (hang a gantry by command)")
        self._support_cfg = load_profile(robot)["support"]
        if support == "suspended" and self._support_cfg["harness"] is None:
            raise ValueError(f"{robot}: its profile names no support.harness to suspend it by")
        if scene is not None and scene.clearance_m() == 0.0:
            raise ValueError(f"spawn point {scene.spawn_m} is inside a box of {scene.file}: the robot starts there")
        self.scene = scene
        self.support = support
        meta = asset_meta(usd)
        if pose is not None and pose not in POSES:
            raise ValueError(f"pose {pose!r}: one of {POSES}")
        self.start_pose = pose or start_pose(load_profile(robot))
        # The named poses for RESET {"pose": ...}: joint positions (description order) and the upright base height
        # at which the soles touch the ground. Every session starts in start_pose.
        self._poses = {name: initial_pose(usd, robot, name) for name in POSES}
        height = self._poses[self.start_pose][1] + STANDING_CLEARANCE_M + self._lift_m()
        cuda = str(device).startswith("cuda")
        models = drive_models(Description.load(robot))
        self.sim = SimCore(usd, device=device, robot=robot, solver=solver,
                           base_height_m=height, imu=True, contacts=True, seed=seed, cuda_graph=cuda,
                           compile=compile and cuda, imu_history=abi.IMU_SAMPLES, scene=scene,
                           queue_capacity=_queue_capacity(max(m.drive_delay_ns for m in models.values()),
                                                          bus_period_ns,
                                                          int(load_profile(robot)["physics"]["period_ns"])))
        self.device = torch.device(device)
        self._intake = _DemandIntake(self.sim.plant, len(self.sim.plant.map.drive_names), self.device,
                                     cuda_graph=cuda, compile=compile and cuda)
        self.desc = self.sim.desc
        self.robot = self.sim.backend.robot
        self.period_ns = self.sim.period_ns
        self.drives = self.sim.plant.map.drive_names
        self.drive_models = {name: models[name] for name in self.drives}
        # Each drive applies a bus cycle's demand its model's delay after the bus delivered it.
        self._drive_delays_ns = np.array([m.drive_delay_ns for m in self.drive_models.values()], dtype=np.int64)
        self.joints = list(self.desc.joints)
        self.client_timeout_ns = client_timeout_ns
        self.truth_every = truth_every
        self._soles = self._sole_geometry(meta["sole_boxes"])
        contacts = self.sim.backend.contacts
        names = [n.rsplit("/", 1)[-1] for n in contacts.sensor_names]
        self._sole_contacts = torch.tensor([names.index(s["body"]) for s in self._soles], device=self.device)
        self._mass = self.robot.data.body_mass.torch[0]
        # Back-drive friction: the 1:1 drives' joints, their friction enabled and disabled.
        one_to_one = [(k, self.joints.index(d.joints[0])) for k, d in
                      enumerate(self.desc.drives[n] for n in self.drives) if d.crank is None]
        self._backdrive_drives = np.array([k for k, _ in one_to_one])
        self._backdrive_joint_ids = [self.robot.joint_names.index(self.joints[j]) for _, j in one_to_one]
        self._friction_on = torch.tensor([[self.desc.joints[self.joints[j]].friction for _, j in one_to_one]],
                                         device=self.device)
        # A family without a recorded backdrive_torque keeps the enabled friction.
        self._friction_off = torch.tensor([[float(self.desc.family_value(
            self.desc.drives[self.drives[k]].family, "backdrive_torque",
            self.desc.joints[self.joints[j]].friction)) for k, j in one_to_one]], device=self.device)
        self._drive_off: np.ndarray | None = None
        # Device-to-host copies: into pinned buffers, waited for on a blocking event (the host thread sleeps
        # instead of spinning while the GPU finishes: it shares the machine with Controls).
        self._pinned: dict[str, torch.Tensor] = {}
        self._copied = torch.cuda.Event(blocking=True) if self.device.type == "cuda" else None
        self._segment_args = dict(
            path=segment, robot_id=robot, description_digest=self.desc.digest, sim_revision=sim_revision,
            num_drives=len(self.drives), num_joints=len(self.joints), physics_period_ns=self.period_ns,
            bus_period_ns=bus_period_ns, descriptor=self._descriptor(support), ready=False)
        self.seg = None
        self._faults: dict[tuple[str, int | None], tuple[int | None, object]] = {}
        self._disturbances: list[_Disturbance] = []
        self._joint_torques: dict[int, tuple[float, float, int | None]] = {}  # joint -> (Nm, Hz, end ns)
        self._joint_torque_active = False
        self._stop = False
        self._poses_stale = False  # the pose was written since the last step: body poses are the old ones
        self._rope_due: Callable[[], None] | None = None  # a checked rope, hung after the next step
        self._harness_hung = False  # the rope hung or due is the harness cable
        self._warm_up()
        self._open_session()

    def _open_session(self) -> None:
        """A new segment (new session id) at the path, published before it is marked ready."""
        self.seg = abi.Segment.create(**self._segment_args)
        self.session_id = self.seg.session_id
        self.step_count = 0
        self.late_steps = 0
        self._last_late_ns = -KEEPING_UP_WINDOW_NS
        self.paused = False
        self.attached = False
        self.detached = False
        self._client_mono_ns = 0
        self._demand_seq = 0
        self._demands_taken = 0
        self._demand_overruns = 0
        self._command_seq = 0
        # Held until the client attaches.
        self._prescribed: tuple[torch.Tensor, torch.Tensor | None] | None = self._named_pose(self.start_pose)
        self._write_pose(*self._prescribed)
        if self.support == "suspended":
            self._hang_harness()
        self._set_drives_off(np.ones(len(self.drives), dtype=bool))  # no client, no demands
        self._truth_due = True
        self._publish()
        self.seg.mark_ready()

    def new_session(self) -> None:
        """End the current session and serve the next client: the robot back in its start pose with the drives
        idle, gantry, disturbances, faults and pause cleared, on a new segment at the same path. Plant time runs
        on."""
        self.seg.end()
        self.seg.close()
        for key in list(self._faults):
            self._clear_fault(key)
        self._release_rope()
        self.sim.imu.set_noise(True)
        self._reset({"pose": self.start_pose})
        self._open_session()

    def _warm_up(self, steps: int = 3) -> None:
        """Step a few times and put the state back: the first steps compile kernels (seconds on a CPU), which must
        not happen while a client watches the heartbeat. The harness cable hangs for the later steps, so the rope's
        force path compiles too."""
        r = self.robot
        q, qd = self.sim.joint_state()
        q, qd = q.clone(), qd.clone()
        root, vel = r.data.root_link_pose_w.torch.clone(), r.data.root_link_vel_w.torch.clone()
        for i in range(steps):
            self.sim.step()
            if i == 0 and self._support_cfg["harness"] is not None:
                self._harness_cable(dry_run=False)
        self.sim.gantry.release(torch.tensor([0], device=self.device))
        r.write_root_pose_to_sim_index(root_pose=root)
        r.write_root_velocity_to_sim_index(root_velocity=vel)
        self.sim.write_joint_state(q, qd)
        env = torch.tensor([0], device=self.device)
        self.sim.plant.reset(env)  # plant time keeps running: the IMU's sample clock depends on it
        self.sim.feedback = None
        self.sim.imu.reset(env, self.sim.t_ns, keep_seq=True)

    # -- descriptor and geometry ------------------------------------------------------------------

    def _descriptor(self, support: str) -> dict:
        return {
            "robot": self.desc.robot, "drives": self.drives, "joints": self.joints, "support": support,
            "bodies": list(self.robot.body_names), "description": self.desc.identity(),
            "physics": {"engine": f"isaaclab/{self.sim.solver}", "period_ns": self.period_ns,
                        "integrator": self.sim.integrator},
            "units": {"drive": "physical drive-output frame (crank angle for ankle cranks), SI",
                      "joint": "URDF joint coordinates", "imu": "sensor frame, rad/s and m/s^2 (specific force)",
                      "time": "sim_time_ns and IMU stamps are plant time; *_mono_ns is CLOCK_MONOTONIC"},
            "imu": imu_descriptor(self.desc),
            "soles": [s["body"] for s in self._soles],
            "commands": {str(k): v for k, v in abi.COMMANDS.items()}, "faults": list(abi.FAULTS),
            "drive_models": self._drive_model_table(),
            "fidelity_gaps": fidelity_gaps(self.drive_models),
            "scene": None if self.scene is None else self.scene.describe(),
        }

    def _drive_model_table(self) -> list[dict]:
        """Each drive model with the drives it is, for the descriptor."""
        table: dict[str, dict] = {}
        for name, model in self.drive_models.items():
            table.setdefault(model.family, model.describe() | {"drives": []})["drives"].append(name)
        return list(table.values())

    def _sole_geometry(self, boxes: list[dict]) -> list[dict]:
        """Each sole plate in its body's frame: corners, bottom-face corners, axes and half extents."""
        signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
        face = torch.tensor([[x, y, -1] for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))], dtype=torch.float32)
        out = []
        for side in ("l", "r"):
            box = next(b for b in boxes if b["side"] == side)
            center, half = torch.tensor(box["center"]), torch.tensor(box["half_extents"])
            rotation = torch.tensor(box["rotation"], dtype=torch.float32)
            out.append({"body": box["body"], "id": self.robot.body_names.index(box["body"]),
                        "corners": (center + (signs * half) @ rotation.T).to(self.device, torch.float32),
                        "face": (center + (face * half) @ rotation.T).to(self.device, torch.float32),
                        "axes": rotation.T.contiguous().to(self.device),  # rows: the plate's x, y, z axes
                        "half": half.to(self.device, torch.float32)})
        return out

    def _lowest_sole_z(self) -> float:
        pose = self.robot.data.body_link_pose_w.torch[0]
        lowest = [(pose[s["id"], :3] + _quat_rotate(pose[s["id"], 3:].expand(8, 4), s["corners"]))[:, 2].min()
                  for s in self._soles]
        return float(torch.stack(lowest).min())

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
                   abi.CMD_INJECT_FAULT: self._inject, abi.CMD_PRESCRIBE_POSE: self._prescribe,
                   abi.CMD_DETACH: self._detach, abi.CMD_SET_FIDELITY: self._set_fidelity}.get(command)
        if handler is None:
            self.seg.ack(seq, abi.ACK_UNKNOWN, f"unknown command {command}")
            return
        try:
            message = handler(args) or ""
        except (KeyError, TypeError, ValueError) as e:
            self.seg.ack(seq, abi.ACK_REJECTED, f"{type(e).__name__}: {e}")
            return
        except Exception as e:  # noqa: BLE001 - a bad command must not take the host down
            self.seg.ack(seq, abi.ACK_FAILED, f"{type(e).__name__}: {e}")
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
            raise ValueError("a suspended session hangs at its gantry; reset it to a named pose")
        p = torch.tensor([pose], dtype=torch.float32, device=self.device)
        if p.shape != (1, 7) or not torch.isfinite(p).all() or abs(float(p[0, 3:].norm()) - 1) > 1e-4:
            raise ValueError("base_pose is [x, y, z, qx, qy, qz, qw] with a unit quaternion")
        return p

    def _named_pose(self, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        from humanoid_sim.model.poses import STANDING_CLEARANCE_M

        q_desc, height = self._poses[name]
        names = list(self.desc.joints)
        q = torch.tensor([[float(q_desc[names.index(n)]) for n in self.joints]], device=self.device)
        z = height + STANDING_CLEARANCE_M + self._lift_m()
        return q, torch.tensor([[0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0]], device=self.device)

    def _lift_m(self) -> float:
        return SUSPENDED_LIFT_M if self.support == "suspended" else 0.0

    def _harness_cable(self, *, dry_run: bool) -> None:
        """The gantry cable at the harness hook, its unstretched length short of the overhead distance by the
        stretch that carries the robot's weight, so it holds the robot where it hangs."""
        cfg = self._support_cfg
        if cfg["harness"] is None:
            raise ValueError(f"{self.desc.robot}: its profile names no support.harness to suspend it by")
        mass = float(self._mass.sum())
        stiffness = mass * GRAVITY_M_S2 / cfg["full_weight_stretch_m"]
        self.sim.gantry.hang_cable(
            torch.tensor([0], device=self.device), self.sim.t_ns, attachment_bodies=cfg["harness"]["shoulders"],
            overhead_m=cfg["overhead_m"], length_breakpoints_s_m=[(0.0, cfg["overhead_m"] - cfg["full_weight_stretch_m"])],
            stiffness_n_m=stiffness, damping_n_s_m=2.0 * math.sqrt(stiffness * mass), dry_run=dry_run)

    def _hang(self, hang: Callable[..., None], *, harness: bool) -> None:
        """Hang a rope (``hang(dry_run=...)``): checked now, hung now while the body poses are current, else right
        after the next step (a pose written since the last step moved the robot without updating them)."""
        hang(dry_run=True)
        self._harness_hung = harness
        if self._poses_stale:
            self._rope_due = lambda: hang(dry_run=False)
        else:
            self._rope_due = None
            hang(dry_run=False)

    def _hang_harness(self) -> None:
        self._hang(self._harness_cable, harness=True)

    def _release_rope(self) -> None:
        self.sim.gantry.release(torch.tensor([0], device=self.device))
        self._rope_due = None
        self._harness_hung = False

    def _reset(self, args: dict) -> str:
        if "joint_pos" in args:
            q, base = self._joint_vector(args["joint_pos"]), None
            if args.get("base_pose") is not None:
                base = self._base_pose(args["base_pose"])
        elif args.get("pose", "home") in self._poses:
            q, base = self._named_pose(args.get("pose", "home"))
        elif args["pose"] == "current":
            q, base = self.sim.joint_state()[0], None
        else:
            raise ValueError(f"unknown pose {args['pose']!r}")
        roped = self._rope_due is not None or bool(self.sim.gantry.mode[0] != NONE)
        harness = self._harness_hung or (self.support == "suspended" and (base is not None or not roped))
        self._write_pose(q, base)
        env = torch.tensor([0], device=self.device)
        # A rope's anchor stays where it was hung: the harness is rehung where the robot now is (a suspended
        # session's robot goes back on it when the reset moves the base); a client's rope is released when the
        # reset moved the base away from it.
        if harness:
            self._release_rope()
            self._hang_harness()
        elif roped and base is not None:
            self._release_rope()
        self.sim.plant.reset(env)
        self.sim.imu.reset(env, self.sim.t_ns, keep_seq=True)
        self._prescribed = None
        self._disturbances.clear()
        self._joint_torques.clear()  # the plant's reset zeroed the external torque
        self._demand_seq = 0  # the client's next demands take effect again
        return "reset"

    def _write_pose(self, q: torch.Tensor, base: torch.Tensor | None) -> None:
        self._poses_stale = True
        if base is not None:
            self.robot.write_root_pose_to_sim_index(root_pose=base)
            self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=self.device))
        self.sim.write_joint_state(q, torch.zeros_like(q))

    def _set_fidelity(self, args: dict) -> str:
        imu = args["imu"]
        if imu not in ("ideal", "real"):
            raise ValueError(f"imu fidelity {imu!r}: ideal or real")
        self.sim.imu.set_noise(imu == "real")
        return f"imu {imu}"

    def _detach(self, args: dict) -> str:
        self.detached = True
        return "detached"

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
            self._release_rope()
            return "free"
        if kind == "suspended":
            self._hang_harness()
            return "suspended"
        if kind != "gantry":
            raise ValueError(f"unknown support {kind!r}")
        t_ns, gantry = self.sim.t_ns, self.sim.gantry
        if "cable" in args:
            c = args["cable"]
            attachment = args["attachment_local_m"]
            overhead, lengths = float(c["overhead_distance_m"]), c["length_breakpoints_s_m"]
            stiffness, damping = float(c["stiffness_n_m"]), float(c["damping_n_s_m"])

            def hang(*, dry_run):
                gantry.hang_cable(env, t_ns, attachment_b=attachment, overhead_m=overhead,
                                  length_breakpoints_s_m=lengths, stiffness_n_m=stiffness, damping_n_s_m=damping,
                                  dry_run=dry_run)
        else:
            tensions = args["tension_breakpoints_s_n"]
            points, bodies = args.get("attachment_local_m"), args.get("attachment_bodies")

            def hang(*, dry_run):
                gantry.hang_tension(env, t_ns, tensions, attachment_b=points, attachment_bodies=bodies,
                                    dry_run=dry_run)
        self._hang(hang, harness=False)
        return "gantry"

    def _disturb(self, args: dict) -> str:
        if "joint" in args or "drive" in args:
            return self._disturb_joint(args)
        body = self.robot.body_names.index(args["body"])
        force = torch.tensor(args.get("force_n", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        torque = torch.tensor(args.get("torque_nm", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self.device)
        duration = float(args["duration_s"])
        if force.shape != (3,) or torque.shape != (3,) or not (0 < duration < math.inf) or not (
                torch.isfinite(force).all() and torch.isfinite(torque).all()):
            raise ValueError("force_n and torque_nm are finite 3-vectors, duration_s positive and finite")
        end = self.sim.t_ns + int(round(duration * NS_PER_S))
        self._disturbances.append(_Disturbance(body, force.view(1, 1, 3), torque.view(1, 1, 3), end))
        return f"until {end} ns"

    def _disturb_joint(self, args: dict) -> str:
        """A torque on one joint (``joint``, or the joint of a 1:1 ``drive``), constant or a sine at
        ``frequency_hz``, outside the drive's feedback; ``torque_nm`` 0 clears it."""
        if "drive" in args:
            drive = self.desc.drives.get(args["drive"])
            if drive is None or drive.crank is not None or len(drive.joints) != 1:
                raise ValueError(f"drive {args['drive']!r}: a 1:1 drive of this robot")
            name = drive.joints[0]
        else:
            name = args["joint"]
        if name not in self.joints:
            raise ValueError(f"unknown joint {name!r}")
        torque, frequency = float(args["torque_nm"]), float(args.get("frequency_hz", 0.0))
        duration = args.get("duration_s")
        if not (math.isfinite(torque) and math.isfinite(frequency) and frequency >= 0) or (
                duration is not None and not 0 < float(duration) < math.inf):
            raise ValueError("torque_nm finite, frequency_hz non-negative, duration_s positive or null")
        j = self.joints.index(name)
        if torque == 0.0:
            self._joint_torques.pop(j, None)
            return "cleared"
        end = None if duration is None else self.sim.t_ns + int(round(float(duration) * NS_PER_S))
        self._joint_torques[j] = (torque, frequency, end)
        return f"{name} until {end} ns" if end is not None else f"{name} until cleared"

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
        if duration is not None and not 0 < float(duration) < math.inf:
            raise ValueError("duration_s is positive and finite, or null (until cleared)")
        end = None if duration is None else self.sim.t_ns + int(round(float(duration) * NS_PER_S))
        if not math.isfinite(value):
            raise ValueError(f"{fault} value must be finite")
        if fault == "power_sag" and not 0.0 <= value <= 1.0:
            raise ValueError("power_sag value is the torque scale, in [0, 1]")
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
        clearance = args.get("floor_clearance_m")
        if clearance is not None:
            if base is None or not (math.isfinite(float(clearance)) and float(clearance) >= 0):
                raise ValueError("floor_clearance_m needs a base_pose and a finite non-negative height")
            self._write_pose(q, base)
            base = base.clone()
            base[0, 2] += float(clearance) - self._lowest_sole_z()
        self._prescribed = (q, base)
        self._write_pose(q, base)
        return "prescribed"

    # -- demands ----------------------------------------------------------------------------------

    def _take_demands(self) -> None:
        """Every ring record not taken yet, oldest first, each sent at its plant time (the client's send time through
        the last feedback's clock mapping)."""
        try:
            got = self.seg.regions["demands"].read()
        except abi.StaleRead:  # the client is mid-write: take it next step
            return
        if got is None or got[0] == self._demand_seq:
            return
        seq, rec = got
        if int(rec["session_id"]) != self.session_id:
            return  # a client of an earlier session
        self._demand_seq = seq
        count = int(rec["count"])
        if count <= self._demands_taken:
            return
        if not self.attached:
            self.attached = True
            self._prescribed = None  # the start pose is released when the client arrives
        self._client_mono_ns = int(rec["written_mono_ns"])
        first = max(self._demands_taken, count - abi.DEMAND_RING)
        self._demand_overruns += first - self._demands_taken
        n = len(self.drives)
        for k in range(first, count):
            record = rec["ring"][k % abi.DEMAND_RING]
            sent_ns = int(record["sent_mono_ns"]) - self._mono_minus_plant_ns
            self._intake(record["drive"][:n], sent_ns + self._drive_delays_ns)
        self._demands_taken = count
        running = rec["ring"][(count - 1) % abi.DEMAND_RING]["drive"][:n]["law"] != abi.LAW_DISABLED
        self._set_drives_off(~running)

    def _set_drives_off(self, off: np.ndarray) -> None:
        """Back-drive friction on the joints of the 1:1 drives that are off; written only on a change."""
        off = off[self._backdrive_drives]
        if self._drive_off is not None and np.array_equal(off, self._drive_off):
            return
        self._drive_off = off.copy()
        mask = torch.from_numpy(off).to(self.device).unsqueeze(0)
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.where(mask, self._friction_off, self._friction_on),
            joint_ids=self._backdrive_joint_ids)

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
        self._apply_joint_torques(t)
        for key, (end, _) in list(self._faults.items()):
            if end is not None and t >= end:
                self._clear_fault(key)

    def _apply_joint_torques(self, t: int) -> None:
        """The joint disturbances' torques for the step starting at ``t``, into the plant (a copy only while any
        is active, and one to clear the last)."""
        expired = [j for j, (_, _, end) in self._joint_torques.items() if end is not None and t >= end]
        for j in expired:
            del self._joint_torques[j]
        if not self._joint_torques and not self._joint_torque_active:
            return
        value = np.zeros(len(self.joints))
        for j, (torque, frequency, _) in self._joint_torques.items():
            value[j] = torque * math.sin(2 * math.pi * frequency * t / NS_PER_S) if frequency else torque
        self.sim.plant.external_torque[0].copy_(torch.from_numpy(value))
        self._joint_torque_active = bool(self._joint_torques)

    def step(self) -> None:
        """One period: command, demands, physics, publish (no pacing)."""
        self._handle_command()
        self._take_demands()
        if not self.paused:
            self._before_physics()
            self.sim.step()
            self.step_count += 1
            self._poses_stale = False
            if self._rope_due is not None:
                hang, self._rope_due = self._rope_due, None
                hang()
        self._publish()

    def _publish(self) -> None:
        """Feedback every step; truth every ``truth_every`` steps (its ``sim_time_ns`` says when)."""
        self._publish_feedback()
        if self.step_count % self.truth_every == 0 or self._truth_due:
            self._publish_truth()
            self._truth_due = False
        self.seg.heartbeat()

    def _to_host(self, name: str, value: torch.Tensor) -> np.ndarray:
        """``value`` on the host, as NumPy (a pinned buffer per ``name``, reused: copy out before the next call)."""
        if self._copied is None:
            return value.numpy()
        buf = self._pinned.get(name)
        if buf is None or buf.shape != value.shape or buf.dtype != value.dtype:
            buf = self._pinned[name] = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
        buf.copy_(value, non_blocking=True)
        self._copied.record()
        self._copied.synchronize()
        return buf.numpy()

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
        host = self._to_host("feedback", torch.cat([flat, ints]))
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
        rec["demand_records"], rec["demand_overruns"] = self._demands_taken, self._demand_overruns
        self._mono_minus_plant_ns = now - sim.t_ns
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
            face = pose[:3] + _quat_rotate(pose[3:].expand(4, 4), s["face"])
            axes = _quat_rotate(pose[3:].expand(3, 4), s["axes"])  # the plate's axes in the world
            span = 2 * (axes[0, 2].abs() * s["half"][0] + axes[1, 2].abs() * s["half"][1])
            soles.append(torch.cat([pose, corners[:, 2].min().view(1), normal[k].view(1), axes[2], span.view(1),
                                    face[:, :2].reshape(-1)]))
        flat = self._to_host("truth", torch.cat([root, vel, com, q[0], qd[0], sim.gantry.tension_n[:1],
                                                 self._mass.sum().view(1), *soles]).to(torch.float64))
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
            s = take(21)
            t["sole"]["position_m"][k], t["sole"]["quat_xyzw"][k] = s[:3], s[3:7]
            t["sole"]["lowest_z_m"][k], t["sole"]["normal_force_n"][k] = s[7], s[8]
            t["sole"]["plate_normal_w"][k], t["sole"]["plate_face_height_span_m"][k] = s[9:12], s[12]
            t["sole"]["face_xy_m"][k] = s[13:21].reshape(4, 2)
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
        """Serve in real time until ``max_steps`` periods, a stop request, or the client detaches or is lost."""
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
                if self.detached:
                    code = EXIT_DETACHED
                    break
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

