"""The ankle cranks' physical stops as solver constraints: one MJWarp spatial tendon per crank and stop.

Each ankle crank's measured stop interval (``drives.<crank>.position_limit`` in humanoid-urdf's ``actuators.yaml``)
is the ankle's only limit; every joint-space bound derives from it through the linkage (humanoid-urdf
``ankle_model``). The hinges' box (``Description.calibration_travel``) is the axis hull of that set, so a pose
inside the box can still need a crank past its stop: the box's corners are not reachable.

A crank pressed on a stop holds its rod's crank-side ball at a fixed point of the shank, the tip of the crank at
that angle, and the rigid rod keeps the foot-side ball at the rod's length from it. With the crank angle ``θ`` the
squared tip-to-ball distance is a sinusoid of ``θ``; between the two crank angles that close the linkage it lies on
one side of ``L²`` and outside them on the other. So the crank of a pose sits inside its stop exactly when the
distance from the stopped tip to the foot-side ball lies on the stop's inner side of the rod length (as long as the
linkage's other assembly closes outside the stop interval, which ``test_crank_stops`` checks over the hinges'
whole travel). A spatial tendon from a site at the stopped tip (on the shank body) to a site at the ball (on the
foot body) with that one-sided length limit is the stop, enforced by the solver as MuJoCo's limit constraint
(``solref`` from the profile's ``ankle_stop_solref``, MuJoCo's default ``solimp``). The constraint acts on the two
hinges alone (the tendon's length depends on nothing else), so where the sites sit on their bodies does not change
the dynamics. Nothing writes the joint state between steps, so contacts keep carrying the robot.

Profile: ``physics.mjwarp.ankle_stops``: ``cranks`` (these tendons and the hinges' box) or ``hinges`` (the box
only). Newton imports spatial tendons from MJCF only, so the sites are added to every world as Newton replicates the
scene (Isaac Lab's ``newton_builder_world_hook``) and the tendons before the model is finalized: build the scene and
reset the simulation inside :func:`crank_stop_hook`.
Kamino has no tendons; it bounds the hinges by their box.

The stops' stiffness is not measured (modeled); their positions are the description's.

Per-env stiffness and damping (the randomization parameters ``crank_stop_stiffness`` and ``crank_stop_damping``,
:class:`StopSolref`): on MJWarp the stops' ``solref`` is written per world, to the tendons' limit solref or, for a
native USD, to its crank joints' raw ``solreflimit``. MuJoCo's constraint has stiffness ``k = 1 / (d_max² τ² ζ²)`` and
damping ``b = 2 / (d_max τ)`` for the time constant ``τ`` and damping ratio ``ζ`` of ``solref`` (``d_max`` the impedance
at full penetration), so scaling ``k`` by ``s`` and ``b`` by ``c`` is ``τ' = τ / c`` and ``ζ' = ζ c / √s``
(:func:`scaled_solref`). MuJoCo raises a time constant below twice the step to that, so a stop at the profile's
``ankle_stop_solref`` cannot get stiffer by shortening it.

Kamino's joint limits are hard constraints with no softness, and a torque on a crank joint goes through the drive's
bounded implicit PD row (the crank's effort limit, 30 N m, would cap what a stop can hold). Its soft stops are
therefore an explicit torque pair on the crank joint's two bodies (:class:`KaminoCrankStops`, the wrench composer
Isaac Lab clears after every physics step), outside that row: past the crank's window a one-sided spring and damper,
``k`` times the overshoot plus ``b`` times the speed, never pulling the crank back out. The crank joint itself is
pushed past its window, so the drive's encoder reads the overshoot as on the robot. The hard limit stays as a
backstop, ``backstop_rad`` further out. An explicit torque has a stability limit at the solver step
(:func:`stability_measure`); the plant checks every env's realized ``k`` and ``b`` against it.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from humanoid_sim.model.description import Description
from humanoid_sim.model.urdf import PreparedUrdf
from humanoid_sim.transmissions.ankle import crank_geometry

ANKLE_STOPS = ("cranks", "hinges")
# The two hinge axes of the universal joint must meet in one point and be perpendicular there (ankle_model's
# pitch +Y, roll +X about a common centre). URDF exports round origins to 1e-6 m and axes to 1e-6.
AXES_MEET_TOLERANCE_M = 1e-5
AXES_PERPENDICULAR_TOLERANCE = 1e-5
SITE_RADIUS_M = 0.004  # drawn size of the sites (they take no part in the physics)
TENDON_LABEL = "crank_stop"
RAW_SOLREF_MODE = 1  # newton's SOLREF_MODE_RAW: the joint's authored mujoco.solreflimit goes to MuJoCo unchanged
# Callback order of the tendons' release at PHYSICS_READY: before Isaac Lab's assets initialize (their order is 10).
RELEASE_ORDER = 0


@dataclass(frozen=True)
class CrankStop:
    """One stop of one crank as a spatial tendon from ``tip`` (shank body frame) to ``ball`` (foot body frame)."""

    side: str
    drive: str
    bound: str  # "lower" | "upper": which end of the drive's position_limit
    stop_rad: float  # crank angle at the stop, relative to the linkage neutral (device frame)
    shank: str  # body the cranks turn on
    tip: np.ndarray  # [3] the stopped crank's rod ball, shank body frame (m)
    foot: str  # body the rods drive
    ball: np.ndarray  # [3] the rod's foot-side ball, foot body frame (m)
    rod_m: float
    longer: bool  # inside the stop while the tip-to-ball distance is at least the rod length (else at most)
    reach_m: float  # the most the two sites can be apart: the sum of their distances from the joint's centre

    @property
    def label(self) -> str:
        return f"{TENDON_LABEL}_{self.drive}_{self.bound}"

    def length_range_m(self) -> tuple[float, float]:
        """The tendon's length range: one end at the rod length, the other where the sites never get (as far apart
        as they can be, or together); MuJoCo limits a tendon from both ends."""
        return (self.rod_m, self.reach_m) if self.longer else (0.0, self.rod_m)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation about a unit ``axis`` through the origin (Rodrigues)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    t = np.eye(4)
    t[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return t


@dataclass(frozen=True)
class AnkleChain:
    """The URDF chain from an ankle's shank body to its foot body: pitch hinge, then roll hinge."""

    shank: str
    foot: str
    pitch_frame: np.ndarray  # [4, 4] pitch joint frame in the shank body frame
    pitch_axis: np.ndarray  # [3] in the pitch joint frame
    roll_frame: np.ndarray  # [4, 4] roll joint frame in the pitch child's body frame
    roll_axis: np.ndarray

    @classmethod
    def from_urdf(cls, urdf: PreparedUrdf, pitch: str, roll: str) -> AnkleChain:
        shank, pitch_frame, pitch_axis = urdf.joint_frame(pitch)
        middle, roll_frame, roll_axis = urdf.joint_frame(roll)
        children = {j.get("name"): j.find("child").get("link") for j in urdf.root.findall("joint")}
        if middle != children[pitch]:
            raise ValueError(f"{roll} does not hang from {pitch}'s child body")
        return cls(shank, children[roll], pitch_frame, pitch_axis, roll_frame, roll_axis)

    def foot_in_shank(self, q_pitch: float, q_roll: float) -> np.ndarray:
        """``[4, 4]`` the foot body frame in the shank body frame at hinge angles (URDF)."""
        return (self.pitch_frame @ _rotation(self.pitch_axis, q_pitch) @ self.roll_frame
                @ _rotation(self.roll_axis, q_roll))


def _closest_points(p1, d1, p2, d2) -> tuple[np.ndarray, np.ndarray]:
    """The closest points of the lines ``p1 + s d1`` and ``p2 + t d2`` (unit, not parallel)."""
    w = p1 - p2
    b = d1 @ d2
    s = (b * (d2 @ w) - (d1 @ w)) / (1 - b * b)
    t = ((d2 @ w) - b * (d1 @ w)) / (1 - b * b)
    return p1 + s * d1, p2 + t * d2


def crank_stops(desc: Description, urdf: PreparedUrdf | None = None) -> list[CrankStop]:
    """The four stops of each ankle linkage (left first; upper crank then lower, lower bound then upper): sites in
    the description URDF's body frames, rod lengths and limit sides from ``actuators.yaml`` ``ankle_model``.

    ankle_model's linkage frame has its origin at the universal joint's centre, pitch about +Y and roll about the
    pitched +X, and turns the foot by ``Ry(p) Rx(r)`` at kinematic angles ``(p, r) = sign (q - neutral)``. In the
    shank body frame at the linkage neutral its +Y is ``sign_pitch`` times the pitch axis and its +X ``sign_roll``
    times the roll axis, so ``R(+Y, p) = R(pitch axis, q - neutral)`` and likewise for roll."""
    if urdf is None:
        from humanoid_sim.model.asset import load_profile

        urdf = PreparedUrdf(desc, load_profile(desc.robot))
    out = []
    for lk in desc.linkages:
        chain = AnkleChain.from_urdf(urdf, *lk.joints)
        sign = np.asarray(desc.ankle_kinematic_sign(lk.side), dtype=np.float64)
        # The pose the geometry is given at: the linkage neutral, or for a native description the USD's joint zero.
        native = desc.usd_path is not None
        neutral = np.zeros(2) if native else np.asarray(desc.ankle_neutral(lk.side), dtype=np.float64)
        # Both hinge axes in the shank frame at the neutral, and where they meet.
        rot_p = chain.pitch_frame[:3, :3]
        axis_p = rot_p @ chain.pitch_axis
        middle = chain.pitch_frame @ _rotation(chain.pitch_axis, neutral[0]) @ chain.roll_frame
        axis_r = middle[:3, :3] @ chain.roll_axis
        a, b = _closest_points(chain.pitch_frame[:3, 3], axis_p, middle[:3, 3], axis_r)
        if np.linalg.norm(a - b) > AXES_MEET_TOLERANCE_M:
            raise ValueError(f"ankle {lk.side}: the hinge axes pass {np.linalg.norm(a - b):.2e} m apart")
        if abs(axis_p @ axis_r) > AXES_PERPENDICULAR_TOLERANCE:
            raise ValueError(f"ankle {lk.side}: the hinge axes are not perpendicular ({axis_p @ axis_r:.2e})")
        centre = 0.5 * (a + b)
        y = sign[0] * axis_p
        x = sign[1] * axis_r
        frame = np.column_stack([x, y, np.cross(x, y)])  # linkage frame → shank frame
        foot_at_neutral = chain.foot_in_shank(*neutral)
        to_foot = np.linalg.inv(foot_at_neutral)
        geometry = desc.ankle_geometry(lk.side)
        for crank, drive in (("upper", lk.upper), ("lower", lk.lower)):
            g = crank_geometry(geometry[crank])
            ball_shank = centre + frame @ g.p_anchor0
            ball = (to_foot @ np.append(ball_shank, 1.0))[:3]
            limits = desc.drives[drive].position_limit_device
            if limits is None:
                raise ValueError(f"{drive}: a crank without position_limit")
            for bound, stop in zip(("lower", "upper"), limits, strict=True):
                theta = desc.drives[drive].shaft_angle(float(stop)) if native else g.theta0 + float(stop)
                tip_link = g.center + g.radius * (np.cos(theta) * g.u_hat + np.sin(theta) * g.v_hat)
                inner = np.linalg.norm(tip_link - g.p_anchor0) - g.length  # at the neutral, inside every stop
                if inner == 0.0:
                    raise ValueError(f"{drive}: the neutral sits on its {bound} stop")
                reach = float(np.linalg.norm(tip_link) + np.linalg.norm(g.p_anchor0))
                out.append(CrankStop(lk.side, drive, bound, float(stop), chain.shank, centre + frame @ tip_link,
                                     chain.foot, ball, g.length, bool(inner > 0), reach))
    return out


def tendon_length_m(chain: AnkleChain, stop: CrankStop, q_pitch, q_roll) -> np.ndarray:
    """The stop's tendon length at hinge angles (URDF; arrays broadcast), through the URDF chain."""
    q_pitch, q_roll = np.broadcast_arrays(np.asarray(q_pitch, dtype=np.float64), np.asarray(q_roll, dtype=np.float64))
    out = np.empty(q_pitch.shape)
    ball = np.append(stop.ball, 1.0)
    for i in np.ndindex(q_pitch.shape):
        out[i] = np.linalg.norm((chain.foot_in_shank(q_pitch[i], q_roll[i]) @ ball)[:3] - stop.tip)
    return out


def ankle_stops_setting(physics: dict, solver: str) -> str:
    """The profile's ``physics.mjwarp.ankle_stops`` for an MJWarp run; ``hinges`` on Kamino (no tendons)."""
    if solver != "mjwarp":
        return "hinges"
    kind = physics["mjwarp"]["ankle_stops"]
    if kind not in ANKLE_STOPS:
        raise ValueError(f"ankle_stops {kind!r}: one of {', '.join(ANKLE_STOPS)}")
    return kind


class _WorldHook:
    """Adds the sites of every crank stop to each world as Newton replicates the scene, and their tendons to the
    whole model at once before it is finalized (``MODEL_INIT``). Tendons added world by world would make Newton
    resolve the articulation owner of every new row against every shape so far on each following world: quadratic
    in the number of worlds."""

    def __init__(self, stops: list[CrankStop], solref: tuple[float, float]):
        self.stops = stops
        self.solref = solref
        self.builder = None
        self.sites: list[tuple[int, list[tuple[int, int]]]] = []  # per world: (world, [(tip, ball) per stop])
        self.tendons = 0

    def __call__(self, builder, world: int, position, quaternion) -> None:
        import warp as wp

        bodies = {}
        for b in range(len(builder.body_world) - 1, -1, -1):
            if int(builder.body_world[b]) != world:
                break
            bodies.setdefault(str(builder.body_label[b]).rsplit("/", 1)[-1], []).append(b)
        ends = []
        for stop in self.stops:
            pair = []
            for body, point, end in ((stop.shank, stop.tip, "tip"), (stop.foot, stop.ball, "ball")):
                found = bodies.get(body, [])
                if len(found) != 1:
                    raise RuntimeError(f"world {world}: {len(found)} bodies named {body!r} for {stop.label}")
                pair.append(builder.add_site(found[0], xform=wp.transform(tuple(map(float, point)), wp.quat_identity()),
                                             scale=(SITE_RADIUS_M,) * 3, label=f"{stop.label}_{end}"))
            ends.append(tuple(pair))
        self.builder = builder
        self.sites.append((world, ends))

    def add_tendons(self) -> None:
        """Every world's tendons, in world order (Newton maps a world's tendons by their index modulo the count per
        world). Once."""
        import warp as wp

        if self.builder is None or self.tendons:
            return
        builder = self.builder
        for world, ends in self.sites:
            for stop, (tip, ball) in zip(self.stops, ends, strict=True):
                rows = [builder.add_custom_values(**{"mujoco:tendon_wrap_type": 0, "mujoco:tendon_wrap_shape": shape,
                                                     "mujoco:tendon_wrap_sidesite": -1, "mujoco:tendon_wrap_prm": 0.0})
                        for shape in (tip, ball)]  # site wraps
                builder.add_custom_values(**{
                    "mujoco:tendon_world": world, "mujoco:tendon_type": 1,  # spatial
                    "mujoco:tendon_joint_adr": 0, "mujoco:tendon_joint_num": 0,
                    "mujoco:tendon_wrap_adr": rows[0]["mujoco:tendon_wrap_type"], "mujoco:tendon_wrap_num": 2,
                    "mujoco:tendon_limited": 1, "mujoco:tendon_range": wp.vec2(*stop.length_range_m()),
                    "mujoco:tendon_solref_limit": wp.vec2(*self.solref),
                    "mujoco:tendon_label": stop.label,
                })
                self.tendons += 1


@contextlib.contextmanager
def crank_stop_hook(robot: str, physics: dict, solver: str | None = None):
    """Build an Isaac Lab scene and reset its simulation inside this to give the MJWarp model the crank stops the
    profile asks for (``physics.mjwarp.ankle_stops: cranks``); a no-op for a robot without crank linkages, for
    ``hinges`` and on Kamino. Yields the hook (``None`` when it adds nothing); :func:`check_crank_stops` checks the
    result.

    The tendons belong to no articulation: Newton assigns a spatial tendon to the articulation its sites ride on,
    and Isaac Lab's Newton articulation refuses spatial tendons (it drives fixed tendons only). They are solver
    constraints, not something a task commands, so once the model is final and before the assets initialize
    (``PHYSICS_READY``, ahead of the assets' order) their owner is cleared."""
    solver = solver or physics["solver"]
    desc = Description.load(robot)
    if desc.usd_path is not None:
        yield None  # native crank joints carry their own stops in the built USD
        return
    if ankle_stops_setting(physics, solver) != "cranks" or not desc.linkages:
        yield None
        return
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import NewtonManager

    solref = tuple(float(v) for v in physics["mjwarp"]["ankle_stop_solref"])
    if len(solref) != 2:
        raise ValueError("ankle_stop_solref: [time constant s, damping ratio]")
    hook = _WorldHook(crank_stops(desc), solref)
    init = NewtonManager.register_callback(lambda _: hook.add_tendons(), PhysicsEvent.MODEL_INIT,
                                           name="crank_stops", wrap_weak_ref=False)
    ready = NewtonManager.register_callback(lambda _: _release_tendons(), PhysicsEvent.PHYSICS_READY,
                                            order=RELEASE_ORDER, name="crank_stops", wrap_weak_ref=False)
    try:
        with newton_builder_world_hook(hook):
            yield hook
    finally:
        init.deregister()
        ready.deregister()


def _stop_rows(model) -> list[int]:
    """The model's crank-stop tendon rows (``mujoco:tendon`` frequency)."""
    labels = getattr(getattr(model, "mujoco", None), "tendon_label", None)
    return [i for i, label in enumerate(labels or ()) if str(label).rsplit("/", 1)[-1].startswith(TENDON_LABEL)]


def _release_tendons() -> None:
    """Clear the articulation owner of every crank-stop tendon of the final model (see :func:`crank_stop_hook`)."""
    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    rows = _stop_rows(model)
    if rows:
        owner = model.mujoco.tendon_articulation
        values = owner.numpy()
        values[rows] = -1
        owner.assign(values)


def check_crank_stops(robot: str, physics: dict) -> int:
    """After the simulation is reset: the running MJWarp model has the crank stops the profile asks for, else raise
    (a scene built outside :func:`crank_stop_hook`). Returns the number of stop tendons."""
    from humanoid_sim.physics.isaaclab import running_solver

    solver = running_solver()
    desc = Description.load(robot)
    if desc.usd_path is not None:
        from humanoid_sim.physics.native import check_native_cranks

        # Kamino's soft stops move the hard limit out to the backstop (KaminoCrankStops).
        pads = (0.0, kamino_stop_settings(physics).backstop_rad) if solver == "kamino" else (0.0,)
        check_native_cranks(desc, pads)
        return 0
    expected = 4 * len(desc.linkages) if ankle_stops_setting(physics, solver) == "cranks" else 0
    found = 0
    if solver == "mjwarp":
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager._solver.mj_model
        import mujoco

        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i) or "" for i in range(model.ntendon)]
        found = sum(TENDON_LABEL in n for n in names)
    if found != expected:
        raise RuntimeError(f"the solver model has {found} crank-stop tendons, the profile's ankle_stops "
                           f"asks for {expected}: build the scene inside humanoid_sim.physics.crank_stops."
                           "crank_stop_hook")
    return found


def scaled_solref(solref: torch.Tensor, stiffness: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    """MuJoCo ``solref`` ``[E, K, 2]`` with the constraint's stiffness scaled by ``stiffness`` and its damping by
    ``damping`` (both ``[E, 1]``, positive). Time-constant form (``τ, ζ > 0``): ``τ' = τ / damping`` and
    ``ζ' = ζ damping / √stiffness``. Direct form (``-k, -b``): ``-k stiffness, -b damping``. ``(0, 0)``, MuJoCo's
    "take the model default", has nothing to scale and raises."""
    for name, scale in (("stiffness", stiffness), ("damping", damping)):
        if not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError(f"crank_stop_{name} must be positive and finite")
    s, c = stiffness.unsqueeze(-1), damping.unsqueeze(-1)
    tau, zeta = solref[..., :1], solref[..., 1:]
    if bool(((tau == 0) | (zeta == 0)).any()) or bool(((tau > 0) != (zeta > 0)).any()):
        raise ValueError("the crank stops' solref must be two positive numbers (time constant, damping ratio) "
                         "or two negative ones (stiffness, damping)")
    direct = tau < 0
    return torch.cat([torch.where(direct, tau * s, tau / c), torch.where(direct, zeta * c, zeta * c / s.sqrt())],
                     dim=-1)


class StopSolref:
    """The running MJWarp model's crank-stop ``solref``, ``[E, K, 2]`` (K stops per env), readable and writable.

    ``kind`` is ``tendons`` (the URDF's spatial tendons of :func:`crank_stop_hook`) or ``native`` (the crank joints'
    raw ``solreflimit`` authored into the USD). ``write`` takes effect after ``flag`` is given to the physics
    manager's ``add_model_change``."""

    def __init__(self, robot, model, desc: Description, num_envs: int):
        import warp as wp

        self._wp = wp
        self.num_envs = num_envs
        if desc.usd_path is not None:
            from humanoid_sim.transmissions.drive_map import native_joint_names

            names = native_joint_names(desc)[len(desc.joints):]  # the crank joints follow the semantic ones
            self.kind, self._view, self._model = "native", robot._root_view, model
            self._ids = torch.tensor([robot.joint_names.index(n) for n in names])
            self._solref = "mujoco.solreflimit"
            mode = wp.to_torch(self._view.get_attribute("mujoco.solreflimit_mode", model))[:, 0, self._ids]
            if bool((mode != RAW_SOLREF_MODE).any()):
                raise RuntimeError(f"the crank joints {names} do not carry a raw solreflimit "
                                   f"(mode {mode.unique().tolist()}); the built USD has no mjc:solreflimit on them")
            from newton import ModelFlags

            self.flag = ModelFlags.JOINT_DOF_PROPERTIES
        else:
            rows = _stop_rows(model)
            if not rows or len(rows) % num_envs:
                raise RuntimeError(f"{len(rows)} crank-stop tendons for {num_envs} envs: build the scene inside "
                                   "humanoid_sim.physics.crank_stops.crank_stop_hook with ankle_stops: cranks")
            self.kind = "tendons"
            self._array = wp.to_torch(model.mujoco.tendon_solref_limit)
            self._rows = torch.tensor(rows, device=self._array.device).reshape(num_envs, -1)
            world = getattr(model.mujoco, "tendon_world", None)
            if world is not None:
                owner = wp.to_torch(world)[self._rows]
                if not bool((owner == torch.arange(num_envs, device=owner.device)[:, None]).all()):
                    raise RuntimeError("the crank-stop tendons are not in world order")
            from newton import ModelFlags

            self.flag = ModelFlags.TENDON_PROPERTIES

    def read(self) -> torch.Tensor:
        if self.kind == "tendons":
            return self._array[self._rows].clone()
        full = self._wp.to_torch(self._view.get_attribute(self._solref, self._model))
        return full[:, 0, self._ids.to(full.device)].clone()

    def write(self, solref: torch.Tensor) -> None:
        if self.kind == "tendons":
            self._array[self._rows] = solref.to(self._array.dtype)
            return
        full = self._wp.to_torch(self._view.get_attribute(self._solref, self._model)).clone()
        full[:, 0, self._ids.to(full.device)] = solref.to(full.dtype)
        self._view.set_attribute(self._solref, self._model, self._wp.from_torch(full.contiguous(), dtype=self._wp.vec2))


def check_stop_solref_support(robot: str, physics: dict, solver: str) -> None:
    """Raise unless the running solver has crank stops whose stiffness can be randomized: on a robot with ankle
    linkages, MJWarp with the stops as tendons (``ankle_stops: cranks``) or as the native USD's crank joint limits,
    or Kamino with a native USD (its crank joints carry :class:`KaminoCrankStops`)."""
    desc = Description.load(robot)
    if not desc.linkages:
        raise ValueError(f"{robot} has no ankle linkages, hence no crank stops")
    if solver == "kamino":
        if desc.usd_path is None:
            raise ValueError("crank-stop stiffness on Kamino needs the native ankle: its crank joints carry the "
                             "stops (the reduced model has no cranks)")
        kamino_stop_settings(physics)
        return
    if solver != "mjwarp":
        raise ValueError(f"{solver} has no crank-stop stiffness to randomize: mjwarp or kamino")
    if desc.usd_path is None and ankle_stops_setting(physics, solver) != "cranks":
        raise ValueError("crank-stop stiffness needs the stops as tendons: physics.mjwarp.ankle_stops: cranks")


# -- Kamino: explicit soft stops on the crank joints ----------------------------------------------------------------

STOP_KEYS = ("soft", "stiffness_nm_per_rad", "damping_nm_s_per_rad", "backstop_rad")
# An explicit spring-damper on an inertia I at step dt (stiffness k, damping b) is stable, in the symplectic Euler
# step the solver takes, when k dt²/I + 2 b dt/I < 4 and b dt/I < 2 (Jury's test of its 2x2 map). The plant keeps to
# half the first bound: the realized stiffness and damping may use at most this much of it.
STABLE_MEASURE = 2.0


@dataclass(frozen=True)
class KaminoStopSettings:
    """The profile's ``physics.kamino.crank_stops``: the nominal soft stop of every crank."""

    soft: bool
    stiffness: float  # N m/rad
    damping: float  # N m s/rad
    backstop_rad: float  # how far past the crank's window the hard limit stays


def kamino_stop_settings(physics: dict) -> KaminoStopSettings:
    """The validated ``physics.kamino.crank_stops`` block."""
    block = physics.get("kamino", {}).get("crank_stops")
    if not isinstance(block, dict) or set(block) != set(STOP_KEYS):
        raise ValueError(f"physics.kamino.crank_stops: a mapping with exactly {', '.join(STOP_KEYS)}")
    if not isinstance(block["soft"], bool):
        raise ValueError("physics.kamino.crank_stops.soft must be true or false")
    for key in STOP_KEYS[1:]:
        v = block[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"physics.kamino.crank_stops.{key} must be a positive number, not {v!r}")
    return KaminoStopSettings(block["soft"], float(block["stiffness_nm_per_rad"]),
                              float(block["damping_nm_s_per_rad"]), float(block["backstop_rad"]))


def stability_measure(stiffness: torch.Tensor, damping: torch.Tensor, inertia: torch.Tensor,
                      period_s: float) -> torch.Tensor:
    """``k dt²/I + 2 b dt/I`` of an explicit spring-damper (broadcasting): stable below 4, kept below
    :data:`STABLE_MEASURE`. ``inertia`` is the joint's armature, the least inertia its row can have."""
    return (stiffness * period_s + 2.0 * damping) * period_s / inertia


def stop_torque(q: torch.Tensor, qd: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                stiffness: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    """The soft stops' torque on a joint at ``q, q̇`` (broadcasting): zero inside ``[lower, upper]``; past ``upper`` by
    ``x`` it is ``-(k x + b q̇)``, past ``lower`` ``k x - b q̇``, each clamped at zero so that a stop only pushes the
    joint back in and never holds it out while it moves back."""
    over, under = q - upper, lower - q
    high = -(stiffness * over + damping * qd).clamp(min=0.0) * (over > 0)
    low = (stiffness * under - damping * qd).clamp(min=0.0) * (under > 0)
    return high + low


def _rotate(quat_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` ``[..., 3]`` by unit quaternions ``[..., 4]`` (x, y, z, w)."""
    xyz, w = quat_xyzw[..., :3], quat_xyzw[..., 3:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _matrix(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` of unit quaternions ``[..., 4]`` (x, y, z, w)."""
    x, y, z, w = (quat_xyzw[..., i] for i in range(4))
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1).unflatten(-1, (3, 3))


def _about(axis: np.ndarray, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` about the unit ``axis`` by ``angle`` ``[...]`` (Rodrigues)."""
    k = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
                     dtype=angle.dtype, device=angle.device)
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + torch.sin(angle)[..., None, None] * k + (1 - torch.cos(angle))[..., None, None] * (k @ k)


class KaminoCrankStops:
    """The soft crank stops of a native V1 on Kamino: per env stiffness and damping (``k = nominal * stiffness_scale``,
    likewise ``b``), applied as an explicit torque pair on each crank joint's bodies every physics step.

    The torque is :func:`stop_torque` of the crank joint's state at the step start, about the joint's axis in the
    world, ``+`` on the child body and ``-`` on the parent. It enters Kamino as body wrenches, not through the joint:
    the crank's PD row is bounded by the drive's effort limit and would cap it. Building it moves each crank's hard
    limit ``backstop_rad`` past the window (the limit's existence is fixed for Kamino, its value is not). Call
    :meth:`apply` once per physics step, before it; the wrench composer is cleared after each. Use
    :func:`kamino_crank_stops` to build or find the articulation's one."""

    def __init__(self, robot, desc: Description, settings: KaminoStopSettings, period_s: float):
        from humanoid_sim.model.usd import crank_joints
        from humanoid_sim.transmissions.compliance import output_body

        self.robot, self.settings, self.period_s = robot, settings, float(period_s)
        self.device = torch.device(robot.device)
        self.num_envs = robot.num_instances
        e = self.num_envs
        cranks = crank_joints(desc)
        self.drives = list(cranks)
        self.joints = list(cranks.values())
        self.ids = torch.tensor([robot.joint_names.index(n) for n in self.joints], device=self.device)
        window = torch.tensor([desc.drives[d].shaft_limits for d in self.drives], device=self.device,
                              dtype=torch.float32)
        self.lower, self.upper = window[:, 0], window[:, 1]
        source = desc.usd_model
        bodies = list(robot.body_names)
        parents, children, axes, self._frames = [], [], [], []
        for joint in self.joints:
            parent, child = source.joint_bodies(joint)
            child = output_body(joint) if output_body(joint) in bodies else child  # an asset with compliance
            for name in (parent, child):
                if name not in bodies:
                    raise RuntimeError(f"crank {joint}: the articulation has no body {name!r}")
            parents.append(bodies.index(parent))
            children.append(bodies.index(child))
            frame0, frame1 = source.joint_frames(joint)
            axis = source.joint_axis(joint)
            axes.append(frame0[:3, :3] @ axis)
            self._frames.append((frame0[:3, :3], frame1[:3, :3], axis))
        self.parent = torch.tensor(parents, device=self.device)
        self.child = torch.tensor(children, device=self.device)
        self.axis = torch.tensor(np.array(axes), device=self.device, dtype=torch.float32)  # in the parent body frame
        # The torque pair's bodies: each distinct body once, with the signed share of every crank that acts on it.
        involved = sorted({*parents, *children})
        self._bodies = torch.tensor(involved, device=self.device)
        share = torch.zeros(2 * len(self.joints), len(involved), device=self.device)
        for c, (parent, child) in enumerate(zip(parents, children, strict=True)):
            share[c, involved.index(child)] = 1.0
            share[len(self.joints) + c, involved.index(parent)] = -1.0
        self._share = share
        self._all = torch.arange(e, device=self.device)
        self._no_force = torch.zeros(e, len(involved), 3, device=self.device)
        self.stiffness_scale = torch.ones(e, 1, device=self.device)
        self.damping_scale = torch.ones(e, 1, device=self.device)
        self.torque = torch.zeros(e, len(self.joints), device=self.device)  # the last step's, on the joints
        self.check_frames()
        self._widen_limits()
        self.set_scales()

    # -- setup ------------------------------------------------------------------------------------------------

    def _widen_limits(self) -> None:
        """Move each crank's hard limit ``backstop_rad`` past its window."""
        pad = self.settings.backstop_rad
        limits = torch.stack([self.lower - pad, self.upper + pad], dim=-1)
        self.robot.write_joint_position_limit_to_sim_index(
            limits=limits.unsqueeze(0).expand(self.num_envs, -1, -1).contiguous(), joint_ids=self.ids,
            warn_limit_violation=False)

    def check_frames(self, tolerance: float = 1e-3) -> None:
        """Raise unless each crank child's world rotation is its parent's turned by the joint's axis and angle, as the
        torque's axis assumes (the USD frames the axis comes from, against the solver's state)."""
        pose = self.robot.data.body_link_pose_w.torch
        q = self.robot.data.joint_pos.torch[:, self.ids]
        for c, (r0, r1, axis) in enumerate(self._frames):
            r0, r1 = (torch.tensor(r, device=self.device, dtype=torch.float32) for r in (r0, r1))
            predicted = _matrix(pose[:, self.parent[c], 3:]) @ r0 @ _about(axis, q[:, c]) @ r1.T
            actual = _matrix(pose[:, self.child[c], 3:])
            error = (predicted - actual).abs().max()
            if float(error) > tolerance:
                raise RuntimeError(f"crank {self.joints[c]}: its bodies' rotation differs from the joint's frames and "
                                   f"axis by {float(error):.2e}; the stop torque's axis would be wrong")

    # -- knobs ------------------------------------------------------------------------------------------------

    def set_scales(self, stiffness: torch.Tensor | None = None, damping: torch.Tensor | None = None) -> None:
        """The realized stiffness and damping as ratios ``[E, 1]`` over the profile's nominal (``None``: 1). Raises for
        a non-positive or non-finite ratio, or a stop the solver step cannot carry explicitly."""
        one = torch.ones(self.num_envs, 1, device=self.device)
        new = []
        for name, scale in (("stiffness", stiffness), ("damping", damping)):
            scale = one if scale is None else scale.to(self.device, torch.float32).reshape(self.num_envs, 1)
            if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError(f"crank_stop_{name} must be positive and finite")
            new.append(scale)
        armature = self.robot.data.joint_armature.torch[:, self.ids]
        if not bool((armature > 0).all()):
            raise ValueError("Kamino's soft crank stops are explicit: they need an armature on the crank joints")
        k, b = (self.settings.stiffness * new[0]), (self.settings.damping * new[1])
        measure = stability_measure(k, b, armature, self.period_s)
        if bool((measure > STABLE_MEASURE).any()):
            worst = int(measure.max(dim=1).values.argmax())
            raise ValueError(
                f"crank stops stiffness {float(k[worst, 0]):.0f} N m/rad and damping {float(b[worst, 0]):.1f} "
                f"N m s/rad exceed what an explicit stop carries at the {self.period_s * 1e3:g} ms step on an armature "
                f"of {float(armature[worst].min()):.3g} kg m² (k dt²/I + 2 b dt/I = {float(measure[worst].max()):.2f}, "
                f"at most {STABLE_MEASURE:g})")
        self.stiffness_scale, self.damping_scale = new

    @property
    def stiffness(self) -> torch.Tensor:
        """The realized stiffness ``[E, 1]``, N m/rad."""
        return self.settings.stiffness * self.stiffness_scale

    @property
    def damping(self) -> torch.Tensor:
        """The realized damping ``[E, 1]``, N m s/rad."""
        return self.settings.damping * self.damping_scale

    # -- every physics step -----------------------------------------------------------------------------------

    def torques(self) -> torch.Tensor:
        """The stop torque ``[E, C]`` on each crank joint at the current state."""
        data = self.robot.data
        return stop_torque(data.joint_pos.torch[:, self.ids], data.joint_vel.torch[:, self.ids], self.lower,
                           self.upper, self.stiffness, self.damping)

    def apply(self) -> None:
        """Add this physics step's stop torques to the articulation's wrench composer."""
        tau = self.torques()
        self.torque.copy_(tau)
        quat = self.robot.data.body_link_pose_w.torch[:, self.parent, 3:]  # [E, C, 4]
        axis_w = _rotate(quat, self.axis.expand_as(quat[..., :3]))  # [E, C, 3]
        pair = tau.unsqueeze(-1) * axis_w
        both = torch.cat([pair, pair], dim=1)  # child share +1, parent share -1: rows of self._share
        wrench = torch.einsum("ebx,bu->eux", both, self._share).contiguous()
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=self._no_force, torques=wrench, body_ids=self._bodies, env_ids=self._all, is_global=True)


def kamino_crank_stops(robot, desc: Description, physics: dict, period_s: float, *,
                       require: bool = False) -> KaminoCrankStops | None:
    """The articulation's soft crank stops on Kamino: the existing ones, else new when the profile asks for them
    (``physics.kamino.crank_stops.soft``) or ``require`` (the plant randomizes the stops), else ``None`` (hard limits).
    ``None`` as well for a robot without a native ankle."""
    stops = getattr(robot, "_kamino_crank_stops", None)
    if stops is not None:
        return stops
    if desc.usd_path is None or not desc.linkages:
        return None
    settings = kamino_stop_settings(physics)
    if not (settings.soft or require):
        return None
    stops = robot._kamino_crank_stops = KaminoCrankStops(robot, desc, settings, period_s)
    return stops
