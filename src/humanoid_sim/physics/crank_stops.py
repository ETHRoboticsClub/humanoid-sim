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
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

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
        neutral = np.asarray(desc.ankle_neutral(lk.side), dtype=np.float64)
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
                theta = g.theta0 + float(stop)
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


class _BallArmature:
    """Captures the replicated scene builder, then gives every ball joint coordinate its armature before the model is
    finalized. Newton's USD import reads no armature for spherical joints."""

    def __init__(self, armature: float):
        self.armature = armature
        self.builder = None
        self.joints = 0

    def __call__(self, builder, world: int, position, quaternion) -> None:
        self.builder = builder

    def apply(self) -> None:
        import newton

        b = self.builder
        if b is None:
            raise RuntimeError("ball-joint armature received no Newton scene builder")
        starts = list(b.joint_qd_start) + [len(b.joint_armature)]
        for j, kind in enumerate(b.joint_type):
            if int(kind) == int(newton.JointType.BALL):
                for k in range(starts[j], starts[j + 1]):
                    b.joint_armature[k] = self.armature
                self.joints += 1


@contextlib.contextmanager
def ball_joint_armature(armature: float):
    """Build a scene inside this to give every ball joint coordinate ``armature`` (kg m^2; profile
    ``physics.ball_joint_armature``). A rod between two ball joints spins freely about its own axis on a tiny
    inertia; armature keeps that mode within what the step resolves. A no-op at 0."""
    if armature < 0:
        raise ValueError("ball_joint_armature must be non-negative")
    if not armature:
        yield None
        return
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import NewtonManager

    hook = _BallArmature(armature)
    init = NewtonManager.register_callback(lambda _: hook.apply(), PhysicsEvent.MODEL_INIT,
                                           name="ball_joint_armature", wrap_weak_ref=False)
    try:
        with newton_builder_world_hook(hook):
            yield hook
    finally:
        init.deregister()


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
    if desc.usd_path is not None:  # native crank joints carry their own stops in the built USD
        from humanoid_sim.model.asset import load_profile

        # The robot's profile, as callers may pass only the solver's block of it.
        with ball_joint_armature(float(load_profile(robot)["physics"]["ball_joint_armature"])):
            yield None
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

        check_native_cranks(desc)
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
