"""Typed plant randomization: which parameter, around which nominal, how it is drawn, and when.

A :class:`ParamSpec` draws one plant parameter per env for a selection of joints, bodies or drives:

* ``dist``: ``uniform`` and ``log_uniform`` take ``(low, high)``, ``normal`` takes ``(mean, std)``. A vector
  parameter (``body_com`` and the IMU's per-axis parameters) may instead give one such pair per axis x, y, z,
  e.g. ``[[-0.03, 0.12], [-0.03, 0.03], [-0.03, 0.03]]`` (``select`` still picks IMU axes; their pairs are
  taken by axis); a plain pair applies to every axis;
* ``op``: how a draw combines with the current value: ``scale`` (×), ``add`` (+) or ``set`` (=). Specs apply in
  order, so ``body_mass`` can be scaled on every body and then get a payload added on one;
* level: a curriculum level in [0, 1] shrinks every draw toward its identity (1 for ``scale``, 0 for ``add``,
  the centre of the distribution for ``set``): ``draw(level) = identity + level · (draw − identity)``;
  ``levels: false`` opts a spec out.

Each parameter has a fixed kind (:data:`PARAMS`):

* ``physics`` parameters live in the solver model. Writing them is expensive (on MJWarp every world is
  updated and the host syncs), so they are drawn at startup and redrawn only when the level changes;
* ``device`` parameters live in this repo's torch tensors and are redrawn for the envs being reset (or only at
  startup, with ``scope: startup``).

Nominal values come from the description (through the built model) or are the identity of the parameter;
nothing here carries a robot number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISTRIBUTIONS = ("uniform", "log_uniform", "normal", "discrete_uniform")
OPS = ("scale", "add", "set")
SCOPES = ("startup", "reset")
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Param:
    kind: str  # "physics" | "device"
    target: str  # "body" | "joint" | "drive" | "env" | "axis" (the IMU's three axes)
    width: int = 1  # values per target (3 for a CoM offset)
    unit: str = ""
    doc: str = ""


PARAMS: dict[str, Param] = {
    "body_mass": Param("physics", "body", unit="kg", doc="body mass; inertia scales with it"),
    "body_com": Param("physics", "body", 3, "m", "centre-of-mass offset in the body frame"),
    "body_inertia": Param("physics", "body", doc="scale of the rotational inertia beyond the mass's"),
    "joint_armature": Param("physics", "joint", unit="kg m²", doc="reflected rotor inertia"),
    "joint_friction": Param("physics", "joint", unit="N m", doc="Coulomb joint friction"),
    "joint_damping": Param("physics", "joint", unit="N m s/rad", doc="passive viscous damping"),
    "joint_friction_ratio": Param("device", "joint", doc="dynamic over static Coulomb joint friction, in [0, 1]"),
    "friction": Param("physics", "env", doc="friction coefficient of every robot shape"),
    "shape_friction": Param("physics", "shape", doc="Coulomb friction coefficient per robot collision shape"),
    "restitution": Param("physics", "env", doc="restitution coefficient of every robot shape, in [0, 1]; Kamino only, "
                                               "MuJoCo's contacts have none"),
    "crank_stop_stiffness": Param("physics", "env", doc="stiffness of the ankle cranks' stops over the profile's "
                                                        "ankle_stop_solref; MJWarp only (Kamino's stops are hard)"),
    "crank_stop_damping": Param("physics", "env", doc="damping of the ankle cranks' stops over the profile's "
                                                      "ankle_stop_solref; MJWarp only (Kamino's stops are hard)"),
    "drive_stiffness": Param("device", "drive", doc="drive's realized P gain over the demanded one"),
    "drive_damping": Param("device", "drive", doc="drive's realized D gain over the demanded one"),
    "torque_limit": Param("device", "drive", doc="scale of the torque-speed envelope and torque cap"),
    "latency": Param("device", "drive", unit="s", doc="latency before the drive (the controller's path), demand sent → staged; the drive's own delay follows. Nominal: the profile's devices.command_path less the drive's delay, else 0"),
    "encoder_offset": Param("device", "drive", unit="rad", doc="drive's position reading minus the truth"),
    "imu_gyro_bias": Param("device", "axis", unit="rad/s", doc="gyro bias, sensor frame"),
    "imu_accel_bias": Param("device", "axis", unit="m/s²", doc="accelerometer bias, sensor frame"),
    "imu_misalignment": Param("device", "axis", unit="rad", doc="sensor rotation from its mount, rotation vector"),
    "imu_latency": Param("device", "env", unit="s", doc="IMU sample measured → its earliest delivery to the host. Nominal: the profile's devices.imu_path, else 0"),
}

_SELECTORS = {"body": "bodies", "joint": "joints", "drive": "drives", "shape": "shapes"}


@dataclass(frozen=True)
class ParamSpec:
    param: str
    dist: str
    range: tuple  # (low, high), or one (low, high) per axis of a vector parameter
    op: str
    select: str = ".*"  # regex over the target's names (bodies, joints or drives); full match
    scope: str | None = None  # None: by kind (physics at startup, device at reset); "startup" for device ones
    levels: bool = True
    per_target: bool = True  # one draw per selected target, or one shared by all of an env's targets
    quantum: float | None = None  # SI spacing for discrete_uniform, inclusive endpoints
    buckets: int | None = None  # sample a shared material table, then assign a bucket per target

    def __post_init__(self):
        if self.param not in PARAMS:
            raise ValueError(f"unknown parameter {self.param!r}; known: {sorted(PARAMS)}")
        if self.dist not in DISTRIBUTIONS:
            raise ValueError(f"{self.param}: dist {self.dist!r}; expected one of {DISTRIBUTIONS}")
        if self.op not in OPS:
            raise ValueError(f"{self.param}: op {self.op!r}; expected one of {OPS}")
        if self.scope is not None and self.scope not in SCOPES:
            raise ValueError(f"{self.param}: scope {self.scope!r}; expected one of {SCOPES}")
        if self.kind == "physics" and self.scope == "reset":
            raise ValueError(f"{self.param} is a solver parameter: drawn at startup and on level changes only")
        pairs = self.axis_ranges
        if pairs is None:
            pairs = [self._pair(self.range)]
        else:
            p = PARAMS[self.param]
            width = p.width if p.width > 1 else len(AXES) if p.target == "axis" else 1
            if width == 1:
                raise ValueError(f"{self.param}: per-axis range on a scalar parameter")
            if len(pairs) != width:
                raise ValueError(f"{self.param}: per-axis range needs {width} pairs, got {len(pairs)}")
            if p.target == "axis" and not self.per_target:
                raise ValueError(f"{self.param}: per-axis range needs per_target")
            pairs = [self._pair(p) for p in self.range]
        if self.buckets is not None and (self.kind != "physics" or isinstance(self.buckets, bool)
                                        or not isinstance(self.buckets, int) or self.buckets < 1):
            raise ValueError(f"{self.param}: buckets must be a positive integer for a physics parameter")
        if self.dist == "discrete_uniform":
            import math

            if self.quantum is None or not math.isfinite(self.quantum) or self.quantum <= 0:
                raise ValueError(f"{self.param}: discrete_uniform needs a positive quantum")
            for lo, hi in pairs:
                count = (hi - lo) / self.quantum
                if not math.isfinite(count) or not math.isclose(count, round(count), abs_tol=1e-9):
                    raise ValueError(f"{self.param}: range must contain a whole number of quanta")
        elif self.quantum is not None:
            raise ValueError(f"{self.param}: quantum only applies to discrete_uniform")
        for lo, hi in pairs:
            if self.dist != "normal" and lo > hi:
                raise ValueError(f"{self.param}: range low {lo} > high {hi}")
            if self.dist == "log_uniform" and lo <= 0:
                raise ValueError(f"{self.param}: log_uniform needs a positive range")
        re.compile(self.select)

    def _pair(self, pair) -> tuple[float, float]:
        if isinstance(pair, (str, bytes)) or not hasattr(pair, "__len__") or len(pair) != 2:
            raise ValueError(f"{self.param}: range needs two numbers")
        return pair[0], pair[1]

    @property
    def axis_ranges(self) -> list | None:
        """The per-axis ranges, or ``None`` for a single pair shared by every axis."""
        if len(self.range) and hasattr(self.range[0], "__len__"):
            return list(self.range)
        return None

    @property
    def kind(self) -> str:
        return PARAMS[self.param].kind

    @property
    def startup(self) -> bool:
        return self.kind == "physics" or self.scope == "startup"

    @classmethod
    def from_dict(cls, d: dict) -> ParamSpec:
        """From a mapping such as ``{param: body_mass, bodies: "Component171_1", dist: uniform, range: [1, 5],
        op: add}``. The selector key is ``bodies``, ``joints`` or ``drives`` by the parameter's target."""
        d = dict(d)
        name = d.pop("param")
        if name not in PARAMS:
            raise ValueError(f"unknown parameter {name!r}; known: {sorted(PARAMS)}")
        key = _SELECTORS.get(PARAMS[name].target)
        if key and "select" in d:
            raise ValueError(f"{name}: select {PARAMS[name].target}s with {key!r}")
        select = d.pop(key, ".*") if key else d.pop("select", ".*")  # e.g. IMU axes
        stray = set(d) & (set(_SELECTORS.values()) - {key})
        if stray:
            raise ValueError(f"{name}: selector {sorted(stray)} does not apply to a {PARAMS[name].target} parameter")
        return cls(param=name, select=select, dist=d.pop("dist"), range=_floats(d.pop("range")),
                   op=d.pop("op"), **d)


def _floats(r):
    return tuple(tuple(float(v) for v in a) if hasattr(a, "__len__") else float(a) for a in r)


def parse_specs(items: list[dict] | None) -> list[ParamSpec]:
    return [s if isinstance(s, ParamSpec) else ParamSpec.from_dict(s) for s in items or []]
