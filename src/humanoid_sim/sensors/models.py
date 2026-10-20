"""Named sensor models and a robot's sensor rig.

``profiles/sensors.yaml`` is the catalog: one entry per sensor product (``mti-610``, ``oak-d-w``,
``livox-mid360``), with its kind and the behaviour the product has wherever it is mounted. A robot's profile lists
its rig under ``sensors``: per instance, the model it is, the URDF link it sits at, and what the robot sets around
it (an IMU's configured sample period and how its samples reach the host; a camera's resolution, rate and
calibration). Naming a model gives its whole behaviour; nothing about a product is spelled in a robot's profile.

The plant simulates the rig's ``imu`` (:mod:`humanoid_sim.sensors.imu`). Cameras and lidars are simulated only by
the RTX sensors in the Isaac Sim image (:mod:`humanoid_sim.sensors.rtx`): a lidar always, a camera when its model
places every stream and its rig entry calibrates every stream. :meth:`SensorRig.unsimulated` names the rest, for the
plant alone or with the RTX sensors.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

CATALOG_SCHEMA = "humanoid-sim-sensors/v1"
SIMULATED_KINDS = ("imu",)
RTX_KINDS = ("camera", "lidar")  # what the RTX sensors (Isaac Sim image) simulate
DISTORTION_MODELS = {"plumb_bob": 5, "equidistant": 4}
STREAM_KINDS = ("color", "mono", "depth")  # coefficients: OpenCV's k1 k2 p1 p2 k3, fisheye k1..k4


@dataclass(frozen=True)
class ImuModel:
    name: str
    product: str
    source: str
    gyro_noise_density_rad_s_sqrt_hz: float
    accel_noise_density_m_s2_sqrt_hz: float
    gyro_turn_on_bias_rad_s: float  # per axis, 1 sigma, drawn at every reset
    accel_turn_on_bias_m_s2: float
    gyro_bias_instability_rad_s: float  # stationary sigma of the in-run Gauss-Markov drift
    accel_bias_instability_m_s2: float
    bias_correlation_time_ns: int

    def __post_init__(self):
        _check(self, positive_ns=("bias_correlation_time_ns",),
               non_negative=[f.name for f in fields(self) if f.type == "float"])


@dataclass(frozen=True)
class CameraModel:
    """A camera product. Its frame is its reference imager's optical centre (x forward, y left, z up, as Controls'
    camera calibration has it); ``stream_offsets_m`` places every stream's optical centre in it, nominally (a unit's
    calibration in the rig replaces them). ``stream_kinds`` gives each stream as Controls' backend publishes it:
    ``color``, ``mono`` (grey) or ``depth`` (metres along the optical axis, within ``depth_range_m``)."""

    name: str
    product: str
    source: str
    streams: tuple[str, ...]
    stereo_baseline_m: float | None = None
    stream_kinds: dict[str, str] | None = None
    stream_offsets_m: dict[str, tuple[float, float, float]] | None = None
    depth_range_m: tuple[float, float] | None = None

    def __post_init__(self):
        where = f"camera model {self.name!r}"
        _check_offsets(self.stream_offsets_m or {}, self.streams, where)
        kinds = self.stream_kinds or {}
        if set(kinds) - set(self.streams) or not set(kinds.values()) <= set(STREAM_KINDS):
            raise ValueError(f"{where}: stream_kinds {kinds}: expected streams it has, each one of {STREAM_KINDS}")
        if self.depth_range_m is not None and not 0 <= self.depth_range_m[0] < self.depth_range_m[1]:
            raise ValueError(f"camera model {self.name!r}: depth_range_m {self.depth_range_m}: expected [min, max]")


@dataclass(frozen=True)
class LidarModel:
    name: str
    product: str
    source: str
    frame_period_ns: int
    point_rate_hz: float
    range_min_m: float
    range_max_m: float
    horizontal_fov_rad: float
    elevation_min_rad: float
    elevation_max_rad: float
    range_sigma_m: float
    angular_sigma_rad: float

    def __post_init__(self):
        _check(self, positive_ns=("frame_period_ns",),
               non_negative=("point_rate_hz", "range_min_m", "range_max_m", "horizontal_fov_rad", "range_sigma_m",
                             "angular_sigma_rad"))


@dataclass(frozen=True)
class ImuConfig:
    """How the robot runs its IMU: the configured sample period and the delivery to the host."""

    sample_period_ns: int
    burst_period_ns: tuple[int, int]  # uniform per burst
    in_burst_spacing_ns: int

    def __post_init__(self):
        _check(self, positive_ns=("sample_period_ns",))
        b = self.burst_period_ns
        if len(b) != 2 or not all(_is_ns(v) and v > 0 for v in b) or b[0] > b[1]:
            raise ValueError(f"burst_period_ns {b}: expected [low, high] positive integer ns, low <= high")
        if not _is_ns(self.in_burst_spacing_ns) or self.in_burst_spacing_ns < 0:
            raise ValueError(f"in_burst_spacing_ns {self.in_burst_spacing_ns}: expected integer ns >= 0")


@dataclass(frozen=True)
class CameraConfig:
    """How the robot runs its camera: every stream at one resolution and rate; per stream the unit's pinhole
    calibration ``(fx, fy, cx, cy)`` in pixels (OpenCV's: pixel (u, v) centred at u, v), its lens distortion
    (``{model: plumb_bob | equidistant, coefficients: [...]}``, none when absent) and, where the unit's calibration
    gives it, its optical centre in the camera frame (``stream_offsets_m``, replacing the model's nominal one)."""

    width: int
    height: int
    frame_period_ns: int
    intrinsics: dict[str, tuple[float, float, float, float]]
    distortion: dict[str, dict] | None = None
    stream_offsets_m: dict[str, tuple[float, float, float]] | None = None

    def __post_init__(self):
        _check(self, positive_ns=("frame_period_ns",))
        if not (isinstance(self.width, int) and isinstance(self.height, int) and self.width > 0 and self.height > 0):
            raise ValueError(f"camera resolution {self.width}x{self.height}: expected positive integers")
        for stream, k in self.intrinsics.items():
            if len(k) != 4:
                raise ValueError(f"intrinsics {stream!r}: expected [fx, fy, cx, cy]")
        for stream, d in (self.distortion or {}).items():
            n = DISTORTION_MODELS.get(d.get("model") if isinstance(d, dict) else None)
            if n is None or set(d) != {"model", "coefficients"} or len(d["coefficients"]) != n:
                raise ValueError(f"distortion {stream!r}: expected {{model: one of {sorted(DISTORTION_MODELS)}, "
                                 f"coefficients: [...]}} with the model's number of coefficients")
        _check_offsets(self.stream_offsets_m or {}, None, "camera stream_offsets_m")


@dataclass(frozen=True)
class LidarConfig:
    pass


def _check_offsets(offsets: dict, streams, where: str) -> None:
    for stream, xyz in offsets.items():
        if streams is not None and stream not in streams:
            raise ValueError(f"{where}: offset for a stream it does not have: {stream!r}")
        if len(xyz) != 3:
            raise ValueError(f"{where}: offset {stream!r}: expected [x, y, z] m")


def _is_ns(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check(obj, *, positive_ns=(), non_negative=()):
    for name in positive_ns:
        v = getattr(obj, name)
        if not _is_ns(v) or v <= 0:
            raise ValueError(f"{name} {v!r}: expected a positive integer of ns")
    for name in non_negative:
        if getattr(obj, name) < 0:
            raise ValueError(f"{name} {getattr(obj, name)!r}: expected >= 0")


MODELS = {"imu": ImuModel, "camera": CameraModel, "lidar": LidarModel}
CONFIGS = {"imu": ImuConfig, "camera": CameraConfig, "lidar": LidarConfig}


@dataclass(frozen=True)
class Sensor:
    name: str
    kind: str
    model: ImuModel | CameraModel | LidarModel
    frame: str  # the URDF link it sits at
    config: ImuConfig | CameraConfig | LidarConfig

    @property
    def simulated(self) -> bool:
        return self.kind in SIMULATED_KINDS

    @property
    def rtx_ready(self) -> bool:
        """Whether the RTX sensors can simulate it: a lidar always; a camera whose every stream has a kind, an optical
        centre and intrinsics, and a depth range when one is depth."""
        if self.kind == "lidar":
            return True
        if self.kind != "camera":
            return False
        m, streams = self.model, set(self.model.streams)
        placed = {*(m.stream_offsets_m or {}), *(self.config.stream_offsets_m or {})}
        kinds = m.stream_kinds or {}
        return (streams <= set(self.config.intrinsics) and streams <= placed and streams <= set(kinds)
                and ("depth" not in kinds.values() or m.depth_range_m is not None))


def _built(cls, block: dict, where: str, **given):
    """``cls`` from ``block``: every field without a default present, no stray keys, lists as tuples."""
    names = {f.name for f in fields(cls)}
    stray = sorted(set(block) - names)
    if stray:
        raise ValueError(f"{where}: unknown keys {stray}; expected {sorted(names - set(given))}")
    values = {**given, **block}
    missing = sorted(f.name for f in fields(cls)
                     if f.default is MISSING and f.default_factory is MISSING and f.name not in values)
    if missing:
        raise ValueError(f"{where}: missing {missing}")
    return cls(**{k: _frozen(v) for k, v in values.items()})


def _mapping(block, where: str) -> dict:
    if not isinstance(block, dict):
        raise ValueError(f"{where}: expected a mapping, got {block!r}")
    return block


def _frozen(value):
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, dict):
        return {k: _frozen(v) for k, v in value.items()}
    return value


def load_catalog(root: Path | None = None) -> dict[str, ImuModel | CameraModel | LidarModel]:
    """Every sensor model in ``profiles/sensors.yaml``, by name."""
    root = root or Path(__file__).resolve().parents[3] / "profiles"
    doc = yaml.safe_load((root / "sensors.yaml").read_text())
    if not isinstance(doc, dict) or doc.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"sensors.yaml: expected schema {CATALOG_SCHEMA!r}")
    catalog = {}
    for name, block in _mapping(doc.get("models"), "sensors.yaml models").items():
        block = dict(_mapping(block, f"sensor model {name!r}"))
        kind = block.pop("kind", None)
        if kind not in MODELS:
            raise ValueError(f"sensor model {name!r}: kind {kind!r}; expected one of {sorted(MODELS)}")
        catalog[name] = _built(MODELS[kind], block, f"sensor model {name!r}", name=name)
    return catalog


class SensorRig:
    """A robot's sensors, by instance name, from its profile's ``sensors`` block."""

    def __init__(self, sensors: dict[str, Sensor]):
        self.sensors = sensors

    @classmethod
    def load(cls, profile: dict, catalog: dict | None = None) -> SensorRig:
        catalog = load_catalog() if catalog is None else catalog
        sensors = {}
        for name, block in (profile.get("sensors") or {}).items():
            where = f"{profile.get('robot', '?')} sensor {name!r}"
            block = dict(_mapping(block, where))
            model_name, frame = block.pop("model", None), block.pop("frame", None)
            if model_name not in catalog:
                raise ValueError(f"{where}: unknown model {model_name!r}; known: {sorted(catalog)}")
            if not frame:
                raise ValueError(f"{where}: no frame (the URDF link it sits at)")
            model = catalog[model_name]
            kind = next(k for k, c in MODELS.items() if isinstance(model, c))
            config = _built(CONFIGS[kind], block, where)
            if kind == "camera":
                given = set(config.intrinsics) | set(config.distortion or {}) | set(config.stream_offsets_m or {})
                if given - set(model.streams):
                    raise ValueError(f"{where}: calibration for streams a {model.name} does not have: "
                                     f"{sorted(given - set(model.streams))}")
            sensors[name] = Sensor(name, kind, model, frame, config)
        return cls(sensors)

    @property
    def imu(self) -> Sensor:
        """The IMU the plant simulates: the instance named ``imu``."""
        sensor = self.sensors.get("imu")
        if sensor is None or sensor.kind != "imu":
            raise ValueError(f"the rig has no IMU named 'imu'; it has {sorted(self.sensors)}")
        return sensor

    def unsimulated(self, rtx: bool = False) -> list[str]:
        """The instances nothing simulates: by the plant alone, or with ``rtx`` the RTX sensors as well."""
        return [n for n, s in self.sensors.items()
                if not (s.simulated or (rtx and s.kind in RTX_KINDS and s.rtx_ready))]

    def merged(self, profile_sensors: dict, catalog: dict | None = None) -> SensorRig:
        """This rig with the instances of another ``sensors`` block added (the block's names must be new)."""
        extra = SensorRig.load({"robot": "extra", "sensors": profile_sensors}, catalog)
        clash = sorted(set(extra.sensors) & set(self.sensors))
        if clash:
            raise ValueError(f"sensors {clash} are already in the rig")
        return SensorRig({**self.sensors, **extra.sensors})

    def check_frames(self, desc) -> None:
        """Every sensor sits at a link of the description, and the IMU at the one the physics mounts it on (the
        child of ``imu_joint``)."""
        links = desc.link_names()
        for name, s in self.sensors.items():
            if s.frame not in links:
                raise ValueError(f"sensor {name!r}: {desc.urdf_path.name} has no link {s.frame!r}")
        if "imu" in self.sensors and self.imu.frame != desc.imu_link():
            raise ValueError(f"the IMU sits at {self.imu.frame!r}, but imu_joint mounts {desc.imu_link()!r}")


__all__ = ["CameraConfig", "CameraModel", "ImuConfig", "ImuModel", "LidarConfig", "LidarModel", "RTX_KINDS", "Sensor",
           "SensorRig", "load_catalog"]
