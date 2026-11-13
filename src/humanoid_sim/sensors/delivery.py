"""When a camera's frames reach the robot's stack, if at all: the latency from a frame's exposure to its arrival and
the frames that are lost on the way. Applied when a session asks for it (``SessionCfg.camera_effects``:
:data:`DELIVERY_EFFECTS`); without it every frame is delivered as soon as it is rendered.

* **Latency.** Each frame arrives ``latency_ns`` after its exposure, drawn uniformly in the camera instance's
  ``(low, high)`` (:class:`~humanoid_sim.sensors.models.Camera`) per frame. Frames do not overtake each other: a
  frame never arrives before the previous one of its stream. The stamp stays the exposure time, as the devices stamp
  their frames; only the delivery moves.
* **Dropped frames.** Each frame is lost with the instance's ``drop_probability`` and never delivered. The streams of
  one sync group (a stereo pair) travel together: both are delivered or both lost. The other streams (colour, depth)
  are drawn on their own.

A frame that the renderer hands over later than its arrival (the render took longer than the latency) is delivered
when it is ready: the modeled latency is a floor, and the session reports how late its frames came.

Draws come from their own generators, one for the latency and one for the losses per camera, seeded from the session's
seed and the camera id, so neither changes the imaging noise, the depth errors or each other.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable

import numpy as np

DELIVERY_EFFECTS = ("latency", "drops")
_TAG = 0xDE11  # this module's seed stream, apart from the imaging's


def delivery_seeds(seed: int, camera_id: str) -> tuple[np.random.SeedSequence, np.random.SeedSequence]:
    """A camera's latency and loss seeds, from the session's."""
    latency, drops = np.random.SeedSequence([seed, zlib.crc32(camera_id.encode()), _TAG]).spawn(2)
    return latency, drops


class FrameDelivery:
    """The delivery of one camera's frames (``spec``: its :class:`~humanoid_sim.sensors.geometry.CameraSpec`): each
    frame's arrival, ``latency_ns`` ``(low, high)`` after its exposure, or its loss, with ``drop_probability``."""

    def __init__(self, spec, latency_ns: tuple[int, int], drop_probability: float, seed: int):
        from humanoid_sim.sensors.models import frame_drop_probability, frame_latency

        where = f"camera {spec.camera_id!r}"
        self.camera_id = spec.camera_id
        self.latency_ns = frame_latency(f"{where} latency_ns", latency_ns)
        self.drop_probability = frame_drop_probability(f"{where} drop_probability", drop_probability)
        self.group = {s.name: s.sync_group or s.name for s in spec.streams}
        latency, drops = delivery_seeds(seed, spec.camera_id)
        self._latency_rng, self._drop_rng = np.random.default_rng(latency), np.random.default_rng(drops)
        self._last_arrival: dict[str, int] = {}
        self._decided: dict[str, tuple[int, int | None]] = {}
        self.frames = 0
        self.dropped = 0
        self.late_ns: list[int] = []  # how much later than its arrival each delivered frame went out

    def decide(self, stream: str, stamp_ns: int) -> int | None:
        """The plant time the stream's frame exposed at ``stamp_ns`` arrives at; None when it is lost. The streams of
        a sync group share one draw per exposure."""
        group = self.group[stream]
        self.frames += 1
        decided = self._decided.get(group)
        if decided is None or decided[0] != stamp_ns:
            # Both draws every exposure, so turning losses on leaves the latencies of the other frames as they were.
            low, high = self.latency_ns
            latency = low if high == low else int(self._latency_rng.integers(low, high, endpoint=True))
            lost = self.drop_probability > 0.0 and bool(self._drop_rng.random() < self.drop_probability)
            arrival = None
            if not lost:
                arrival = max(stamp_ns + latency, self._last_arrival.get(group, stamp_ns))
                self._last_arrival[group] = arrival
            decided = self._decided[group] = (stamp_ns, arrival)
        if decided[1] is None:
            self.dropped += 1
        return decided[1]

    def delivered(self, arrival_ns: int, at_ns: int) -> None:
        """Note that a frame due at ``arrival_ns`` went out at plant time ``at_ns``."""
        self.late_ns.append(max(at_ns - arrival_ns, 0))

    def describe(self) -> dict:
        late = np.asarray(self.late_ns, dtype=np.float64) / 1e6
        return {"latency_ns": list(self.latency_ns), "drop_probability": self.drop_probability,
                "frames": self.frames, "dropped": self.dropped, "delivered": len(self.late_ns),
                "late_ms": None if not len(late) else {"median": float(np.median(late)),
                                                       "p95": float(np.percentile(late, 95)),
                                                       "max": float(late.max())}}


def latency_from_ms(values) -> tuple[int, int] | None:
    """``--camera-latency-ms``: one value (fixed) or two (low, high), in ms, as integer ns; None for none given."""
    from humanoid_sim.sensors.models import frame_latency

    if values is None:
        return None
    values = [float(v) for v in values]
    if len(values) not in (1, 2):
        raise ValueError(f"camera latency {values} ms: expected one value or (low, high)")
    ns = [int(round(v * 1e6)) for v in values]
    return frame_latency("camera latency (ns)", (ns[0], ns[-1]))


def check_overrides(effects: Iterable[str], latency_ns=None, drop_probability=None) -> None:
    """Refuse an override of an effect that is off: it would do nothing. Its value is checked as an instance's."""
    from humanoid_sim.sensors.models import frame_drop_probability, frame_latency

    effects = set(effects)
    if latency_ns is not None:
        frame_latency("camera latency_ns", latency_ns)
        if "latency" not in effects:
            raise ValueError("a camera latency was given but the 'latency' camera effect is off")
    if drop_probability is not None:
        frame_drop_probability("camera drop_probability", drop_probability)
        if "drops" not in effects:
            raise ValueError("a camera drop probability was given but the 'drops' camera effect is off")


def delivery_settings(camera, effects: Iterable[str], latency_ns=None, drop_probability=None
                      ) -> tuple[tuple[int, int], float]:
    """The latency and loss of ``camera`` (its model instance) under ``effects``: the instance's, or ``latency_ns`` /
    ``drop_probability`` in their place (:func:`check_overrides`); none for an effect that is off."""
    check_overrides(effects, latency_ns, drop_probability)
    effects = set(effects)
    latency = (0, 0)
    if "latency" in effects:
        latency = camera.latency_ns if latency_ns is None else tuple(latency_ns)
    drops = 0.0
    if "drops" in effects:
        drops = camera.drop_probability if drop_probability is None else drop_probability
    return latency, drops


__all__ = ["DELIVERY_EFFECTS", "FrameDelivery", "check_overrides", "delivery_seeds", "delivery_settings",
           "latency_from_ms"]
