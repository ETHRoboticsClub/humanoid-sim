"""How the viewers draw a gantry: the rope from its anchor to the harness hook, coloured by how much it carries, the
anchor, a plumb line under it, the crane's travel, bridge and trolley, a status label, and the tension as signals.
``SceneRecorder`` (``signals/scene.py``, training scenes) and ``view-runtime`` (``runtime/view.py``) both draw through :func:`log_env`, so a rope looks the same in either.

The rope is classed from its tension against the robot's weight ``W`` (``mass · g``):

* ``SLACK`` (grey): the cable is at or below its unstretched length and pulls nothing;
* ``TAUT`` (green): it carries up to the full weight, the stretch ``W / k`` of a cable that holds the robot just up;
* ``OVER`` (red): it carries more than the full weight, or is stretched past ``W / k``, which a hung robot only
  reaches when it was hung off its anchor, falls, or is pulled;
* ``CAUGHT`` (red): the tension spiked over ``CAUGHT_SHARE`` of the weight, as when a falling robot is stopped.

Entity paths, under the ``world`` and ``sim`` prefixes the caller gives: ``<world>/{rope,anchor,plumb,status}``,
``<world>/{travel,bridge,trolley}`` with a crane, and ``<sim>/{tension_n,tension_share,length_m,rest_length_m}``
(scalars; ``tension_share`` is the tension over ``W``).
"""

from __future__ import annotations

import numpy as np

SLACK, TAUT, OVER, CAUGHT = 0, 1, 2, 3
STATE_WORDS = {SLACK: "slack", TAUT: "taut", OVER: "over-stretched", CAUGHT: "caught"}
STATE_RGB = {SLACK: (150, 150, 150), TAUT: (60, 190, 90), OVER: (225, 60, 50), CAUGHT: (225, 60, 50)}

FULL_WEIGHT_SHARE = 1.0  # tension over weight at the full-weight stretch: more is over-stretched
CAUGHT_SHARE = 2.0  # tension over weight beyond which the rope has caught the robot: twice what it hangs by

ROPE_RADIUS_M = 0.006
ANCHOR_RADIUS_M = 0.05
ANCHOR_RGB = (240, 240, 240)
PLUMB_RGB = (110, 110, 130)  # straight down from the anchor, as far as the rope is long
CRANE_RGB = (90, 90, 200)
TRAVEL_RGB = (120, 120, 120)
BRIDGE_RADIUS_M = 0.03
TROLLEY_RADIUS_M = 0.08
LABEL_RADIUS_M = 0.01
LABEL_LIFT_M = 0.6  # a label with no rope floats this far above the robot's base
NO_GANTRY = "no gantry"
LABEL_RGB = (245, 245, 245)  # the label is drawn on a grey plate: the rope carries the state colour


def full_weight_stretch_m(weight_n, stiffness_n_m):
    """How far a cable of ``stiffness_n_m`` stretches under the robot's full weight."""
    return np.asarray(weight_n, dtype=np.float64) / np.asarray(stiffness_n_m, dtype=np.float64)


def rope_state(tension_n, weight_n, *, extension_m=None, stretch_m=None):
    """The class (``SLACK``, ``TAUT``, ``OVER`` or ``CAUGHT``) of a rope, elementwise. ``extension_m`` (length less
    unstretched length) decides slack when given, else a tension of zero does; ``stretch_m`` (the full-weight
    stretch) adds over-stretched beyond it, whatever the tension, as a damper can hide it."""
    tension = np.asarray(tension_n, dtype=np.float64)
    weight = np.asarray(weight_n, dtype=np.float64)
    slack = tension <= 0.0 if extension_m is None else np.asarray(extension_m) <= 0.0
    over = tension > FULL_WEIGHT_SHARE * weight
    if extension_m is not None and stretch_m is not None:
        over = over | (np.asarray(extension_m) > np.asarray(stretch_m))
    state = np.where(over, OVER, TAUT)
    state = np.where(tension > CAUGHT_SHARE * weight, CAUGHT, state)
    return np.where(slack, SLACK, state)


def status_text(state: int, tension_n: float, weight_n: float, note: str = "") -> str:
    """``gantry: taut 0.88 W`` (tension over weight), ``gantry: slack``; ``note`` (the caller's phase or outcome)
    follows after a bar."""
    text = f"gantry: {STATE_WORDS[int(state)]}"
    if state != SLACK:
        text += f" {tension_n / weight_n:.2f} W"
    return f"{text} | {note}" if note else text


def log_label(rr, path: str, at, text: str, rgb) -> None:
    rr.log(path, rr.Points3D([np.asarray(at, dtype=np.float64)], radii=LABEL_RADIUS_M, colors=[rgb], labels=[text]))


def log_no_gantry(rr, world: str, base_w) -> None:
    """The label of a robot with no gantry (and no stale rope drawn from a gantry it had)."""
    for name in ("rope", "anchor", "plumb"):
        rr.log(f"{world}/{name}", rr.Clear(recursive=False))
    log_label(rr, f"{world}/status", np.asarray(base_w) + [0.0, 0.0, LABEL_LIFT_M], NO_GANTRY, LABEL_RGB)


def log_env(rr, world: str, sim: str, *, top, hook, rest_m: float, tension_n: float, weight_n: float,
            stretch_m: float | None = None, note: str = "", base_w=None) -> int:
    """One robot's rope. ``top`` and ``hook`` are the world points (NaN when no cable hangs, as with a tension
    program: then only the label, over ``base_w``), ``rest_m`` the unstretched length (NaN if unknown),
    ``stretch_m`` the full-weight stretch if known. Returns the rope's class."""
    if not weight_n > 0.0:
        raise ValueError(f"the robot's weight must be positive, got {weight_n}")
    top, hook = np.asarray(top, dtype=np.float64), np.asarray(hook, dtype=np.float64)
    roped = bool(np.isfinite(top).all() and np.isfinite(hook).all())
    length = float(np.linalg.norm(top - hook)) if roped else float("nan")
    extension = length - rest_m if roped and np.isfinite(rest_m) else None
    state = int(rope_state(tension_n, weight_n, extension_m=extension, stretch_m=stretch_m))
    text = status_text(state, tension_n, weight_n, note)
    rr.log(f"{sim}/tension_n", rr.Scalars(float(tension_n)))
    rr.log(f"{sim}/tension_share", rr.Scalars(float(tension_n / weight_n)))
    if roped:
        rgb = STATE_RGB[state]
        rr.log(f"{world}/rope", rr.LineStrips3D([[top, hook]], colors=[rgb], radii=ROPE_RADIUS_M))
        rr.log(f"{world}/anchor", rr.Points3D([top], radii=ANCHOR_RADIUS_M, colors=[ANCHOR_RGB]))
        plumb = top - [0.0, 0.0, rest_m if np.isfinite(rest_m) else length]
        rr.log(f"{world}/plumb", rr.LineStrips3D([[top, plumb]], colors=[PLUMB_RGB]))
        rr.log(f"{sim}/length_m", rr.Scalars(length))
        if np.isfinite(rest_m):
            rr.log(f"{sim}/rest_length_m", rr.Scalars(float(rest_m)))
        log_label(rr, f"{world}/status", top, text, LABEL_RGB)
    else:
        for name in ("rope", "anchor", "plumb"):
            rr.log(f"{world}/{name}", rr.Clear(recursive=False))
        log_label(rr, f"{world}/status", np.asarray(base_w) + [0.0, 0.0, LABEL_LIFT_M], text, LABEL_RGB)
    return state


def log_travel(rr, path: str, rig, offset=(0.0, 0.0, 0.0)) -> None:
    """The crane's reach, a static rectangle at the hoist's height, moved by ``offset``."""
    outline = rig.travel_outline_w() + np.asarray(offset)
    rr.log(path, rr.LineStrips3D([np.vstack([outline, outline[:1]])], colors=[TRAVEL_RGB]), static=True)


def log_crane(rr, world: str, rig, position, offset=(0.0, 0.0, 0.0)) -> None:
    """The bridge (a line along y at the trolley's x) and the trolley (a point) at the crane's ``position``
    (rig x, y, hook height), moved by ``offset``."""
    x, y = position[:2]
    y0, y1 = rig.travel_y_m
    off = np.asarray(offset)
    rr.log(f"{world}/bridge", rr.LineStrips3D([[rig.to_world((x, y0, 0)) + off, rig.to_world((x, y1, 0)) + off]],
                                              colors=[CRANE_RGB], radii=BRIDGE_RADIUS_M))
    rr.log(f"{world}/trolley", rr.Points3D([rig.to_world((x, y, 0)) + off], radii=TROLLEY_RADIUS_M, colors=[CRANE_RGB]))
