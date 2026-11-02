"""Validate recorded hardware gains; replay matches numerical PD at reset."""
import numpy as np


def gain_support(capture, plant, *, gains=None):
    gains = capture.gains() if gains is None else gains
    names = list(plant.map.drive_names)
    missing = set(names) - set(gains)
    if missing:
        raise ValueError(f"missing recorded gains for plant drives {sorted(missing)}")
    recorded = np.array([(gains[n].stiffness, gains[n].damping) for n in names])
    if not np.isfinite(recorded).all() or (recorded[:, 0] <= 0).any() or (recorded[:, 1] < 0).any():
        raise ValueError("replay requires finite positive P and nonnegative D")
    return {"mode": "capture", "drives": names, "recorded_p_d": recorded.tolist(),
            "criterion": "native solver PD matches recorded drive gains at each replay reset",
            "reference_source": "recorded capture gains; not fitted or promoted"}


def check_run_gain_support(plant):
    warned = bool(plant.gain_warnings.any().item())
    if warned:
        raise RuntimeError("recorded-gain replay exceeded numerical solver gain support")
    return {"drive_warning": warned}
