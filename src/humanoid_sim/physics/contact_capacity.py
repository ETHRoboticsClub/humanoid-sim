"""Detect Kamino contact truncation on device, before another plant step."""

import torch
import warp as wp


class ContactCapacityGuard:
    """Fail instead of continuing with contacts beyond Kamino's per-world cap.

    Kamino's Newton-to-Kamino conversion culls speculative contacts (positive margin-shifted distance), reserves
    per-world capacity for the rest and raises a device flag for every contact it drops; the flag is cleared before
    each conversion. The guard asserts on that flag after each step, so it counts exactly the contacts Kamino keeps.
    Counting Newton's whole pool instead also counts the speculative contacts within the shapes' contact gap and
    stops on falls that fit. The asynchronous assertion adds no host sync.
    """

    def __init__(self, overflow: wp.array):
        self._overflow = wp.to_torch(overflow)

    def check(self):
        torch._assert_async(self._overflow[0] == 0, "Kamino contact capacity exceeded; increase max_contacts_per_world")


def kamino_overflow_flag(solver) -> wp.array | None:
    """The overflow flag of ``solver``'s Kamino contact buffers, or None when the model admits no contacts."""
    contacts = getattr(solver, "_contacts_kamino", None)
    if contacts is None:
        return None
    flag = getattr(getattr(contacts, "_data", None), "_contact_overflow_warning_emitted", None)
    if flag is None:
        raise RuntimeError("Kamino's contacts expose no overflow flag: revalidate the contact capacity guard "
                           "against this Newton")
    return flag


def contact_capacity_guard(physics: dict, num_envs: int, device):
    from humanoid_sim.physics.isaaclab import running_solver

    if running_solver() != "kamino":
        return None
    from isaaclab_newton.physics import NewtonManager

    flag = kamino_overflow_flag(NewtonManager._solver)
    return None if flag is None else ContactCapacityGuard(flag)
