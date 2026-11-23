"""Detect Kamino contact truncation on device, before another plant step."""

import torch
import warp as wp


def _kamino_contacts(solver):
    """``solver``'s Kamino contact buffers, or None when the model admits no contacts."""
    contacts = getattr(solver, "_contacts_kamino", None)
    if contacts is None or contacts.model_max_contacts_host == 0:
        return None
    return contacts


def _running_kamino_solver():
    """The running Kamino solver, or None when the solver is another."""
    from humanoid_sim.physics.isaaclab import running_solver

    if running_solver() != "kamino":
        return None
    from isaaclab_newton.physics import NewtonManager

    return NewtonManager._solver


def kamino_active_contacts(solver=None) -> tuple[torch.Tensor, int] | None:
    """``([worlds] active contacts of the last physics step, per-world capacity)`` of the Kamino ``solver`` (default:
    the running one), as a zero-copy view of Kamino's own counters (no device read), or None when the solver is not
    Kamino or the model admits no contacts. The counts are the contacts Kamino keeps: it culls speculative ones
    (still apart) before it reserves capacity, and a world that ran out stays at its capacity."""
    contacts = _kamino_contacts(_running_kamino_solver() if solver is None else solver)
    if contacts is None:
        return None
    return wp.to_torch(contacts.world_active_contacts), int(max(contacts.world_max_contacts_host))


def kamino_contact_overflow(solver=None) -> torch.Tensor | None:
    """``[worlds]`` bool, on the device: the worlds that dropped a contact in the last physics step because they ran
    out of capacity (``max_contacts_per_world``), or None when the solver is not Kamino or the model admits no contacts.
    Kamino keeps one device flag for the whole model, cleared before each contact conversion, and holds a world that
    ran out at exactly its capacity: a world is overflowed when the flag is set and the world sits at its capacity.
    (A world that fills its capacity exactly, with nothing dropped, is also named while another world drops; the
    ending of that episode costs one rare reset.) Read it after every physics step to cover a control step: the flag
    describes the last step only. No host sync."""
    solver = _running_kamino_solver() if solver is None else solver
    contacts = _kamino_contacts(solver)
    if contacts is None:
        return None
    flag = kamino_overflow_flag(solver)
    full = wp.to_torch(contacts.world_active_contacts) >= wp.to_torch(contacts.world_max_contacts)
    return full & (wp.to_torch(flag)[0] != 0)


class ContactCapacityGuard:
    """What a plant does when a Kamino world has more contacts than its capacity.

    Kamino's Newton-to-Kamino conversion culls speculative contacts (positive margin-shifted distance), reserves
    per-world capacity for the rest and raises a device flag for every contact it drops; the flag is cleared before
    each conversion. The guard reads that flag after each step, so it counts exactly the contacts Kamino keeps.
    Counting Newton's whole pool instead also counts the speculative contacts within the shapes' contact gap and
    stops on falls that fit. The asynchronous assertion adds no host sync.

    ``strict`` (the default): an overflow stops the run with an assertion, so nothing continues on dropped contacts.
    Not strict: the plant's consumer reads the overflowed worlds (:func:`kamino_contact_overflow`, after each physics
    step) and ends their episodes; the guard still stops on a drop in no world at its capacity (the model's contact
    pool ran out), which no episode can answer for.
    """

    def __init__(self, solver, *, strict: bool = True):
        self._solver, self.strict = solver, strict
        self._flag = wp.to_torch(kamino_overflow_flag(solver))

    def check(self):
        dropped = self._flag[0] != 0
        if self.strict:
            torch._assert_async(~dropped, "Kamino contact capacity exceeded; increase max_contacts_per_world")
        else:
            worlds = kamino_contact_overflow(self._solver)
            attributed = torch.zeros_like(dropped) if worlds is None else worlds.any()
            torch._assert_async(~dropped | attributed,
                                "Kamino contact capacity exceeded in the model's pool, in no single world")


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


def contact_capacity_guard(physics: dict, num_envs: int, device, *, strict: bool = True):
    solver = _running_kamino_solver()
    if solver is None or kamino_overflow_flag(solver) is None:
        return None
    return ContactCapacityGuard(solver, strict=strict)


# Kamino's dense Delassus operator (and its factor) is one float32 array over every world, each world's block sized by
# its largest possible constraint count, indexed with int32 offsets in every dense kernel (Newton 1.6.1, and Newton 1.8,
# which raises on it): worlds x rows^2 must stay below 2^31.
KAMINO_DENSE_LIMIT = 2**31 - 1


def kamino_constraint_rows(solver) -> dict[str, int]:
    """The largest world's constraint rows of a built SolverKamino, by kind: ``bilateral`` (joints, with each implicit
    joint dynamics row), ``bounded`` (bounded implicit-PD effort rows), ``limits``, ``contacts`` (three per contact of
    the per-world capacity), their ``total``, and ``max_worlds``: the most worlds the dense operator holds at it."""
    impl = solver._solver_kamino
    info = impl._model.info
    bilateral = max(info.num_joint_bilateral_cts.numpy().tolist())
    bounded = max(info.num_joint_bounded_cts.numpy().tolist())
    limits = impl._limits
    nl = max(limits.world_max_limits_host) if limits is not None and limits.model_max_limits_host > 0 else 0
    contacts = getattr(solver, "_contacts_kamino", None)
    nc = max(contacts.world_max_contacts_host) if contacts is not None and contacts.model_max_contacts_host > 0 else 0
    total = int(bilateral + bounded + nl + 3 * nc)
    return {"bilateral": int(bilateral), "bounded": int(bounded), "limits": int(nl), "contacts": int(3 * nc),
            "total": total, "max_worlds": KAMINO_DENSE_LIMIT // total**2}
