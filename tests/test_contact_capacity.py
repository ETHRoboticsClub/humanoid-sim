"""Contact budgets apply per world and must not silently truncate collisions."""

from types import SimpleNamespace

import pytest

wp = pytest.importorskip("warp")


def _solver(flag, active=None, capacity=None):
    """A stand-in for the Kamino solver's contact buffers: the overflow flag, the worlds' counts and capacities."""
    contacts = SimpleNamespace(_data=SimpleNamespace(_contact_overflow_warning_emitted=flag), model_max_contacts_host=1)
    if active is not None:
        contacts.world_active_contacts = wp.array(active, dtype=wp.int32, device="cpu")
        contacts.world_max_contacts = wp.array(capacity, dtype=wp.int32, device="cpu")
        contacts.world_max_contacts_host = list(capacity)
        contacts.model_max_contacts_host = sum(capacity)
    return SimpleNamespace(_contacts_kamino=contacts)


def _flag(value=0):
    return wp.array([value], dtype=wp.int32, device="cpu")


def test_strict_guard_follows_kaminos_overflow_flag():
    from humanoid_sim.physics.contact_capacity import ContactCapacityGuard

    flag = _flag()
    guard = ContactCapacityGuard(_solver(flag, [3, 4], [4, 4]))
    guard.check()  # no contact dropped in the last conversion
    flag.fill_(1)  # Kamino dropped a contact: a world ran out of capacity
    with pytest.raises(RuntimeError, match="contact capacity exceeded"):
        guard.check()
    flag.zero_()  # Kamino clears the flag before each conversion
    guard.check()


def test_flagging_guard_leaves_a_worlds_overflow_to_its_consumer():
    from humanoid_sim.physics.contact_capacity import ContactCapacityGuard, kamino_contact_overflow

    flag = _flag(1)
    solver = _solver(flag, [3, 4, 2], [4, 4, 4])  # world 1 ran out; Kamino held it at its capacity
    ContactCapacityGuard(solver, strict=False).check()  # no assertion: the world is named instead
    assert kamino_contact_overflow(solver).tolist() == [False, True, False]


def test_flagging_guard_still_stops_on_a_drop_no_world_explains():
    from humanoid_sim.physics.contact_capacity import ContactCapacityGuard

    solver = _solver(_flag(1), [3, 2], [4, 4])  # the flag is up, every world has room: the model's pool ran out
    with pytest.raises(RuntimeError, match="in no single world"):
        ContactCapacityGuard(solver, strict=False).check()


def test_a_world_at_its_capacity_is_not_overflowed_while_nothing_dropped():
    from humanoid_sim.physics.contact_capacity import kamino_contact_overflow

    flag = _flag(0)
    solver = _solver(flag, [4, 4], [4, 4])
    assert kamino_contact_overflow(solver).tolist() == [False, False]
    flag.fill_(1)
    assert kamino_contact_overflow(solver).tolist() == [True, True]  # a full world while another drops: named, too


def test_active_contacts_are_a_view_of_kaminos_counters():
    from humanoid_sim.physics.contact_capacity import kamino_active_contacts

    solver = _solver(_flag(), [3, 4], [4, 6])
    active, capacity = kamino_active_contacts(solver)
    assert active.tolist() == [3, 4] and capacity == 6
    solver._contacts_kamino.world_active_contacts.fill_(1)  # the same memory: no copy
    assert active.tolist() == [1, 1]
    assert kamino_active_contacts(SimpleNamespace(_contacts_kamino=None)) is None


def test_the_overflow_helpers_name_no_worlds_without_contacts():
    from humanoid_sim.physics.contact_capacity import kamino_contact_overflow

    assert kamino_contact_overflow(SimpleNamespace(_contacts_kamino=None)) is None


def test_contact_capacity_guard_needs_the_flag():
    from humanoid_sim.physics.contact_capacity import kamino_overflow_flag

    assert kamino_overflow_flag(SimpleNamespace(_contacts_kamino=None)) is None  # the model admits no contacts
    with pytest.raises(RuntimeError, match="no overflow flag"):
        kamino_overflow_flag(SimpleNamespace(_contacts_kamino=SimpleNamespace(_data=SimpleNamespace())))


def test_the_factory_builds_a_strict_guard_by_default_and_a_flagging_one_on_request(monkeypatch):
    from humanoid_sim.physics import contact_capacity

    flag = _flag(1)
    solver = _solver(flag, [4, 1], [4, 4])
    monkeypatch.setattr(contact_capacity, "_running_kamino_solver", lambda: solver)
    with pytest.raises(RuntimeError, match="contact capacity exceeded; increase"):
        contact_capacity.contact_capacity_guard({}, 2, "cpu").check()
    contact_capacity.contact_capacity_guard({}, 2, "cpu", strict=False).check()  # world 0 is named instead
    monkeypatch.setattr(contact_capacity, "_running_kamino_solver", lambda: None)
    assert contact_capacity.contact_capacity_guard({}, 2, "cpu") is None  # another solver: no guard
