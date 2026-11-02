"""Contact budgets apply per world and must not silently truncate collisions."""

from types import SimpleNamespace

import pytest


def _solver(flag):
    return SimpleNamespace(_contacts_kamino=SimpleNamespace(_data=SimpleNamespace(_contact_overflow_warning_emitted=flag)))


def test_contact_capacity_guard_follows_kaminos_overflow_flag():
    wp = pytest.importorskip("warp")
    from humanoid_sim.physics.contact_capacity import ContactCapacityGuard, kamino_overflow_flag

    flag = wp.zeros(1, dtype=wp.int32, device="cpu")
    guard = ContactCapacityGuard(kamino_overflow_flag(_solver(flag)))
    guard.check()  # no contact dropped in the last conversion
    flag.fill_(1)  # Kamino dropped a contact: a world ran out of capacity
    with pytest.raises(RuntimeError, match="contact capacity exceeded"):
        guard.check()
    flag.zero_()  # Kamino clears the flag before each conversion
    guard.check()


def test_contact_capacity_guard_needs_the_flag():
    pytest.importorskip("warp")
    from humanoid_sim.physics.contact_capacity import kamino_overflow_flag

    assert kamino_overflow_flag(SimpleNamespace(_contacts_kamino=None)) is None  # the model admits no contacts
    with pytest.raises(RuntimeError, match="no overflow flag"):
        kamino_overflow_flag(SimpleNamespace(_contacts_kamino=SimpleNamespace(_data=SimpleNamespace())))
