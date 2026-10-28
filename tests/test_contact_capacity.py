"""Contact budgets apply per world and must not silently truncate collisions."""

from types import SimpleNamespace

import pytest


def test_contact_capacity_counts_each_world_and_ignores_unused_pool_entries():
    wp = pytest.importorskip("warp")
    from humanoid_sim.physics.contact_capacity import ContactCapacityGuard

    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([4], dtype=int, device="cpu"),
        rigid_contact_shape0=wp.array([0, 1, 2, 0, 1], dtype=int, device="cpu"),
        rigid_contact_shape1=wp.array([1, 0, 0, 0, 0], dtype=int, device="cpu"))
    model = SimpleNamespace(shape_world=wp.array([-1, 0, 1], dtype=int, device="cpu"))
    manager = SimpleNamespace(get_contacts=lambda: contacts, get_model=lambda: model)
    guard = ContactCapacityGuard(manager, 2, 2, "cpu")
    guard.check()  # two contacts in world 0, one in world 1; global-static pair is ignored
    guard.check()  # counts reset rather than accumulating across steps
    contacts.rigid_contact_count.assign([5])
    with pytest.raises(RuntimeError, match="contact capacity exceeded"):
        guard.check()
