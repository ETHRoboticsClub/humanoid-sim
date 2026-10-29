"""Detect Kamino contact truncation on device, before another plant step."""

import torch
import warp as wp


@wp.kernel
def _count_contacts(count: wp.array[int], shape_a: wp.array[int], shape_b: wp.array[int],
                    shape_world: wp.array[int], per_world: wp.array[int]):
    i = wp.tid()
    if i < count[0]:
        world = wp.max(shape_world[shape_a[i]], shape_world[shape_b[i]])
        if world >= 0:
            wp.atomic_add(per_world, world, 1)


class ContactCapacityGuard:
    """Fail instead of continuing with contacts beyond Kamino's per-world cap.

    Newton's collision pool precedes Kamino's truncated per-world buffers. Count
    that pool with a device kernel; the asynchronous assertion adds no host sync.
    """

    def __init__(self, manager, num_envs: int, capacity: int, device):
        self.manager = manager
        self.capacity = capacity
        self.counts = wp.zeros(num_envs, dtype=int, device=str(device))
        self._counts_torch = wp.to_torch(self.counts)

    def check(self):
        contacts = self.manager.get_contacts()
        if contacts is None:
            return
        self.counts.zero_()
        wp.launch(_count_contacts, dim=contacts.rigid_contact_shape0.shape[0], inputs=[
            contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
            self.manager.get_model().shape_world, self.counts], device=self.counts.device)
        torch._assert_async((self._counts_torch <= self.capacity).all(),
                            "Kamino contact capacity exceeded; increase max_contacts_per_world")


def contact_capacity_guard(physics: dict, num_envs: int, device):
    from humanoid_sim.physics.isaaclab import running_solver

    if running_solver() != "kamino":
        return None
    from isaaclab_newton.physics import NewtonManager

    return ContactCapacityGuard(NewtonManager, num_envs, int(physics["kamino"]["max_contacts_per_world"]), device)
