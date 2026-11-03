"""Scoped Newton model construction shared by training and runtime scenes."""

from contextlib import contextmanager

from humanoid_sim.physics.crank_stops import crank_stop_hook


class _FixedJointCollapse:
    """Capture the replicated builder and merge rigid assemblies before finalization."""

    def __init__(self):
        self.builder = None

    def world(self, builder, world, position, quaternion):
        self.builder = builder

    def finalize(self, _):
        if self.builder is None:
            raise RuntimeError("fixed-joint collapse received no Newton scene builder")
        collapse_fixed_joints(self.builder)


def collapse_fixed_joints(builder):
    """Merge geometry with Newton, then assemble inertias about each merged COM.

    The pinned Newton collapse translates child inertia by the body-origin
    offset, although imported USD inertias are about each body's COM. Rebuild
    compound tensors from the original inertias with the parallel-axis theorem.
    This is a model-build operation; no state or library implementation changes
    during stepping.
    """
    import numpy as np
    import warp as wp

    mass = np.asarray(builder.body_mass, dtype=np.float64).copy()
    com = np.asarray(builder.body_com, dtype=np.float64).reshape(-1, 3).copy()
    inertia = np.asarray(builder.body_inertia, dtype=np.float64).reshape(-1, 3, 3).copy()
    # A world-fixed articulation root remains a named body for sensors and supports.
    # Merge its rigid children, but do not absorb the root into the static world.
    anchors = [i for i, parent in enumerate(builder.joint_parent) if parent < 0]
    mapping = builder.collapse_fixed_joints(joints_to_keep=anchors)
    groups = {}
    for child, parent in mapping["body_merged_parent"].items():
        if parent >= 0:
            groups.setdefault(parent, []).append(child)

    for parent, children in groups.items():
        new_id = mapping["body_remap"][parent]
        centre = np.asarray(builder.body_com[new_id], dtype=np.float64)
        compound = np.zeros((3, 3))
        for body in [parent, *children]:
            frame = mapping["body_merged_transform"].get(body, wp.transform_identity())
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(frame))).reshape(3, 3)
            offset = np.asarray(wp.transform_point(frame, wp.vec3(com[body]))) - centre
            compound += rotation @ inertia[body] @ rotation.T
            compound += mass[body] * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
        tensor = wp.mat33(compound)
        builder.body_inertia[new_id] = tensor
        builder.body_inv_inertia[new_id] = wp.inverse(tensor) if builder.body_mass[new_id] > 0 else wp.mat33(0.)
    return mapping


@contextmanager
def model_build(robot: str, physics: dict, solver: str | None = None):
    """Build and reset a scene with its profile's model transformations.

    ``physics.<solver>.collapse_fixed_joints`` merges fixed bodies, accumulating
    their mass and inertia and transforming attached shapes and joints. Body
    selectors and randomization then address the surviving rigid assemblies.
    The callback is scoped to this build, including when construction raises.
    """
    solver = solver or physics["solver"]
    collapse = physics[solver].get("collapse_fixed_joints", False)
    if not isinstance(collapse, bool):
        raise ValueError("collapse_fixed_joints must be a boolean")
    with crank_stop_hook(robot, physics, solver) as stops:
        if not collapse:
            yield stops
            return
        if stops is not None:
            raise ValueError("fixed-joint collapse with surrogate crank tendons is unsupported")
        from isaaclab.physics import PhysicsEvent
        from isaaclab_newton.cloner import newton_builder_world_hook
        from isaaclab_newton.physics import NewtonManager

        hook = _FixedJointCollapse()
        handle = NewtonManager.register_callback(hook.finalize, PhysicsEvent.MODEL_INIT,
                                                  name="fixed_joint_collapse", wrap_weak_ref=False)
        try:
            with newton_builder_world_hook(hook.world):
                yield stops
        finally:
            handle.deregister()
