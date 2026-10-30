"""Validate physical crank stops and enabled native loop closures."""


def check_native_cranks(desc):
    import numpy as np
    from isaaclab_newton.physics import NewtonManager
    from pxr import Usd, UsdPhysics

    from humanoid_sim.model.usd import crank_joints

    model = NewtonManager.get_model()
    labels = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
    starts = model.joint_qd_start.numpy()
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    counts = set()
    for drive, name in crank_joints(desc).items():
        indices = [index for index, label in enumerate(labels) if label == name]
        if not indices:
            raise RuntimeError(f"native USD articulation lacks crank {name}")
        counts.add(len(indices))
        for index in indices:
            actual = [lower[starts[index]], upper[starts[index]]]
            if not np.allclose(actual, desc.drives[drive].shaft_limits, atol=1e-6, rtol=0):
                raise RuntimeError(f"native crank {name} has incorrect physical stops: {actual}")
    stage = Usd.Stage.Open(str(desc.usd_path))
    enabled = model.joint_enabled.numpy()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or not UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
            continue
        indices = [index for index, label in enumerate(labels) if label == prim.GetName()]
        if {len(indices)} != counts or not np.all(enabled[indices]):
            raise RuntimeError(f"native USD articulation lacks enabled loop closures for {prim.GetName()}")
