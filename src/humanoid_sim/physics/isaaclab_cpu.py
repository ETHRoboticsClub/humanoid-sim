"""Isaac Lab MJWarp manager that can run Newton's MuJoCo-C path (``use_mujoco_cpu``).

Isaac Lab 3.0 (``release/3.0.0`` @ 3271268) sizes its contact buffer with
``SolverMuJoCo.get_max_contact_count()``, which Newton 1.6.1 does not implement on the MuJoCo-C path, so
``use_mujoco_cpu=True`` fails at ``sim.reset()``. This manager allocates an empty contact report
instead. Consequence: Isaac Lab contact sensors get no data on this path (``update_contacts`` is also
unimplemented there); the physics is unaffected. Selected through the solver config's ``class_type``,
the extension point Isaac Lab provides. Remove once either upstream implements the call.
"""

from __future__ import annotations

from isaaclab.physics import PhysicsManager
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_newton.physics.newton_manager import NewtonManager
from newton import Contacts


class MujocoCpuManager(NewtonMJWarpManager):
    @classmethod
    def _initialize_contacts(cls) -> None:
        if cls._solver is not None and cls._solver.use_mujoco_cpu and not cls._needs_collision_pipeline:
            NewtonManager._contacts = Contacts(rigid_contact_max=0, soft_contact_max=0,
                                               device=PhysicsManager._device)
            return
        super()._initialize_contacts()


@configclass
class MujocoCpuSolverCfg(MJWarpSolverCfg):
    class_type: type = MujocoCpuManager
    use_mujoco_cpu: bool = True
