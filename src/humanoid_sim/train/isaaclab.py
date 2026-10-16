"""Isaac Lab manager-based hook: an action term that owns the :class:`~humanoid_sim.train.plant.RobotPlant`.

Subclass :class:`PlantAction` and implement :meth:`PlantAction.demands` (policy actions → drive demands and the
time they leave the master). The term submits them once per environment step, runs the plant before every
physics step and resets it with the env (after the reset events have written the state). Other terms reach
the plant as ``env.action_manager.get_term(<name>).plant``. Imports Isaac Lab: load it in the Isaac Lab image.
"""

from __future__ import annotations

from dataclasses import MISSING, field

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.train.plant import RobotPlant, TrainPlantCfg


class PlantAction(ActionTerm):
    cfg: PlantActionCfg

    def __init__(self, cfg: PlantActionCfg, env):
        super().__init__(cfg, env)
        scene = env.scene
        self.plant = RobotPlant(self._asset, cfg.plant,
                                imu=scene[cfg.imu_name] if cfg.imu_name else None,
                                contacts=scene[cfg.contacts_name] if cfg.contacts_name else None,
                                physics_manager=env.sim.physics_manager)
        self._raw = torch.zeros(env.num_envs, self.action_dim, device=env.device)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def demands(self, actions: torch.Tensor) -> tuple[DriveDemand, torch.Tensor]:
        """Drive demands ``[E, D]`` for these actions and the time ``[E]`` they leave the master (ns, plant
        clock; ``self.plant.t_ns`` is now)."""
        raise NotImplementedError

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        demand, sent_ns = self.demands(actions)
        self.plant.submit(demand, sent_ns)

    def apply_actions(self):
        self.plant.pre_physics()

    def reset(self, env_ids=None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw.index_fill_(0, env_ids, 0.0)
        self.plant.reset(env_ids)


@configclass
class PlantActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PlantAction
    asset_name: str = "robot"
    action_dim: int = MISSING
    plant: TrainPlantCfg = field(default_factory=TrainPlantCfg)
    imu_name: str | None = "imu"
    contacts_name: str | None = "contacts"
