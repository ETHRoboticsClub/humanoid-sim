"""Isaac Lab manager-based hook: an action term that owns the :class:`~humanoid_sim.train.plant.RobotPlant`.

Subclass :class:`PlantAction` and implement :meth:`PlantAction.demands` (policy actions → drive demands and the
time they leave the master). The term submits them once per environment step, runs the plant before every
physics step and resets it with the env (after the reset events have written the state). Other terms reach
the plant as ``env.action_manager.get_term(<name>).plant``. Imports Isaac Lab: load it in the Isaac Lab image.

With ``compute_delay`` set, each step's demands leave later by the robot computer's policy compute time, drawn per
env and step from a measured distribution (:mod:`humanoid_sim.timing.compute`): the time the policy on the robot
takes from its tick to its demand.
"""

from __future__ import annotations

import hashlib
from dataclasses import MISSING, field

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from humanoid_sim.devices.demand import DriveDemand
from humanoid_sim.timing.compute import ComputeDelay
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
        self.compute_delay = compute_delay(cfg.compute_delay, self.plant.profile)
        self.last_compute_delay_ns = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        # its own stream: the draws do not repeat the randomization's, which uses the plant seed itself
        stream = hashlib.sha256(f"{self.plant.cfg.seed}/compute_delay".encode()).digest()
        self._compute_gen = torch.Generator(device=env.device).manual_seed(int.from_bytes(stream[:8], "little") >> 1)

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
        if self.compute_delay is not None:
            self.last_compute_delay_ns = self.compute_delay.sample(self.num_envs, device=self.device,
                                                                   generator=self._compute_gen)
            sent_ns = sent_ns.to(torch.int64) + self.last_compute_delay_ns
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
    # The policy's compute time on the robot, added to each step's sent time: (workload, load) from the robot
    # profile's platform.compute block, or a ComputeDelay. None: the demands leave at the time demands() gives.
    compute_delay: tuple[str, str] | ComputeDelay | None = None


def compute_delay(spec, profile: dict) -> ComputeDelay | None:
    """The :class:`ComputeDelay` a ``PlantActionCfg.compute_delay`` names."""
    if spec is None or isinstance(spec, ComputeDelay):
        return spec
    workload, load = spec
    return ComputeDelay.from_profile(profile, workload, load)
