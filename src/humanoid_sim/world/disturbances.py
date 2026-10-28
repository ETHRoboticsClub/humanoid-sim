"""Force pushes on a body: horizontal in the world frame, random heading, at random intervals.

A push of ``magnitude_n`` (uniform) on ``body`` lasts ``duration_s`` (uniform); the next one starts ``interval_s``
(uniform) after it ends. Forces go through the articulation's permanent wrench composer (graph-safe, no host
syncs).
Call :meth:`ForcePush.step` once per physics or policy step with the plant time; :meth:`ForcePush.reset` for the
envs being reset. A level in [0, 1] scales the magnitude (curricula).
"""

from __future__ import annotations

import math

import torch

from humanoid_sim.timing.clock import NS_PER_S


class VelocityPush:
    """Periodically set root horizontal velocity, preserving vertical and angular velocity.

    The caller chooses the interval and uniform velocity range. Times are ns;
    masks and writes are batched without extracting active indices to the host.
    """

    def __init__(self, robot, *, interval_ns: int, velocity_range_m_s: tuple[float, float], seed: int = 0):
        if interval_ns <= 0 or velocity_range_m_s[0] > velocity_range_m_s[1]:
            raise ValueError("velocity push needs a positive interval and ordered velocity range")
        self.robot, self.interval_ns, self.velocity_range = robot, interval_ns, velocity_range_m_s
        self.next_ns = torch.full((robot.num_instances,), interval_ns, device=robot.device, dtype=torch.long)
        self.generator = torch.Generator(device=robot.device).manual_seed(seed)
        self.ids = torch.arange(robot.num_instances, device=robot.device)

    def reset(self, ids: torch.Tensor, t_ns: int) -> None:
        self.next_ns[ids] = t_ns + self.interval_ns

    def step(self, t_ns: int) -> None:
        active = self.next_ns <= t_ns
        velocity = self.robot.data.root_com_vel_w.torch.clone()
        lo, hi = self.velocity_range
        drawn = lo + (hi - lo) * torch.rand(velocity.shape[0], 2, device=velocity.device, generator=self.generator)
        velocity[:, :2] = torch.where(active[:, None], drawn, velocity[:, :2])
        self.robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=self.ids)
        self.next_ns = torch.where(active, t_ns + self.interval_ns, self.next_ns)


class ForcePush:
    def __init__(self, robot, body: str, *, interval_s: tuple[float, float], duration_s: tuple[float, float],
                 magnitude_n: tuple[float, float], level: float = 1.0, seed: int = 0):
        self.robot = robot
        self.body_id = robot.body_names.index(body)
        self.interval, self.duration, self.magnitude = interval_s, duration_s, magnitude_n
        self.level = level
        e, dev = robot.num_instances, torch.device(robot.device)
        self.num_envs, self.device = e, dev
        self.generator = torch.Generator(device=dev).manual_seed(seed)
        self.next_ns = torch.zeros(e, dtype=torch.int64, device=dev)
        self.end_ns = torch.zeros(e, dtype=torch.int64, device=dev)
        self.force = torch.zeros(e, 3, device=dev)
        self._all = torch.arange(e, device=dev)
        self._body = torch.tensor([self.body_id], device=dev)

    def _uniform(self, n: int, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(n, generator=self.generator, device=self.device)

    def reset(self, env_ids: torch.Tensor, t_ns: int) -> None:
        self.next_ns[env_ids] = t_ns + (self._uniform(len(env_ids), self.interval) * NS_PER_S).to(torch.int64)
        self.end_ns[env_ids] = t_ns
        self.force.index_fill_(0, env_ids, 0.0)
    def step(self, t_ns: int) -> None:
        start = self.next_ns <= t_ns
        heading = self._uniform(self.num_envs, (0.0, 2 * math.pi))
        size = self._uniform(self.num_envs, self.magnitude) * self.level
        push = torch.stack([size * torch.cos(heading), size * torch.sin(heading), torch.zeros_like(size)], dim=1)
        self.force = torch.where(start.unsqueeze(1), push, self.force)
        end = t_ns + (self._uniform(self.num_envs, self.duration) * NS_PER_S).to(torch.int64)
        self.end_ns = torch.where(start, end, self.end_ns)
        gap = (self._uniform(self.num_envs, self.interval) * NS_PER_S).to(torch.int64)
        self.next_ns = torch.where(start, end + gap, self.next_ns)
        self.force = torch.where((self.end_ns <= t_ns).unsqueeze(1), torch.zeros_like(self.force), self.force)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.force.unsqueeze(1), torques=torch.zeros_like(self.force).unsqueeze(1), body_ids=self._body,
            env_ids=self._all, is_global=True)
