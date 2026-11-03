"""One source for the friction of a disabled drive: the runtime, the training plant and a replay write the same solver
friction for the same drive state. The robot is a recorder, so no solver is needed."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from humanoid_sim.core import PlantCfg, PlantCore
from humanoid_sim.devices.demand import LAW_DISABLED, DriveDemand
from humanoid_sim.model.description import Description
from humanoid_sim.runtime.host import RuntimeHost
from humanoid_sim.sim import SimCore
from humanoid_sim.train.plant import RobotPlant

OFFSET = 4  # the articulation's joint order differs from the plant's


class _Robot:
    """The solver's per-joint friction ``[E, OFFSET + J]``, written through the same call as Isaac Lab's."""

    def __init__(self, enabled: torch.Tensor):
        self.friction = torch.cat([torch.full((len(enabled), OFFSET), -1.0), enabled.float()], 1)

    def write_joint_friction_coefficient_to_sim_index(self, joint_friction_coeff, joint_ids):
        self.friction[:, joint_ids] = joint_friction_coeff


class _Path:
    """One front end over a fresh plant: ``submit(off)`` tells it which drives are off (``off[env][drive]``) the way
    that front end learns it, then runs what it runs before a physics step; ``reset(env_ids)`` is its reset."""

    def __init__(self, kind: str, desc: Description, envs: int):
        self.kind = kind
        self.core = core = PlantCore(desc, list(desc.joints), envs, PlantCfg())
        self.robot = _Robot(core.enabled_friction)
        ids = torch.arange(len(core.solver_map.joint_names)) + OFFSET
        if kind == "training":
            self.front = RobotPlant.__new__(RobotPlant)
            self.front.core, self.front.robot, self.front.solver_joint_ids = core, self.robot, ids
            self._write = self.front._write_friction
        else:  # replay and runtime both write through SimCore, before its backend step
            self.front = SimCore.__new__(SimCore)
            self.front.plant, self.front.backend = core, SimpleNamespace(robot=self.robot)
            self.front._to_solver = ids
            self._write = self.front.write_friction
        if kind == "runtime":
            self.host = RuntimeHost.__new__(RuntimeHost)
            self.host.sim, self.host._drive_off = self.front, None

    def submit(self, off: list[list[bool]]) -> None:
        core, n = self.core, self.core.map.num_drives
        if self.kind == "runtime":  # one env; the demand reaches the plant through a CUDA graph
            self.host._set_drives_off(np.array(off[0]))
        else:
            demand = DriveDemand.pd(torch.zeros(len(off), n), core.drive_stiffness.expand(len(off), -1),
                                    core.drive_damping.expand(len(off), -1))
            demand.law[torch.tensor(off, dtype=torch.bool)] = LAW_DISABLED
            core.submit(demand, torch.zeros(len(off), n, dtype=torch.long))
        self._write()

    def reset(self, env_ids: list[int]) -> None:
        self.core.reset(torch.tensor(env_ids))
        if self.kind == "runtime":
            self.host._drive_off = None  # as the host does after every plant reset
        self._write()

    def joints(self) -> torch.Tensor:
        """What the solver holds for the plant's joints ``[E, J]``."""
        return self.robot.friction[:, OFFSET:]


KINDS = ("replay", "training", "runtime")


@pytest.fixture(scope="module")
def desc():
    return Description.load("ethrc_v1")


def _expected(core: PlantCore, off_row: list[bool], env: int = 0) -> torch.Tensor:
    f = core.enabled_friction[env].clone()
    for slot, (drive, joint) in enumerate(zip(core.backdrive_drives.tolist(), core.backdrive_joints.tolist())):
        if off_row[drive]:
            f[joint] = core.backdrive_friction[slot]
    return f.float()


def _states(n: int) -> dict[str, list[bool]]:
    return {"all on": [False] * n, "all off": [True] * n, "alternate": [bool(k % 2) for k in range(n)]}


def test_the_description_has_back_drive_joints(desc):
    core = PlantCore(desc, list(desc.joints), 1, PlantCfg())
    if not len(core.backdrive_joints):
        pytest.skip("the description records no backdrive torque")
    assert not torch.equal(core.joint_friction()[0], core.enabled_friction[0])


@pytest.mark.parametrize("state", ["all on", "all off", "alternate"])
def test_every_path_writes_the_same_friction_for_the_same_state(desc, state):
    results = {}
    for kind in KINDS:
        path = _Path(kind, desc, 1)
        off = _states(path.core.map.num_drives)[state]
        path.submit([off])
        torch.testing.assert_close(path.joints()[0], _expected(path.core, off), msg=lambda m, k=kind: f"{k}: {m}")
        assert (path.robot.friction[:, :OFFSET] == -1).all()  # joints that are not the plant's are not touched
        results[kind] = path.joints().clone()
    torch.testing.assert_close(results["replay"], results["training"])
    torch.testing.assert_close(results["replay"], results["runtime"])


def test_every_path_follows_a_change_of_state(desc):
    for kind in KINDS:
        path = _Path(kind, desc, 1)
        for state in ("all on", "all off", "alternate", "all on"):
            off = _states(path.core.map.num_drives)[state]
            path.submit([off])
            torch.testing.assert_close(path.joints()[0], _expected(path.core, off),
                                       msg=lambda m, k=kind, s=state: f"{k} {s}: {m}")


def test_each_env_has_its_own_friction_in_replay_and_training(desc):
    for kind in ("replay", "training"):
        path = _Path(kind, desc, 3)
        states = _states(path.core.map.num_drives)
        rows = [states["all on"], states["all off"], states["alternate"]]
        path.submit(rows)
        for env, row in enumerate(rows):
            torch.testing.assert_close(path.joints()[env], _expected(path.core, row, env),
                                       msg=lambda m, k=kind, e=env: f"{k} env {e}: {m}")


def test_a_reset_turns_that_env_s_drives_off_in_every_path(desc):
    for kind in KINDS:
        envs = 1 if kind == "runtime" else 2
        path = _Path(kind, desc, envs)
        on = [False] * path.core.map.num_drives
        path.submit([on] * envs)
        path.reset([0])
        torch.testing.assert_close(path.joints()[0], _expected(path.core, [True] * len(on)),
                                   msg=lambda m, k=kind: f"{k}: {m}")
        if envs == 2:
            torch.testing.assert_close(path.joints()[1], _expected(path.core, on, 1), msg=lambda m: f"{kind}: {m}")


def test_a_randomized_or_fitted_friction_is_the_enabled_one_in_every_path(desc):
    for kind in KINDS:
        path = _Path(kind, desc, 1)
        n = path.core.map.num_drives
        path.core.enabled_friction *= 3.0  # what a randomization or a fit writes to the solver
        path.core.friction_changed = True
        off = _states(n)["alternate"]
        path.submit([off])
        j = path.core.backdrive_joints  # the other joints' friction is the solver's own, written by whoever set it
        torch.testing.assert_close(path.joints()[0, j], _expected(path.core, off)[j], msg=lambda m, k=kind: f"{k}: {m}")


def test_the_friction_is_written_only_after_a_change(desc):
    for kind in KINDS:
        path = _Path(kind, desc, 1)
        path.submit([[True] * path.core.map.num_drives])
        assert not path.core.friction_changed
        path.robot.friction[:] = -2.0  # a second write would overwrite this
        path._write()
        assert (path.robot.friction == -2.0).all()
