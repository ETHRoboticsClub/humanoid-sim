"""The capture contract a PACE fit checks (sysid/contract.py), on synthetic captures."""

import numpy as np
import pytest

from humanoid_sim.replay.exc import CommandTrace, DriveGains, DriveTrace, ImuTrace, sample_hold

PERIOD = 20_000_000
DELAY = 6_000_000


@pytest.fixture(scope="module")
def plant():
    import torch

    from humanoid_sim.core import PlantCfg, PlantCore
    from humanoid_sim.model.description import Description

    desc = Description.load()
    return PlantCore(desc, list(desc.joints), 1, PlantCfg(dtype=torch.float64))


class Cap:
    """Commands every 20 ms for 10 s; every drive's 0x6062 and position follow them 6 ms later."""

    name = "synthetic"

    def __init__(self, plant, *, offset=0.0, demand=True, cmd_shift_ns=0, drop=None, imu=True):
        from humanoid_sim.replay.replay import drive_targets

        joints = list(plant.desc.joints)
        n = 500
        t_cmd = np.arange(n, dtype=np.int64) * PERIOD
        self.cmds = CommandTrace(t_cmd + cmd_shift_ns, joints,
                                 0.05 * np.sin(np.arange(n)[:, None] / 7 + np.arange(len(joints))[None]))
        targets, _ = drive_targets(plant, CommandTrace(t_cmd, joints, self.cmds.positions_rad))
        self.t = np.arange(0, n * PERIOD, 5_000_000, dtype=np.int64)
        self._pdo = {}
        for i, d in enumerate(plant.map.drive_names):
            if d == drop:
                continue
            x = sample_hold(t_cmd, targets[:, i], self.t - DELAY)
            dem = x + (offset if i == 0 else 0.0) if demand else np.full(x.shape, np.nan)
            self._pdo[d] = DriveTrace(d, self.t, {"position_demand_rad": dem, "native_position_rad": x})
        self._imu = imu

    def pdo(self):
        return self._pdo

    def commands(self):
        return self.cmds

    def applied_commands(self):
        return None  # a bag: frame stamps only

    def gains(self):
        return {d: DriveGains(100.0, 5.0, 0, 0) for d in self._pdo}

    def imu(self):
        if not self._imu:
            raise ValueError("no /rt/imu in the bag")
        n = len(self.t)
        return ImuTrace(self.t, np.arange(n), np.tile([0, 0, 0, 1.0], (n, 1)), np.zeros((n, 3)), np.zeros((n, 3)))

    def time_span_ns(self):
        return int(self.t[0]), int(self.t[-1])


def check(plant, cap, **kw):
    from humanoid_sim.sysid.contract import check_capture

    return check_capture(cap, plant, bias_bound_rad=0.02, delay_bounds_ns=(0, 20_000_000), **kw)


def test_a_capture_that_meets_the_contract_passes(plant):
    c = check(plant, Cap(plant))
    assert c.ok and not c.warnings, c.report()
    assert c.rebuild["best_shift_ns"] == DELAY


@pytest.mark.parametrize("kw, error", [
    (dict(demand=False), "no 0x6062"),
    (dict(offset=0.05), "beyond the bias bound"),
    (dict(cmd_shift_ns=-30_000_000), "outside the fit's delay range"),
    (dict(cmd_shift_ns=-60_000_000_000), "overlap for 0.0 s"),
    (dict(imu=False), "IMU"),
])
def test_violations_refuse(plant, kw, error):
    c = check(plant, Cap(plant, **kw))
    assert not c.ok and any(error in e for e in c.errors), c.report()


def test_a_missing_drive_refuses_and_an_unscored_one_needs_only_stream(plant):
    drive = plant.map.drive_names[0]
    assert "no feedback from drives" in check(plant, Cap(plant, drop=drive)).errors[0]
    assert check(plant, Cap(plant, offset=0.05), scored=plant.map.drive_names[1:]).ok
    assert check(plant, Cap(plant, imu=False), need_imu=False).ok
