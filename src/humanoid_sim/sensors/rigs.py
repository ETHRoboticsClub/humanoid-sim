"""Each robot's sensors, as on the robot (read from the running stacks, 2026-10-20). Sensors Controls does not read
are left out: the G1's torso IMU, its Mid-360's own IMU and its head D435 (on the lab unit the camera service finds
no camera)."""

from __future__ import annotations

from humanoid_sim.sensors.models import Imu, LivoxMid360, Mti610, OakDW, Rig, UnitreeG1Imu


def ethrc_v1() -> Rig:
    return Rig(
        # 400 Hz over CAN (Controls ethrc_mti_can), each sample reaching the host as it is sent: kernel stamps of
        # /imu/data_raw on the V1 (2026-10-20, 2000 samples) are 2.47 ms apart (median; p1 2.33, p99 3.12 ms), none
        # missed. The exc-* captures' bursts were the earlier serial path.
        imu=Mti610("imu_link", sample_period_ns=2_500_000),
        # As Controls' camera service opens it, with the unit's calibration (camera_info on the V1, 2026-10-20).
        camera0=OakDW("camera0_camera", width=640, height=400, frame_period_ns=33_333_333,
                      intrinsics={"rgb": (283.087462, 282.805448, 325.969479, 198.375709),
                                  "mono_left": (283.604271, 283.556541, 321.744194, 196.769579)}),
    )


def g1() -> Rig:
    return Rig(
        # LowState at 500 Hz, each sample delivered as it is taken (modeled; the delivery is unmeasured).
        imu=UnitreeG1Imu("imu_link", sample_period_ns=2_000_000),
        mid360=LivoxMid360("mid360_link"),
    )


RIGS = {"ethrc_v1": ethrc_v1, "g1_29dof_rev_1_0": g1, "g1_29dof_with_hand_rev_1_0": g1}


def rig(robot: str) -> Rig:
    """The robot's sensors; none for a robot without a rig here."""
    make = RIGS.get(robot)
    return make() if make else Rig()


def imu(desc) -> Imu:
    """The robot's IMU, its rig checked against the description."""
    r = rig(desc.robot)
    r.check_frames(desc)
    return r.imu


__all__ = ["RIGS", "Imu", "imu", "rig"]
