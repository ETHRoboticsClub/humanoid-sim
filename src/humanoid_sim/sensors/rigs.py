"""Each robot's sensors, as on the robot (read from the running stacks, 2026-10-20), and extra rigs that add sensors a
robot's rig does not declare (``EXTRAS``, by name: ``examples/rtx_sensors.py --sensors NAME``). Sensors Controls does not
read are left out of a robot's rig: the G1's torso IMU, its Mid-360's own IMU and its head D435 (on the lab unit the
camera service finds no camera)."""

from __future__ import annotations

from humanoid_sim.sensors.models import (
    Imu,
    LivoxMid360,
    Mti610,
    OakDW,
    RealSenseD435i,
    Rig,
    UnitreeG1Imu,
    ZedMini,
)

V1_MONO_LEFT = (283.604271, 283.556541, 321.744194, 196.769579)


def ethrc_v1() -> Rig:
    return Rig(
        # 400 Hz over CAN (Controls ethrc_mti_can), each sample reaching the host as it is sent: kernel stamps of
        # /imu/data_raw on the V1 (2026-10-20, 2000 samples) are 2.47 ms apart (median; p1 2.33, p99 3.12 ms), none
        # missed. The exc-* captures' bursts were the earlier serial path.
        imu=Mti610("imu_link", sample_period_ns=2_500_000),
        # As Controls' camera service opens it, with the unit's calibration (camera_info on the V1, 2026-10-20) for
        # rgb and mono_left. mono_right (the same OV9282 sensor) and depth (aligned to the rectified left image) are
        # modeled with mono_left's pinhole until the unit's are read.
        camera0=OakDW("camera0_camera", width=640, height=400, frame_period_ns=33_333_333,
                      intrinsics={"rgb": (283.087462, 282.805448, 325.969479, 198.375709),
                                  "mono_left": V1_MONO_LEFT, "mono_right": V1_MONO_LEFT, "depth": V1_MONO_LEFT}),
    )


def g1() -> Rig:
    return Rig(
        # LowState at 500 Hz, each sample delivered as it is taken (modeled; the delivery is unmeasured).
        imu=UnitreeG1Imu("imu_link", sample_period_ns=2_000_000),
        mid360=LivoxMid360("mid360_link"),
    )


def g1_cameras() -> Rig:
    """The cameras the lab G1 is reported to carry, at the description's camera links, until its rig declares them."""
    return Rig(
        # The helmet mount: the left eye's optical centre (the G1 URDF's camera0_camera). HD720 at 30 Hz as Controls'
        # zed backend opens it, with the factory calibration of the lab's unit (SN19945568, as Controls'
        # test_zed_backend.py carries it): fisheye (equidistant) lenses, the right eye's place from [STEREO]
        # (Baseline 62.8272 mm, TY -0.181381 mm, TZ 0.233734 mm, placed as zed_backend.parse_calibration places it).
        zed=ZedMini(
            "camera0_camera", width=1280, height=720, frame_period_ns=33_333_333,
            intrinsics={"left": (771.4, 771.14, 629.115, 353.0685), "right": (772.01, 771.55, 641.75, 346.6085)},
            distortion={"left": ("equidistant", (-0.0358597, 0.0185157, -0.00496448, 0.00194007)),
                        "right": ("equidistant", (-0.0368591, 0.0234426, -0.0128113, 0.00625172))},
            stream_offsets_m={"right": (0.000233734, -0.0628272, 0.000181381)},
        ),
        # Unitree's d435_joint on the torso. 640 x 480 at 30 Hz as Controls' realsense adapter opens it. No unit's
        # calibration: centred pinholes with the datasheet's fields of view, colour 69 deg across
        # (fx = 320 / tan(34.5 deg)), the infrared imagers and depth 87 x 58 deg (fx = 320 / tan(43.5 deg),
        # fy = 240 / tan(29 deg)).
        d435=RealSenseD435i(
            "d435_link", width=640, height=480, frame_period_ns=33_333_333,
            intrinsics={"rgb": (465.6029, 465.6029, 320.0, 240.0),
                        "mono_left": (337.2096, 432.9715, 320.0, 240.0),
                        "mono_right": (337.2096, 432.9715, 320.0, 240.0),
                        "depth": (337.2096, 432.9715, 320.0, 240.0)},
        ),
    )


RIGS = {"ethrc_v1": ethrc_v1, "g1_29dof_rev_1_0": g1, "g1_29dof_with_hand_rev_1_0": g1}
EXTRAS = {"g1-cameras": g1_cameras}


def rig(robot: str) -> Rig:
    """The robot's sensors; none for a robot without a rig here."""
    make = RIGS.get(robot)
    return make() if make else Rig()


def load(robot: str, desc, extra: str | None = None) -> Rig:
    """The robot's rig with the extra rig named ``extra`` added, every frame checked against the description."""
    r = rig(robot)
    if extra:
        if extra not in EXTRAS:
            raise ValueError(f"unknown extra rig {extra!r}; known: {sorted(EXTRAS)}")
        r = r.extended(EXTRAS[extra]())
    r.check_frames(desc)
    return r


def imu(desc) -> Imu:
    """The robot's IMU, its rig checked against the description."""
    return load(desc.robot, desc).imu


__all__ = ["EXTRAS", "RIGS", "imu", "load", "rig"]
