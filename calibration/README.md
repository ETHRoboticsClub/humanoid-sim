# Device calibrations

The factory calibrations of the robots' cameras, as read off each unit, named by serial. The sensor rigs in
`src/humanoid_sim/sensors/rigs.py` take their lens and stream values from these units; a test checks the ZED Mini's.

| File | Device | Robot | Read |
| --- | --- | --- | --- |
| `oak-d-w_19443010313B604800.json` | Luxonis OAK-D-W (`OAK-D-W-97`), serial 19443010313B604800 | V1 | 2026-11-07, depthai 2.33.0.0 |
| `zed-mini_SN19945568.conf` | Stereolabs ZED Mini, serial 19945568 | G1 (the lab unit) | 2026-11-16 |

## OAK-D-W

The JSON holds the device's `readCalibration()` (`calibration`, the one in use) and `readFactoryCalibration()`
(`factory`); on this unit they are the same. The calibration is stored at 1280 x 800: halve focal lengths and
principal points for the 640 x 400 streams Controls opens. Read with depthai inside Controls' hardware image
(`/opt/ethrc/cameras-runtime`), with the camera service stopped, since one process holds the device at a time.

## ZED Mini

Stereolabs keeps each unit's factory calibration on its server, by serial; Controls' camera service downloads it once
into the `camera-calibration` volume (`/var/lib/ethrc-cameras/SN<serial>.conf`), and this file is a copy of that one.
The serial is the one the camera's HID interface reports over USB (`/sys/bus/usb/devices/*/serial`, vendor `2b03`).
Controls opens the camera at HD720, so the `*_CAM_HD` and `[STEREO]` `*_HD` entries are the ones used: the lenses are
equidistant (k1 to k4), and `[STEREO]` places the right eye in the left eye's optical frame (Baseline, TY, TZ in mm)
turned by the rotation vector (RX, CV, RZ in rad).
