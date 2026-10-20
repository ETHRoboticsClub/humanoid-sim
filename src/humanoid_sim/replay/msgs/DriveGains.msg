# Stiff position-loop gains the backend holds in force, latched on
# /drive_gains/state and republished on every change and every 250 ms.
#
# kp/kd are the SI gains IN FORCE for each of `motors` (the lease's values for
# held drives, the baseline elsewhere), as the drives carry them: rounded to the
# backend's resolution. `motor_applied` says per drive
# whether it carries them: false while the hardware master is still draining
# the paced SDO writes, and false while DAMPING or the POWERED body mode has
# the drive at P=0 (the stiff pair is restored when that is released).
# `applied` is the conjunction over every drive and requires that no ramp is
# running. `failed` latches when a write exhausted its retries; the drives in
# `failed_motors` then hold UNKNOWN gains and need manual intervention.
#
# While a HOLD with ramp_s runs, kp/kd are the ramp's current sample, the target
# pairs its end, and ramp_remaining_s counts down to 0; without a ramp the
# targets equal the in-force pairs.
std_msgs/Header header
string holder          # "" when the baseline is in force
bool applied
bool failed
string[] motors
float64[] kp_nm_per_rad
float64[] kd_nm_s_per_rad
float64[] baseline_kp_nm_per_rad     # the gains the backend reverts to
float64[] baseline_kd_nm_s_per_rad
bool[] motor_applied
string[] failed_motors
float64[] target_kp_nm_per_rad       # the pair each drive is moving to; == kp/kd when not ramping
float64[] target_kd_nm_s_per_rad
float64 ramp_remaining_s
# The smallest step the drives take; a requested gain is carried rounded to it.
# 0 when the backend carries gains as requested.
float64 kp_resolution_nm_per_rad
float64 kd_resolution_nm_s_per_rad
