# Every drive's latest cyclic feedback as parallel sequences, published at the
# 200 Hz consumer rate. The writer is RELIABLE KeepLast(REPAIR_DEPTH): control
# consumers subscribe best-effort depth 1 and see only the newest sample, never
# a retransmission or a sample held behind a repair; the excitation capture
# subscribes reliable and has lost samples repaired from that history. A
# keep-last writer never blocks on a slow reader (Cyclone pins its WHC
# watermarks for keep-last). High-rate telemetry and excitation capture use this
# flat message because deserializing a nested DriveStatus[] (a Header and a Time
# per drive) costs more. One object per publish, plain sequences.
# DriveStatusArray keeps every drive's full MotorStatusMessage for diagnostics
# and telemetry. Drives whose PDO reading is stale are withheld here exactly as
# there. Entry i of every sequence belongs to device[i].
# The authoritative topic name; the EtherCAT master, the simulator and the
# consumers reference this constant rather than their own literal.
string TOPIC="/ethrc/drive_feedback"
# Writer history: 200 ms at 200 Hz, longer than a NACK round trip under load.
uint16 REPAIR_DEPTH=40

# Publish time.
std_msgs/Header header
# Sanitized drive name, the prefix of its /<device>/status topic.
string[] device
# The drive's sample instant (ROS time, ns): the header stamp of its per-device status.
int64[] source_stamp_ns
# MotorStatusMessage.actual_position / actual_velocity, drive coordinates.
float64[] position_rad
float64[] velocity_rad_s
# CiA-402 statusword.
uint16[] status_word
# The drive's joint torque estimate (0x3672, from motor current; datasheet
# steady-state error < 2 N m) and instantaneous current (0x30D1:02), same cycle.
# NaN where a producer has no such measurement.
float64[] torque_nm
float64[] current_a
# Motor encoder (0x60E4:01) through the gear, joint-side rad, zero at drive
# power-up: relative only, but 140x (HEJ50) to 560x (HEJ90) finer than
# position_rad. NaN where unavailable.
float64[] motor_position_rad
# The drive's position-loop input (0x6062, joint-side rad, the reference after
# the drive's own interpolation) and current-loop set value (0x30D0, A): what
# the drive acts on, as opposed to what the master sent. NaN where unavailable.
float64[] position_demand_rad
float64[] current_demand_a
