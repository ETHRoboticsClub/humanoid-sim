# Ordered authority and command frame consumed first by the body router, then
# forwarded to the final controller for independent enforcement. One authority
# publisher emits every lifecycle and command frame for its runtime.
std_msgs/Header header

# A receiver holds one grant. A GRANT naming the active one (same runtime,
# session, grant and source, newer sequence) reinstalls it at the final
# controller, which drops its grant during a native joint zero; any other is refused.
uint8 KIND_GRANT=1
uint8 KIND_COMMAND=2
uint8 KIND_REVOKE=3
uint8 KIND_HOLD=4
uint8 KIND_RENEW=5
# Ends any grant and fences commands. Receivers check only robot_id and
# stop_profile_id: no session, grant, stamp age or sequence order refuses a stop,
# and an out-of-order one never lowers the sequence floor.
uint8 KIND_STOP=7
uint8 KIND_MODE=8
uint8 KIND_ARM=9
uint8 KIND_MANUAL=10
# Releases the body emergency-stop latch the final controller owns (BodyEstop.srv
# sets it). Validated like ARM: it must name the live receiver grant, and it
# carries no command, mode, stop, lifetime or clear_latches field. Clearing
# never arms; the robot API sends it as the first step of an ARM with
# clear_latches. The controller answers APPLIED once the release is durably
# recorded (or no latch stood), REJECTED with the cause otherwise.
uint8 KIND_CLEAR_ESTOP=11

uint8 MODE_POWERED=1
uint8 MODE_STIFF=2
# MODE captures feedback in the controller, never browser-supplied positions.
uint8 body_mode
uint8 kind
# ARM only: operator-approved trip clearing for this armed session. A trip (a
# command outside the drive's window, a sustained torque overload or another
# master guard; the final controller owns them all) lasts at most one armed
# session: with clear_latches the Arm releases every such trip and re-enables
# the drive; without it the Arm is refused while any trip stands, naming the
# tripped drives (identity arm_error). Neither touches lost feedback, drive
# faults or failed gain writes, which keep their own gates. While the body
# emergency-stop latch stands every Arm is refused whatever clear_latches says;
# only KIND_CLEAR_ESTOP releases it.
bool clear_latches

string robot_id
string authority_runtime_id
string component_session_id
string grant_id
string source_id

# Strictly increasing for every frame in one authority runtime. Zero is invalid.
uint64 sequence

# Client input sequence for MANUAL_CONTROL; zero for non-input commands.
uint64 input_sequence

# Receiver-local lifetime from receipt. Renewals replace the deadline without
# changing a reference. Zero is valid for REVOKE, HOLD, STOP and MODE.
uint64 valid_for_ns

# Present on COMMAND or MANUAL. MANUAL carries only explicitly selected joints.
# Joint coordinates are semantic radians.
string[] joint_ids
float64[] positions_rad
# COMMAND only: the final receiver refuses the complete frame if any safety
# projection would substitute a different endpoint. It never bypasses a gate.
bool require_exact_positions
# COMMAND only, set by the router: the subset of joint_ids it added to keep
# uncommanded drives at their reference. The final receiver drops a hold joint
# whose drive (or mechanism partner) is tripped or latched and refuses the frame
# only when a requested joint touches such a drive. Empty from the authority.
string[] hold_joint_ids

# Present on REVOKE and STOP. The receiver implements HOLD locally; other profiles
# still fence commands here and are confirmed by their mechanism readback.
string stop_profile_id
