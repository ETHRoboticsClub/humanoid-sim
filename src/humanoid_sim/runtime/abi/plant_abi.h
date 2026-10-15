// Runtime segment ABI between humanoid-sim's real-time host and Controls' plant host: the same layout as
// plant_abi.py (which documents the protocol). Self-contained C11/C++17; vendored unchanged by Controls.
//
// Every region starts with seq (u64) and crc (u32). A writer makes seq odd, writes the payload (the bytes
// after the 16-byte region head), stores the CRC-32 (zlib) of the payload and makes seq even again, with
// release ordering. A reader copies the payload between two equal, even acquire-loads of seq and checks the
// CRC. Little-endian, natural alignment, every region on a 64-byte boundary.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HSIM_PLANT_ABI_MAJOR 1
#define HSIM_PLANT_ABI_MINOR 0
#define HSIM_PLANT_MAGIC "HSIMPLNT"
#define HSIM_PLANT_LAYOUT_HASH 0x93d27dd415b5c26bULL
#define HSIM_PLANT_SEGMENT_BYTES 27648

#define HSIM_PLANT_MAX_DRIVES 32
#define HSIM_PLANT_MAX_JOINTS 64
#define HSIM_PLANT_IMU_SAMPLES 32
#define HSIM_PLANT_DESCRIPTOR_BYTES 16384
#define HSIM_PLANT_MAILBOX_BYTES 4096
#define HSIM_PLANT_ACK_MESSAGE_BYTES 256
#define HSIM_PLANT_SOLES 2

enum { HSIM_STATE_CREATING = 0, HSIM_STATE_READY = 1, HSIM_STATE_ENDED = 2 };

enum {
  HSIM_CMD_NONE = 0,
  HSIM_CMD_RESET = 1,
  HSIM_CMD_PAUSE = 2,
  HSIM_CMD_RESUME = 3,
  HSIM_CMD_SET_SUPPORT = 4,
  HSIM_CMD_APPLY_DISTURBANCE = 5,
  HSIM_CMD_INJECT_FAULT = 6,
  HSIM_CMD_PRESCRIBE_POSE = 7,
};

enum { HSIM_ACK_OK = 0, HSIM_ACK_REJECTED = 1, HSIM_ACK_UNKNOWN = 2, HSIM_ACK_FAILED = 3 };

typedef struct {
  char magic[8];
  uint16_t abi_major, abi_minor;
  uint32_t header_bytes;
  uint64_t layout_hash, segment_bytes, session_id;
  uint32_t state, num_drives, num_joints, imu_samples;
  uint64_t physics_period_ns, bus_period_ns, heartbeat_mono_ns;
  char robot_id[32];
  char description_digest[64];
  char sim_revision[48];
  uint64_t offsets[6]; // descriptor, demands, feedback, mailbox, ack, truth
} HsimHeader;

// FakeDriveDemand (Controls fake_bank.h), at fixed widths.
typedef struct {
  int32_t torque_enabled;
  int8_t mode; // 0x6061: 6 homing, -64 JVPT, -62 freeze, 0 none
  uint8_t homing_hold;
  uint8_t _pad0[2];
  double target_position_rad, target_velocity_rad_s, feedforward_torque_nm;
  uint32_t p_gain_raw, i_gain_raw, d_gain_raw, max_torque_mnm;
} HsimDriveDemand;

// fake_bank_feedback's arguments.
typedef struct {
  double position_rad, velocity_rad_s, applied_torque_nm, demanded_torque_nm;
} HsimDriveFeedback;

typedef struct {
  uint64_t seq;       // per-session sample number, from 1
  int64_t sample_ns;  // measurement time, sim time
  int64_t stamp_ns;   // host-receipt time, sim time
  double gyro_rad_s[3];
  double specific_force_m_s2[3];
} HsimImuSample;

typedef struct {
  double position_m[3];
  double quat_xyzw[4];
  double lowest_z_m;
  double normal_force_n;
} HsimSole;

#define HSIM_REGION_HEAD \
  uint64_t seq;          \
  uint32_t crc;          \
  uint32_t _pad0;

typedef struct {
  HSIM_REGION_HEAD
  uint32_t length;
  uint32_t _pad1;
  uint8_t json[HSIM_PLANT_DESCRIPTOR_BYTES];
} HsimDescriptor;

typedef struct {
  HSIM_REGION_HEAD
  uint64_t cycle, written_mono_ns, session_id;
  uint32_t num_drives, _pad1;
  HsimDriveDemand drive[HSIM_PLANT_MAX_DRIVES];
} HsimDemands;

typedef struct {
  HSIM_REGION_HEAD
  uint64_t step;
  int64_t sim_time_ns;
  uint64_t written_mono_ns, late_steps;
  uint32_t keeping_up, paused, num_drives, imu_count;
  uint64_t imu_total;
  HsimDriveFeedback drive[HSIM_PLANT_MAX_DRIVES];
  HsimImuSample imu[HSIM_PLANT_IMU_SAMPLES]; // newest first; imu_count valid
} HsimFeedback;

typedef struct {
  HSIM_REGION_HEAD
  uint32_t command, length;
  uint64_t written_mono_ns;
  uint8_t args[HSIM_PLANT_MAILBOX_BYTES]; // JSON object, `length` bytes
} HsimMailbox;

typedef struct {
  HSIM_REGION_HEAD
  uint64_t acked;
  int32_t result;
  uint32_t _pad1;
  char message[HSIM_PLANT_ACK_MESSAGE_BYTES];
} HsimAck;

typedef struct {
  HSIM_REGION_HEAD
  int64_t sim_time_ns;
  double base_position_m[3], base_quat_xyzw[4], base_lin_vel_m_s[3], base_ang_vel_rad_s[3], com_m[3];
  double gantry_tension_n, mass_kg;
  uint32_t num_joints, _pad1;
  double joint_pos_rad[HSIM_PLANT_MAX_JOINTS], joint_vel_rad_s[HSIM_PLANT_MAX_JOINTS];
  HsimSole sole[HSIM_PLANT_SOLES]; // left, right
} HsimTruth;

#undef HSIM_REGION_HEAD

#define HSIM_OFFSET_DESCRIPTOR 320
#define HSIM_OFFSET_DEMANDS 16768
#define HSIM_OFFSET_FEEDBACK 18368
#define HSIM_OFFSET_MAILBOX 21824
#define HSIM_OFFSET_ACK 25984
#define HSIM_OFFSET_TRUTH 26304

#ifdef __cplusplus
static_assert(sizeof(HsimHeader) == 272, "header layout");
static_assert(sizeof(HsimFeedback) == 3400, "feedback layout");
#else
_Static_assert(sizeof(HsimHeader) == 272, "header layout");
_Static_assert(sizeof(HsimFeedback) == 3400, "feedback layout");
#endif

#ifdef __cplusplus
}
#endif
