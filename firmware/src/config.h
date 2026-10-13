#pragma once
/**
 * Constants of the firmware. Pure constants (no Arduino header), so the maths can be tested on a PC.
 *
 * The numbers marked "= mdp/actions.py" must equal the ones of the simulator the policy was trained in
 * (src/Rotary_Pendulum_RL/rotary/mdp/actions.py, rotary_cfg.py): change one side, change the other.
 */
#include <math.h>
#include <stdint.h>

// ── Pins (ESP32-WROOM-32) ─────────────────────────────────────────────────────────────
constexpr uint8_t PIN_PWM_POSITIVE = 18;   // BTS7960 RPWM: PWM here drives the motor in the positive direction
constexpr uint8_t PIN_PWM_NEGATIVE = 19;   // BTS7960 LPWM: PWM here drives the motor in the negative direction
constexpr uint8_t PIN_ARM_A = 32;          // motor encoder, channel A (interrupt) and B
constexpr uint8_t PIN_ARM_B = 33;
constexpr uint8_t PIN_PENDULUM_A = 25;     // pendulum encoder, channel A (interrupt) and B
constexpr uint8_t PIN_PENDULUM_B = 26;

// ── Motor driver ──────────────────────────────────────────────────────────────────────
constexpr uint32_t PWM_FREQ_HZ = 512;      // the PWM carrier
constexpr uint8_t  PWM_BITS = 16;
constexpr uint32_t PWM_MAX = 65535;        // = rotary_cfg.PWM_MAX
constexpr float    V_MAX = 12.0f;          // [V] supply: 100 % duty = rotary_cfg.QUBE_V_MAX

// ── Encoders ──────────────────────────────────────────────────────────────────────────
// The interrupt fires on every edge of channel A only: 2 edges x 512 lines = 1024 counts per turn (6.1 mrad).
constexpr float COUNTS_PER_REV = 1024.0f;
constexpr float RAD_PER_COUNT = 2.0f * (float)M_PI / COUNTS_PER_REV;

// Direction of each sensor and of the motor, +1 or -1. The simulator's angles are positive in the direction of the joint
// axes of the URDF; bring-up (firmware/readme.md) says how to find the sign that makes a positive angle match.
constexpr float ARM_SIGN = 1.0f;
constexpr float PENDULUM_SIGN = 1.0f;
constexpr float MOTOR_SIGN = 1.0f;

// ── Loop rates ────────────────────────────────────────────────────────────────────────
constexpr uint32_t INNER_HZ = 500;         // = rotary_cfg.INNER_HZ: encoders, inner (pendulum) PID, PWM
constexpr uint32_t OUTER_HZ = 250;         // = actions.OUTER_HZ: outer (arm) PID
constexpr uint32_t POLICY_HZ = 100;        // = rotary_cfg.POLICY_HZ: the PC writes gains and reads the state at this rate
constexpr uint32_t TICK_US = 1000000 / INNER_HZ;
constexpr uint32_t OUTER_EVERY = INNER_HZ / OUTER_HZ;     // inner ticks per outer tick
constexpr uint32_t POLICY_EVERY = INNER_HZ / POLICY_HZ;   // inner ticks per state message
constexpr float INNER_DT = 1.0f / INNER_HZ;
constexpr float OUTER_DT = 1.0f / OUTER_HZ;

// ── Velocity estimate ─────────────────────────────────────────────────────────────────
// The difference of two counts, through the low-pass 50 / (s + 50): the filter of the Quanser Simulink files.
constexpr float VELOCITY_FILTER_RAD_S = 50.0f;
constexpr float VELOCITY_ALPHA = INNER_DT / (INNER_DT + 1.0f / VELOCITY_FILTER_RAD_S);

// ── Gain ranges: the policy writes a in [-1, 1], the gain is a times these ── = mdp/actions.py ──
constexpr float ARM_KP_MAX = 1.0f;         // [rad of lean per rad]
constexpr float ARM_KI_MAX = 1.0f;         // [rad of lean per rad s]
constexpr float ARM_KD_MAX = 0.1f;         // [rad of lean per rad/s]
constexpr float PENDULUM_KP_MAX = 100.0f;  // [V/rad]
constexpr float PENDULUM_KI_MAX = 100.0f;  // [V/(rad s)]
constexpr float PENDULUM_KD_MAX = 10.0f;   // [V s/rad]
constexpr float TILT_MAX = 10.0f * (float)M_PI / 180.0f;  // [rad] largest lean the outer loop may ask for

// ── Safety ────────────────────────────────────────────────────────────────────────────
constexpr uint32_t WATCHDOG_MS = 100;      // no gains from the PC for this long: motor off
constexpr float ARM_LIMIT_RAD = (float)M_PI;   // the arm turned half a turn: motor off (the simulator ends the episode here)
