/**
 * Firmware of the rotary pendulum (ESP32 + BTS7960 + QUBE-Servo 2 mechanics).
 *
 * The ESP32 runs the two PIDs in cascade at 500 / 250 Hz; the PC runs the policy at 100 Hz and writes the gains.
 * The loops are the ones the policy was trained with (cascade_pid.h = mdp/actions.py in the simulator).
 *
 *   every 2 ms   (500 Hz)   read the encoders -> [every 2nd tick: outer PID] -> inner PID -> voltage -> PWM
 *   every 10 ms  (100 Hz)   send the state to the PC
 *   whenever a line arrives from the PC (about every 10 ms) see below
 *
 * Serial, 921600 baud, one text line per message, '\n' at the end:
 *
 *   PC -> ESP32   G <command> <a1> <a2> <a3> <a4> <a5> <a6>   command: arm angle to hold [rad]; a1..a6: the policy's
 *                                                            outputs in [-1, 1] (Kp1 Ki1 Kd1 Kp2 Ki2 Kd2)
 *                 R                                          zero the encoders and the PIDs, clear a fault (the arm
 *                                                            centred, the pendulum hanging still)
 *   ESP32 -> PC   S <ms> <theta> <alpha> <theta_dot> <alpha_dot> <voltage> <setpoint> <status>
 *                   theta, alpha: arm and pendulum angle in [-pi, pi] (alpha 0 hanging, +-pi upright) [rad]
 *                   theta_dot, alpha_dot [rad/s]; voltage the inner PID applied [V]; setpoint theta2* [rad]
 *                   status 0 running, 1 waiting for gains or the PC went quiet, 2 arm turned half a turn (motor off)
 *
 * The motor stays off until the first G, goes off when the PC is silent for WATCHDOG_MS, and after a fault stays off
 * until R.
 */
#include <Arduino.h>
#include <stdio.h>
#include <stdlib.h>

#include "cascade_pid.h"
#include "config.h"
#include "encoder.h"
#include "joint_state.h"
#include "motor.h"

enum Status { RUNNING = 0, WAITING = 1, ARM_LIMIT = 2 };

static CascadePid pid;
static JointState arm;
static JointState pendulum;

static float command = 0.0f;           // arm angle to hold [rad], from the PC
static uint32_t lastGainsMs = 0;       // when the last G arrived
static bool haveGains = false;
static bool armFault = false;

static uint32_t nextTickUs = 0;
static uint32_t tick = 0;

static char line[96];
static size_t lineLength = 0;

static void zeroEverything() {
  encodersZero();
  pid.reset();
  arm.reset(armCounts(), ARM_SIGN);
  pendulum.reset(pendulumCounts(), PENDULUM_SIGN);
  armFault = false;
  haveGains = false;
}

/** One message from the PC. */
static void handleLine(const char* text) {
  if (text[0] == 'R') {
    zeroEverything();
  } else if (text[0] == 'G') {
    float value[7];
    char* end = (char*)text + 1;
    for (int i = 0; i < 7; i++) {
      value[i] = strtof(end, &end);
    }
    command = value[0];
    pid.setGains(value + 1);
    lastGainsMs = millis();
    haveGains = true;
  }
}

/** Collect the bytes that arrived, without waiting for more. */
static void readSerial() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n') {
      line[lineLength] = '\0';
      handleLine(line);
      lineLength = 0;
    } else if (lineLength < sizeof(line) - 1) {
      line[lineLength++] = c;
    }
  }
}

static void sendState(float voltage, Status status) {
  char message[128];
  const int length = snprintf(message, sizeof(message), "S %lu %.4f %.4f %.4f %.4f %.3f %.4f %d\n",
                              (unsigned long)millis(), arm.angle, pendulum.angle, arm.velocity, pendulum.velocity,
                              voltage, pid.setpoint(), (int)status);
  Serial.write((const uint8_t*)message, length);
}

/** One inner tick: sensors, loops, motor. */
static void controlTick() {
  arm.update(armCounts());
  pendulum.update(pendulumCounts());

  if (fabsf(armCounts() * RAD_PER_COUNT) > ARM_LIMIT_RAD) armFault = true;
  const bool quiet = !haveGains || (millis() - lastGainsMs) > WATCHDOG_MS;

  Status status = RUNNING;
  float voltage = 0.0f;
  if (armFault) {
    status = ARM_LIMIT;
  } else if (quiet) {
    status = WAITING;
  } else {
    if (tick % OUTER_EVERY == 0) pid.outer(arm.angle, arm.velocity, command);
    voltage = pid.inner(pendulum.angle, pendulum.velocity);
  }
  motorWriteVoltage(voltage);

  if (tick % POLICY_EVERY == 0) sendState(voltage, status);
  tick++;
}

void setup() {
  Serial.begin(921600);
  motorBegin();
  encodersBegin();
  zeroEverything();
  nextTickUs = micros() + TICK_US;
}

void loop() {
  readSerial();

  const uint32_t now = micros();
  if ((int32_t)(now - nextTickUs) >= 0) {
    nextTickUs += TICK_US;
    if ((int32_t)(now - nextTickUs) >= (int32_t)(5 * TICK_US)) nextTickUs = now + TICK_US;   // far behind: do not catch up
    controlTick();
  }
}
