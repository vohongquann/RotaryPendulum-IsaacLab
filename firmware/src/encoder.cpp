#include "encoder.h"

#include <Arduino.h>

#include "config.h"

static volatile int32_t armCount = 0;
static volatile int32_t pendulumCount = 0;

// On each edge of channel A: count up if B equals A, down if not.
void IRAM_ATTR onArmEdge() { armCount += (digitalRead(PIN_ARM_A) == digitalRead(PIN_ARM_B)) ? 1 : -1; }
void IRAM_ATTR onPendulumEdge() {
  pendulumCount += (digitalRead(PIN_PENDULUM_A) == digitalRead(PIN_PENDULUM_B)) ? 1 : -1;
}

void encodersBegin() {
  pinMode(PIN_ARM_A, INPUT_PULLUP);
  pinMode(PIN_ARM_B, INPUT_PULLUP);
  pinMode(PIN_PENDULUM_A, INPUT_PULLUP);
  pinMode(PIN_PENDULUM_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_ARM_A), onArmEdge, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_PENDULUM_A), onPendulumEdge, CHANGE);
}

void encodersZero() {
  noInterrupts();
  armCount = 0;
  pendulumCount = 0;
  interrupts();
}

int32_t armCounts() { return armCount; }
int32_t pendulumCounts() { return pendulumCount; }
