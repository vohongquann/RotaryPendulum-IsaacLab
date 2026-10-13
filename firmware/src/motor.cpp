#include "motor.h"

#include <Arduino.h>

#include "cascade_pid.h"
#include "config.h"

void motorBegin() {
  ledcAttach(PIN_PWM_POSITIVE, PWM_FREQ_HZ, PWM_BITS);
  ledcAttach(PIN_PWM_NEGATIVE, PWM_FREQ_HZ, PWM_BITS);
  motorWriteVoltage(0.0f);
}

void motorWriteVoltage(float volts) {
  volts = clampf(MOTOR_SIGN * volts, -V_MAX, V_MAX);
  const uint32_t pwm = (uint32_t)lroundf(fabsf(volts) / V_MAX * PWM_MAX);   // = duty_to_pwm in the simulator
  ledcWrite(PIN_PWM_POSITIVE, volts > 0.0f ? pwm : 0);
  ledcWrite(PIN_PWM_NEGATIVE, volts < 0.0f ? pwm : 0);
}
