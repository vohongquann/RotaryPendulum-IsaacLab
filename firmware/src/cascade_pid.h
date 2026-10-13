#pragma once
/**
 * The two PIDs in cascade, the same equations as CascadePidAction in the simulator
 * (src/Rotary_Pendulum_RL/rotary/mdp/actions.py):
 *
 *   outer, 250 Hz   e1 = wrap(arm angle - command)
 *                   I1 += e1 dt                         cut at +-TILT_MAX / ARM_KI_MAX
 *                   lean = Kp1 e1 + Ki1 I1 + Kd1 w1     w1: arm speed
 *                   setpoint = pi + clamp(lean, +-TILT_MAX)
 *
 *   inner, 500 Hz   e2 = wrap(pendulum angle - setpoint)
 *                   I2 += e2 dt                         cut at +-V_MAX / PENDULUM_KI_MAX
 *                   V = -(Kp2 e2 + Ki2 I2 + Kd2 w2)     w2: pendulum speed; cut at +-V_MAX
 *
 * The six gains come from the policy as numbers a in [-1, 1].
 */
#include <math.h>

#include "config.h"

inline float clampf(float x, float low, float high) { return x < low ? low : (x > high ? high : x); }

inline float wrapToPi(float x) {
  const float two_pi = 2.0f * (float)M_PI;
  x = fmodf(x + (float)M_PI, two_pi);
  if (x < 0.0f) x += two_pi;
  return x - (float)M_PI;
}

class CascadePid {
 public:
  CascadePid() { reset(); }

  /** Gains to zero, integrals to zero, setpoint back to upright. */
  void reset() {
    for (int i = 0; i < 6; i++) gain_[i] = 0.0f;
    armIntegral_ = 0.0f;
    pendulumIntegral_ = 0.0f;
    setpoint_ = (float)M_PI;
  }

  /** The policy's six outputs a in [-1, 1]: Kp1 Ki1 Kd1 (arm), Kp2 Ki2 Kd2 (pendulum). */
  void setGains(const float a[6]) {
    static const float gainMax[6] = {ARM_KP_MAX, ARM_KI_MAX, ARM_KD_MAX, PENDULUM_KP_MAX, PENDULUM_KI_MAX, PENDULUM_KD_MAX};
    for (int i = 0; i < 6; i++) gain_[i] = clampf(a[i], -1.0f, 1.0f) * gainMax[i];
  }

  /** Outer loop, every 1 / OUTER_HZ: the arm error sets the pendulum setpoint. */
  void outer(float armAngle, float armSpeed, float command) {
    const float error = wrapToPi(armAngle - command);
    const float integralLimit = TILT_MAX / ARM_KI_MAX;
    armIntegral_ = clampf(armIntegral_ + error * OUTER_DT, -integralLimit, integralLimit);

    const float lean = gain_[0] * error + gain_[1] * armIntegral_ + gain_[2] * armSpeed;
    setpoint_ = (float)M_PI + clampf(lean, -TILT_MAX, TILT_MAX);
  }

  /** Inner loop, every 1 / INNER_HZ: the pendulum error sets the motor voltage [V]. */
  float inner(float pendulumAngle, float pendulumSpeed) {
    const float error = wrapToPi(pendulumAngle - setpoint_);
    const float integralLimit = V_MAX / PENDULUM_KI_MAX;
    pendulumIntegral_ = clampf(pendulumIntegral_ + error * INNER_DT, -integralLimit, integralLimit);

    const float voltage = -(gain_[3] * error + gain_[4] * pendulumIntegral_ + gain_[5] * pendulumSpeed);
    return clampf(voltage, -V_MAX, V_MAX);
  }

  float setpoint() const { return setpoint_; }

 private:
  float gain_[6];            // Kp1 Ki1 Kd1 Kp2 Ki2 Kd2
  float armIntegral_;        // I1 [rad s]
  float pendulumIntegral_;   // I2 [rad s]
  float setpoint_;           // theta2* [rad]
};
