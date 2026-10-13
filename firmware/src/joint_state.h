#pragma once
/**
 * Angle and velocity of one joint from its encoder counts, once per inner tick (every 1 / INNER_HZ):
 *
 *   angle    = counts x RAD_PER_COUNT, wrapped to [-pi, pi]
 *   raw      = (counts - last counts) x RAD_PER_COUNT / dt          difference of two counts
 *   velocity += alpha (raw - velocity)                              the low-pass 50 / (s + 50)
 *
 * The velocity comes from the counts, not from the wrapped angle: the angle jumps by 2 pi at +-pi, the counts do not.
 */
#include <stdint.h>

#include "cascade_pid.h"
#include "config.h"

class JointState {
 public:
  float angle = 0.0f;      // [-pi, pi] [rad]
  float velocity = 0.0f;   // [rad/s]

  /** Start from ``counts`` with zero velocity. */
  void reset(int32_t counts, float sign) {
    sign_ = sign;
    lastCounts_ = counts;
    angle = wrapToPi(sign_ * counts * RAD_PER_COUNT);
    velocity = 0.0f;
  }

  void update(int32_t counts) {
    const float raw = sign_ * (counts - lastCounts_) * RAD_PER_COUNT / INNER_DT;
    velocity += VELOCITY_ALPHA * (raw - velocity);
    angle = wrapToPi(sign_ * counts * RAD_PER_COUNT);
    lastCounts_ = counts;
  }

 private:
  float sign_ = 1.0f;
  int32_t lastCounts_ = 0;
};
