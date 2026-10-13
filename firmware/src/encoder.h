#pragma once
/** Quadrature counts of the two encoders, kept by interrupts on channel A. */
#include <stdint.h>

void encodersBegin();
void encodersZero();          // the arm centred and the pendulum hanging still: both counts to 0
int32_t armCounts();
int32_t pendulumCounts();
