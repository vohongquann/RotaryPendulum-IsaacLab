#pragma once
/** The motor: a voltage in [-V_MAX, V_MAX] becomes a 16-bit PWM on one of the two BTS7960 inputs. */

void motorBegin();
void motorWriteVoltage(float volts);   // positive: PWM_POSITIVE, negative: PWM_NEGATIVE, the other pin low
