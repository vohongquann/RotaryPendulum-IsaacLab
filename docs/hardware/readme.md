# QUBE-Servo 2 motor: sources and what they give

**The real chain of this robot** (github.com/vohongquann/rotary-pendulum-matlab, `embedded/`, `docs/datasheets/`): an ESP32 writes a
16-bit PWM at 512 Hz to a BTS7960 H-bridge, which drives the Allied Motion CL40 motor of the QUBE-Servo 2 from a 12 V supply;
the encoders (E8P, 512 lines) are counted on channel A only, 1024 counts per turn. The Quanser amplifier, its +-10 V range and
its 2 A limit (below) are not in this chain: the voltage at 100 % duty is the supply, 12 V. The PWM carrier (1.95 ms period)
is slow against the electrical time constant (0.14 ms), so the current ripples inside each period, but the circuit is linear
and the average current obeys the average-voltage equation of `motor.py` (the low side of the bridge conducts both ways).

Everything the motor model ([motor.py](../../src/Rotary_Pendulum_RL/rotary/mdp/motor.py),
[rotary_cfg.py](../../src/Rotary_Pendulum_RL/rotary/rotary_cfg.py)) is built on, with where each number comes from.
"Not published" means no source below states it: measure it on your robot.

## Sources

| Source | What it gives |
|---|---|
| [Allied Motion CL Series datasheet](https://www.alliedmotion.com/wp-content/uploads/documents/Allied_Motion_Datasheet-CLSeries_20171002.pdf) (2017), table "CL40 (7 W)", column 16705 | The motor. 18 V: R 8.4 ohm, L 1.16 mH, k_t 42 mNm/A, J 4E-6 kg m^2, no-load 4050 RPM and 22 mA, rated 3050 RPM and 540 mA, starting 2.1 A and 89 mNm, mechanical time constant 19 ms, thermal resistance winding-housing 5.5 and housing-ambient 13 C/W |
| [QUBE-Servo User Manual](http://wwwlehre.dhbw-stuttgart.de/~flaemig/MATLAB/Quanser_CubeServo/QUBE-Servo%20User%20Manual.pdf) v1.1, section 2.2 and table 2.2 | Confirms the motor (CL40 model 16705) and the same constants; PWM voltage-controlled amplifier, output +-10 V, 2 A peak, 0.5 A continuous (the thermal limit of the motor); input +-10 V; encoders US Digital E8P-512, 2048 counts/rev in quadrature = 0.176 deg; caution: a stall above 5 V can damage the motor |
| [QUARC driver page, qube_servo2_usb](https://docs.quanser.com/quarc/documentation/qube_servo2_usb.html) | Analog output (motor voltage) range +-15 V; analog input 0 = motor current in A, range +-3 A, 12 bit; digital output 0 = amplifier enable; digital inputs: amplifier fault (overheating), stall detected (voltage above 5 V *including deadband compensation*), stall error (3 s); a tachometer in counts/s (other input 14000); the card has **no PWM outputs** |
| [NI QUBE-Servo 2 for myRIO datasheet](qube-servo2/ni_qube_servo_2_for_myrio_datasheet.pdf) | Nominal 18 V, 0.54 A, 4050 RPM; encoder "512 counts/rev, non-quadrature" (= 2048 in quadrature) |
| [Quanser blog, Simscape model of the QUBE-Servo 2](https://www.quanser.com/blog/simscape-qube-servo-2/) | The only published friction values: Stribeck torque about 2.03E-4 N m, Coulomb torque about 1.02E-4 N m, identified from measured current and speed; the datasheet model answers faster than the hardware, which the author puts on R and k_t |
| [Lorraine lab sheet 1](https://w3.cran.univ-lorraine.fr/perso/hugues.garnier/Enseignement/Mobile_Robots/QUBE-servo2_Lab1.pdf) and [lab sheet 2](http://w3.cran.univ-lorraine.fr/perso/hugues.garnier/Enseignement/Mobile_Robots/QUBE-servo2_Lab2.pdf) | The velocity filter 50 s/(s+50); step-response identification of the motor (first order from voltage to speed); the sheets warn of dry friction in the shaft; Quanser's preset arm and pendulum damping for the rotary pendulum: 0.0015 and 0.0005 N m s/rad |
| [Quanser, instrumentation and modelling with QUBE-Servo 2](https://www.quanser.com/blog/instrumentation-and-modelling-with-qube-servo-2/) | Lists the labs: step response, frequency response, parameter estimation (R and the back-EMF constant measured, because units differ from the datasheet), friction identification (Coulomb and viscous); the QUBE-Servo 2 has a current sensor, the first QUBE-Servo did not |
| [quanser-openai-driver](https://github.com/BlueRiverTech/quanser-openai-driver) (`gym_brt/quanser/qube_simulator.py`) | A simulator of the real Furuta: R 8.4, k_t = k_e = 0.042, D_r 2.7E-4, D_p 5E-5 N m s/rad; no voltage saturation and no Coulomb friction in it. `rotary_cfg.py` takes 2.75E-4 and 5.05E-5 from the vision-based-furuta-pendulum repo of the Data-Science-in-Mechanical-Engineering group (not re-checked here) |

Local copies of three of these (datasheet, user manual, lab sheet 1) are kept next to this file but not committed: they are
not ours to redistribute.

## What follows from them

Checked against the datasheet numbers (the check is an arithmetic one, not a measurement):

- The constants are consistent: back-EMF at no load (18 V - 8.4 x 0.022 A) / 424 rad/s = 0.0420 V s/rad = k_t; mechanical
  time constant J R / k_t^2 = 19.05 ms (datasheet 19 ms); starting torque 0.042 x 2.1 A = 88 mN m (datasheet 89).
- Electrical time constant L / R = 0.138 ms. The physics step is 2 ms, so the winding is integrated exactly (as `motor.py`
  does); the 512 Hz PWM of the ESP32 is slower than that, see the top: the model uses the average voltage.
- No-load friction of the motor alone: k_t x 22 mA = 0.92 mN m at 4050 RPM, i.e. about 2.2E-6 N m s/rad if it were all
  viscous. The Coulomb friction of the Quanser blog (0.1 mN m) is worth 20 mV of voltage (R x tau / k_t), so the dead zone from
  friction is tiny against a +-12 V range. A visible dead zone on the real robot would come from the driver (BTS7960 dead
  time, its voltage drop) or the PWM resolution, not from the brushes.
- The equivalent arm damping in `rotary_cfg.py` (2.75E-4) is 100 times the damping of the motor alone, and with it the model's
  open-loop speed gain is 1 / (k_e + R B / k_t) = 10.3 rad/s/V, against 23.8 rad/s/V (= 1 / k_e) when only the datasheet
  friction acts. This number came from fitting the whole arm with its pendulum module, not from the motor. A step-response
  test of the arm decides which is right.
- Winding heating: 0.5 A continuous gives 0.5^2 x 8.4 = 2.1 W, a 39 K rise with the two thermal resistances, copper 0.393 %/K,
  so R grows by about 15 %. At the 0.2 A rms the cascade PID uses it is 2.4 %. The assumed spread of the model (R x 0.9 to
  x 1.2, k x 0.9 to x 1.1) covers that.
- Stall at 12 V is 12 / 8.4 = 1.43 A (1.59 A with R x 0.9). The BTS7960 has no current limit worth speaking of (43 A), so
  nothing but the supply limits it; the Quanser manual warns that a stall above 5 V can damage the motor.

## Not published (measure it)

| What | Why it matters | Quick test |
|---|---|---|
| Voltage at the winding against the duty (BTS7960 drop, dead time, supply sag under load) | the +-12 V of the policy is a duty of a supply, not a regulated voltage | scope or multimeter across the motor at 10, 25, 50 and 100 % duty, once at rest and once with the arm spinning |
| Real R, k_t, k_e of your unit | the blog says the datasheet model is too fast | stall test at 1 V (R = V / I), speed at 5 V open loop (k_e = (V - R I) / w) |
| Friction of the arm with the pendulum module on | sets the arm damping | free decay of the arm with the motor off; ramp the duty from 0 until the arm just moves |
| Current (the BTS7960 has IS pins, unused so far) | the only way to check torque, not just speed | multimeter in series, or read R_IS / L_IS with an ADC |
| Latency of your microcontroller loop (read, compute, write) | the policy was trained with none | toggle a pin and look at it with a scope |
