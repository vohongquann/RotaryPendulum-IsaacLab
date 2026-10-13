# Firmware (ESP32)

Runs the two PIDs of the simulator on the real robot. The policy runs on a PC at 100 Hz and writes the six gains over USB
serial; the ESP32 does everything else at 500 Hz.

```
PC, 100 Hz:  12 observations -> policy -> a1..a6 ──G line──►  ESP32, 500 Hz:  encoders -> outer PID (250 Hz) -> inner PID -> PWM -> BTS7960 -> motor
             ◄──S line── theta, alpha, theta_dot, alpha_dot, voltage, setpoint, status
```

| File | What it does | Simulator counterpart |
|---|---|---|
| [src/config.h](src/config.h) | pins, PWM, encoder resolution, rates, gain ranges, safety limits | `rotary_cfg.py`, the constants of `mdp/actions.py` |
| [src/cascade_pid.h](src/cascade_pid.h) | the outer and inner PID, anti-windup, gain scaling | `CascadePidAction.arm_pid`, `pendulum_pid`, `process_actions` |
| [src/joint_state.h](src/joint_state.h) | angle in [-pi, pi] and filtered velocity from the counts | the joint angle and velocity the simulator reads |
| [src/encoder.cpp](src/encoder.cpp) | quadrature counts by interrupt on channel A | |
| [src/motor.cpp](src/motor.cpp) | voltage -> 16-bit PWM on RPWM or LPWM | `duty_to_pwm`, `QubeServo2Motor` |
| [src/main.ino](src/main.ino) | the 500 Hz tick, serial messages, watchdog | `apply_actions` |

`cascade_pid.h` was compared with the simulator's own `arm_pid` and `pendulum_pid` on 4000 random ticks: the voltage differs
by at most 0.2 mV and the setpoint by 1e-6 rad. The Arduino files were syntax-checked, not run on a board.

## Serial protocol (921600 baud, one text line per message)

| Direction | Line | Meaning |
|---|---|---|
| PC -> ESP32 | `G <cmd> <a1> <a2> <a3> <a4> <a5> <a6>` | `cmd`: arm angle to hold [rad]; `a1..a6`: the policy outputs in [-1, 1] (Kp1 Ki1 Kd1 Kp2 Ki2 Kd2), scaled by the ranges in `config.h` |
| PC -> ESP32 | `R` | zero the encoders and PIDs and clear a fault: do it with the arm centred and the pendulum hanging still |
| ESP32 -> PC | `S <ms> <theta> <alpha> <theta_dot> <alpha_dot> <voltage> <setpoint> <status>` | every 10 ms; angles in [-pi, pi] (alpha 0 hanging, +-pi upright) |

`status`: 0 running, 1 waiting for gains (or the PC went quiet for 100 ms: motor off), 2 the arm turned half a turn (motor
off until `R`).

The policy's 12 observations, from an `S` line and the last `G`:
`wrap(theta - cmd)`, `cmd`, `sin(alpha)`, `cos(alpha)`, `theta_dot`, `alpha_dot`, then `a1..a6` of the previous step. The
PC must answer every `S` line with a `G` line.

## Before the first run

1. **Signs.** `ARM_SIGN`, `PENDULUM_SIGN`, `MOTOR_SIGN` in `config.h` are `+1`. The simulator's angle is positive when the
   joint turns about its URDF axis. Check on the robot: with the motor off, turn the arm by hand and the pendulum by hand and
   read `theta` and `alpha` in the `S` lines; a positive voltage must turn the arm towards positive `theta`. Flip the sign that
   disagrees. If the policy still pushes the wrong way, look at `alpha`: hanging is 0 and upright is +-pi.
2. **Supply.** `V_MAX` is 12 V: the voltage at 100 % duty, so the supply must be 12 V under load.
3. **Zero.** Send `R` with the arm at the centre of its range and the pendulum hanging still.
4. **Rates.** The simulator runs the inner PID at exactly 500 Hz and the outer at 250 Hz; the loop uses `micros()` and runs
   the ticks as they fall due. If `S` lines show jitter, move the loop to a hardware timer.

## What the simulator does not have

The policy was trained with an ideal angle and velocity inside the PIDs, and with Gaussian noise (angle 2 mrad, velocity
0.1 rad/s) on what the policy sees. This firmware has the real thing: 1024 counts per turn (6.1 mrad), a velocity from the
difference of two counts through 50 / (s + 50), and a 1 to 2 ms delay between the encoders and the PWM. The velocity noise it
gives is about 0.09 rad/s, near the 0.1 used in training. If the real robot shakes, compare the velocity of the `S` lines with
a tachometer or a slow video before changing the gains.

## Build

Arduino IDE or `arduino-cli` with the ESP32 core 3.x (`ledcAttach(pin, freq, bits)`). The IDE wants the sketch folder to be
named like the sketch: copy `src/` to a folder called `main`, or rename `main.ino`.
