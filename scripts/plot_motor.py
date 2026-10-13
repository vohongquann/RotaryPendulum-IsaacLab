"""Torque curves of the motor model of ``rotary/mdp/motor.py`` -> docs/media/motor_torque.png.

    python scripts/plot_motor.py                # writes docs/media/motor_torque.png
    python scripts/plot_motor.py --show         # open a window instead
"""
import argparse
from pathlib import Path

import matplotlib
import torch

from Rotary_Pendulum_RL.rotary import rotary_cfg as C
from Rotary_Pendulum_RL.rotary.mdp.motor import QubeServo2Motor

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "docs" / "media" / "motor_torque.png"))
parser.add_argument("--show", action="store_true")
args = parser.parse_args()
if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.002                        # the physics step
POINTS = 200


def settled_torque(pwm: torch.Tensor, arm_speed: torch.Tensor) -> torch.Tensor:
    """Torque once the winding has settled (20 steps), for a batch of PWM values and arm speeds."""
    motor = QubeServo2Motor(POINTS, "cpu", randomize=False)
    for _ in range(20):
        torque = motor.step(pwm, arm_speed, DT)
    return torque


fig, (vs_voltage, vs_speed, step_response) = plt.subplots(1, 3, figsize=(15, 3.8))

# Torque against the voltage the inner PID writes, one line per arm speed: tau = kt (v - ke w) / R.
voltage = torch.linspace(-C.QUBE_V_MAX, C.QUBE_V_MAX, POINTS)
for speed in (0.0, 10.0, 50.0, 100.0):
    torque = settled_torque(voltage / C.QUBE_V_MAX * C.PWM_MAX, torch.full((POINTS,), speed))
    vs_voltage.plot(voltage, torque * 1000, label=f"arm at {speed:.0f} rad/s")
vs_voltage.set_xlabel("voltage written by the inner PID [V]")
vs_voltage.set_ylabel("motor torque [mN m]")
vs_voltage.set_title("Torque against voltage (settled)")
vs_voltage.grid(alpha=0.3)
vs_voltage.legend(fontsize=8)

# Torque against the arm speed, one line per duty.
speed = torch.linspace(0.0, 440.0, POINTS)
for duty in (0.25, 0.5, 0.75, 1.0):
    torque = settled_torque(torch.full((POINTS,), duty * C.PWM_MAX), speed)
    vs_speed.plot(speed, torque * 1000, label=f"duty {duty:.2f} ({duty * C.QUBE_V_MAX:.1f} V)")
vs_speed.set_xlabel("arm shaft speed [rad/s]")
vs_speed.set_ylabel("motor torque [mN m]")
vs_speed.set_title("Torque against speed (settled)")
vs_speed.grid(alpha=0.3)
vs_speed.legend()

# The winding after a full-duty step at rest, in ten steps of a tenth of the electrical time constant.
time_constant = C.QUBE_LM_H / C.QUBE_RM_OHM
motor = QubeServo2Motor(1, "cpu", randomize=False)
times, currents = [], []
for n in range(10):
    motor.step(torch.tensor([float(C.PWM_MAX)]), torch.zeros(1), time_constant / 10)
    times.append((n + 1) * time_constant / 10 * 1e3)
    currents.append(motor.current.item())
step_response.plot(times, currents, "o-")
step_response.axhline(C.QUBE_V_MAX / C.QUBE_RM_OHM, color="k", ls="--", lw=0.8, label="stall current v / R")
step_response.set_xlabel("time [ms] (one time constant L / R = 0.14 ms)")
step_response.set_ylabel("winding current [A]")
step_response.set_title("Current after a full-duty step")
step_response.grid(alpha=0.3)
step_response.legend()

fig.tight_layout()
if args.show:
    plt.show()
else:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
