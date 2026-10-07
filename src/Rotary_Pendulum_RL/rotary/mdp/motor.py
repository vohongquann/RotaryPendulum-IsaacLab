"""The QUBE-Servo 2 motor, from the PWM written by the firmware to the torque on the arm shaft.

    python src/Rotary_Pendulum_RL/rotary/mdp/motor.py                    # writes docs/media/motor_torque.png
    python src/Rotary_Pendulum_RL/rotary/mdp/motor.py --show             # open a window instead

On the real hardware the H-bridge, the winding and the rotor turn the PWM into a torque and nobody computes it; in the
simulation it is computed here, once per physics step. There is no thrust on this robot: the counterpart of the drone's
``PWM -> thrust`` curve is ``PWM -> voltage -> current -> torque``:

    PWM (signed counts)  -->  v = PWM / PWM_MAX * V_max                       average voltage of the H-bridge
    v, shaft speed w     -->  L di/dt = v - R i - k_e w                       winding, solved exactly over one step
    i                    -->  tau = k_t i                                     torque on the shaft

The viscous friction of the arm and the rotor inertia are not here: they are joint properties of the articulation
(``rotary_cfg.ROTARY_CFG``). The back-EMF term ``k_e w`` is what makes the torque fall as the arm speeds up (the electrical
damping ``k_t k_e / R`` = 2.1e-4 N m s/rad, as large as the viscous friction of the arm).

Not modelled because it is not published for the QUBE-Servo 2: the current limit of the amplifier, the dead-time of the
bridge and the Coulomb friction of the brushes.
"""
import math

import torch

from Rotary_Pendulum_RL.rotary import rotary_cfg as C


def duty_to_pwm(duty: torch.Tensor, pwm_max: int = C.PWM_MAX) -> torch.Tensor:
    """Duty cycle in [-1, 1] -> signed integer PWM counts (as floats), the quantisation of the timer."""
    return torch.round(duty.clamp(-1.0, 1.0) * pwm_max)


def pwm_to_voltage(pwm: torch.Tensor, pwm_max: int = C.PWM_MAX, v_max: float = C.QUBE_V_MAX) -> torch.Tensor:
    """Signed PWM counts -> average voltage across the winding [V]."""
    return pwm / pwm_max * v_max


class QubeServo2Motor:
    """Winding current and torque of ``num_envs`` motors.

    ``k_scale`` multiplies the torque and back-EMF constants (the same physical constant) and ``r_scale`` the resistance;
    both are drawn every episode (``resample``): the real motor differs from the datasheet and its resistance grows as it
    heats, so the controller is trained on a spread of motors, not on one exact copy (domain randomisation). The ranges are
    ASSUMPTIONS, not published.
    """

    def __init__(self, num_envs: int, device, randomize: bool = True,
                 k_range: tuple[float, float] = (0.9, 1.1), r_range: tuple[float, float] = (0.9, 1.2)):
        self.device = device
        self.randomize, self.k_range, self.r_range = randomize, k_range, r_range
        self.current = torch.zeros(num_envs, device=device)      # i [A]
        self.torque = torch.zeros(num_envs, device=device)       # k_t i [N m], what the last step applied
        self.voltage = torch.zeros(num_envs, device=device)      # v [V], what the last step applied
        self.k_scale = torch.ones(num_envs, device=device)
        self.r_scale = torch.ones(num_envs, device=device)
        self.resample(torch.arange(num_envs, device=device))

    def resample(self, env_ids) -> None:
        n = len(env_ids)
        if self.randomize:
            self.k_scale[env_ids] = self.k_range[0] + torch.rand(n, device=self.device) * (self.k_range[1] - self.k_range[0])
            self.r_scale[env_ids] = self.r_range[0] + torch.rand(n, device=self.device) * (self.r_range[1] - self.r_range[0])
        else:
            self.k_scale[env_ids] = 1.0
            self.r_scale[env_ids] = 1.0

    def reset(self, env_ids) -> None:
        """Motor at rest (no current), new random constants."""
        self.current[env_ids] = 0.0
        self.torque[env_ids] = 0.0
        self.voltage[env_ids] = 0.0
        self.resample(env_ids)

    def step(self, pwm: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the winding by ``dt`` [s] with the PWM ``pwm`` (N,) held and the shaft speed ``omega`` (N,) [rad/s];
        returns the torque [N m] to apply on the arm for this step.

        The step is exact for a constant ``v`` and ``omega``: ``i`` goes from its value to the steady state
        ``(v - k_e w) / R`` with the electrical time constant ``L / R`` (0.14 ms), shorter than the physics step, so an
        explicit Euler step would be unstable.
        """
        k = C.QUBE_KT * self.k_scale
        r = C.QUBE_RM_OHM * self.r_scale
        v = pwm_to_voltage(pwm)
        steady = (v - C.QUBE_KE * self.k_scale * omega) / r
        self.current = steady + (self.current - steady) * torch.exp(-dt * r / C.QUBE_LM_H)
        self.voltage[:] = v
        self.torque[:] = k * self.current
        return self.torque


def quantize_angle(angle: torch.Tensor, counts_per_rev: int = C.QUBE_ENCODER_CPR) -> torch.Tensor:
    """What the encoder reports: the angle rounded to a whole number of counts [rad]."""
    step = 2.0 * math.pi / counts_per_rev
    return torch.round(angle / step) * step


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[4] / "docs" / "media" / "motor_torque.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (volt, left, right) = plt.subplots(1, 3, figsize=(15, 3.8))

    # First: steady torque against the voltage the policy writes, one line per arm speed: tau = k_t (v - k_e w) / R.
    voltage = torch.linspace(-C.QUBE_V_MAX, C.QUBE_V_MAX, 200)
    for omega_now in (0.0, 10.0, 50.0, 100.0):
        motor = QubeServo2Motor(200, "cpu", randomize=False)
        pwm = voltage / C.QUBE_V_MAX * C.PWM_MAX
        for _ in range(20):
            torque = motor.step(pwm, torch.full((200,), omega_now), 0.002)
        volt.plot(voltage, torque * 1000, label=f"arm at {omega_now:.0f} rad/s")
    volt.axvline(10.0, color="k", ls="--", lw=0.8)
    volt.axvline(-10.0, color="k", ls="--", lw=0.8, label="+-10 V (amplifier range in the Quanser manual)")
    volt.set_xlabel("voltage written by the policy [V]")
    volt.set_ylabel("motor torque [mN m]")
    volt.set_title("Torque against voltage (steady)")
    volt.grid(alpha=0.3)
    volt.legend(fontsize=8)

    # Left: steady torque against shaft speed, one line per duty (the winding settled): tau = k_t (v - k_e w) / R.
    omega = torch.linspace(0.0, 440.0, 200)
    for duty in (0.25, 0.5, 0.75, 1.0):
        motor = QubeServo2Motor(200, "cpu", randomize=False)
        pwm = torch.full((200,), duty * C.PWM_MAX)
        for _ in range(20):
            torque = motor.step(pwm, omega, 0.002)
        left.plot(omega, torque * 1000, label=f"duty {duty:.2f} ({duty * C.QUBE_V_MAX:.1f} V)")
    left.set_xlabel("arm shaft speed [rad/s]")
    left.set_ylabel("motor torque [mN m]")
    left.set_title("Torque against speed (steady)")
    left.grid(alpha=0.3)
    left.legend()

    # Right: the winding after a full-duty step at rest, at the physics step of the simulation.
    motor = QubeServo2Motor(1, "cpu", randomize=False)
    times, currents = [], []
    for n in range(10):
        motor.step(torch.tensor([float(C.PWM_MAX)]), torch.zeros(1), C.QUBE_LM_H / C.QUBE_RM_OHM / 10)
        times.append((n + 1) * C.QUBE_LM_H / C.QUBE_RM_OHM / 10 * 1e3)
        currents.append(motor.current.item())
    right.plot(times, currents, "o-")
    right.axhline(C.QUBE_V_MAX / C.QUBE_RM_OHM, color="k", ls="--", lw=0.8, label="stall current v / R")
    right.set_xlabel("time [ms] (one time constant L / R = 0.14 ms)")
    right.set_ylabel("winding current [A]")
    right.set_title("Current after a full-duty step")
    right.grid(alpha=0.3)
    right.legend()
    fig.tight_layout()
    if args.show:
        plt.show()
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"wrote {args.output}")
