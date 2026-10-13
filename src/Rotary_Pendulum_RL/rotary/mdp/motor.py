"""The QUBE-Servo 2 motor, from the PWM the firmware writes to the torque on the arm shaft.

On the real robot the H-bridge, the winding and the rotor do this and nobody computes it. In the simulation it is computed
here, once per physics step, in three lines:

    1. voltage   v = PWM / PWM_MAX * V_MAX                  average voltage the H-bridge puts on the winding
    2. current   L di/dt = v - R i - ke w                    winding with its back-EMF (w: arm speed), solved exactly
    3. torque    tau = kt i                                  torque on the arm shaft

The rotor inertia and the viscous friction of the arm are joint properties of the articulation (``rotary_cfg.ROTARY_CFG``).
The back-EMF ke w is why the torque falls as the arm speeds up: the electrical damping kt ke / R = 2.1e-4 N m s/rad is as
large as the viscous friction of the arm.

Plot of the torque curves: ``python scripts/plot_motor.py``.
"""
import torch

from Rotary_Pendulum_RL.rotary import rotary_cfg as C


def duty_to_pwm(duty: torch.Tensor, pwm_max: int = C.PWM_MAX) -> torch.Tensor:
    """Duty cycle in [-1, 1] -> signed PWM counts (whole numbers, kept as floats): what the timer can write."""
    return torch.round(duty.clamp(-1.0, 1.0) * pwm_max)


def pwm_to_voltage(pwm: torch.Tensor, pwm_max: int = C.PWM_MAX, v_max: float = C.QUBE_V_MAX) -> torch.Tensor:
    """Signed PWM counts -> average voltage across the winding [V]."""
    return pwm / pwm_max * v_max


class QubeServo2Motor:
    """The winding current and torque of ``num_envs`` motors.

    Every episode each motor gets its own constants (domain randomisation), because the real motor differs from the
    datasheet and its resistance grows as it heats:

        k_scale   multiplies the torque constant and the back-EMF constant (the same physical constant), in ``k_range``
        r_scale   multiplies the resistance, in ``r_range``

    The ranges are ASSUMPTIONS, not published.
    """

    def __init__(self, num_envs: int, device, randomize: bool = True,
                 k_range: tuple[float, float] = (0.9, 1.1), r_range: tuple[float, float] = (0.9, 1.2)):
        self.device = device
        self.randomize = randomize
        self.k_range = k_range
        self.r_range = r_range

        self.k_scale = torch.ones(num_envs, device=device)
        self.r_scale = torch.ones(num_envs, device=device)

        self.voltage = torch.zeros(num_envs, device=device)    # v [V]    applied in the last step
        self.current = torch.zeros(num_envs, device=device)    # i [A]    winding current
        self.torque = torch.zeros(num_envs, device=device)     # tau [N m] applied in the last step

        self.resample(torch.arange(num_envs, device=device))

    def resample(self, env_ids) -> None:
        """Draw new motor constants for ``env_ids`` (an index tensor or a slice)."""
        k_scale = self.k_scale[env_ids]
        r_scale = self.r_scale[env_ids]
        if self.randomize:
            k_scale.uniform_(*self.k_range)
            r_scale.uniform_(*self.r_range)
        else:
            k_scale.fill_(1.0)
            r_scale.fill_(1.0)
        self.k_scale[env_ids] = k_scale
        self.r_scale[env_ids] = r_scale

    def reset(self, env_ids) -> None:
        """Motor at rest (no current, no torque) with new random constants."""
        self.voltage[env_ids] = 0.0
        self.current[env_ids] = 0.0
        self.torque[env_ids] = 0.0
        self.resample(env_ids)

    def step(self, pwm: torch.Tensor, arm_speed: torch.Tensor, dt: float) -> torch.Tensor:
        """Run the winding for ``dt`` seconds with the PWM ``pwm`` (N,) held and the arm turning at ``arm_speed`` (N,) [rad/s].
        Returns the torque [N m] the motor puts on the arm during this step.

        With v and w held constant the current moves from its old value to its steady value with the electrical time
        constant L / R = 0.14 ms:

            i_steady = (v - ke w) / R
            i_new    = i_steady + (i_old - i_steady) exp(-dt R / L)

        This is the exact solution; an explicit Euler step would be unstable because dt (2 ms) is 14 time constants.
        """
        resistance = C.QUBE_RM_OHM * self.r_scale
        torque_constant = C.QUBE_KT * self.k_scale
        back_emf_constant = C.QUBE_KE * self.k_scale

        voltage = pwm_to_voltage(pwm)
        back_emf = back_emf_constant * arm_speed
        steady_current = (voltage - back_emf) / resistance
        decay = torch.exp(-dt * resistance / C.QUBE_LM_H)

        self.current = steady_current + (self.current - steady_current) * decay
        self.voltage[:] = voltage
        self.torque[:] = torque_constant * self.current
        return self.torque
