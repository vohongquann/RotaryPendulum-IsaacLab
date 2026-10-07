"""What the firmware measures: the encoder angles and a velocity estimated from them.

The simulator knows the exact angle and velocity of both joints; the microcontroller only has the encoder counts. The
policy reads the values of this file, never the simulator state, so the sim2real gap of the sensing (the
quantisation of 2 pi / 2048 = 3.07 mrad and the delay of the velocity filter) is in the training, not found on the hardware.
"""
import math

import torch

from Rotary_Pendulum_RL.rotary import rotary_cfg as C
from Rotary_Pendulum_RL.rotary.mdp.motor import quantize_angle


class JointEncoders:
    """Angle and velocity of ``n`` joints as the firmware sees them, updated once per encoder period (``ENCODER_HZ``).

        theta_q = round(theta / (2 pi / CPR)) * (2 pi / CPR)
        w_raw   = (theta_q - theta_q_prev) / dt                       finite difference
        w_hat  <- w_hat + a (w_raw - w_hat),   a = dt / (dt + 1 / w_f)   the high-pass w_f s / (s + w_f) of Quanser, ``VELOCITY_FILTER_RAD_S``
    """

    def __init__(self, num_envs: int, num_joints: int, dt: float, device, cutoff_rad_s: float = C.VELOCITY_FILTER_RAD_S):
        self.dt = dt
        self.alpha = dt / (dt + 1.0 / cutoff_rad_s)
        self.angle = torch.zeros(num_envs, num_joints, device=device)       # theta_q [rad]
        self.velocity = torch.zeros(num_envs, num_joints, device=device)    # w_hat [rad/s]

    def reset(self, env_ids, angle: torch.Tensor) -> None:
        """Start from the true ``angle`` (rows ``env_ids``) with zero velocity: no spike from a stale previous angle."""
        self.angle[env_ids] = quantize_angle(angle)
        self.velocity[env_ids] = 0.0

    def update(self, angle: torch.Tensor) -> None:
        """Read the true joint angles (N, n) once, as the firmware does in its timer tick."""
        angle_q = quantize_angle(angle)
        raw = (angle_q - self.angle) / self.dt
        self.velocity += self.alpha * (raw - self.velocity)
        self.angle[:] = angle_q
