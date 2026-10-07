"""Action term: the policy writes the gains of two PIDs in cascade; the inner PID writes the motor voltage, as the firmware
would. Explained step by step in guide/07_cascade_pid.md, the Isaac Lab hook in guide/08_action_term.md.

    process_actions, 100 Hz:  a (N, 6) in [-1, 1] -> gains = a * (ARM_KP_MAX, ARM_KI_MAX, ARM_KD_MAX,
                                                                  PENDULUM_KP_MAX, PENDULUM_KI_MAX, PENDULUM_KD_MAX)
    apply_actions, every physics step (500 Hz):
        read_encoders                   500 Hz (``ENCODER_HZ``)
        arm_pid       (outer)  250 Hz   arm error to the command -> pendulum setpoint theta2*
        pendulum_pid  (inner)  500 Hz   pendulum error to theta2* -> voltage, cut at +-QUBE_V_MAX -> PWM
        drive_motor                     500 Hz: PWM, true shaft speed -> ``motor.QubeServo2Motor`` -> torque -> arm joint

The encoders (``sensors.JointEncoders``) are read inside the action term, like the firmware reads them in its timer tick; the
observations read them from here. The pendulum joint is passive.
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from Rotary_Pendulum_RL.rotary import rotary_cfg as C
from Rotary_Pendulum_RL.rotary.mdp.motor import QubeServo2Motor, duty_to_pwm
from Rotary_Pendulum_RL.rotary.mdp.sensors import JointEncoders

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Gain ranges: the policy writes a in [-1, 1], the gain is a times the largest value.
TILT_MAX = math.radians(10.0)
"""Largest lean of the pendulum the outer loop may ask for [rad]: a larger setpoint would drop the pendulum out of the
inner loop's reach."""
ARM_KP_MAX = 1.0
"""Outer loop, largest |Kp| [rad/rad]: the full lean at an arm error equal to it (10 deg)."""
ARM_KI_MAX = 1.0
"""Outer loop, largest |Ki| [rad/(rad s)]: the full lean once the arm error integral reaches 0.17 rad s."""
ARM_KD_MAX = 0.1
"""Outer loop, largest |Kd| [rad s/rad]: the full lean at an arm speed of 1.7 rad/s."""
PENDULUM_KP_MAX = 100.0
"""Inner loop, largest |Kp| [V/rad]: 12 V at a pendulum error of 0.12 rad (7 deg)."""
PENDULUM_KI_MAX = 100.0
"""Inner loop, largest |Ki| [V/(rad s)]: 12 V once the error integral reaches 0.12 rad s."""
PENDULUM_KD_MAX = 10.0
"""Inner loop, largest |Kd| [V s/rad]: 12 V at a pendulum speed of 1.2 rad/s."""
OUTER_HZ = 250.0
"""Rate of the outer (arm) loop. The inner (pendulum) loop runs on every encoder read, ``ENCODER_HZ``."""


class CascadePidAction(ActionTerm):
    """Positive gains are the stable sign of both loops (fixed-gain check from upright: inner Kp2 50, Kd2 2, outer Kp1 0.2,
    Kd1 0.02 hold the arm on a 0.3 rad command within 0.5 deg)."""

    cfg: "CascadePidActionCfg"

    def __init__(self, cfg: "CascadePidActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)                       # sets self.cfg, self._env, self._asset (the robot)
        self._arm_id = self._asset.find_joints(C.ARM_JOINT)[0]
        self._pendulum_id = self._asset.find_joints(C.PENDULUM_JOINT)[0]
        self._joint_ids = self._arm_id + self._pendulum_id          # order of ``encoders`` columns: arm, pendulum

        # Physics steps between two runs of each loop: 1 for the encoders and the inner loop (500 Hz), 2 for the outer loop
        # (250 Hz). Both must be whole numbers of physics steps, and the outer loop must fall on an encoder read.
        physics_dt = env.physics_dt
        self._encoder_period = round(1.0 / (C.ENCODER_HZ * physics_dt))
        self._outer_period = round(1.0 / (OUTER_HZ * physics_dt))
        if abs(self._encoder_period * physics_dt * C.ENCODER_HZ - 1.0) > 1e-6:
            raise ValueError(f"the physics rate {1.0 / physics_dt} Hz is not a multiple of the encoder rate {C.ENCODER_HZ} Hz")
        if self._outer_period % self._encoder_period != 0:
            raise ValueError(f"the outer loop ({OUTER_HZ} Hz) must run on an encoder read ({C.ENCODER_HZ} Hz)")
        self.outer_dt = self._outer_period * physics_dt
        self.inner_dt = self._encoder_period * physics_dt

        self.motor = QubeServo2Motor(self.num_envs, self.device, randomize=cfg.randomize_motor)
        self.encoders = JointEncoders(self.num_envs, 2, self.inner_dt, self.device)
        self._gain_max = torch.tensor(
            [ARM_KP_MAX, ARM_KI_MAX, ARM_KD_MAX, PENDULUM_KP_MAX, PENDULUM_KI_MAX, PENDULUM_KD_MAX], device=self.device
        )
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.gains = torch.zeros(self.num_envs, 6, device=self.device)        # Kp1, Ki1, Kd1, Kp2, Ki2, Kd2
        self.arm_integral = torch.zeros(self.num_envs, device=self.device)    # I1 [rad s]
        self.pendulum_integral = torch.zeros(self.num_envs, device=self.device)  # I2 [rad s]
        self.pendulum_setpoint = torch.full((self.num_envs,), math.pi, device=self.device)  # theta2* [rad]
        self._voltage = torch.zeros(self.num_envs, device=self.device)        # V [V], written by the inner loop
        self._pwm = torch.zeros(self.num_envs, device=self.device)
        # Physics steps since the start, never reset: each loop runs when it is a multiple of its period, like a timer tick.
        self._tick = 0

    @property
    def joint_ids(self) -> list[int]:
        """Arm and pendulum joint ids: the column order of ``encoders``."""
        return self._joint_ids

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The motor voltage the inner loop applies (N, 1) [V]."""
        return self._voltage.unsqueeze(-1)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self.gains[env_ids] = 0.0
        self.arm_integral[env_ids] = 0.0
        self.pendulum_integral[env_ids] = 0.0
        self.pendulum_setpoint[env_ids] = math.pi
        self._voltage[env_ids] = 0.0
        self._pwm[env_ids] = 0.0
        self.motor.reset(env_ids)
        self.encoders.reset(env_ids, self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids])

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self.gains[:] = actions.clamp(-1.0, 1.0) * self._gain_max

    def apply_actions(self):
        if self._tick % self._encoder_period == 0:
            self.read_encoders()
            if self._tick % self._outer_period == 0:
                self.arm_pid()
            self.pendulum_pid()
        self.drive_motor()
        self._tick += 1

    def read_encoders(self):
        """Read the true joint angles once: quantised angle and filtered velocity in ``encoders``."""
        self.encoders.update(self._asset.data.joint_pos.torch[:, self._joint_ids])

    def arm_pid(self):
        """Outer loop: e1 = measured arm angle - commanded angle (wrapped), I1 += e1 dt,
        theta2* = pi + (Kp1 e1 + Ki1 I1 + Kd1 w1), the lean cut at +-TILT_MAX. w1 is the filtered encoder velocity
        (the command is held, so it is de1/dt). I1 is cut where Ki1 I1 alone reaches TILT_MAX at the largest Ki1."""
        command = self._env.command_manager.get_command(self.cfg.command_name)[:, 0]
        error = wrap_to_pi(self.encoders.angle[:, 0] - command)
        limit = TILT_MAX / ARM_KI_MAX
        self.arm_integral[:] = (self.arm_integral + error * self.outer_dt).clamp(-limit, limit)
        kp, ki, kd = self.gains[:, 0], self.gains[:, 1], self.gains[:, 2]
        lean = kp * error + ki * self.arm_integral + kd * self.encoders.velocity[:, 0]
        self.pendulum_setpoint[:] = math.pi + lean.clamp(-TILT_MAX, TILT_MAX)

    def pendulum_pid(self):
        """Inner loop: e2 = measured pendulum angle - theta2* (wrapped: 0 on the setpoint, about +-pi hanging),
        I2 += e2 dt, V = -(Kp2 e2 + Ki2 I2 + Kd2 w2), cut at +-QUBE_V_MAX -> PWM. I2 is cut where Ki2 I2 alone reaches
        QUBE_V_MAX at the largest Ki2."""
        error = wrap_to_pi(self.encoders.angle[:, 1] - self.pendulum_setpoint)
        limit = C.QUBE_V_MAX / PENDULUM_KI_MAX
        self.pendulum_integral[:] = (self.pendulum_integral + error * self.inner_dt).clamp(-limit, limit)
        kp, ki, kd = self.gains[:, 3], self.gains[:, 4], self.gains[:, 5]
        voltage = -(kp * error + ki * self.pendulum_integral + kd * self.encoders.velocity[:, 1])
        self._voltage[:] = voltage.clamp(-C.QUBE_V_MAX, C.QUBE_V_MAX)
        self._pwm[:] = duty_to_pwm(self._voltage / C.QUBE_V_MAX)

    def drive_motor(self):
        """One physics step of the motor with the PWM held: torque from the PWM and the true arm speed, written as the
        effort of the arm joint."""
        arm_velocity = self._asset.data.joint_vel.torch[:, self._arm_id[0]]
        torque = self.motor.step(self._pwm, arm_velocity, self._env.physics_dt)
        self._asset.set_joint_effort_target_index(target=torque.unsqueeze(-1), joint_ids=self._arm_id)


@configclass
class CascadePidActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = CascadePidAction

    asset_name: str = MISSING
    command_name: str = "heading_cmd"   # the commanded arm angle, target of the outer loop
    randomize_motor: bool = True       # draw k_t, k_e and R every episode (``QubeServo2Motor``)
