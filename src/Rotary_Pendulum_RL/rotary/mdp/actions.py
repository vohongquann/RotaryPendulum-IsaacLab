"""Action term: the policy writes the gains of two PIDs in cascade, the inner PID writes the motor voltage, as the firmware
would. Explained step by step in guide/07_cascade_pid.md, the Isaac Lab hook in guide/08_action_term.md.

Isaac Lab calls the term twice:

    process_actions   once per policy step (100 Hz)     a in [-1, 1]^6  ->  the six gains
    apply_actions     once per physics step (500 Hz)    runs the loops below

and ``apply_actions`` runs, in this order:

    arm_pid        outer loop  250 Hz   arm angle error to the command   ->  pendulum setpoint theta2*
    pendulum_pid   inner loop  500 Hz   pendulum angle error to theta2*  ->  voltage V, cut at +-V_MAX  ->  PWM
    drive_motor    every step           PWM and the arm speed  ->  ``motor.QubeServo2Motor``  ->  torque on the arm

Both PIDs read the joint angles and velocities of the simulator as they are. The pendulum joint is passive.
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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# The policy writes a gain a in [-1, 1]; the gain is a times the largest value below.
ARM_KP_MAX = 1.0           # outer loop [rad of lean per rad of arm error]: full lean at an error of 10 deg
ARM_KI_MAX = 1.0           # outer loop [rad of lean per rad s]: full lean at an error integral of 0.17 rad s
ARM_KD_MAX = 0.1           # outer loop [rad of lean per rad/s]: full lean at an arm speed of 1.7 rad/s
PENDULUM_KP_MAX = 100.0    # inner loop [V/rad]: 12 V at a pendulum error of 0.12 rad (7 deg)
PENDULUM_KI_MAX = 100.0    # inner loop [V/(rad s)]: 12 V at an error integral of 0.12 rad s
PENDULUM_KD_MAX = 10.0     # inner loop [V s/rad]: 12 V at a pendulum speed of 1.2 rad/s

# Largest lean of the pendulum the outer loop may ask for: a larger setpoint would drop the pendulum out of the inner
# loop's reach.
TILT_MAX = math.radians(10.0)

OUTER_HZ = 250.0           # rate of the outer (arm) loop; the inner (pendulum) loop runs at ``C.INNER_HZ``


class CascadePidAction(ActionTerm):
    """6 gains in, one motor voltage out (see the module docstring)."""

    cfg: "CascadePidActionCfg"

    def __init__(self, cfg: "CascadePidActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)                       # sets self.cfg, self._env, self._asset (the robot)

        # The joints: ``joint_ids`` is [arm, pendulum], the order the observation terms rely on.
        self._arm_id = self._asset.find_joints(C.ARM_JOINT)[0]
        self._pendulum_id = self._asset.find_joints(C.PENDULUM_JOINT)[0]
        self._joint_ids = self._arm_id + self._pendulum_id

        # How often each loop runs, in physics steps (1 = every step). Both must be whole numbers, and every outer
        # tick must also be an inner tick.
        physics_dt = env.physics_dt
        self._inner_period = round(1.0 / (C.INNER_HZ * physics_dt))
        self._outer_period = round(1.0 / (OUTER_HZ * physics_dt))
        if abs(self._inner_period * physics_dt * C.INNER_HZ - 1.0) > 1e-6:
            raise ValueError(f"the physics rate {1.0 / physics_dt} Hz is not a multiple of the inner loop rate {C.INNER_HZ} Hz")
        if self._outer_period % self._inner_period != 0:
            raise ValueError(f"the outer loop ({OUTER_HZ} Hz) must run on an inner loop tick ({C.INNER_HZ} Hz)")
        self.inner_dt = self._inner_period * physics_dt
        self.outer_dt = self._outer_period * physics_dt
        self._tick = 0              # physics steps since the start, never reset: each loop runs on its own multiples

        # The gains, as the policy last wrote them, and the largest value of each.
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)       # a in [-1, 1]
        self.gains = torch.zeros(self.num_envs, 6, device=self.device)              # Kp1 Ki1 Kd1 Kp2 Ki2 Kd2
        self._gain_max = torch.tensor(
            [ARM_KP_MAX, ARM_KI_MAX, ARM_KD_MAX, PENDULUM_KP_MAX, PENDULUM_KI_MAX, PENDULUM_KD_MAX], device=self.device
        )

        # State of the loops.
        self.arm_integral = torch.zeros(self.num_envs, device=self.device)          # I1 [rad s]
        self.pendulum_integral = torch.zeros(self.num_envs, device=self.device)     # I2 [rad s]
        self.pendulum_setpoint = torch.full((self.num_envs,), math.pi, device=self.device)   # theta2* [rad]
        self._voltage = torch.zeros(self.num_envs, device=self.device)              # V [V], written by the inner loop
        self._pwm = torch.zeros(self.num_envs, device=self.device)                  # the PWM counts of V

        self.motor = QubeServo2Motor(self.num_envs, self.device, randomize=cfg.randomize_motor)

    # ── what Isaac Lab reads ────────────────────────────────────────────────────────────

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

    @property
    def joint_ids(self) -> list[int]:
        """Arm and pendulum joint ids, in this order."""
        return self._joint_ids

    # ── called by Isaac Lab ─────────────────────────────────────────────────────────────

    def reset(self, env_ids=None):
        """New episode for ``env_ids`` (None: all): gains, integrals, setpoint, voltage and motor back to rest."""
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[env_ids] = 0.0
        self.gains[env_ids] = 0.0
        self.arm_integral[env_ids] = 0.0
        self.pendulum_integral[env_ids] = 0.0
        self.pendulum_setpoint[env_ids] = math.pi
        self._voltage[env_ids] = 0.0
        self._pwm[env_ids] = 0.0
        self.motor.reset(env_ids)

    def process_actions(self, actions: torch.Tensor):
        """Policy step (100 Hz): the network output becomes the gains, held until the next policy step."""
        self._raw_actions[:] = actions
        self.gains[:] = actions.clamp(-1.0, 1.0) * self._gain_max

    def apply_actions(self):
        """Physics step (500 Hz): the loops that are due, then the motor."""
        if self._tick % self._inner_period == 0:
            if self._tick % self._outer_period == 0:
                self.arm_pid()
            self.pendulum_pid()
        self.drive_motor()
        self._tick += 1

    # ── the loops ───────────────────────────────────────────────────────────────────────

    def arm_pid(self):
        """Outer loop. The arm angle error to the command gives the pendulum setpoint:

            e1      = wrap(arm angle - command)
            I1     += e1 dt                       cut where Ki1 I1 alone would reach TILT_MAX at the largest Ki1
            lean    = Kp1 e1 + Ki1 I1 + Kd1 w1    w1: arm speed (the command is held, so w1 is d e1 / dt)
            theta2* = pi + clamp(lean, +-TILT_MAX)
        """
        arm_angle = self._asset.data.joint_pos.torch[:, self._arm_id[0]]
        arm_speed = self._asset.data.joint_vel.torch[:, self._arm_id[0]]
        command = self._env.command_manager.get_command(self.cfg.command_name)[:, 0]
        kp, ki, kd = self.gains[:, 0], self.gains[:, 1], self.gains[:, 2]

        error = wrap_to_pi(arm_angle - command)
        integral_limit = TILT_MAX / ARM_KI_MAX
        self.arm_integral[:] = (self.arm_integral + error * self.outer_dt).clamp(-integral_limit, integral_limit)

        lean = kp * error + ki * self.arm_integral + kd * arm_speed
        self.pendulum_setpoint[:] = math.pi + lean.clamp(-TILT_MAX, TILT_MAX)

    def pendulum_pid(self):
        """Inner loop. The pendulum angle error to the setpoint gives the voltage:

            e2      = wrap(pendulum angle - theta2*)       0 on the setpoint, about +-pi when hanging
            I2     += e2 dt                                 cut where Ki2 I2 alone would reach V_MAX at the largest Ki2
            V       = -(Kp2 e2 + Ki2 I2 + Kd2 w2)           w2: pendulum speed; cut at +-V_MAX
            PWM     = V / V_MAX * PWM_MAX
        """
        pendulum_angle = self._asset.data.joint_pos.torch[:, self._pendulum_id[0]]
        pendulum_speed = self._asset.data.joint_vel.torch[:, self._pendulum_id[0]]
        kp, ki, kd = self.gains[:, 3], self.gains[:, 4], self.gains[:, 5]

        error = wrap_to_pi(pendulum_angle - self.pendulum_setpoint)
        integral_limit = C.QUBE_V_MAX / PENDULUM_KI_MAX
        self.pendulum_integral[:] = (self.pendulum_integral + error * self.inner_dt).clamp(-integral_limit, integral_limit)

        voltage = -(kp * error + ki * self.pendulum_integral + kd * pendulum_speed)
        self._voltage[:] = voltage.clamp(-C.QUBE_V_MAX, C.QUBE_V_MAX)
        self._pwm[:] = duty_to_pwm(self._voltage / C.QUBE_V_MAX)

    def drive_motor(self):
        """The motor for one physics step with the PWM held: torque from the PWM and the arm speed, written to the arm joint."""
        arm_speed = self._asset.data.joint_vel.torch[:, self._arm_id[0]]
        torque = self.motor.step(self._pwm, arm_speed, self._env.physics_dt)
        self._asset.set_joint_effort_target_index(target=torque.unsqueeze(-1), joint_ids=self._arm_id)


@configclass
class CascadePidActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = CascadePidAction

    asset_name: str = MISSING
    command_name: str = "heading_cmd"    # the commanded arm angle, the target of the outer loop
    randomize_motor: bool = True         # draw k_t, k_e and R every episode (``QubeServo2Motor``)
