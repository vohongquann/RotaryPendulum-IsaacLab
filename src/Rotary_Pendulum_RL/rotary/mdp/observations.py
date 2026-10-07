"""Observation terms. The policy reads what the firmware measures (quantised encoder angle, filtered velocity, from
``sensors.JointEncoders`` through the action term); the critic is given the true joint state (privileged)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ACTION_NAME = "motor"


# Column 0 of the encoder / joint arrays is the arm, column 1 the pendulum (``term.joint_ids`` order).
# ``measured=True``: the encoders as the firmware sees them (the actor); ``False``: the simulator state (the critic).


def arm_angle_error(env: ManagerBasedRLEnv, command_name: str, measured: bool = True) -> torch.Tensor:
    """Arm angle minus the commanded angle, wrapped to [-pi, pi] [rad] (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    if measured:
        arm_angle = term.encoders.angle[:, 0]
    else:
        arm_angle = env.scene[term.cfg.asset_name].data.joint_pos.torch[:, term.joint_ids[0]]
    return wrap_to_pi(arm_angle - env.command_manager.get_command(command_name)[:, 0]).unsqueeze(-1)


def pendulum_angle_sin(env: ManagerBasedRLEnv, measured: bool = True) -> torch.Tensor:
    """sin of the pendulum angle (0 hanging, +-pi upright) (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    if measured:
        pendulum_angle = term.encoders.angle[:, 1]
    else:
        pendulum_angle = env.scene[term.cfg.asset_name].data.joint_pos.torch[:, term.joint_ids[1]]
    return torch.sin(pendulum_angle).unsqueeze(-1)


def pendulum_angle_cos(env: ManagerBasedRLEnv, measured: bool = True) -> torch.Tensor:
    """cos of the pendulum angle (1 hanging, -1 upright) (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    if measured:
        pendulum_angle = term.encoders.angle[:, 1]
    else:
        pendulum_angle = env.scene[term.cfg.asset_name].data.joint_pos.torch[:, term.joint_ids[1]]
    return torch.cos(pendulum_angle).unsqueeze(-1)


def arm_velocity(env: ManagerBasedRLEnv, measured: bool = True) -> torch.Tensor:
    """Arm angular velocity [rad/s] (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    if measured:
        return term.encoders.velocity[:, 0:1]
    return env.scene[term.cfg.asset_name].data.joint_vel.torch[:, term.joint_ids[0:1]]


def pendulum_velocity(env: ManagerBasedRLEnv, measured: bool = True) -> torch.Tensor:
    """Pendulum angular velocity [rad/s] (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    if measured:
        return term.encoders.velocity[:, 1:2]
    return env.scene[term.cfg.asset_name].data.joint_vel.torch[:, term.joint_ids[1:2]]
