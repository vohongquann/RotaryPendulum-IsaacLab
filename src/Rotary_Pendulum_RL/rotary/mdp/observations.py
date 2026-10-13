"""Observation terms, read from the simulator state."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ACTION_NAME = "motor"


# ``term.joint_ids`` is [arm, pendulum]; the action term is looked up only for that order.


def arm_angle_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Arm angle minus the commanded angle, wrapped to [-pi, pi] [rad] (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    arm_angle = env.scene[term.cfg.asset_name].data.joint_pos.torch[:, term.joint_ids[0]]
    return wrap_to_pi(arm_angle - env.command_manager.get_command(command_name)[:, 0]).unsqueeze(-1)


def pendulum_angle_sin(env: ManagerBasedRLEnv) -> torch.Tensor:
    """sin of the pendulum angle (0 hanging, +-pi upright) (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    pendulum_angle = env.scene[term.cfg.asset_name].data.joint_pos.torch[:, term.joint_ids[1]]
    return torch.sin(pendulum_angle).unsqueeze(-1)


def pendulum_angle_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """cos of the pendulum angle (1 hanging, -1 upright) (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    pendulum_angle = env.scene[term.cfg.asset_name].data.joint_pos.torch[:, term.joint_ids[1]]
    return torch.cos(pendulum_angle).unsqueeze(-1)


def arm_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Arm angular velocity [rad/s] (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    return env.scene[term.cfg.asset_name].data.joint_vel.torch[:, term.joint_ids[0:1]]


def pendulum_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pendulum angular velocity [rad/s] (N, 1)."""
    term = env.action_manager.get_term(ACTION_NAME)
    return env.scene[term.cfg.asset_name].data.joint_vel.torch[:, term.joint_ids[1:2]]
