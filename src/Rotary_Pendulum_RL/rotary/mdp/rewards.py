"""Reward terms the Isaac Lab library does not have. The velocity penalties are the library's own
(``isaac_mdp.joint_vel_l2``), see ``rl_control/pid_env_cfg.py``.

theta2 is the pendulum angle: 0 hanging, +-pi upright. The joints are not limited, so theta2 keeps counting when the pendulum
spins; every term wraps the angle error first (``wrap_to_pi``).
The arm and pendulum joints are passed as ``SceneEntityCfg(..., joint_names=[...])`` so no joint order is assumed.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pendulum_upright_exp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """``exp(-(e / std)^2)``, e = pendulum angle - pi (wrapped) [rad]: 1 with the pendulum up, exp(-(pi / std)^2) hanging."""
    pendulum_angle = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids].squeeze(-1)
    error = wrap_to_pi(pendulum_angle - math.pi)
    return torch.exp(-torch.square(error / std))


def arm_angle_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """``exp(-(e / std)^2)``, e = arm angle - commanded angle (wrapped) [rad]: 1 with the arm on the command."""
    arm_angle = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids].squeeze(-1)
    error = wrap_to_pi(arm_angle - env.command_manager.get_command(command_name)[:, 0])
    return torch.exp(-torch.square(error / std))


def arm_angle_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, pendulum_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, std: float,
) -> torch.Tensor:
    """Same as ``arm_angle_exp`` with a narrow ``std``, only while the pendulum is upright (cos theta2 < -0.99): it still pulls
    when the error is a few degrees, where the wide ``arm_angle_exp`` is flat."""
    arm_angle = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids].squeeze(-1)
    pendulum_angle = env.scene[pendulum_cfg.name].data.joint_pos.torch[:, pendulum_cfg.joint_ids].squeeze(-1)
    error = wrap_to_pi(arm_angle - env.command_manager.get_command(command_name)[:, 0])
    tracking = torch.exp(-torch.square(error / std))
    return torch.where(torch.cos(pendulum_angle) < -0.99, tracking, torch.zeros_like(tracking))


def motor_voltage_l2(env: ManagerBasedRLEnv, action_name: str = "motor") -> torch.Tensor:
    """Square of the motor voltage the inner PID applies [V^2]. ``isaac_mdp.action_l2`` would square the action, which is
    the gains, not the voltage."""
    voltage = env.action_manager.get_term(action_name).processed_actions[:, 0]
    return torch.square(voltage)
