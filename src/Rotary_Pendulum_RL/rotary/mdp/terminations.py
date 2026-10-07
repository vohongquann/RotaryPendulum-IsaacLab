"""Termination terms."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_when_pivot_exceeds_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_pivot_angle: float = math.pi / 2) -> torch.Tensor:
    """The arm is more than ``max_pivot_angle`` from its start (theta1 is 0 at the start)."""
    theta1 = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids].squeeze(-1)
    return theta1.abs() > max_pivot_angle

