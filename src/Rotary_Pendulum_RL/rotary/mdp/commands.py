"""Command: the arm angle the policy holds while the pendulum stays upright."""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from Rotary_Pendulum_RL.rotary import rotary_cfg as C

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PivotHeadingCommand(CommandTerm):
    """Target arm angle [rad] (N, 1), uniform in ``+-cfg.heading_range``, redrawn every ``resampling_time_range`` seconds."""

    cfg: "PivotHeadingCommandCfg"

    def __init__(self, cfg: "PivotHeadingCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]
        self._arm_id = self._robot.find_joints(C.ARM_JOINT)[0]
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        arm_angle = self._robot.data.joint_pos.torch[:, self._arm_id[0]]
        self.metrics["error"] = wrap_to_pi(arm_angle - self._command[:, 0]).abs()

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.heading_range
        self._command[env_ids, 0] = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * r

    def _update_command(self):
        pass


@configclass
class PivotHeadingCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = PivotHeadingCommand

    asset_name: str = MISSING
    heading_range: float = math.pi / 2
