# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for the project."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])

# Rotary pendulum tasks (Isaac-Rotary-Pendulum-*): importing the package runs its gym.register() calls.
import Rotary_Pendulum_RL.rotary.rl_control  # noqa: E402, F401
