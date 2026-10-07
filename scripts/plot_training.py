"""Draw the training curves of a run from its TensorBoard events (no Isaac Sim needed).

    conda activate env_isaaclab
    python scripts/plot_training.py logs/rsl_rl/rotary_pendulum_pid/<run> --out docs/media/training_pid.png

Writes docs/media/training.png (``--out`` to change).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("run", help="run folder with the events file")
parser.add_argument("--out", default="docs/media/training.png")
args = parser.parse_args()

events = EventAccumulator(args.run, size_guidance={"scalars": 0})
events.Reload()


def series(tag):
    values = events.Scalars(tag)
    return np.array([v.step for v in values]), np.array([v.value for v in values])


# Reference palette, light mode: categorical slots in fixed order, text and grid tokens.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REWARD_TERMS = ["pendulum_upright", "arm_angle", "arm_tracking", "arm_velocity", "pendulum_velocity", "voltage"]
TERMINATIONS = ["time_out", "pivot_limit"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlelocation": "left", "axes.titlesize": 10, "axes.titlecolor": INK})
fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), facecolor=SURFACE)
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

steps, values = series("Train/mean_reward")
axes[0, 0].plot(steps, values, color=SLOTS[0], linewidth=1.6)
axes[0, 0].set_title("Mean episode return")
steps, values = series("Train/mean_episode_length")
axes[0, 1].plot(steps, values * 0.01, color=SLOTS[0], linewidth=1.6)
axes[0, 1].set_title("Mean episode length [s] (10 s = no early end)")
for color, term in zip(SLOTS, REWARD_TERMS):
    steps, values = series(f"Episode_Reward/{term}")
    axes[1, 0].plot(steps, values, color=color, linewidth=1.6, label=term)
axes[1, 0].set_title("Reward terms (sum per episode, weighted)")
axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
for color, term in zip(SLOTS, TERMINATIONS):
    steps, values = series(f"Episode_Termination/{term}")
    axes[1, 1].plot(steps, values, color=color, linewidth=1.6, label=term)
axes[1, 1].set_title("Share of episode ends by cause")
axes[1, 1].legend(frameon=False, fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("PPO iteration")
fig.suptitle(f"Training, run {Path(args.run).name}", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=150, facecolor=SURFACE)
print(f"wrote {args.out}")
