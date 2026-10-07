"""Run a trained policy through a fixed scenario on many envs, print the metrics and draw the response chart of the README.

Scenario (the same for every env, 22 s): start hanging (random +-0.2 rad, +-0.5 rad/s, as in training), arm command 0
until 6 s, then steps to +60, -60, +30 and 0 deg every 4 s. The motor constants are drawn per env as in training, so the
spread of the plot is the spread of the motor and of the start.

    conda activate env_isaaclab
    python scripts/evaluate_policy.py --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt

Writes <out>/response_pid.png and <out>/response_pid_metrics.json (with the gains the policy chose).
"""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--checkpoint", required=True, help="model_N.pt of a run")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42, help="same seed = same starts and motor constants")
parser.add_argument("--out", default="docs/media", help="folder of the chart and the metrics")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from Rotary_Pendulum_RL.rotary import rotary_cfg as C  # noqa: E402
from Rotary_Pendulum_RL.rotary.rl_control.agents.rsl_rl_ppo_cfg import PidPPORunnerCfg  # noqa: E402
from Rotary_Pendulum_RL.rotary.rl_control.pid_env_cfg import HEADING_COMMAND, PidEnvCfg  # noqa: E402

COMMANDS_DEG = [(0.0, 0.0), (6.0, 60.0), (10.0, -60.0), (14.0, 30.0), (18.0, 0.0)]   # (start time [s], arm command [deg])
DURATION_S = 22.0
UPRIGHT_DEG = 10.0          # the pendulum counts as up within 10 deg of vertical
SETTLED_DEG = 5.0           # the arm counts as settled within 5 deg of the command

env_cfg = PidEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.episode_length_s = DURATION_S + 1.0
env_cfg.commands.heading_cmd.resampling_time_range = (1.0e9, 1.0e9)      # the scenario sets the command
env = ManagerBasedRLEnv(env_cfg)
agent_cfg = PidPPORunnerCfg()
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = create_rsl_rl_runner(wrapped, agent_cfg)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

robot = env.scene["robot"]
arm_id = robot.find_joints(C.ARM_JOINT)[0][0]
pendulum_id = robot.find_joints(C.PENDULUM_JOINT)[0][0]
motor_term = env.action_manager.get_term("motor")
command = env.command_manager.get_term(HEADING_COMMAND).command

steps = round(DURATION_S / env.step_dt)
time_s = np.arange(steps) * env.step_dt
command_deg = np.zeros(steps)
for start, value in COMMANDS_DEG:
    command_deg[time_s >= start] = value
arm_deg, pendulum_error_deg, voltage, current = (np.zeros((steps, args.num_envs)) for _ in range(4))
gains = np.zeros((steps, args.num_envs, 6))      # Kp1, Ki1, Kd1 (arm), Kp2, Ki2, Kd2 (pendulum)
failed = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)

command[:, 0] = 0.0
obs = wrapped.get_observations()
with torch.inference_mode():
    for k in range(steps):
        command[:, 0] = math.radians(command_deg[k])
        obs, _, dones, _ = wrapped.step(policy(obs))
        failed |= env.termination_manager.terminated
        joint_pos = robot.data.joint_pos.torch
        arm_deg[k] = torch.rad2deg(joint_pos[:, arm_id]).cpu().numpy()
        pendulum_error = torch.atan2(torch.sin(joint_pos[:, pendulum_id] - math.pi), torch.cos(joint_pos[:, pendulum_id] - math.pi))
        pendulum_error_deg[k] = torch.rad2deg(pendulum_error.abs()).cpu().numpy()
        voltage[k] = motor_term.processed_actions[:, 0].cpu().numpy()
        current[k] = motor_term.motor.current.cpu().numpy()
        gains[k] = motor_term.gains.cpu().numpy()
failed = failed.cpu().numpy()
ok = ~failed

# Metrics, on the envs that never ended early. A time to reach a band is the time after which the signal stays inside it;
# NaN if it is still outside at the end of the phase.
up = pendulum_error_deg < UPRIGHT_DEG
first_step = time_s >= COMMANDS_DEG[1][0]
swing_up_s = np.full(args.num_envs, np.nan)
for e in range(args.num_envs):
    down = np.where(~up[~first_step, e])[0]
    if len(down) == 0:
        swing_up_s[e] = 0.0
    elif down[-1] + 1 < np.sum(~first_step):
        swing_up_s[e] = time_s[down[-1] + 1]
balanced = ok & up[first_step].all(axis=0)
segments = []
for (start, value), end in zip(COMMANDS_DEG[1:], [c[0] for c in COMMANDS_DEG[2:]] + [DURATION_S]):
    in_segment = (time_s >= start) & (time_s < end)
    segment_time = time_s[in_segment]
    error = arm_deg[in_segment][:, ok] - value
    settle = np.full(error.shape[1], np.nan)
    for e in range(error.shape[1]):
        outside = np.where(np.abs(error[:, e]) > SETTLED_DEG)[0]
        if len(outside) == 0:
            settle[e] = 0.0
        elif outside[-1] + 1 < len(segment_time):
            settle[e] = segment_time[outside[-1] + 1] - start
    previous = command_deg[np.argmax(in_segment) - 1]
    overshoot = np.max(error * np.sign(value - previous), axis=0)
    steady = np.abs(error[segment_time >= end - 1.0]).mean(axis=0)
    segments.append({"command_deg": value, "from_deg": previous, "settled_pct": float(100.0 * np.isfinite(settle).mean()),
                     "settling_s_median": float(np.nanmedian(settle)), "settling_s_p95": float(np.nanpercentile(settle, 95)),
                     "overshoot_deg_median": float(np.median(overshoot)), "overshoot_deg_p95": float(np.percentile(overshoot, 95)),
                     "steady_error_deg_median": float(np.median(steady)), "steady_error_deg_p95": float(np.percentile(steady, 95))})
balance_phase = first_step
metrics = {
    "checkpoint": args.checkpoint, "num_envs": args.num_envs,
    "ended_early": int(failed.sum()),
    "balanced_whole_time_pct": float(100.0 * balanced.mean()),
    "swung_up_pct": float(100.0 * np.isfinite(swing_up_s[ok]).mean()),
    "swing_up_s_median": float(np.nanmedian(swing_up_s[ok])), "swing_up_s_p95": float(np.nanpercentile(swing_up_s[ok], 95)),
    "arm_steps": segments,
    "current_rms_a_balance": float(np.sqrt(np.mean(current[balance_phase][:, ok] ** 2))),
    "current_rms_a_swing_up": float(np.sqrt(np.mean(current[~balance_phase][:, ok] ** 2))),
    "current_peak_a": float(np.abs(current[:, ok]).max()),
    "voltage_rms_v_balance": float(np.sqrt(np.mean(voltage[balance_phase][:, ok] ** 2))),
    "voltage_peak_v": float(np.abs(voltage[:, ok]).max()),
}
for i, name in enumerate(["arm_kp", "arm_ki", "arm_kd", "pendulum_kp", "pendulum_ki", "pendulum_kd"]):
    for phase, mask in (("swing_up", ~balance_phase), ("balance", balance_phase)):
        g = gains[mask][:, ok, i]
        metrics[f"{name}_{phase}"] = {"median": float(np.median(g)), "p5": float(np.percentile(g, 5)),
                                      "p95": float(np.percentile(g, 95))}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "response_pid_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# Chart: four panels on one time axis, median and the 5-95 % band of the envs that did not end early.
SERIES = "#2a78d6"          # categorical slot 1 of the reference palette
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlelocation": "left", "axes.titlesize": 10, "axes.titlecolor": INK})
fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.4), sharex=True, facecolor=SURFACE)
panels = [
    (arm_deg, "Arm angle θ₁ [deg]", "Arm follows the command"),
    (pendulum_error_deg, "Pendulum from upright [deg]", "Pendulum: swing-up from hanging (180°), then balanced"),
    (voltage, "Voltage [V]", f"Motor voltage the inner PID writes (limit ±{C.QUBE_V_MAX:.0f} V)"),
    (current, "Current [A]", "Motor current (amplifier: 2 A peak, 0.5 A continuous)"),
]
for ax, (data, ylabel, title) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for start, _ in COMMANDS_DEG[1:]:
        ax.axvline(start, color=GRID, linewidth=1.0, zorder=0)
    ax.fill_between(time_s, np.percentile(data[:, ok], 5, axis=1), np.percentile(data[:, ok], 95, axis=1),
                    color=SERIES, alpha=0.18, linewidth=0, label="5–95 % of envs")
    ax.plot(time_s, np.median(data[:, ok], axis=1), color=SERIES, linewidth=1.6, label="median")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
axes[0].step(time_s, command_deg, where="post", color=INK, linestyle="--", linewidth=1.0, label="command")
axes[0].legend(loc="upper right", frameon=False, ncol=3)
axes[1].axhline(UPRIGHT_DEG, color=MUTED, linestyle=":", linewidth=1.0)
axes[1].set_ylim(-5, 185)
for limit in (-C.QUBE_V_MAX, C.QUBE_V_MAX):
    axes[2].axhline(limit, color=MUTED, linestyle=":", linewidth=1.0)
for limit in (-2.0, 2.0):
    axes[3].axhline(limit, color=MUTED, linestyle=":", linewidth=1.0)
axes[3].set_xlabel("Time [s]")
axes[3].set_xlim(0, DURATION_S)
fig.suptitle(f"Policy response, {int(ok.sum())} of {args.num_envs} simulated robots (motor constants drawn per robot)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(out / "response_pid.png", dpi=150, facecolor=SURFACE)
print(f"wrote {out / 'response_pid.png'}")

env.close()
app.close()
