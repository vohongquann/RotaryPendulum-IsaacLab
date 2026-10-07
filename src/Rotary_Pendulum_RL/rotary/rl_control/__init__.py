r"""RL task of the rotary (Furuta) pendulum with the QUBE-Servo 2 motor.

    Isaac-Rotary-Pendulum-PID   swing up, balance and hold the arm at a commanded angle through two PIDs in cascade: the
                                policy writes their gains; the outer PID (arm, 250 Hz) sets the pendulum angle, the inner
                                PID (pendulum, 500 Hz) the voltage                                (10 s episode, 100 Hz)

The voltage becomes a PWM, the motor model turns the PWM into torque (500 Hz physics). See ``mdp/actions.py`` and
``mdp/motor.py``. PPO settings: ``agents/rsl_rl_ppo_cfg.py``.

Commands, from the repo root. ``--video_length`` is in policy steps (100 per second of video).

    isaaclab train --rl_library rsl_rl --task Isaac-Rotary-Pendulum-PID --num_envs 4096
    isaaclab train --rl_library rsl_rl --task Isaac-Rotary-Pendulum-PID --num_envs 4096 \
        --checkpoint logs/rsl_rl/rotary_pendulum_pid/<run>/model_<n>.pt                     # resumes a run
    isaaclab play  --rl_library rsl_rl --task Isaac-Rotary-Pendulum-PID --num_envs 4 --viz kit \
        --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt
"""
import gymnasium as gym

gym.register(
    id="Isaac-Rotary-Pendulum-PID",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pid_env_cfg:PidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PidPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
