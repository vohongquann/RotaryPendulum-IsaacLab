"""PPO (rsl_rl 5.x) settings of the ``PID`` task.

The actor sees the ``policy`` group (what the encoders measure), the critic the ``critic`` group (true state).
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from Rotary_Pendulum_RL.rotary import rotary_cfg as C


@configclass
class PidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """The policy writes the six gains of the cascade scaled to [-1, 1] (``mdp.CascadePidAction``): action noise 0.5 at the
    start, a quarter of the range. 100 Hz policy: gamma 0.995 looks about 2 s ahead, the time of a swing-up; rollouts of
    1 s. Entropy 0.001: a larger one keeps the action noise high once the pendulum is up, too loud to hold the arm."""

    num_steps_per_env = round(1.0 * C.POLICY_HZ)
    max_iterations = 1500
    save_interval = 100
    experiment_name = "rotary_pendulum_pid"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[64, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[64, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
