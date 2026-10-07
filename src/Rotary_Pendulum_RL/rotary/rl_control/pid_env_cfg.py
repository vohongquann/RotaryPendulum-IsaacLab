"""Swing up, balance and hold the arm at a commanded angle through two PIDs in cascade: ``Isaac-Rotary-Pendulum-PID``.

    scene:        ground, light, the rotary pendulum (``rotary_cfg.ROTARY_CFG``), hanging down
    action:       6 gains (mdp.CascadePidAction), 100 Hz: the outer PID ``arm_pid`` (250 Hz) turns the arm angle error
                  into a pendulum angle setpoint theta2*, the inner PID ``pendulum_pid`` (500 Hz) turns the pendulum error
                  to theta2* into the motor voltage -> PWM -> QUBE-Servo 2 motor -> torque; both run on the measured
                  angles and filtered velocities, like the firmware would (guide/07_cascade_pid.md)
                      command theta1*, arm angle -> RL gains + PID (outer) -> theta2* -> RL gains + PID (inner) -> volts
    observation:  arm angle error to the command, the commanded arm angle, sin and cos of the pendulum, both velocities,
                  measured like the firmware does (encoder counts, filtered velocity), and the last 6 gains (12); the
                  critic gets the same from the true state
    reward:       6 terms: pendulum up, arm on the command (wide and narrow), velocity and voltage penalties

Physics runs at ``PHYSICS_HZ`` (500 Hz), the encoders at ``ENCODER_HZ`` (500 Hz), the policy at ``POLICY_HZ`` (100 Hz).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from Rotary_Pendulum_RL.rotary import mdp
from Rotary_Pendulum_RL.rotary import rotary_cfg as C

ARM = SceneEntityCfg("robot", joint_names=[C.ARM_JOINT])
PENDULUM = SceneEntityCfg("robot", joint_names=[C.PENDULUM_JOINT])
HEADING_COMMAND = "heading_cmd"


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    )
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    robot = C.ROTARY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    motor = mdp.CascadePidActionCfg(asset_name="robot", command_name=HEADING_COMMAND)  # the observation terms look the term up by this name


@configclass
class PolicyCfg(ObsGroup):
    """Observation (12): arm angle error to the command, the commanded arm angle, sin and cos of the pendulum, both velocities
    (the encoders, as the firmware sees them), the last 6 gains. The state of the Quanser LQR (arm angle, pendulum angle,
    both velocities) with the arm angle taken from the command."""

    arm_angle_error = ObsTerm(func=mdp.arm_angle_error, params={"command_name": HEADING_COMMAND})
    arm_command = ObsTerm(func=isaac_mdp.generated_commands, params={"command_name": HEADING_COMMAND})
    pendulum_angle_sin = ObsTerm(func=mdp.pendulum_angle_sin)
    pendulum_angle_cos = ObsTerm(func=mdp.pendulum_angle_cos)
    arm_velocity = ObsTerm(func=mdp.arm_velocity)
    pendulum_velocity = ObsTerm(func=mdp.pendulum_velocity)
    last_action = ObsTerm(func=isaac_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticCfg(PolicyCfg):
    """Same terms from the true state (privileged)."""

    def __post_init__(self):
        super().__post_init__()
        for term in (
            self.arm_angle_error,
            self.pendulum_angle_sin,
            self.pendulum_angle_cos,
            self.arm_velocity,
            self.pendulum_velocity,
        ):
            term.params["measured"] = False


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    heading_cmd = mdp.PivotHeadingCommandCfg(
        asset_name="robot", resampling_time_range=(3.0, 6.0), heading_range=math.pi / 2
    )


@configclass
class RandomStartEventCfg:
    """Random start (+-0.2 rad, +-0.5 rad/s) for exploration."""

    reset_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "position_range": (-0.2, 0.2), "velocity_range": (-0.5, 0.5)},
    )


@configclass
class RewardsCfg:
    """Six terms, each logged on its own (``Episode_Reward/<name>``). The goals pay in (0, 1], the penalties are negative.
    The episode ends if the arm turns more than half a turn; that loses the reward to come.

    The goal and penalty scales are those of a single quadratic cost (Menzenbach 2019, MathWorks QUBE-Servo 2 example),
    exp(-c (e' Q e + R V^2)) with Q = diag(1, 1, 0.02, 0.005), R = 0.003, c = 0.216 (reward 1e-4 at the worst case: errors
    of pi, velocities of 30 rad/s, 12 V): std = 1 / sqrt(c) = 2.15 rad, penalty weights c * Q and c * R."""

    # Goals: the pendulum up (0.12 hanging) and the arm on the command.
    pendulum_upright = RewTerm(func=mdp.pendulum_upright_exp, weight=1.0, params={"asset_cfg": PENDULUM, "std": 2.15})
    arm_angle = RewTerm(
        func=mdp.arm_angle_exp, weight=1.0, params={"command_name": HEADING_COMMAND, "asset_cfg": ARM, "std": 2.15}
    )
    # With the wide term alone the arm stayed 5 deg off on average (7.5 deg for 95 %): near the target it is flat. This
    # narrow term (std 0.1 rad = 6 deg) pays for the last degrees, only while the pendulum is up.
    arm_tracking = RewTerm(
        func=mdp.arm_angle_tracking_exp,
        weight=0.5,
        params={"command_name": HEADING_COMMAND, "pendulum_cfg": PENDULUM, "asset_cfg": ARM, "std": 0.1},
    )
    # Penalties: velocities squared (library term), and the motor voltage squared (the action is gains, not volts).
    arm_velocity = RewTerm(func=isaac_mdp.joint_vel_l2, weight=-0.0043, params={"asset_cfg": ARM})
    pendulum_velocity = RewTerm(func=isaac_mdp.joint_vel_l2, weight=-0.0011, params={"asset_cfg": PENDULUM})
    voltage = RewTerm(func=mdp.motor_voltage_l2, weight=-0.00065)


@configclass
class TerminationsCfg:
    """Timeout, or the arm more than half a turn from its start: it would wind up the cable and spin for ever."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    pivot_limit = DoneTerm(
        func=mdp.reset_when_pivot_exceeds_limit, params={"asset_cfg": ARM, "max_pivot_angle": math.pi}
    )


@configclass
class PidEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=1.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: RandomStartEventCfg = RandomStartEventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / C.PHYSICS_HZ
        self.decimation = int(round(C.PHYSICS_HZ / C.POLICY_HZ))
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (-0.8, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)
