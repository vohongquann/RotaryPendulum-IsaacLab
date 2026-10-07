"""Configuration of the rotary (Furuta) pendulum: the QUBE-Servo 2, ``assets/data/qube_servo2/qube_servo2.urdf``.

The URDF is built from the CAD of the robot (``scripts/build_qube_urdf.py``); Isaac Lab turns it into a USD at the first start.
Bodies ``base`` (fixed), ``arm`` and ``pendulum``, revolute joints:

    Revolute_1   base -> arm        axis z, driven by the DC motor          (theta1 = 0: the arm at its start position)
    Revolute_2   arm -> pendulum    axis along the arm, passive             (theta2 = 0: hanging down, +-pi: upright)

The motor is the one of the Quanser QUBE-Servo 2 (``mdp/motor.py`` simulates it from the PWM up); every number that
comes from a source is listed here with the source. A number marked ASSUMPTION is not published: set it from your own
hardware before trusting a sim2real result.
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from Rotary_Pendulum_RL.assets import ROTARY_PENDULUM_RL_ASSETS_DIR

_URDF_PATH = ROTARY_PENDULUM_RL_ASSETS_DIR / "qube_servo2" / "qube_servo2.urdf"

ARM_JOINT = "Revolute_1"
PENDULUM_JOINT = "Revolute_2"

# ── Loop rates ──────────────────────────────────────────────────────────────────────────
"""Physics step 1 / PHYSICS_HZ; the motor current is integrated at this rate."""
PHYSICS_HZ = 500.0
"""Rate at which the microcontroller reads the encoders (angle counts, filtered velocity) every 1 / ENCODER_HZ; the inner
(pendulum) PID runs on every read (``mdp/actions.py``)."""
ENCODER_HZ = 500.0
"""Rate of the policy, which writes the 6 PID gains: every 1 / POLICY_HZ. Must divide PHYSICS_HZ."""
POLICY_HZ = 100.0

# ── QUBE-Servo 2 motor ──────────────────────────────────────────────────────────────────
# The motor is the Allied Motion CL40 (7 W) coreless DC motor, 18 V winding, model 9904 120 16705 (Allied Motion CL Series
# datasheet, 2017: R 8.4 ohm, L 1.16 mH, k_t 42 mNm/A, J 4E-6 kg m^2, rated 540 mA, no-load 4050 RPM, starting current 2.1 A,
# starting torque 89 mNm, mechanical time constant 19 ms). Same numbers in the Quanser QUBE-Servo 2 workbook, in the sim-to-real
# Furuta work of Data-Science-in-Mechanical-Engineering/vision-based-furuta-pendulum (``gym_brt/quanser/qube_simulator.py``) and in
# the NI QUBE-Servo 2 datasheet (nominal 18 V, 0.54 A, 4050 RPM no load; 512 counts/rev per channel, 4X quadrature).
"""Terminal resistance [ohm]."""
QUBE_RM_OHM = 8.4
"""Rotor inductance [H]."""
QUBE_LM_H = 1.16e-3
"""Torque constant [N m / A]. The identified value of the vision-based-furuta-pendulum repo is 0.046 (+9 %)."""
QUBE_KT = 0.042
"""Back-EMF constant [V / (rad/s)]; equal to the torque constant in SI units."""
QUBE_KE = 0.042
"""Rotor inertia [kg m^2]; added to the arm joint as armature."""
QUBE_JM = 4.0e-6
"""Nominal voltage of the winding [V] (Allied Motion 16705, NI datasheet): the voltage at which the rated 540 mA, 4050 RPM and
89 mNm starting torque are given. The robot is not driven that high, see ``QUBE_V_MAX``."""
QUBE_V_NOMINAL = 18.0
"""Largest voltage the inner PID may write [V], the full scale of the PWM (100 % duty): +-12 V, the action limit of the SAC policies
that run on the real QUBE-Servo 2 in github.com/vohongquann/rotary-pendulum-matlab (``env_swingup.m``, ``env_balance.m``). It sits
between the +-10 V recommended and the +-15 V maximum output range of the amplifier (``params_system.m``; Quanser QUARC
documentation, QUBE-Servo user manual), and keeps the stall current at 12 / 8.4 = 1.43 A, under the 2 A peak of the amplifier."""
QUBE_V_MAX = 12.0
"""Encoder counts per revolution, quadrature decoding (512 lines x 4); the angle is quantised to 2 pi / this."""
QUBE_ENCODER_CPR = 2048

# ── Electronics and firmware (ASSUMPTION: not published, match them to your microcontroller) ────────
"""Full-scale PWM command (16-bit timer): duty = PWM / PWM_MAX, sign = direction pin."""
PWM_MAX = 65535
"""Cut-off [rad/s] of the first-order high-pass 50 s / (s + 50) that estimates the velocity of each encoder: the filter of the
Quanser QUBE-Servo 2 Simulink files (Rotary Pendulum Interface block, as taught in the Lorraine lab sheet), 50 rad/s = 7.96 Hz."""
VELOCITY_FILTER_RAD_S = 50.0

# ── Mechanics ───────────────────────────────────────────────────────────────────────────
# Masses, inertias and geometry come from the URDF (``assets/data/qube_servo2/qube_servo2.urdf``, built from the CAD by
# ``scripts/build_qube_urdf.py``): arm 0.095 kg, pendulum link 0.024 kg (QUBE-Servo 2 workbook), arm axis to pendulum shaft
# 85.9 mm (workbook 85 mm), pendulum 129 mm.
"""Equivalent viscous damping of the arm joint [N m s/rad]: the identified value of the vision-based-furuta-pendulum repo."""
ARM_DAMPING = 2.75e-4
"""Equivalent viscous damping of the pendulum joint [N m s/rad]: same source."""
PENDULUM_DAMPING = 5.05e-5
"""Height of the motor housing [m]: the base of the URDF hangs 0.117 m below its top face, so the robot is raised by this much to
stand on the ground plane."""
BASE_HEIGHT_M = 0.117

ROTARY_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_URDF_PATH),
        usd_dir=str(_URDF_PATH.parent / "usd"),
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none", gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    # Pendulum hanging down; the arm at 0.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BASE_HEIGHT_M),
        joint_pos={ARM_JOINT: 0.0, PENDULUM_JOINT: 0.0},
    ),
    actuators={
        # The motor torque is not a joint drive: ``mdp/actions.py`` computes it from the PWM and writes it as an effort,
        # so the drive is off (stiffness 0, damping 0) and only the rotor armature and the viscous friction remain.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[ARM_JOINT],
            stiffness=0.0,
            damping=0.0,
            armature=QUBE_JM,
            viscous_friction=ARM_DAMPING,
        ),
        "pendulum": ImplicitActuatorCfg(
            joint_names_expr=[PENDULUM_JOINT],
            stiffness=0.0,
            damping=0.0,
            viscous_friction=PENDULUM_DAMPING,
        ),
    },
)
