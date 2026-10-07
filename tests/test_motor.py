"""QUBE-Servo 2 motor model: PWM -> voltage -> current -> torque, and the encoder."""
import math

import torch

from Rotary_Pendulum_RL.rotary import rotary_cfg as C
from Rotary_Pendulum_RL.rotary.mdp.motor import QubeServo2Motor, duty_to_pwm, pwm_to_voltage, quantize_angle
from Rotary_Pendulum_RL.rotary.mdp.sensors import JointEncoders


def motor(**kwargs):
    return QubeServo2Motor(1, "cpu", randomize=False, **kwargs)


def test_pwm_to_voltage_is_linear_and_signed():
    assert pwm_to_voltage(torch.tensor([C.PWM_MAX])).item() == C.QUBE_V_MAX
    assert pwm_to_voltage(torch.tensor([-C.PWM_MAX / 2])).item() == -C.QUBE_V_MAX / 2
    assert duty_to_pwm(torch.tensor([2.0, -2.0, 0.5])).tolist() == [C.PWM_MAX, -C.PWM_MAX, round(0.5 * C.PWM_MAX)]


def test_stall_torque_is_kt_v_over_r():
    m, pwm = motor(), torch.tensor([float(C.PWM_MAX)])
    for _ in range(50):                                   # 100 ms, 700 electrical time constants
        torque = m.step(pwm, torch.zeros(1), 0.002)
    assert math.isclose(torque.item(), C.QUBE_KT * C.QUBE_V_MAX / C.QUBE_RM_OHM, rel_tol=1e-6)    # 0.060 N m at 12 V


def test_no_load_speed_has_no_torque():
    """At w = v / k_e the back-EMF cancels the voltage: no current, no torque."""
    m, pwm = motor(), torch.tensor([float(C.PWM_MAX)])
    omega = torch.tensor([C.QUBE_V_MAX / C.QUBE_KE])      # 286 rad/s at 12 V
    for _ in range(50):
        torque = m.step(pwm, omega, 0.002)
    assert abs(torque.item()) < 1e-6
    assert math.isclose(C.QUBE_V_NOMINAL / C.QUBE_KE * 60 / (2 * math.pi), 4050, rel_tol=0.03)   # datasheet: 4050 RPM at 18 V


def test_current_follows_the_electrical_time_constant():
    """One time constant L / R after a voltage step the current is 63 % of its final value."""
    m, pwm = motor(), torch.tensor([float(C.PWM_MAX)])
    tau_e = C.QUBE_LM_H / C.QUBE_RM_OHM
    m.step(pwm, torch.zeros(1), tau_e)
    assert math.isclose(m.current.item(), (1 - math.exp(-1)) * C.QUBE_V_MAX / C.QUBE_RM_OHM, rel_tol=1e-6)


def test_step_is_stable_for_a_step_longer_than_the_time_constant():
    m, pwm = motor(), torch.tensor([float(C.PWM_MAX)])
    for _ in range(20):
        m.step(pwm, torch.zeros(1), 0.01)                 # 70 time constants per step: explicit Euler would blow up
    assert 0.0 < m.current.item() <= C.QUBE_V_MAX / C.QUBE_RM_OHM + 1e-6


def test_randomisation_stays_in_range_and_reset_clears_the_winding():
    m = QubeServo2Motor(256, "cpu")
    assert m.k_scale.min() >= 0.9 and m.k_scale.max() <= 1.1 and m.r_scale.min() >= 0.9 and m.r_scale.max() <= 1.2
    m.step(torch.full((256,), float(C.PWM_MAX)), torch.zeros(256), 0.002)
    m.reset(torch.arange(10))
    assert m.current[:10].abs().max() == 0 and m.current[10:].abs().min() > 0


def test_encoder_quantises_to_counts():
    step = 2 * math.pi / C.QUBE_ENCODER_CPR
    q = quantize_angle(torch.tensor([0.4 * step, 0.6 * step, -1.4 * step]))
    assert torch.allclose(q, torch.tensor([0.0, step, -step]))


def test_velocity_estimate_converges_to_a_constant_speed():
    dt = 1.0 / C.ENCODER_HZ
    enc = JointEncoders(1, 1, dt, "cpu")
    angle = torch.zeros(1, 1)
    for _ in range(100):
        angle += 5.0 * dt                                  # 5 rad/s
        enc.update(angle)
    assert abs(enc.velocity.item() - 5.0) < 1.0            # the quantisation (1.5 rad/s per count at 500 Hz) shows up as noise
