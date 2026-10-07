# Media

| File | What it shows | How it is made |
|---|---|---|
| `motor_torque.png` | torque against voltage per arm speed; torque against speed per duty; current after a full-duty step (page 2) | `python src/Rotary_Pendulum_RL/rotary/mdp/motor.py` |
| `demo_pid.mp4`, `demo_pid.gif` | one robot, swing-up then the arm command 0, -45, -90, -45, 0, 45, 90, 45, 0 deg (3 s each); green sphere = target pendulum tip; θ1*, θ1, θ2 in the bottom-left corner; 1920x1080, RTX path tracing with sun shadows (page 7) | `python scripts/record_demo.py --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt --gif` |
| `training_pid.png` | training curves of the PID run: return, episode length, reward terms, termination causes (README) | `python scripts/plot_training.py logs/rsl_rl/rotary_pendulum_pid/<run> --out docs/media/training_pid.png` |
| `response_pid.png`, `response_pid_metrics.json` | the PID policy on 256 simulated robots (seed 42): swing-up, arm steps, voltage, current, the gains it chose (README, page 7) | `python scripts/evaluate_policy.py --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt` |
