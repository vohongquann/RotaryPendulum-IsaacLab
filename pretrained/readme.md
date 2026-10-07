# Pretrained policies

Trained weights shipped with the repo, so a fresh clone can play, evaluate and record without training first.

## rotary_pendulum_pid

`Isaac-Rotary-Pendulum-PID`: the policy writes the 6 gains of the cascade PID (guide pages 7 and 8). PPO (rsl_rl 5.5.1),
4096 envs, 1500 iterations, 64 min on an RTX 3060; copied from the run `logs/rsl_rl/rotary_pendulum_pid/2026-10-07_08-36-09`.
Simulation results: [README](../README.md#results-simulation).

| File | What it is |
|---|---|
| `model_1499.pt` | rsl_rl checkpoint (actor, critic, optimizer, observation normalisers): what `--checkpoint` takes |
| `params/env.yaml`, `params/agent.yaml` | the env and PPO settings of the run |
| `exported/policy.pt`, `exported/policy.onnx` (+ `.onnx.data`) | the actor alone with its observation normaliser (TorchScript / ONNX), written by `isaaclab play`: 12 observations in, 6 actions in [−1, 1] out (multiply by the gain ranges of `mdp/actions.py`) |

```bash
isaaclab play --rl_library rsl_rl --task Isaac-Rotary-Pendulum-PID --num_envs 4 --viz kit \
    --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt
python scripts/evaluate_policy.py --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt
python scripts/record_demo.py --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt --gif
```

`isaaclab play` writes `exported/` (and `videos/` with `--video`) next to the checkpoint, so it rewrites the files here. The
network comes out the same, but the new files carry debug information with the file paths of your machine (the shipped ones
had it stripped); restore them with `git checkout pretrained/` before committing.

The weights match the code of this commit. A change of the observation, the action, the motor model or the rates makes them
stale: retrain and replace them.
