"""Record a demo video of a trained policy: swing-up, then the arm command 0 -> -45 -> -90 -> -45 -> 0 -> 45 -> 90 -> 45 -> 0 deg.

    conda activate env_isaaclab
    python scripts/record_demo.py --checkpoint pretrained/rotary_pendulum_pid/model_1499.pt          # the shipped policy
    python scripts/record_demo.py --checkpoint logs/rsl_rl/rotary_pendulum_pid/<run>/model_1499.pt   # your own run
    python scripts/record_demo.py --checkpoint ... --hold 4 --eye -0.58 0 0.43 --lookat 0 0 0.17 --gif
    python scripts/record_demo.py --checkpoint ... --render rt --width 1280 --height 720      # faster, lower quality

One robot, started hanging. Each command is held ``--hold`` seconds; the first one (0 deg) includes the swing-up. A green
sphere marks where the tip of the upright pendulum should be for the current command. theta1* (command), theta1 (arm) and
theta2 (pendulum, 0 hanging, 180 deg upright) are written in the bottom-left corner of every frame. A low sun (distant light) casts the shadows; ``--render pt`` (default) renders
with RTX real-time path tracing at the highest settings (shadows, global illumination, reflections, ambient occlusion,
DLAA, DL denoiser), ``--render rt`` with RTX real-time ray tracing and the same switches. Writes <out>/demo_pid.mp4 (and demo_pid.gif with --gif); the raw clip of
the recorder (<out>/raw/) is deleted once the captions are written.
"""
import argparse
import math
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--checkpoint", required=True, help="model_N.pt of a run")
parser.add_argument("--commands", type=float, nargs="+", default=[0, -45, -90, -45, 0, 45, 90, 45, 0],
                    help="arm commands [deg], in order")
parser.add_argument("--hold", type=float, default=3.0, help="seconds each command is held")
parser.add_argument("--eye", type=float, nargs=3, default=[-0.58, 0.0, 0.43],
                    help="camera position [m]; the default is 20 cm further back than (-0.40, 0, 0.35) along the line of sight")
parser.add_argument("--lookat", type=float, nargs=3, default=[0.0, 0.0, 0.17], help="camera target [m]")
parser.add_argument("--width", type=int, default=1920, help="video width [px]")
parser.add_argument("--height", type=int, default=1080, help="video height [px]")
parser.add_argument("--render", choices=["pt", "rt"], default="pt",
                    help="pt: RTX real-time path tracing (best), rt: RTX real-time ray tracing (faster)")
parser.add_argument("--fps", type=int, default=50, help="video frame rate; the policy runs at 100 Hz, so 50 = every 2nd step")
parser.add_argument("--gif", action="store_true", help="also write a GIF (15 fps, 480 px wide) for the README")
parser.add_argument("--out", default="docs/media", help="folder of the video")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

import imageio.v2 as imageio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_visualizers.kit import KitVisualizerCfg  # noqa: E402

from Rotary_Pendulum_RL.rotary import rotary_cfg as C  # noqa: E402
from Rotary_Pendulum_RL.rotary.rl_control.agents.rsl_rl_ppo_cfg import PidPPORunnerCfg  # noqa: E402
from Rotary_Pendulum_RL.rotary.rl_control.pid_env_cfg import HEADING_COMMAND, PidEnvCfg  # noqa: E402

PENDULUM_LENGTH_M = 0.129    # shaft to tip, from the URDF (scripts/build_qube_urdf.py); only places the target marker
WARMUP_STEPS = 5             # steps run before the clip starts, so the renderer has warmed up
# The sun: a distant light shining along (0.5, -0.6, -1), from above, behind the camera and to its left, so the shadows of
# the arm and the pendulum fall on the motor housing and the ground behind it. The quaternion (x, y, z, w) turns the light's
# -z axis onto that direction (38 deg about (-0.77, -0.64, 0)).
SUN_ROTATION = (-0.25, -0.2084, 0.0, 0.9455)
SUN_INTENSITY = 1800.0
DOME_INTENSITY = 300.0       # the task's dome light (500) dimmed so the shadows show

out = Path(args.out)
raw_dir = out / "raw"
policy_hz = C.POLICY_HZ
frame_stride = max(1, round(policy_hz / args.fps))
steps = round(len(args.commands) * args.hold * policy_hz)

env_cfg = PidEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = len(args.commands) * args.hold + 2.0
env_cfg.commands.heading_cmd.resampling_time_range = (1.0e9, 1.0e9)       # the script sets the command
env_cfg.viewer.eye, env_cfg.viewer.lookat = tuple(args.eye), tuple(args.lookat)
env_cfg.scene.light.spawn.intensity = DOME_INTENSITY
env_cfg.scene.sun = AssetBaseCfg(
    prim_path="/World/Sun",
    spawn=sim_utils.DistantLightCfg(intensity=SUN_INTENSITY, angle=0.53, color=(1.0, 0.96, 0.9)),
    init_state=AssetBaseCfg.InitialStateCfg(rot=SUN_ROTATION),
)
env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(
    headless=True, eye=tuple(args.eye), lookat=tuple(args.lookat), window_width=args.width, window_height=args.height
)]
env_cfg.video_recorders = [VideoRecorderCfg(
    source="visualizer:kit", output_dir=str(raw_dir), output_filename_prefix="demo_pid",
    video_length=steps, step_offset=WARMUP_STEPS, frame_stride=frame_stride, fps=args.fps,
)]

def write_videos(log: np.ndarray, total: int):
    """Write the captions on the recorder's clip (it captured the steps WARMUP_STEPS, WARMUP_STEPS + stride, ... of
    ``log``) into <out>/demo_pid.mp4, and the GIF with --gif. Called inside ``launch_simulation``: closing the Kit app
    ends the process."""
    clips = sorted(raw_dir.glob("demo_pid*.mp4"))
    if not clips:
        raise RuntimeError(f"no clip was written to {raw_dir}")
    reader = imageio.get_reader(clips[-1])
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    mp4_path = out / "demo_pid.mp4"
    writer = imageio.get_writer(mp4_path, fps=args.fps, codec="libx264", quality=9, macro_block_size=1)
    font = None
    for i, frame in enumerate(reader):
        k = min(WARMUP_STEPS + i * frame_stride, total - 1)
        image = Image.fromarray(frame).convert("RGBA")
        size = max(14, image.height // 27)
        if font is None:
            font = ImageFont.truetype(font_path, size) if Path(font_path).exists() else ImageFont.load_default()
        command_deg, arm_deg, pendulum_deg = log[k]
        lines = [f"θ1* = {command_deg:+7.1f}°", f"θ1  = {arm_deg:+7.1f}°", f"θ2  = {pendulum_deg:7.1f}°"]
        # Bottom-left corner, drawn on the frame with a translucent backing so it stays readable on light and dark floor.
        pad, line_height = size // 2, round(size * 1.3)
        box_width = round(font.getlength(lines[0])) + 2 * pad
        box_height = line_height * len(lines) + 2 * pad
        left, top = pad * 2, image.height - box_height - pad * 2
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle([left, top, left + box_width, top + box_height], radius=pad, fill=(0, 0, 0, 140))
        for row, text in enumerate(lines):
            draw.text((left + pad, top + pad + row * line_height), text, fill=(255, 255, 255, 255), font=font)
        writer.append_data(np.asarray(Image.alpha_composite(image, overlay).convert("RGB")))
    writer.close()
    reader.close()
    shutil.rmtree(raw_dir)
    print(f"wrote {mp4_path}")

    if args.gif:
        gif_path = out / "demo_pid.gif"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4_path), "-vf",
                        "fps=15,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse",
                        "-loop", "0", str(gif_path)], check=True)
        print(f"wrote {gif_path}")


with launch_simulation(env_cfg, args):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    from isaaclab.app.settings_manager import get_settings_manager
    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

    # Highest quality: every lighting feature on, DLAA (anti-aliasing at full resolution), the DL denoiser. The
    # anti-aliasing call selects the real-time ray tracing mode, so path tracing is selected after it.
    apply_isaac_rtx_global_settings(IsaacRtxRendererGlobalSettingsCfg(
        enable_shadows=True, enable_direct_lighting=True, samples_per_pixel=8, enable_global_illumination=True,
        enable_reflections=True, enable_translucency=True, enable_ambient_occlusion=True, enable_dl_denoiser=True,
        antialiasing_mode="DLAA", max_bounces=8, enable_cached_raytracing=False,
    ))
    if args.render == "pt":
        get_settings_manager().set("/rtx/rendermode", "RealTimePathTracing")

    env = ManagerBasedRLEnv(env_cfg)
    agent_cfg = PidPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(wrapped, agent_cfg)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    robot = env.scene["robot"]
    arm_joint = robot.find_joints(C.ARM_JOINT)[0][0]
    pendulum_joint = robot.find_joints(C.PENDULUM_JOINT)[0][0]
    arm_body = robot.find_bodies("arm")[0][0]
    pendulum_body = robot.find_bodies("pendulum")[0][0]
    command = env.command_manager.get_term(HEADING_COMMAND).command
    target_marker = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/ArmTarget",
        markers={"target": sim_utils.SphereCfg(
            radius=0.008, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)))},
    ))

    def show_target(target_rad: float):
        """Put the marker where the pendulum tip is when the arm is on ``target_rad`` and the pendulum is upright: the
        current shaft position turned about the motor axis by the arm error, raised by the pendulum length."""
        body_pos = robot.data.body_link_pos_w.torch[0]
        hub, shaft = body_pos[arm_body], body_pos[pendulum_body]
        turn = target_rad - robot.data.joint_pos.torch[0, arm_joint].item()
        dx, dy = (shaft[0] - hub[0]).item(), (shaft[1] - hub[1]).item()
        tip = [hub[0].item() + dx * math.cos(turn) - dy * math.sin(turn),
               hub[1].item() + dx * math.sin(turn) + dy * math.cos(turn),
               shaft[2].item() + PENDULUM_LENGTH_M]
        target_marker.visualize(translations=torch.tensor([tip], device=env.device))

    # Per-step log for the captions [deg]: command theta1*, arm theta1, pendulum theta2 in [0, 360) (180 = upright).
    total = WARMUP_STEPS + steps + 2        # a few more than the clip so the recorder closes it
    log = np.zeros((total, 3))
    command[:, 0] = math.radians(args.commands[0])
    obs = wrapped.get_observations()
    with torch.inference_mode():
        for k in range(total):
            segment = min(max(k - WARMUP_STEPS, 0) // round(args.hold * policy_hz), len(args.commands) - 1)
            target = math.radians(args.commands[segment])
            command[:, 0] = target
            show_target(target)
            obs, _, dones, _ = wrapped.step(policy(obs))
            if dones.any():
                print(f"[WARN] the episode ended at step {k} ({k / policy_hz:.2f} s): the video shows a reset")
            joint_pos = robot.data.joint_pos.torch[0]
            pendulum_angle = math.degrees(joint_pos[pendulum_joint].item()) % 360.0      # 0 hanging, 180 upright
            log[k] = (args.commands[segment], math.degrees(joint_pos[arm_joint].item()), pendulum_angle)
    env.close()
    write_videos(log, total)
