# G1 Wave Demo v6 - FINAL - Prominent Wave Animation
#
# Kullanım:
#   cd C:\IsaacLab
#   .\isaaclab.bat -p source\isaaclab_tasks\isaaclab_tasks\direct\isaac_g1_vlm_rl\play_wave.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 16 --load_run 2025-12-24_13-11-23 --wave_hand right

from __future__ import annotations

import argparse
import math
import torch
import os
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained G1 policy with wave animation")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--wave_hand", type=str, default="right", choices=["left", "right", "both", "none"])
parser.add_argument("--wave_freq", type=float, default=2.5)  # Biraz daha hızlı

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.video = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


def find_checkpoint(run_dir: str, checkpoint: str = None) -> str:
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    model_files = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in: {run_dir}")
    if checkpoint:
        if checkpoint in model_files:
            return os.path.join(run_dir, checkpoint)
        raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
    model_files.sort(key=lambda x: int(re.search(r'model_(\d+)', x).group(1)))
    return os.path.join(run_dir, model_files[-1])


def main():
    print("\n" + "=" * 70)
    print("  G1 WALK + WAVE DEMO v6 - Prominent Wave Animation")
    print("=" * 70 + "\n")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Get joint names
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_names = robot.joint_names

    # G1 Joint Mapping (from debug output):
    # [ 5] left_shoulder_pitch_joint
    # [ 6] right_shoulder_pitch_joint
    # [ 9] left_shoulder_roll_joint
    # [10] right_shoulder_roll_joint
    # [13] left_shoulder_yaw_joint
    # [14] right_shoulder_yaw_joint
    # [17] left_elbow_pitch_joint  <-- Dirsek bükme
    # [18] right_elbow_pitch_joint
    # [21] left_elbow_roll_joint   <-- Dirsek döndürme
    # [22] right_elbow_roll_joint

    ARM_JOINTS = {
        "left_shoulder_pitch": 5,
        "right_shoulder_pitch": 6,
        "left_shoulder_roll": 9,
        "right_shoulder_roll": 10,
        "left_shoulder_yaw": 13,
        "right_shoulder_yaw": 14,
        "left_elbow_pitch": 17,
        "right_elbow_pitch": 18,
        "left_elbow_roll": 21,
        "right_elbow_roll": 22,
    }

    print("Using hardcoded arm indices:")
    for name, idx in sorted(ARM_JOINTS.items(), key=lambda x: x[1]):
        print(f"  [{idx:2d}] {name}")
    print()

    # Checkpoint
    run_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
    resume_path = find_checkpoint(run_dir, args_cli.checkpoint)
    print(f"[INFO] Loading: {resume_path}")

    # RSL-RL wrapper
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env)

    # Agent config
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    agent_cfg = G1FlatPPORunnerCfg()

    # Runner
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    print(f"[INFO] Num envs: {args_cli.num_envs}")
    print(f"[INFO] Wave hand: {args_cli.wave_hand}")
    print(f"[INFO] Wave freq: {args_cli.wave_freq} Hz")

    # Get observations
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    sim_time = 0.0
    step_count = 0
    dt = env.unwrapped.step_dt

    print("\n" + "-" * 50)
    print(" Running... Ctrl+C to exit")
    print(" Robots will WALK and WAVE! 👋")
    print("-" * 50 + "\n")

    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(obs)

            # === WAVE ANIMATION ===
            if args_cli.wave_hand != "none":
                # Sinusoidal wave - daha belirgin
                wave = math.sin(2.0 * math.pi * args_cli.wave_freq * sim_time)

                # === SAĞ KOL ===
                if args_cli.wave_hand in ["right", "both"]:
                    # Kolu YUKARI kaldır (shoulder_pitch negatif = yukarı)
                    actions[:, ARM_JOINTS["right_shoulder_pitch"]] = -2.5  # Daha yukarı!

                    # Kolu YANA aç (shoulder_roll)
                    actions[:, ARM_JOINTS["right_shoulder_roll"]] = -0.8 + wave * 0.4

                    # Kolu DÖNDÜR (shoulder_yaw) - sallama hareketi
                    actions[:, ARM_JOINTS["right_shoulder_yaw"]] = wave * 0.5

                    # Dirseği BÜK (elbow_pitch) - el yukarıda
                    actions[:, ARM_JOINTS["right_elbow_pitch"]] = -1.5 + wave * 0.3

                    # Dirsek roll
                    actions[:, ARM_JOINTS["right_elbow_roll"]] = wave * 0.2

                # === SOL KOL ===
                if args_cli.wave_hand in ["left", "both"]:
                    actions[:, ARM_JOINTS["left_shoulder_pitch"]] = -2.5
                    actions[:, ARM_JOINTS["left_shoulder_roll"]] = 0.8 + wave * 0.4
                    actions[:, ARM_JOINTS["left_shoulder_yaw"]] = -wave * 0.5
                    actions[:, ARM_JOINTS["left_elbow_pitch"]] = -1.5 - wave * 0.3
                    actions[:, ARM_JOINTS["left_elbow_roll"]] = -wave * 0.2

            # Step
            step_result = env.step(actions)
            if isinstance(step_result, tuple):
                obs = step_result[0]
            else:
                obs = step_result
            if isinstance(obs, tuple):
                obs = obs[0]

            sim_time += dt
            step_count += 1

            if step_count % 500 == 0:
                print(f"[Step {step_count}] Time: {sim_time:.2f}s")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()