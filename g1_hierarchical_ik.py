# Copyright (c) 2025, VLM-RL G1 Project
# G1 Hierarchical Control: PPO Locomotion + DifferentialIK + Debug Visualization

"""
G1 Hierarchical Control with Hand Trajectory Visualization
===========================================================

Lower Body: PPO Locomotion Policy (trained 20K iterations)
Upper Body: DifferentialIKController with PhysX Jacobians
Visualization: Debug draw for hand trajectory

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p <path>\g1_hierarchical_ik.py --num_envs 4 --load_run 2025-12-27_00-29-54
"""

import argparse
import os
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from collections import deque

# ==== Isaac Lab App Launcher ====
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 Hierarchical Control: PPO + IK + Visualization")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--load_run", type=str, required=True, help="Locomotion policy run folder")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--ik_method", type=str, default="dls", choices=["dls", "pinv", "svd", "trans"])
parser.add_argument("--target_mode", type=str, default="circle",
                    choices=["circle", "static", "wave", "reach", "track_object"])
parser.add_argument("--arm", type=str, default="right", choices=["left", "right", "both"])
parser.add_argument("--draw_trajectory", action="store_true", default=True, help="Draw hand trajectory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==== Post-Launch Imports ====
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.envs import ManagerBasedRLEnv

# Debug Draw
try:
    from isaacsim.util.debug_draw import _debug_draw

    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    print("[Warning] Debug draw not available")

# Isaac Lab tasks
import isaaclab_tasks  # noqa: F401

##############################################################################
# G1 ARM CONFIGURATION
##############################################################################

G1_ARM_JOINTS = {
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
    ],
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_roll_joint",
    ],
}

G1_EE_BODIES = {
    "right": "right_palm_link",
    "left": "left_palm_link",
}

ARM_JOINT_INDICES = {
    "right": [6, 10, 14, 18, 22],
    "left": [5, 9, 13, 17, 21],
}


##############################################################################
# CUSTOM ACTOR NETWORK (Compatible with RSL-RL checkpoint)
##############################################################################

class CustomActorCritic(nn.Module):
    """
    Custom ActorCritic that can load RSL-RL checkpoints regardless of API version.
    """

    def __init__(
            self,
            num_obs: int,
            num_actions: int,
            actor_hidden_dims: List[int] = [512, 256, 128],
            critic_hidden_dims: List[int] = [512, 256, 128],
            activation: str = "elu",
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions

        # Activation function
        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ELU()

        # Build actor network
        actor_layers = []
        prev_dim = num_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(act_fn)
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic network
        critic_layers = []
        prev_dim = num_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(act_fn)
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action noise (log std)
        self.std = nn.Parameter(torch.ones(num_actions))

        print(f"[Policy] Created CustomActorCritic: obs={num_obs}, actions={num_actions}")
        print(f"[Policy] Actor: {actor_hidden_dims} -> {num_actions}")

    def forward(self, obs):
        return self.actor(obs)

    def act_inference(self, obs):
        """Get deterministic action for inference."""
        with torch.no_grad():
            return self.actor(obs)

    def load_rsl_rl_checkpoint(self, checkpoint_path: str, device: str):
        """Load weights from RSL-RL checkpoint."""
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "model_state_dict" not in data:
            raise ValueError("Checkpoint does not contain 'model_state_dict'")

        state_dict = data["model_state_dict"]

        # Map RSL-RL keys to our model
        new_state_dict = {}

        for key, value in state_dict.items():
            # Actor layers: actor.0.weight -> actor.0.weight
            if key.startswith("actor."):
                new_state_dict[key] = value
            # Critic layers: critic.0.weight -> critic.0.weight
            elif key.startswith("critic."):
                new_state_dict[key] = value
            # Std parameter
            elif key == "std":
                new_state_dict["std"] = value
            elif key == "log_std":
                # Convert log_std to std if needed
                new_state_dict["std"] = torch.exp(value)

        # Load with strict=False to handle any mismatches
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        if missing:
            print(f"[Policy] Missing keys: {missing}")
        if unexpected:
            print(f"[Policy] Unexpected keys: {unexpected}")

        return True


##############################################################################
# CHECKPOINT FINDER
##############################################################################

def find_checkpoint(run_dir: str, checkpoint_name: str = None) -> str:
    """Find checkpoint file in run directory."""
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if checkpoint_name:
        path = os.path.join(run_dir, checkpoint_name)
        if os.path.exists(path):
            return path

    checkpoints = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return os.path.join(run_dir, checkpoints[-1])


##############################################################################
# TRAJECTORY VISUALIZER
##############################################################################

class TrajectoryVisualizer:
    """Visualize hand trajectory using debug draw."""

    def __init__(self, num_envs: int, max_points: int = 200):
        self.num_envs = num_envs
        self.max_points = max_points
        self.enabled = DEBUG_DRAW_AVAILABLE

        # Store trajectory points for each environment
        self.trajectories = [deque(maxlen=max_points) for _ in range(num_envs)]
        self.target_trajectories = [deque(maxlen=max_points) for _ in range(num_envs)]

        # Colors (RGBA)
        self.hand_colors = [
            (1.0, 0.0, 0.0, 1.0),  # Red - env 0
            (0.0, 1.0, 0.0, 1.0),  # Green - env 1
            (0.0, 0.0, 1.0, 1.0),  # Blue - env 2
            (1.0, 1.0, 0.0, 1.0),  # Yellow - env 3
        ]
        self.target_colors = [
            (1.0, 0.5, 0.5, 0.5),  # Light red
            (0.5, 1.0, 0.5, 0.5),  # Light green
            (0.5, 0.5, 1.0, 0.5),  # Light blue
            (1.0, 1.0, 0.5, 0.5),  # Light yellow
        ]

        if self.enabled:
            self.draw = _debug_draw.acquire_debug_draw_interface()
            print("[Viz] Trajectory visualizer initialized")
        else:
            self.draw = None
            print("[Viz] Debug draw not available")

    def add_point(self, env_id: int, hand_pos: torch.Tensor, target_pos: torch.Tensor):
        """Add a point to the trajectory."""
        if not self.enabled:
            return

        # Convert to list for storage
        hand_pt = hand_pos.cpu().tolist() if torch.is_tensor(hand_pos) else hand_pos
        target_pt = target_pos.cpu().tolist() if torch.is_tensor(target_pos) else target_pos

        self.trajectories[env_id].append(hand_pt)
        self.target_trajectories[env_id].append(target_pt)

    def draw_all(self):
        """Draw all trajectories."""
        if not self.enabled or self.draw is None:
            return

        try:
            # Clear previous drawings
            self.draw.clear_lines()
            self.draw.clear_points()

            for env_id in range(self.num_envs):
                traj = list(self.trajectories[env_id])
                target_traj = list(self.target_trajectories[env_id])

                # Draw hand trajectory as connected lines
                if len(traj) > 1:
                    for i in range(len(traj) - 1):
                        p1 = traj[i]
                        p2 = traj[i + 1]
                        color = self.hand_colors[env_id % len(self.hand_colors)]
                        self.draw.draw_line(p1, color, p2, color)

                # Draw target trajectory (dashed effect with points)
                if len(target_traj) > 1:
                    for i in range(0, len(target_traj), 3):  # Every 3rd point
                        pt = target_traj[i]
                        color = self.target_colors[env_id % len(self.target_colors)]
                        self.draw.draw_point(pt, color, 5.0)

                # Draw current hand position (larger point)
                if traj:
                    current = traj[-1]
                    color = self.hand_colors[env_id % len(self.hand_colors)]
                    self.draw.draw_point(current, color, 15.0)

                # Draw current target (sphere-like with multiple points)
                if target_traj:
                    target = target_traj[-1]
                    color = (1.0, 1.0, 1.0, 1.0)  # White
                    self.draw.draw_point(target, color, 20.0)

        except Exception as e:
            print(f"[Viz] Draw error: {e}")

    def reset(self, env_ids: torch.Tensor = None):
        """Clear trajectories for reset environments."""
        if env_ids is None:
            for traj in self.trajectories:
                traj.clear()
            for traj in self.target_trajectories:
                traj.clear()
        else:
            for idx in env_ids.tolist():
                if idx < len(self.trajectories):
                    self.trajectories[idx].clear()
                    self.target_trajectories[idx].clear()


##############################################################################
# ARM IK CONTROLLER
##############################################################################

class G1ArmIKController:
    """DifferentialIK controller for G1 arm."""

    def __init__(
            self,
            num_envs: int,
            device: str,
            arm: str = "right",
            ik_method: str = "dls",
    ):
        self.num_envs = num_envs
        self.device = device
        self.arm = arm

        self.ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params={"lambda_val": 0.1} if ik_method == "dls" else {"k_val": 1.0},
        )

        self.controller = DifferentialIKController(
            self.ik_cfg,
            num_envs=num_envs,
            device=device,
        )

        self.ee_body_idx = None
        self.arm_joint_ids = None
        self.ee_jacobi_idx = None
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_quat = torch.zeros(num_envs, 4, device=device)
        self.target_quat[:, 0] = 1.0
        self.initialized = False

        print(f"[IK] Created G1ArmIKController for {arm} arm (method: {ik_method})")

    def initialize_from_robot(self, robot, scene):
        """Initialize from robot articulation."""
        try:
            ee_name = G1_EE_BODIES[self.arm]
            body_names = robot.body_names if hasattr(robot, 'body_names') else []

            if ee_name in body_names:
                self.ee_body_idx = body_names.index(ee_name)
                print(f"[IK] Found {ee_name} at body index {self.ee_body_idx}")
            else:
                self.ee_body_idx = 29  # Fallback
                print(f"[IK] Using fallback EE index {self.ee_body_idx}")

            joint_names = robot.joint_names if hasattr(robot, 'joint_names') else []
            self.arm_joint_ids = []
            for jname in G1_ARM_JOINTS[self.arm]:
                if jname in joint_names:
                    self.arm_joint_ids.append(joint_names.index(jname))

            if len(self.arm_joint_ids) < 5:
                self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]
                print(f"[IK] Using default arm joint indices: {self.arm_joint_ids}")
            else:
                print(f"[IK] Found arm joint indices: {self.arm_joint_ids}")

            self.ee_jacobi_idx = self.ee_body_idx - 1
            self.initialized = True

        except Exception as e:
            print(f"[IK] Init error: {e}")
            self.ee_body_idx = 29
            self.arm_joint_ids = ARM_JOINT_INDICES[self.arm]
            self.ee_jacobi_idx = 28
            self.initialized = True

    def set_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor = None):
        """Set target pose."""
        self.target_pos = target_pos.clone()

        if target_quat is None:
            target_quat = torch.zeros(self.num_envs, 4, device=self.device)
            target_quat[:, 0] = 1.0

        self.target_quat = target_quat.clone()
        pose_command = torch.cat([target_pos, target_quat], dim=-1)
        self.controller.set_command(pose_command)

    def compute(self, robot, jacobian: torch.Tensor = None) -> torch.Tensor:
        """Compute IK solution."""
        if not self.initialized:
            return torch.zeros(self.num_envs, len(self.arm_joint_ids), device=self.device)

        try:
            joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]

            ee_pose_w = robot.data.body_state_w[:, self.ee_body_idx, 0:7]
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            root_pose_w = robot.data.root_state_w[:, 0:7]
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
            )

            if jacobian is None:
                full_jacobian = robot.root_physx_view.get_jacobians()
                jacobian = full_jacobian[:, self.ee_jacobi_idx, :, :]
                jacobian = jacobian[:, :, self.arm_joint_ids]

            joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            return joint_pos_des

        except Exception as e:
            print(f"[IK] Compute error: {e}")
            return robot.data.joint_pos[:, self.arm_joint_ids]

    def get_ee_pos_world(self, robot) -> torch.Tensor:
        """Get current end-effector position in world frame."""
        return robot.data.body_state_w[:, self.ee_body_idx, 0:3]

    def reset(self, env_ids: torch.Tensor = None):
        """Reset controller."""
        if env_ids is None:
            self.target_pos.zero_()
            self.target_quat.zero_()
            self.target_quat[:, 0] = 1.0
        else:
            self.target_pos[env_ids] = 0.0
            self.target_quat[env_ids] = 0.0
            self.target_quat[env_ids, 0] = 1.0
        self.controller.reset(env_ids)


##############################################################################
# TARGET GENERATOR
##############################################################################

class TargetGenerator:
    """Generate target trajectories for end-effector."""

    def __init__(self, num_envs: int, device: str, mode: str = "circle", arm: str = "right"):
        self.num_envs = num_envs
        self.device = device
        self.mode = mode
        self.arm = arm

        y_offset = -0.25 if arm == "right" else 0.25
        self.base_position = torch.tensor([0.35, y_offset, 0.55], device=device)
        self.radius = 0.12
        self.freq = 0.4

    def get_target(self, time: float) -> torch.Tensor:
        """Get target position at given time."""
        pos = self.base_position.unsqueeze(0).expand(self.num_envs, -1).clone()

        if self.mode == "circle":
            angle = 2 * math.pi * self.freq * time
            pos[:, 0] += self.radius * math.cos(angle)
            pos[:, 2] += self.radius * math.sin(angle)
        elif self.mode == "wave":
            wave = math.sin(2 * math.pi * self.freq * time)
            pos[:, 1] += wave * self.radius
        elif self.mode == "reach":
            wave = math.sin(2 * math.pi * 0.3 * time)
            pos[:, 2] += wave * 0.15
            pos[:, 0] += (1 + wave) * 0.05

        return pos


##############################################################################
# MAIN
##############################################################################

def main():
    """Main simulation loop."""

    print("=" * 70)
    print("  G1 Hierarchical Control with Trajectory Visualization")
    print("  Lower Body: PPO Locomotion")
    print("  Upper Body: DifferentialIK")
    print("  Visualization: Hand + Target Trajectories")
    print("=" * 70)

    # ==== Environment Setup ====
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.action_manager.total_action_dim

    print(f"\n[Env] Obs dim: {obs_dim}, Action dim: {action_dim}")

    # ==== Get Robot ====
    robot = None
    scene = env.scene
    if hasattr(scene, 'articulations') and 'robot' in scene.articulations:
        robot = scene.articulations['robot']
        print("[Env] Robot articulation found!")

    # ==== Load Policy ====
    policy = None
    try:
        run_dir = os.path.join("logs", "rsl_rl", "g1_flat", args_cli.load_run)
        checkpoint_path = find_checkpoint(run_dir, args_cli.checkpoint)
        print(f"\n[Policy] Loading: {checkpoint_path}")

        # Create custom policy
        policy = CustomActorCritic(
            num_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ).to(env.device)

        # Load checkpoint
        policy.load_rsl_rl_checkpoint(checkpoint_path, env.device)
        policy.eval()
        print("[Policy] ✓ Locomotion policy loaded successfully!")

    except Exception as e:
        print(f"[Policy] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        policy = None

    # ==== Create IK Controller ====
    arm = args_cli.arm
    arm_ik = G1ArmIKController(env.num_envs, env.device, arm=arm, ik_method=args_cli.ik_method)
    if robot is not None:
        arm_ik.initialize_from_robot(robot, scene)

    # ==== Create Target Generator ====
    target_gen = TargetGenerator(env.num_envs, env.device, args_cli.target_mode, arm=arm)

    # ==== Create Trajectory Visualizer ====
    visualizer = TrajectoryVisualizer(env.num_envs, max_points=300)

    # ==== Reset ====
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    actions = torch.zeros(env.num_envs, action_dim, device=env.device)
    arm_joint_ids = ARM_JOINT_INDICES[arm]

    print("\n" + "-" * 70)
    print("[Info] Starting simulation... Press Ctrl+C to stop")
    print("[Info] Hand trajectory: Colored lines")
    print("[Info] Target: White dots")
    print("-" * 70)

    sim_time = 0.0
    dt = 0.02
    step_count = 0

    try:
        while simulation_app.is_running():
            # ==== Get Target ====
            target_pos = target_gen.get_target(sim_time)

            # ==== Lower Body: PPO Policy ====
            if policy is not None:
                with torch.no_grad():
                    actions = policy.act_inference(obs)
            else:
                actions.zero_()

            # ==== Upper Body: IK Control ====
            if robot is not None and arm_ik.initialized:
                arm_ik.set_target(target_pos)
                arm_joints = arm_ik.compute(robot)

                # Override arm joints (blend with policy output)
                for i, idx in enumerate(arm_joint_ids):
                    if i < arm_joints.shape[1]:
                        actions[:, idx] = arm_joints[:, i]

                # ==== Visualization ====
                if args_cli.draw_trajectory and step_count % 2 == 0:  # Every other step
                    ee_pos_w = arm_ik.get_ee_pos_world(robot)
                    for env_id in range(env.num_envs):
                        # Transform target to world frame (approximate)
                        root_pos = robot.data.root_state_w[env_id, 0:3]
                        target_world = target_pos[env_id] + root_pos
                        visualizer.add_point(env_id, ee_pos_w[env_id], target_world)

                    visualizer.draw_all()

            # ==== Step ====
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]

            # ==== Handle Resets ====
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                arm_ik.reset(reset_ids)
                visualizer.reset(reset_ids)

            sim_time += dt
            step_count += 1

            # ==== Logging ====
            if step_count % 200 == 0:
                mean_reward = rewards.mean().item()
                alive = (~terminated).float().mean().item() * 100
                target_str = f"[{target_pos[0, 0]:.2f}, {target_pos[0, 1]:.2f}, {target_pos[0, 2]:.2f}]"

                # Get actual EE position for comparison
                if robot is not None:
                    ee_pos = arm_ik.get_ee_pos_world(robot)[0]
                    ee_str = f"[{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]"
                else:
                    ee_str = "N/A"

                print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                      f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                      f"Target: {target_str} | EE: {ee_str}")

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user")

    finally:
        env.close()
        print("[Info] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()