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

# Debug Draw - Isaac Lab's marker visualization
try:
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
    import isaaclab.sim as sim_utils

    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    print("[Warning] Visualization markers not available")

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

        # Use assign=True to handle any remaining mismatches
        try:
            self.load_state_dict(new_state_dict, strict=True)
            print("[Policy] Loaded with strict=True")
        except RuntimeError as e:
            print(f"[Policy] Strict load failed, trying manual assignment: {e}")
            # Manual assignment for each parameter
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    if param.shape == new_state_dict[name].shape:
                        param.data.copy_(new_state_dict[name])
                    else:
                        print(
                            f"[Policy] Shape mismatch for {name}: model={param.shape}, ckpt={new_state_dict[name].shape}")

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
# TRAJECTORY VISUALIZER (Simplified - logs to console)
##############################################################################

class TrajectoryVisualizer:
    """Simple trajectory tracker - logs positions to console."""

    def __init__(self, num_envs: int, max_points: int = 200):
        self.num_envs = num_envs
        self.max_points = max_points
        self.enabled = True

        # Store trajectory points for each environment
        self.trajectories = [deque(maxlen=max_points) for _ in range(num_envs)]
        self.target_trajectories = [deque(maxlen=max_points) for _ in range(num_envs)]

        print("[Viz] Trajectory tracker initialized (console mode)")

    def add_point(self, env_id: int, hand_pos: torch.Tensor, target_pos: torch.Tensor):
        """Add a point to the trajectory."""
        hand_pt = hand_pos.cpu().tolist() if torch.is_tensor(hand_pos) else hand_pos
        target_pt = target_pos.cpu().tolist() if torch.is_tensor(target_pos) else target_pos

        self.trajectories[env_id].append(hand_pt)
        self.target_trajectories[env_id].append(target_pt)

    def draw_all(self):
        """No-op for console mode."""
        pass

    def get_tracking_error(self, env_id: int = 0) -> float:
        """Get distance between current hand pos and target."""
        if not self.trajectories[env_id] or not self.target_trajectories[env_id]:
            return 0.0

        hand = self.trajectories[env_id][-1]
        target = self.target_trajectories[env_id][-1]

        error = math.sqrt(sum((h - t) ** 2 for h, t in zip(hand, target)))
        return error

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
                # Shape: (num_envs, num_bodies-1, 6, num_dofs)
                jacobian = full_jacobian[:, self.ee_jacobi_idx, :, :]
                # Now shape: (num_envs, 6, num_dofs)
                jacobian = jacobian[:, :, self.arm_joint_ids]
                # Now shape: (num_envs, 6, 5) for 5 arm joints

            # Compute IK
            joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            return joint_pos_des

        except Exception as e:
            print(f"[IK] Compute error: {e}")
            import traceback
            traceback.print_exc()
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

        # Create custom policy - MUST match checkpoint architecture!
        # Checkpoint shows: actor.0=[256,123], actor.2=[128,256], actor.4=[128,128], actor.6=[37,128]
        policy = CustomActorCritic(
            num_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[256, 128, 128],
            critic_hidden_dims=[256, 128, 128],
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
    print("[Info] Lower body: PPO locomotion policy")
    print("[Info] Upper body: IK circle trajectory")
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

            # ==== Upper Body: Arm Control ====
            # DifferentialIK has issues (joint_diff=0), using simple analytical control instead
            if robot is not None and arm_ik.initialized:
                # Get target in base frame
                target_b = target_pos[0]  # [x, y, z]

                # Simple analytical IK for reaching motion
                # Target: [0.35 to 0.47, -0.25, 0.44 to 0.67] (circle in XZ plane)

                # Shoulder pitch: controls forward/back reach
                # More negative = arm goes forward/up
                # Target x is ~0.35-0.47, we want to reach forward
                shoulder_pitch = -1.5 - (target_b[0].item() - 0.35) * 2.0  # Range: -1.5 to -1.74

                # Shoulder roll: controls arm spread (negative = away from body for right arm)
                shoulder_roll = -0.3

                # Shoulder yaw: rotation around arm axis
                shoulder_yaw = 0.0

                # Elbow pitch: controls elbow bend
                # Higher target z = less bend needed
                elbow_pitch = -1.2 + (target_b[2].item() - 0.5) * 1.5  # Adjust based on height

                # Elbow roll
                elbow_roll = 0.0

                # Apply to actions
                # arm_joint_ids = [6, 10, 14, 18, 22]
                # [right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch, right_elbow_roll]
                actions[:, 6] = shoulder_pitch
                actions[:, 10] = shoulder_roll
                actions[:, 14] = shoulder_yaw
                actions[:, 18] = elbow_pitch
                actions[:, 22] = elbow_roll

                # Debug logging
                if step_count % 100 == 1:
                    ee_pos_w = robot.data.body_state_w[0, arm_ik.ee_body_idx, 0:3]
                    root_pos_w = robot.data.root_state_w[0, 0:3]
                    ee_pos_b = ee_pos_w - root_pos_w
                    error = torch.norm(target_b - ee_pos_b).item()

                    print(f"\n[Arm Control Step {step_count}]")
                    print(f"  Target: [{target_b[0].item():.2f}, {target_b[1].item():.2f}, {target_b[2].item():.2f}]")
                    print(f"  EE pos: [{ee_pos_b[0].item():.2f}, {ee_pos_b[1].item():.2f}, {ee_pos_b[2].item():.2f}]")
                    print(f"  Error:  {error:.3f}m")
                    print(
                        f"  Joints: sh_pitch={shoulder_pitch:.2f}, sh_roll={shoulder_roll:.2f}, el_pitch={elbow_pitch:.2f}")

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
                    track_err = visualizer.get_tracking_error(0)
                else:
                    ee_str = "N/A"
                    track_err = 0.0

                print(f"[Step {step_count:5d}] t={sim_time:6.2f}s | "
                      f"Reward: {mean_reward:7.3f} | Alive: {alive:5.1f}% | "
                      f"TrackErr: {track_err:.3f}m")

    except KeyboardInterrupt:
        print("\n[Info] Stopped by user")

    finally:
        env.close()
        print("[Info] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()