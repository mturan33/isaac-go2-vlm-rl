"""
Go2 + Z1 Arm Import Guide for Isaac Lab
=========================================

Bu guide, Unitree Go2 quadruped'e Z1 arm ekleyip Isaac Lab'da
loco-manipulation için kullanmayı açıklar.

URDF Kaynakları:
1. Go2: Isaac Lab'da hazır (isaaclab_assets)
2. Z1: Unitree GitHub repo veya Isaac Lab assets

Alternatif Arm Seçenekleri:
- Unitree Z1 (6 DoF + gripper) - Resmi, ağır
- Custom lightweight (5 DoF) - Simülasyon için ideal
- Franka-style arm - Isaac Lab'da hazır

Author: VLM-RL Project
Date: December 2024
"""

import os
from typing import Optional, Dict, Any

# Paths
ISAAC_LAB_ROOT = os.environ.get("ISAAC_LAB_PATH", "D:/IsaacLab")
ASSETS_PATH = os.path.join(ISAAC_LAB_ROOT, "source/isaaclab_assets/data")


def check_available_robots():
    """Isaac Lab'da mevcut robot URDF'lerini listele"""
    robots_path = os.path.join(ASSETS_PATH, "Robots")

    if not os.path.exists(robots_path):
        print(f"Robots path not found: {robots_path}")
        print("Make sure ISAAC_LAB_PATH environment variable is set correctly")
        return

    print("Available robots in Isaac Lab:")
    print("=" * 50)

    for category in os.listdir(robots_path):
        cat_path = os.path.join(robots_path, category)
        if os.path.isdir(cat_path):
            print(f"\n{category}/")
            for robot in os.listdir(cat_path):
                robot_path = os.path.join(cat_path, robot)
                if os.path.isdir(robot_path):
                    # Check for USD files
                    usd_files = [f for f in os.listdir(robot_path) if f.endswith('.usd') or f.endswith('.usda')]
                    print(f"  └─ {robot}")
                    for usd in usd_files[:3]:  # Show first 3
                        print(f"      └─ {usd}")


def get_go2_config():
    """
    Go2 robot configuration for Isaac Lab.

    Isaac Lab'da Go2, isaaclab_assets içinde tanımlı.
    """
    config = """
# Go2 Configuration (from isaaclab_assets)
# File: source/isaaclab_assets/isaaclab_assets/robots/unitree.py

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="{ISAACLAB_ASSETS_PATH}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            ".*thigh_joint": 0.8,
            ".*calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness={
                ".*_hip_joint": 25.0,
                ".*_thigh_joint": 25.0,
                ".*_calf_joint": 25.0,
            },
            damping={
                ".*_hip_joint": 0.5,
                ".*_thigh_joint": 0.5,
                ".*_calf_joint": 0.5,
            },
        ),
    },
)
"""
    return config


def get_go2_with_arm_config():
    """
    Go2 + Z1 arm configuration.

    İki yaklaşım:
    1. Composite robot: Go2 + Z1 ayrı asset'ler, attach edilir
    2. Combined URDF: Tek bir unified robot
    """
    config = """
# ===================================================================
# OPTION 1: Composite Robot (Recommended for flexibility)
# ===================================================================

# Go2 base robot (locomotion)
GO2_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="{ISAACLAB_ASSETS_PATH}/Robots/Unitree/Go2/go2.usd",
        # ... (same as above)
    ),
)

# Z1 Arm (manipulation)
Z1_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="{ISAACLAB_ASSETS_PATH}/Robots/Unitree/Z1/z1.usd",
        # USD'de arm base'i Go2'nin body link'ine attach edilecek
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Relative to Go2 body
        joint_pos={
            "joint1": 0.0,
            "joint2": -1.57,  # -90 degrees
            "joint3": 1.57,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    actuators={
        "arm": IdealPDActuatorCfg(
            joint_names_expr=["joint.*"],
            effort_limit=30.0,
            velocity_limit=2.0,
            stiffness=40.0,
            damping=4.0,
        ),
        "gripper": IdealPDActuatorCfg(
            joint_names_expr=["gripper.*"],
            effort_limit=10.0,
            velocity_limit=1.0,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)


# ===================================================================
# OPTION 2: Isaac Lab Scene ile Attach
# ===================================================================

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation

class Go2Z1Scene(InteractiveSceneCfg):
    \"\"\"Go2 + Z1 combined robot scene\"\"\"

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Go2 base
    robot_base = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{ISAACLAB_ASSETS_PATH}/Robots/Unitree/Go2/go2.usd",
        ),
    )

    # Z1 arm attached to Go2
    robot_arm = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/arm",  # Attached to body
        spawn=sim_utils.UsdFileCfg(
            usd_path="{ISAACLAB_ASSETS_PATH}/Robots/Unitree/Z1/z1.usd",
        ),
    )


# ===================================================================
# OPTION 3: Combined URDF/USD (Most integrated)
# ===================================================================

# Bu yaklaşım için custom URDF oluşturmanız gerekir:
# 1. Go2 URDF'i indir
# 2. Z1 URDF'i indir  
# 3. Z1'i Go2'nin body link'ine child olarak ekle
# 4. URDF → USD dönüşümü yap

# Örnek combined URDF structure:
COMBINED_URDF_TEMPLATE = '''
<?xml version="1.0"?>
<robot name="go2_z1">
    <!-- Include Go2 base -->
    <xacro:include filename="go2.urdf.xacro"/>

    <!-- Attach Z1 arm to body -->
    <joint name="arm_base_joint" type="fixed">
        <parent link="body"/>
        <child link="arm_base_link"/>
        <origin xyz="0.2 0.0 0.05" rpy="0 0 0"/>
    </joint>

    <!-- Include Z1 arm -->
    <xacro:include filename="z1.urdf.xacro"/>
</robot>
'''
"""
    return config


def create_go2_z1_env_config():
    """
    Go2 + Z1 için RL environment configuration.
    """
    config = """
# ===================================================================
# Environment Configuration for Go2 + Z1 Loco-manipulation
# ===================================================================

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class Go2Z1EnvCfg(DirectRLEnvCfg):
    \"\"\"Go2 + Z1 Loco-manipulation Environment\"\"\"

    # Scene
    scene: InteractiveSceneCfg = Go2Z1SceneCfg()

    # Simulation
    decimation = 4  # 250Hz / 4 = 62.5Hz policy
    sim = SimulationCfg(
        dt=1/250,
        render_interval=decimation,
    )

    # Environment
    episode_length_s = 20.0

    # Observations
    num_observations = 48 + 12 + 1024  # proprio + arm + height_map
    num_actions = 12 + 7  # legs + arm (6 joints + gripper)

    # Rewards
    reward_cfg = {
        # Locomotion
        "distance_to_goal": 1.0,
        "heading_alignment": 0.3,
        "leg_energy": -0.01,
        "action_smoothness": -0.005,

        # Manipulation (when grasping)
        "end_effector_position": 0.5,
        "grasp_success": 2.0,
        "object_displacement": 1.0,

        # Coordination
        "arm_collision": -1.0,  # Arm hitting legs
        "stable_base": 0.2,     # Keep base stable during manipulation
    }


# ===================================================================
# Observation Space
# ===================================================================

class Go2Z1Observations:
    \"\"\"
    Go2 + Z1 için observation space.

    Locomotion policy:
    - proprio: 48D (joint pos/vel, base vel, gravity)
    - height_map: 32x32 = 1024D (veya CNN encoded 64D)
    - goal_info: 10D (position, type, etc.)

    Manipulation policy (arm):
    - arm_proprio: 14D (6 joint pos + 6 joint vel + gripper + force)
    - end_effector: 6D (position + orientation)
    - target_object: 6D (relative position + size)
    \"\"\"

    @staticmethod
    def get_locomotion_obs(env):
        \"\"\"Locomotion policy observations (12 DoF legs)\"\"\"
        obs = torch.cat([
            env.robot.data.joint_pos[:, :12],      # Leg joint positions
            env.robot.data.joint_vel[:, :12],      # Leg joint velocities
            env.robot.data.root_lin_vel_b,         # Base linear velocity
            env.robot.data.root_ang_vel_b,         # Base angular velocity
            env.robot.data.projected_gravity,      # Gravity direction
        ], dim=-1)
        return obs

    @staticmethod
    def get_manipulation_obs(env):
        \"\"\"Manipulation policy observations (arm)\"\"\"
        obs = torch.cat([
            env.robot.data.joint_pos[:, 12:19],    # Arm joint positions (6 + gripper)
            env.robot.data.joint_vel[:, 12:19],    # Arm joint velocities
            env.end_effector_pos,                  # EE position
            env.target_object_pos - env.end_effector_pos,  # Relative target
        ], dim=-1)
        return obs


# ===================================================================
# Action Space
# ===================================================================

class Go2Z1Actions:
    \"\"\"
    Go2 + Z1 action space.

    Option 1: Unified policy (19 actions)
    Option 2: Hierarchical (separate leg + arm policies)

    Hierarchy recommended for:
    - Easier training (smaller action spaces)
    - Better interpretability
    - Can reuse pretrained locomotion policy
    \"\"\"

    # Unified
    NUM_LEG_ACTIONS = 12
    NUM_ARM_ACTIONS = 6
    NUM_GRIPPER_ACTIONS = 1
    TOTAL_ACTIONS = NUM_LEG_ACTIONS + NUM_ARM_ACTIONS + NUM_GRIPPER_ACTIONS

    # Action limits
    LEG_ACTION_SCALE = 0.5   # ±0.5 rad from default
    ARM_ACTION_SCALE = 1.0   # ±1.0 rad 
    GRIPPER_SCALE = 1.0      # 0=open, 1=closed
"""
    return config


def check_z1_urdf_sources():
    """Z1 URDF kaynaklarını listele"""
    sources = """
# ===================================================================
# Z1 Arm URDF/USD Sources
# ===================================================================

1. Isaac Lab Assets (Official)
   Path: {ISAACLAB_ASSETS_PATH}/Robots/Unitree/Z1/
   Note: Isaac Lab 1.0+ içinde olabilir

2. Unitree GitHub
   URL: https://github.com/unitreerobotics/z1_description
   Contains: URDF, meshes, launch files

3. Isaac Sim Assets (via Nucleus)
   Path: omniverse://localhost/NVIDIA/Assets/Isaac/2024.1/Isaac/Robots/Unitree/

4. Custom from CAD
   - SolidWorks/Fusion360 → URDF (urdf_exporter)
   - URDF → USD (Isaac Sim URDF importer)

# ===================================================================
# URDF → USD Conversion
# ===================================================================

# Option A: Isaac Sim GUI
1. Open Isaac Sim
2. Isaac Utils → Workflows → URDF Importer
3. Select URDF file
4. Configure joint limits, collision meshes
5. Export as USD

# Option B: Python Script
from isaaclab.app import AppLauncher
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.urdf")

from omni.importer.urdf import _urdf
urdf_config = _urdf.ImportConfig()
urdf_config.merge_fixed_joints = True
urdf_config.fix_base = False

result = _urdf.import_robot(
    "path/to/z1.urdf",
    urdf_config,
    "/World/Robot"
)
"""
    print(sources)


# Practical steps
def print_integration_steps():
    """Go2 + Z1 entegrasyonu için adım adım guide"""
    steps = """
╔════════════════════════════════════════════════════════════════════╗
║            Go2 + Z1 Integration Steps                              ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  STEP 1: Verify Go2 works standalone                               ║
║  ─────────────────────────────────────                             ║
║  python -m isaaclab.envs.manager_based.locomotion.velocity.go2     ║
║                                                                    ║
║  STEP 2: Locate Z1 URDF/USD                                        ║
║  ─────────────────────────────────                                 ║
║  Check: Isaac Lab assets, Unitree GitHub, Nucleus                  ║
║                                                                    ║
║  STEP 3: Test Z1 arm standalone                                    ║
║  ─────────────────────────────────                                 ║
║  Create simple scene with just Z1, verify joints work              ║
║                                                                    ║
║  STEP 4: Combine in scene                                          ║
║  ──────────────────────────                                        ║
║  Option A: Attach Z1 to Go2 body via scene config                  ║
║  Option B: Create combined URDF/USD                                ║
║                                                                    ║
║  STEP 5: Create combined env config                                ║
║  ──────────────────────────────────                                ║
║  - Define observation space (legs + arm)                           ║
║  - Define action space (12 leg + 7 arm)                            ║
║  - Add arm-specific rewards                                        ║
║                                                                    ║
║  STEP 6: Train with curriculum                                     ║
║  ──────────────────────────────                                    ║
║  Stage 1: Locomotion only (arm locked)                             ║
║  Stage 2: Static manipulation (base locked)                        ║
║  Stage 3: Combined loco-manipulation                               ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""
    print(steps)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Go2 + Z1 Arm Integration Guide")
    print("=" * 60)

    print("\n1. Checking available robots in Isaac Lab...")
    check_available_robots()

    print("\n2. Z1 URDF sources:")
    check_z1_urdf_sources()

    print("\n3. Integration steps:")
    print_integration_steps()

    print("\n4. Go2 config example:")
    print(get_go2_config()[:500] + "...")

    print("\n5. Go2 + Z1 config example:")
    print(get_go2_with_arm_config()[:800] + "...")