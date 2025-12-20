"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v2 with Camera)
===========================================================

VLM (Florence-2) ile nesne tespiti yaparak Go2 robotu hedefe yönlendirir.

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_vlm_rl/vlm_isaac_demo_v2.py ^
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 ^
        --checkpoint "logs/rsl_rl/unitree_go2_flat/<TIMESTAMP>/model_500.pt"

Kontroller:
    SPACE - Hedefi değiştir
    R     - Robot reset
    V     - VLM durumunu göster
    ESC   - Çık
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import importlib.util

# ============================================================
# Flash Attention Bypass (MUST BE FIRST)
# ============================================================
def setup_flash_attn_bypass():
    """Flash attention bypass for Windows."""
    fake_flash_attn = types.ModuleType('flash_attn')
    fake_flash_attn.__file__ = __file__
    fake_flash_attn.__path__ = []
    fake_flash_attn.__package__ = 'flash_attn'
    fake_spec = importlib.util.spec_from_loader('flash_attn', loader=None)
    fake_flash_attn.__spec__ = fake_spec
    fake_flash_attn.flash_attn_func = None

    fake_bert_padding = types.ModuleType('flash_attn.bert_padding')
    fake_bert_padding.__file__ = __file__
    fake_bert_padding.__package__ = 'flash_attn.bert_padding'
    fake_bert_padding.__spec__ = importlib.util.spec_from_loader('flash_attn.bert_padding', loader=None)
    fake_bert_padding.index_first_axis = lambda *a, **k: None
    fake_bert_padding.pad_input = lambda *a, **k: None
    fake_bert_padding.unpad_input = lambda *a, **k: None

    sys.modules['flash_attn'] = fake_flash_attn
    sys.modules['flash_attn.bert_padding'] = fake_bert_padding

    try:
        from transformers.utils import import_utils
        import_utils.is_flash_attn_2_available = lambda: False
    except:
        pass

    print("[PATCH] Flash attention bypass installed")

setup_flash_attn_bypass()
# ============================================================

from isaaclab.app import AppLauncher

# Argument parser
parser = argparse.ArgumentParser(description="VLM Navigation Demo for Go2")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0",
                   help="Isaac Lab task name")
parser.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_vlm", action="store_true", help="Disable VLM for testing")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric for debugging")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab imports (after AppLauncher)
import carb
import omni.appwindow
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


# ============================================================
# Camera Manager - Isaac Sim Native Camera
# ============================================================
class CameraManager:
    """Manages a camera in Isaac Sim for VLM input."""

    def __init__(self, resolution=(256, 256)):
        self.resolution = resolution
        self.camera_prim = None
        self.render_product = None
        self.rgb_annotator = None
        self._initialized = False

    def create_camera(self, robot_prim_path: str):
        """Create a camera attached to the robot or world."""
        try:
            import omni.replicator.core as rep

            stage = omni.usd.get_context().get_stage()

            # Create camera prim - positioned to see the scene from above-behind robot
            camera_path = "/World/VLMCamera"

            # Delete if exists
            if stage.GetPrimAtPath(camera_path):
                stage.RemovePrim(camera_path)

            # Create camera
            camera = UsdGeom.Camera.Define(stage, camera_path)

            # Set camera properties
            camera.GetFocalLengthAttr().Set(18.0)  # Wide angle
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            # Position camera - bird's eye view looking at scene
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()

            # Position: behind and above, looking at origin
            translate = xform.AddTranslateOp()
            translate.Set(Gf.Vec3d(0.0, -5.0, 4.0))  # Behind and above

            # Rotation: tilt down to see the ground
            rotate = xform.AddRotateXYZOp()
            rotate.Set(Gf.Vec3d(45.0, 0.0, 0.0))  # 45 degree tilt

            self.camera_prim = camera.GetPrim()
            print(f"[CAMERA] Created camera at {camera_path}")

            # Create render product for capturing
            self.render_product = rep.create.render_product(
                camera_path,
                self.resolution
            )

            # Create RGB annotator
            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])

            self._initialized = True
            print(f"[CAMERA] Render product ready: {self.resolution}")
            return True

        except Exception as e:
            print(f"[CAMERA] Failed to create camera: {e}")
            self._try_viewport_fallback()
            return False

    def _try_viewport_fallback(self):
        """Try viewport capture as fallback."""
        print("[CAMERA] Trying viewport capture fallback...")
        self._use_viewport = True
        self._initialized = True

    def capture(self) -> np.ndarray:
        """Capture RGB image from camera."""
        if not self._initialized:
            return None

        try:
            # Method 1: Replicator annotator
            if self.rgb_annotator is not None:
                import omni.replicator.core as rep
                rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

                data = self.rgb_annotator.get_data()
                if data is not None and len(data) > 0:
                    # Convert to numpy array (H, W, 4) RGBA
                    img = np.array(data)
                    if img.ndim == 3 and img.shape[2] >= 3:
                        return img[:, :, :3]  # RGB only

            # Method 2: Viewport fallback
            if hasattr(self, '_use_viewport') and self._use_viewport:
                return self._capture_viewport()

        except Exception as e:
            pass

        return None

    def _capture_viewport(self) -> np.ndarray:
        """Capture from active viewport."""
        try:
            import omni.kit.viewport.utility as vp_utils

            viewport = vp_utils.get_active_viewport()
            if viewport is None:
                return None

            # Try different capture methods
            frame = viewport.get_frame_data()
            if frame is not None:
                return np.array(frame)

        except:
            pass
        return None


# ============================================================
# VLM Navigator
# ============================================================
class VLMNavigator:
    """Florence-2 based object grounding."""

    COLOR_MAP = {
        "mavi": "blue", "kırmızı": "red", "yeşil": "green",
        "sarı": "yellow", "turuncu": "orange", "mor": "purple",
        "beyaz": "white", "siyah": "black", "pembe": "pink",
    }

    OBJECT_MAP = {
        "kutu": "box", "top": "ball", "sandalye": "chair",
        "masa": "table", "koni": "cone", "koltuk": "sofa",
    }

    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
        from PIL import Image

        self.Image = Image
        self.device = device

        model_id = "microsoft/Florence-2-base"
        print(f"[VLM] Loading {model_id}...")

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # Warmup
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.find_object(dummy, "red box")

        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[VLM] GPU Memory: {mem:.2f} GB - Ready!")

    def parse_command(self, command: str):
        cmd = command.lower()
        color, obj = "", "object"
        for tr, en in self.COLOR_MAP.items():
            if tr in cmd:
                color = en
                break
        for tr, en in self.OBJECT_MAP.items():
            if tr in cmd:
                obj = en
                break
        return color, obj

    def find_object(self, image: np.ndarray, command: str):
        """Find object in image using VLM grounding."""
        import time
        t0 = time.time()

        color, obj = self.parse_command(command)
        target = f"{color} {obj}".strip()

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

        pil = self.Image.fromarray(image)
        w, h = pil.size

        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(text=task + target, images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        text = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

        dt = time.time() - t0

        result = {
            "found": False,
            "target": target,
            "x": 0.0,
            "distance": 1.0,
            "bbox": None,
            "time_ms": dt * 1000,
        }

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            result["x"] = (cx / w) * 2 - 1  # -1 to 1 (left to right)
            area = (x2-x1) * (y2-y1) / (w*h)
            result["distance"] = max(0.1, 1.0 - area * 5)
            result["found"] = True
            result["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

        return result


# ============================================================
# Policy Network
# ============================================================
class ActorNetwork(nn.Module):
    """Actor network for locomotion policy."""

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list = [512, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = num_obs
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_actions))

        self.actor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class EmpiricalNormalization(nn.Module):
    """Running observation normalization."""

    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.epsilon = epsilon

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)


# ============================================================
# VLM Controller
# ============================================================
class VLMController:
    """Converts VLM output to velocity commands."""

    def __init__(self, device):
        self.device = device
        self._command = torch.zeros(1, 3, device=device)
        self._search_direction = 1.0  # 1 or -1 for search rotation

        # Navigation targets (Turkish commands)
        self.targets = [
            "mavi kutuya git",
            "kırmızı topa git",
            "yeşil kutuya git",
            "sarı koniye git",
            "turuncu kutuya git",
        ]
        self.target_idx = 0
        self.current_target = self.targets[0]

        # VLM state
        self.vlm_result = None
        self.vlm_interval = 15  # Run VLM every N steps
        self.search_steps = 0
        self.max_search_steps = 200  # Change direction after this many search steps

    def next_target(self):
        self.target_idx = (self.target_idx + 1) % len(self.targets)
        self.current_target = self.targets[self.target_idx]
        self.vlm_result = None  # Reset VLM state
        self.search_steps = 0
        print(f"\n[TARGET] New target: {self.current_target}")
        return self.current_target

    def update_from_vlm(self, vlm_result):
        """Update velocity command from VLM result."""
        self.vlm_result = vlm_result

        if not vlm_result["found"]:
            # Spin to search for object
            self.search_steps += 1
            if self.search_steps > self.max_search_steps:
                self._search_direction *= -1
                self.search_steps = 0
                print(f"[VLM] Reversing search direction")

            self._command[0] = torch.tensor([0.0, 0.0, 0.4 * self._search_direction], device=self.device)
        else:
            self.search_steps = 0
            x = vlm_result["x"]  # -1 (left) to 1 (right)
            dist = vlm_result["distance"]  # 0 (close) to 1 (far)

            # Check if target reached
            if dist < 0.25 and abs(x) < 0.2:
                self._command[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                print(f"\n[VLM] ★★★ TARGET REACHED! ★★★")
            else:
                # Navigate towards target
                angular = -x * 0.6  # Turn towards target (negative because x positive = object on right = turn right = negative yaw)

                # Slow down if turning a lot
                turn_factor = max(0.3, 1.0 - abs(x))
                linear = min(0.4 + dist * 0.3, 0.6) * turn_factor

                self._command[0] = torch.tensor([linear, 0.0, angular], device=self.device)

    def get_command(self) -> torch.Tensor:
        return self._command

    def get_status_string(self) -> str:
        if self.vlm_result is None:
            return "[VLM] Waiting for camera..."
        r = self.vlm_result
        if r["found"]:
            return f"[VLM] FOUND '{r['target']}' | x={r['x']:.2f} | dist={r['distance']:.2f} | {r['time_ms']:.0f}ms"
        else:
            return f"[VLM] SEARCHING for '{r['target']}' | {r['time_ms']:.0f}ms"


# ============================================================
# Keyboard Handler
# ============================================================
class KeyboardHandler:
    """Simple keyboard input handler."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

        self.space_pressed = False
        self.reset_pressed = False
        self.vlm_toggle = False
        self.quit = False

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "SPACE":
                self.space_pressed = True
            elif event.input.name == "R":
                self.reset_pressed = True
            elif event.input.name == "V":
                self.vlm_toggle = True
            elif event.input.name == "ESCAPE":
                self.quit = True
        return True

    def consume_space(self):
        if self.space_pressed:
            self.space_pressed = False
            return True
        return False

    def consume_reset(self):
        if self.reset_pressed:
            self.reset_pressed = False
            return True
        return False

    def consume_vlm_toggle(self):
        if self.vlm_toggle:
            self.vlm_toggle = False
            return True
        return False


# ============================================================
# Object Spawning
# ============================================================
def spawn_target_objects():
    """Spawn colored objects in the scene for VLM navigation."""
    stage = omni.usd.get_context().get_stage()

    # Create parent prim
    targets_path = "/World/Targets"
    if not stage.GetPrimAtPath(targets_path):
        UsdGeom.Xform.Define(stage, targets_path)

    # Define objects with positions and colors
    objects = [
        {"name": "blue_box", "type": "cube", "pos": (3.0, 2.0, 0.3), "scale": 0.3, "color": (0.1, 0.3, 0.9)},
        {"name": "red_ball", "type": "sphere", "pos": (-2.0, 3.0, 0.25), "scale": 0.25, "color": (0.9, 0.1, 0.1)},
        {"name": "green_box", "type": "cube", "pos": (2.0, -2.5, 0.3), "scale": 0.3, "color": (0.1, 0.8, 0.2)},
        {"name": "yellow_cone", "type": "cone", "pos": (-3.0, -2.0, 0.4), "scale": 0.4, "color": (0.9, 0.9, 0.1)},
        {"name": "orange_box", "type": "cube", "pos": (0.0, 4.0, 0.35), "scale": 0.35, "color": (1.0, 0.5, 0.0)},
    ]

    for obj in objects:
        prim_path = f"{targets_path}/{obj['name']}"

        # Skip if already exists
        if stage.GetPrimAtPath(prim_path):
            continue

        # Create geometry
        if obj["type"] == "cube":
            geom = UsdGeom.Cube.Define(stage, prim_path)
            geom.GetSizeAttr().Set(obj["scale"] * 2)
        elif obj["type"] == "sphere":
            geom = UsdGeom.Sphere.Define(stage, prim_path)
            geom.GetRadiusAttr().Set(obj["scale"])
        elif obj["type"] == "cone":
            geom = UsdGeom.Cone.Define(stage, prim_path)
            geom.GetRadiusAttr().Set(obj["scale"])
            geom.GetHeightAttr().Set(obj["scale"] * 2)

        # Set position
        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*obj["pos"]))

        # Create and apply material with color
        mat_path = f"{prim_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*obj["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material to geometry
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)

        print(f"[SPAWN] Created {obj['name']} at {obj['pos']}")

    print(f"[SPAWN] Total {len(objects)} objects spawned!")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("       VLM Navigation Demo - Isaac Lab + Go2 (v2)")
    print("="*60)

    # Create environment
    print(f"[ENV] Creating: {args_cli.task}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    num_obs = unwrapped.observation_space["policy"].shape[1]
    num_actions = unwrapped.action_space.shape[1]
    device = unwrapped.device

    print(f"[ENV] Observation dim: {num_obs}")
    print(f"[ENV] Action dim: {num_actions}")
    print(f"[ENV] Device: {device}")

    # Load policy
    print(f"\n[POLICY] Loading: {args_cli.checkpoint}")
    actor = None
    obs_normalizer = None

    try:
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)

        # Detect architecture
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Check observation dimension compatibility
        first_layer_key = "actor.0.weight"
        if first_layer_key in state_dict:
            checkpoint_obs_dim = state_dict[first_layer_key].shape[1]
            if checkpoint_obs_dim != num_obs:
                print(f"[WARNING] Observation dim mismatch: checkpoint={checkpoint_obs_dim}, env={num_obs}")
                print(f"[WARNING] Policy may not work correctly!")

        # Find hidden dims
        hidden_dims = []
        for i in range(10):
            key = f"actor.{i*2}.weight"
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])
        if hidden_dims:
            hidden_dims = hidden_dims[:-1]
        if not hidden_dims:
            hidden_dims = [512, 256, 128]

        print(f"[POLICY] Hidden dims: {hidden_dims}")

        actor = ActorNetwork(num_obs, num_actions, hidden_dims).to(device)
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()

        # Load normalizer if available
        if "obs_normalizer" in checkpoint:
            obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
            print("[POLICY] Observation normalizer loaded")

        print("[POLICY] Loaded successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}")
        print("[ERROR] Cannot run without policy. Please provide a valid checkpoint.")
        env.close()
        simulation_app.close()
        return

    # Reset environment first
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn target objects
    print("\n[SPAWN] Creating target objects...")
    spawn_target_objects()

    # Create camera
    print("\n[CAMERA] Setting up camera...")
    camera = CameraManager(resolution=(256, 256))
    robot_path = "/World/envs/env_0/Robot"
    camera_ready = camera.create_camera(robot_path)

    if not camera_ready:
        print("[WARNING] Camera setup failed, VLM will be disabled")

    # Initialize VLM
    vlm = None
    if not args_cli.disable_vlm and camera_ready:
        print("\n[VLM] Initializing Florence-2...")
        try:
            vlm = VLMNavigator(device=str(device))
        except Exception as e:
            print(f"[VLM] Failed to load: {e}")
            vlm = None
    elif not camera_ready:
        print("[VLM] Disabled - no camera available")

    # Controllers
    vlm_ctrl = VLMController(device)
    keyboard = KeyboardHandler()

    # Print controls
    print("\n" + "="*60)
    print("                    CONTROLS")
    print("="*60)
    print("  SPACE     - Change navigation target")
    print("  R         - Reset robot")
    print("  V         - Print VLM status")
    print("  ESC       - Quit")
    print("="*60)
    print(f"\n[START] Target: {vlm_ctrl.current_target}\n")

    step = 0
    last_vlm_step = -999

    # Main loop
    while simulation_app.is_running() and not keyboard.quit:

        # Handle keyboard
        if keyboard.consume_space():
            vlm_ctrl.next_target()

        if keyboard.consume_reset():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[RESET] Robot reset")

        if keyboard.consume_vlm_toggle():
            print(f"\n{vlm_ctrl.get_status_string()}")

        # Run VLM periodically
        if vlm is not None and (step - last_vlm_step) >= vlm_ctrl.vlm_interval:
            camera_img = camera.capture()

            if camera_img is not None:
                vlm_result = vlm.find_object(camera_img, vlm_ctrl.current_target)
                vlm_ctrl.update_from_vlm(vlm_result)
                last_vlm_step = step

                # Print status every VLM update
                print(f"\r{vlm_ctrl.get_status_string()}    ", end="", flush=True)
            else:
                if step % 100 == 0:
                    print("[CAMERA] No image captured", flush=True)

        # Get velocity command from VLM controller
        cmd = vlm_ctrl.get_command()

        # Set velocity command in environment
        try:
            if hasattr(unwrapped, "command_manager"):
                cmd_term = unwrapped.command_manager.get_term("base_velocity")
                if cmd_term is not None and hasattr(cmd_term, "vel_command_b"):
                    cmd_term.vel_command_b[:] = cmd
                    # Prevent resampling
                    if hasattr(cmd_term, "command_counter"):
                        cmd_term.command_counter[:] = 0
                    if hasattr(cmd_term, "time_left"):
                        cmd_term.time_left[:] = 9999.0
        except Exception as e:
            if step == 0:
                print(f"[CMD] Error setting command: {e}")

        # Get action from policy
        with torch.no_grad():
            obs_input = obs_normalizer.normalize(obs) if obs_normalizer else obs
            actions = actor(obs_input)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # Handle episode end
        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[ENV] Episode ended, resetting...")

        step += 1

    # Cleanup
    print("\n[EXIT] Closing...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()