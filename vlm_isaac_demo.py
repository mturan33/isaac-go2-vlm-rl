"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v5 - Debug Fixed)
=============================================================

Fixes from v4:
- Better debug output
- Simpler camera viewport (texture-based)
- Fallback motion when VLM fails
- Exception handling for camera capture

Kullanım:
    cd C:\IsaacLab
    .\isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_vlm_rl/vlm_isaac_demo_v5.py ^
        --task Isaac-Velocity-Flat-Unitree-Go2-v0 ^
        --checkpoint "logs/rsl_rl/unitree_go2_flat/2025-12-20_18-58-21/model_999.pt"
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import importlib.util
import traceback

# ============================================================
# Flash Attention Bypass (MUST BE FIRST)
# ============================================================
def setup_flash_attn_bypass():
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

parser = argparse.ArgumentParser(description="VLM Navigation Demo for Go2")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_vlm", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac imports
import carb
import omni.appwindow
import omni.usd
import omni.ui as ui
from pxr import UsdGeom, Gf, UsdShade, Sdf
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


# ============================================================
# VLM Navigator
# ============================================================
class VLMNavigator:
    COLOR_MAP = {
        "mavi": "blue", "kırmızı": "red", "yeşil": "green",
        "sarı": "yellow", "turuncu": "orange",
        "blue": "blue", "red": "red", "green": "green",
        "yellow": "yellow", "orange": "orange",
    }
    OBJECT_MAP = {
        "kutu": "box", "top": "ball", "koni": "cone",
        "box": "box", "ball": "ball", "cone": "cone",
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
            model_id, config=config, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
        self.model.eval()

        # Warmup
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.find_object(dummy, "red box")

        print(f"[VLM] Ready! GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

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
                max_new_tokens=1024, num_beams=3,
            )

        text = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

        result = {"found": False, "target": target, "x": 0.0, "y": 0.0,
                  "distance": 1.0, "bbox": None, "time_ms": (time.time()-t0)*1000}

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1+x2)/2, (y1+y2)/2
            result["x"] = (cx / w) * 2 - 1
            result["y"] = (cy / h) * 2 - 1
            area = (x2-x1) * (y2-y1) / (w*h)
            result["distance"] = max(0.1, 1.0 - area * 5)
            result["found"] = True
            result["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

        return result


# ============================================================
# Robot Camera
# ============================================================
class RobotCamera:
    def __init__(self, resolution=(320, 240)):
        self.resolution = resolution
        self._initialized = False
        self._last_image = None
        self.render_product = None
        self.rgb_annotator = None

    def create_camera(self, robot_base_path: str):
        try:
            import omni.replicator.core as rep
            stage = omni.usd.get_context().get_stage()

            camera_path = f"{robot_base_path}/front_camera"

            # Remove if exists
            if stage.GetPrimAtPath(camera_path):
                stage.RemovePrim(camera_path)

            # Create camera
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.GetFocalLengthAttr().Set(15.0)
            camera.GetHorizontalApertureAttr().Set(20.955)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            # Position: forward and up from robot base
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, 0.25))
            xform.AddRotateYXZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))

            print(f"[CAMERA] Created at {camera_path}")

            # Create render product
            self.render_product = rep.create.render_product(camera_path, self.resolution)
            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])

            self._initialized = True
            print(f"[CAMERA] Render product ready: {self.resolution}")
            return True

        except Exception as e:
            print(f"[CAMERA] Failed: {e}")
            traceback.print_exc()
            return False

    def capture(self) -> np.ndarray:
        if not self._initialized:
            return None
        try:
            import omni.replicator.core as rep
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
            data = self.rgb_annotator.get_data()

            if data is not None and len(data) > 0:
                img = np.array(data)
                if img.ndim == 3:
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    self._last_image = img
                    return img
        except Exception as e:
            print(f"[CAMERA] Capture error: {e}")
        return self._last_image


# ============================================================
# Camera Viewport Window
# ============================================================
class CameraViewport:
    """Simple viewport window using dynamic texture."""

    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self._window = None
        self._byte_provider = None

    def create(self):
        try:
            self._window = ui.Window("Robot Camera", width=self.width+20, height=self.height+50)
            with self._window.frame:
                with ui.VStack():
                    ui.Label("Front Camera View", height=20, alignment=ui.Alignment.CENTER)
                    self._byte_provider = ui.ByteImageProvider()
                    ui.ImageWithProvider(self._byte_provider, width=self.width, height=self.height)
            print("[VIEWPORT] Camera window created")
            return True
        except Exception as e:
            print(f"[VIEWPORT] Failed: {e}")
            return False

    def update(self, image: np.ndarray):
        if self._byte_provider is None or image is None:
            return
        try:
            # Ensure correct format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB to RGBA
                rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = image
                rgba[:, :, 3] = 255
                image = rgba

            self._byte_provider.set_bytes_data(
                image.flatten().tobytes(),
                [image.shape[1], image.shape[0]]
            )
        except Exception as e:
            pass  # Silent fail


# ============================================================
# Policy Network
# ============================================================
class ActorNetwork(nn.Module):
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
    def __init__(self, input_shape: tuple, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.epsilon = epsilon

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)


# ============================================================
# Keyboard Handler
# ============================================================
class KeyboardHandler:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self.keys_pressed = set()
        self.keys_just_pressed = set()

    def _on_key(self, event, *args, **kwargs):
        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key not in self.keys_pressed:
                self.keys_just_pressed.add(key)
            self.keys_pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys_pressed.discard(key)
        return True

    def is_pressed(self, key: str) -> bool:
        return key in self.keys_pressed

    def just_pressed(self, key: str) -> bool:
        if key in self.keys_just_pressed:
            self.keys_just_pressed.discard(key)
            return True
        return False


# ============================================================
# Object Spawning
# ============================================================
def spawn_target_objects():
    stage = omni.usd.get_context().get_stage()
    targets_path = "/World/Targets"
    if not stage.GetPrimAtPath(targets_path):
        UsdGeom.Xform.Define(stage, targets_path)

    objects = [
        {"name": "blue_box", "type": "cube", "pos": (3.0, 2.0, 0.3), "scale": 0.3, "color": (0.1, 0.3, 0.9)},
        {"name": "red_ball", "type": "sphere", "pos": (-2.0, 3.0, 0.25), "scale": 0.25, "color": (0.9, 0.1, 0.1)},
        {"name": "green_box", "type": "cube", "pos": (2.0, -2.5, 0.3), "scale": 0.3, "color": (0.1, 0.8, 0.2)},
        {"name": "yellow_cone", "type": "cone", "pos": (-3.0, -2.0, 0.4), "scale": 0.4, "color": (0.9, 0.9, 0.1)},
        {"name": "orange_box", "type": "cube", "pos": (0.0, 4.0, 0.35), "scale": 0.35, "color": (1.0, 0.5, 0.0)},
    ]

    for obj in objects:
        prim_path = f"{targets_path}/{obj['name']}"
        if stage.GetPrimAtPath(prim_path):
            continue

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

        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*obj["pos"]))

        mat_path = f"{prim_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*obj["color"]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(geom.GetPrim()).Bind(material)

        print(f"[SPAWN] {obj['name']} at {obj['pos']}")

    print(f"[SPAWN] Done!")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("     VLM Navigation Demo - Go2 + Florence-2 (v5)")
    print("="*60)

    # Create environment
    print(f"\n[ENV] Creating: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1,
                            use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    num_obs = unwrapped.observation_space["policy"].shape[1]
    num_actions = unwrapped.action_space.shape[1]
    device = unwrapped.device
    print(f"[ENV] Obs: {num_obs}, Act: {num_actions}, Device: {device}")

    # Load policy
    print(f"\n[POLICY] Loading: {args_cli.checkpoint}")
    try:
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        hidden_dims = []
        for i in range(10):
            key = f"actor.{i*2}.weight"
            if key in state_dict:
                hidden_dims.append(state_dict[key].shape[0])
        hidden_dims = hidden_dims[:-1] if hidden_dims else [512, 256, 128]

        actor = ActorNetwork(num_obs, num_actions, hidden_dims).to(device)
        actor.load_state_dict(state_dict, strict=False)
        actor.eval()

        obs_normalizer = None
        if "obs_normalizer" in checkpoint:
            obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

        print(f"[POLICY] Loaded! Hidden: {hidden_dims}")
    except Exception as e:
        print(f"[ERROR] Policy load failed: {e}")
        env.close()
        simulation_app.close()
        return

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn objects
    print("\n[SPAWN] Creating objects...")
    spawn_target_objects()

    # Create camera
    print("\n[CAMERA] Setting up...")
    camera = RobotCamera(resolution=(320, 240))
    camera_ok = camera.create_camera("/World/envs/env_0/Robot/base")

    # Create viewport
    viewport = CameraViewport(320, 240)
    viewport.create()

    # Initialize VLM
    vlm = None
    if not args_cli.disable_vlm:
        print("\n[VLM] Initializing...")
        try:
            vlm = VLMNavigator(device=str(device))
        except Exception as e:
            print(f"[VLM] Init failed: {e}")
            traceback.print_exc()

    # Navigation state
    targets = ["mavi kutu", "kırmızı top", "yeşil kutu", "sarı koni", "turuncu kutu"]
    target_idx = 0
    current_target = targets[0]

    # Command state
    command = torch.tensor([[0.3, 0.0, 0.3]], device=device)  # Default: forward + turn
    search_dir = 1.0
    search_steps = 0
    target_reached = False

    # Get command term
    cmd_term = None
    if hasattr(unwrapped, "command_manager"):
        cmd_term = unwrapped.command_manager.get_term("base_velocity")
        print(f"[CMD] Command term ready")

    keyboard = KeyboardHandler()

    print("\n" + "="*60)
    print("  SPACE - Next target | R - Reset | ESC - Quit")
    print("="*60)
    print(f"\n[START] Target: {current_target}\n")

    step = 0
    vlm_interval = 15

    # Main loop
    while simulation_app.is_running() and not keyboard.is_pressed("ESCAPE"):

        # Keyboard
        if keyboard.just_pressed("SPACE"):
            target_idx = (target_idx + 1) % len(targets)
            current_target = targets[target_idx]
            target_reached = False
            search_steps = 0
            print(f"\n[TARGET] New: {current_target}")

        if keyboard.just_pressed("R"):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[RESET] Done")

        # Camera capture & viewport update
        img = None
        if camera_ok and step % 5 == 0:
            try:
                img = camera.capture()
                if img is not None:
                    viewport.update(img)
            except Exception as e:
                print(f"[CAMERA] Error: {e}")

        # VLM inference
        if vlm is not None and img is not None and step % vlm_interval == 0:
            try:
                result = vlm.find_object(img, current_target)

                if result["found"]:
                    x = result["x"]
                    dist = result["distance"]

                    if dist < 0.3 and abs(x) < 0.25:
                        command = torch.tensor([[0.0, 0.0, 0.0]], device=device)
                        if not target_reached:
                            print(f"\n[VLM] ★ REACHED: {result['target']} ★")
                            target_reached = True
                    else:
                        target_reached = False
                        angular = -x * 0.6
                        turn_factor = max(0.4, 1.0 - abs(x) * 0.6)
                        linear = (0.3 + dist * 0.3) * turn_factor
                        command = torch.tensor([[linear, 0.0, angular]], device=device)

                    print(f"[VLM] FOUND '{result['target']}' x={x:.2f} d={dist:.2f} | {result['time_ms']:.0f}ms")
                else:
                    search_steps += 1
                    if search_steps > 100:
                        search_dir *= -1
                        search_steps = 0
                    command = torch.tensor([[0.15, 0.0, 0.4 * search_dir]], device=device)
                    target_reached = False
                    print(f"[VLM] SEARCHING '{result['target']}' | {result['time_ms']:.0f}ms")

            except Exception as e:
                print(f"[VLM] Error: {e}")
                traceback.print_exc()

        # Apply command
        if cmd_term is not None and hasattr(cmd_term, 'vel_command_b'):
            cmd_term.vel_command_b[:] = command
            if hasattr(cmd_term, 'command_counter'):
                cmd_term.command_counter[:] = 0

        # Policy inference
        with torch.no_grad():
            obs_input = obs_normalizer.normalize(obs) if obs_normalizer else obs
            actions = actor(obs_input)

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[ENV] Reset")

        step += 1

    print("\n[EXIT] Done")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()