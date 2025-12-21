"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v8)
===============================================
Non-blocking Replicator camera - no rep.orchestrator.step()!
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import importlib.util
import traceback

# Flash Attention Bypass
def setup_flash_attn_bypass():
    fake_flash_attn = types.ModuleType('flash_attn')
    fake_flash_attn.__file__ = __file__
    fake_flash_attn.__path__ = []
    fake_flash_attn.__package__ = 'flash_attn'
    fake_flash_attn.__spec__ = importlib.util.spec_from_loader('flash_attn', loader=None)
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--no_vlm", action="store_true", help="Disable VLM")
parser.add_argument("--no_camera", action="store_true", help="Disable camera")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

# CRITICAL: Enable cameras in AppLauncher
if not args_cli.no_camera:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

# Replicator imports
import omni.replicator.core as rep


# ============================================================
# Non-blocking Camera using Replicator
# ============================================================
class ReplicatorCamera:
    """Camera using Replicator without blocking orchestrator.step()"""

    def __init__(self, camera_path: str, width: int = 320, height: int = 240):
        self.camera_path = camera_path
        self.width = width
        self.height = height
        self.render_product = None
        self.rgb_annotator = None
        self.last_image = None
        self.initialized = False

    def setup(self):
        """Setup camera after simulation starts"""
        try:
            stage = omni.usd.get_context().get_stage()

            # Check if camera prim exists, if not create it
            camera_prim = stage.GetPrimAtPath(self.camera_path)
            if not camera_prim.IsValid():
                print(f"[CAMERA] Creating camera at {self.camera_path}")
                camera = UsdGeom.Camera.Define(stage, self.camera_path)
                camera.GetFocalLengthAttr().Set(24.0)
                camera.GetHorizontalApertureAttr().Set(20.955)
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

                # Set position (world camera looking at scene)
                xform = UsdGeom.Xformable(camera.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -5.0, 3.0))
                # Rotate to look at origin
                xform.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 0.0))

            # Create render product
            self.render_product = rep.create.render_product(
                self.camera_path,
                resolution=(self.width, self.height)
            )

            # Create RGB annotator
            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])

            self.initialized = True
            print(f"[CAMERA] Initialized! Resolution: {self.width}x{self.height}")
            return True

        except Exception as e:
            print(f"[CAMERA] Setup failed: {e}")
            traceback.print_exc()
            return False

    def get_image(self) -> np.ndarray:
        """Get current camera image without blocking"""
        if not self.initialized:
            return None

        try:
            # Get data from annotator (non-blocking - uses last rendered frame)
            data = self.rgb_annotator.get_data()

            if data is not None and len(data) > 0:
                # Convert to numpy array
                if isinstance(data, np.ndarray):
                    img = data
                else:
                    img = np.array(data)

                # Handle different formats
                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA -> RGB
                        img = img[:, :, :3]
                    self.last_image = img
                    return img

            return self.last_image  # Return last valid image

        except Exception as e:
            # Don't spam errors
            return self.last_image


# ============================================================
# VLM Navigator
# ============================================================
class VLMNavigator:
    COLOR_MAP = {"mavi": "blue", "kırmızı": "red", "yeşil": "green", "sarı": "yellow", "turuncu": "orange"}
    OBJECT_MAP = {"kutu": "box", "top": "ball", "koni": "cone"}

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

    def find_object(self, image: np.ndarray, command: str):
        import time
        t0 = time.time()

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

        result = {"found": False, "target": target, "x": 0.0, "distance": 1.0, "time_ms": (time.time()-t0)*1000}

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox
            cx = (x1+x2)/2
            result["x"] = (cx / w) * 2 - 1
            area = (x2-x1) * (y2-y1) / (w*h)
            result["distance"] = max(0.1, 1.0 - area * 5)
            result["found"] = True

        return result


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


# ============================================================
# Keyboard Handler
# ============================================================
class KeyboardHandler:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self.keys_just_pressed = set()

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.keys_just_pressed.add(event.input.name)
        return True

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

    print(f"[SPAWN] Objects created!")


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("     VLM Navigation Demo - Go2 (v8 - Non-blocking Camera)")
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
        print(f"[POLICY] Loaded! Hidden: {hidden_dims}")
    except Exception as e:
        print(f"[ERROR] Policy: {e}")
        return

    # Reset env first (this starts simulation)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn objects
    spawn_target_objects()

    # Setup camera AFTER simulation starts
    camera = None
    if not args_cli.no_camera:
        print("\n[CAMERA] Setting up...")
        camera = ReplicatorCamera(
            camera_path="/World/VLMCamera",
            width=320,
            height=240
        )

        # Wait a few frames for simulation to stabilize
        for _ in range(10):
            with torch.no_grad():
                actions = actor(obs)
            obs_dict, _, _, _, _ = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # Now setup camera
        if not camera.setup():
            print("[CAMERA] Setup failed, continuing without camera")
            camera = None

    # Setup VLM
    vlm = None
    if not args_cli.no_vlm and camera is not None:
        print("\n[VLM] Initializing...")
        try:
            vlm = VLMNavigator(device=str(device))
        except Exception as e:
            print(f"[VLM] Failed: {e}")
            traceback.print_exc()

    # Get command term
    cmd_term = None
    if hasattr(unwrapped, "command_manager"):
        cmd_term = unwrapped.command_manager.get_term("base_velocity")
        print(f"[CMD] Ready: {cmd_term is not None}")

    keyboard = KeyboardHandler()

    # Navigation state
    targets = ["mavi kutu", "kırmızı top", "yeşil kutu", "sarı koni", "turuncu kutu"]
    target_idx = 0
    command = torch.tensor([[0.4, 0.0, 0.3]], device=device, dtype=torch.float32)
    search_dir = 1.0

    print("\n" + "="*60)
    print("  SPACE - Next target | R - Reset | ESC - Quit")
    print("="*60)
    print(f"\n[START] Target: {targets[target_idx]}")
    print(f"[START] VLM: {vlm is not None}, Camera: {camera is not None}\n")

    step = 0
    vlm_interval = 30  # VLM every 30 steps (~0.6s at 50Hz)
    camera_warmup = 50  # Wait for camera to warm up

    # Main loop
    while simulation_app.is_running():

        # Keyboard
        if keyboard.just_pressed("ESCAPE"):
            print("\n[EXIT] ESC pressed")
            break

        if keyboard.just_pressed("SPACE"):
            target_idx = (target_idx + 1) % len(targets)
            print(f"\n[TARGET] {targets[target_idx]}")

        if keyboard.just_pressed("R"):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[RESET]")

        # Debug every 100 steps
        if step % 100 == 0:
            print(f"[STEP {step:5d}] cmd=[{command[0,0]:.2f}, {command[0,1]:.2f}, {command[0,2]:.2f}]")

        # VLM inference (after camera warmup)
        if vlm is not None and camera is not None and step > camera_warmup and step % vlm_interval == 0:
            try:
                img = camera.get_image()

                if img is not None:
                    if step % 100 == 0:
                        print(f"[CAMERA] Got image: {img.shape}, dtype: {img.dtype}")

                    # VLM inference
                    result = vlm.find_object(img, targets[target_idx])

                    if result["found"]:
                        x = result["x"]
                        dist = result["distance"]
                        print(f"[VLM] FOUND '{result['target']}' x={x:.2f} d={dist:.2f} | {result['time_ms']:.0f}ms")

                        if dist < 0.3 and abs(x) < 0.25:
                            command = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
                            print(f"[VLM] ★ TARGET REACHED ★")
                        else:
                            angular = -x * 0.5
                            linear = 0.3 + dist * 0.2
                            command = torch.tensor([[linear, 0.0, angular]], device=device, dtype=torch.float32)
                    else:
                        print(f"[VLM] SEARCHING '{result['target']}' | {result['time_ms']:.0f}ms")
                        command = torch.tensor([[0.15, 0.0, 0.4 * search_dir]], device=device, dtype=torch.float32)
                else:
                    if step % 100 == 0:
                        print(f"[CAMERA] No image yet")

            except Exception as e:
                if step % 100 == 0:
                    print(f"[VLM] Error: {e}")

        # Apply velocity command
        if cmd_term is not None:
            try:
                cmd_term.vel_command_b[:] = command
                if hasattr(cmd_term, 'command_counter'):
                    cmd_term.command_counter[:] = 0
                if hasattr(cmd_term, 'time_left'):
                    cmd_term.time_left[:] = 9999.0
            except Exception as e:
                if step % 100 == 0:
                    print(f"[CMD] Error: {e}")

        # Policy inference
        with torch.no_grad():
            actions = actor(obs)

        # Step environment (this also renders!)
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if (terminated | truncated).any():
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            print("\n[ENV] Episode reset")

        step += 1

    print("\n[EXIT] Done")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()