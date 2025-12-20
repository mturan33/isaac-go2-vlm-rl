"""
Isaac Lab VLM Navigation Demo - Go2 Robot
==========================================

Gerçek Isaac Lab simülasyonunda:
- Go2 robot spawn
- Student policy (depth distillation) yükle
- VLM ile hedef nesne bul
- Robot hedefe yürüsün

Kullanım:
    cd C:\IsaacLab
    isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_vlm_rl/vlm_isaac_demo.py

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
parser.add_argument("--checkpoint", type=str,
                   default="logs/rsl_rl/unitree_go2_flat/model_best.pt",
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
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


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
            result["x"] = (cx / w) * 2 - 1
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

        # Navigation targets
        self.targets = [
            "mavi kutuya git",
            "kırmızı topa git",
            "yeşil sandalyeye git",
            "sarı koniye git",
        ]
        self.target_idx = 0
        self.current_target = self.targets[0]

        # VLM state
        self.vlm_result = None
        self.vlm_interval = 10  # Run VLM every N steps

    def next_target(self):
        self.target_idx = (self.target_idx + 1) % len(self.targets)
        self.current_target = self.targets[self.target_idx]
        print(f"\n[VLM] New target: {self.current_target}")
        return self.current_target

    def update_from_vlm(self, vlm_result):
        """Update velocity command from VLM result."""
        self.vlm_result = vlm_result

        if not vlm_result["found"]:
            # Spin to search
            self._command[0] = torch.tensor([0.0, 0.0, 0.3], device=self.device)
        else:
            x = vlm_result["x"]
            dist = vlm_result["distance"]

            # Check if target reached
            if dist < 0.2 and abs(x) < 0.15:
                self._command[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
                print(f"\n[VLM] TARGET REACHED!")
            else:
                # Navigate towards target
                angular = -x * 0.5  # Turn towards target
                linear = min(0.3 + dist * 0.4, 0.6) * (0.5 if abs(x) > 0.4 else 1.0)
                self._command[0] = torch.tensor([linear, 0.0, angular], device=self.device)

    def get_command(self) -> torch.Tensor:
        return self._command

    def print_status(self):
        if self.vlm_result:
            r = self.vlm_result
            status = "FOUND" if r["found"] else "SEARCHING"
            print(f"\r[VLM] {status} | Target: {r['target']} | x={r['x']:.2f} | "
                  f"dist={r['distance']:.2f} | {r['time_ms']:.0f}ms    ", end="", flush=True)


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
# Main
# ============================================================
def main():
    print("\n" + "="*60)
    print("       VLM Navigation Demo - Isaac Lab + Go2")
    print("="*60)

    # Create environment
    print(f"[ENV] Creating: {args_cli.task}")

    # Parse environment config (required for Isaac Lab)
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
    try:
        checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)

        # Detect architecture
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

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
        obs_normalizer = None
        if "obs_normalizer" in checkpoint:
            obs_normalizer = EmpiricalNormalization((num_obs,)).to(device)
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
            print("[POLICY] Observation normalizer loaded")

        print("[POLICY] Loaded successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to load policy: {e}")
        print("[INFO] Running without policy (zero actions)")
        actor = None
        obs_normalizer = None

    # Initialize VLM
    vlm = None
    if not args_cli.disable_vlm:
        print("\n[VLM] Initializing Florence-2...")
        try:
            vlm = VLMNavigator(device=str(device))
        except Exception as e:
            print(f"[VLM] Failed to load: {e}")
            vlm = None

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

    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    step = 0

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
            vlm_ctrl.print_status()

        # Run VLM periodically
        if vlm is not None and step % vlm_ctrl.vlm_interval == 0:
            # Get camera image from environment
            try:
                # Try to get RGB image from env
                if hasattr(unwrapped, "_render_camera"):
                    camera_img = unwrapped._render_camera()
                elif hasattr(unwrapped, "scene") and hasattr(unwrapped.scene, "sensors"):
                    # Try to access camera sensor
                    for name, sensor in unwrapped.scene.sensors.items():
                        if "camera" in name.lower():
                            camera_img = sensor.data.output["rgb"][0].cpu().numpy()
                            break
                    else:
                        camera_img = None
                else:
                    camera_img = None

                # If no camera, use render
                if camera_img is None:
                    camera_img = env.render()

                if camera_img is not None:
                    vlm_result = vlm.find_object(camera_img, vlm_ctrl.current_target)
                    vlm_ctrl.update_from_vlm(vlm_result)

                    # Print VLM output
                    status = "✓ FOUND" if vlm_result["found"] else "⟳ SEARCH"
                    print(f"\n[VLM] Step {step} | {status} | '{vlm_result['target']}' | "
                          f"x={vlm_result['x']:.2f} | dist={vlm_result['distance']:.2f} | "
                          f"{vlm_result['time_ms']:.0f}ms")

            except Exception as e:
                if step == 0:
                    print(f"[VLM] Camera access error: {e}")

        # Set velocity command from VLM
        cmd = vlm_ctrl.get_command()
        try:
            if hasattr(unwrapped, "_commands"):
                unwrapped._commands[:] = cmd
        except:
            pass

        # Get action from policy
        with torch.no_grad():
            if actor is not None:
                obs_input = obs_normalizer.normalize(obs) if obs_normalizer else obs
                actions = actor(obs_input)
            else:
                actions = torch.zeros(1, num_actions, device=device)

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