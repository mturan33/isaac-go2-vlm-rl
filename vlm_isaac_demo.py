"""
Isaac Lab VLM Navigation Demo - Go2 Robot (v9)
===============================================
v8 üzerine eklenenler:
1. Robot-mounted camera (base frame'e bağlı)
2. OpenCV viewport penceresi (real-time görüntü)
3. Gelişmiş navigation logic

Kullanım:
    # Full demo
    .\isaaclab.bat -p vlm_isaac_demo_v9.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint path/model.pt

    # VLM olmadan
    .\isaaclab.bat -p vlm_isaac_demo_v9.py --task ... --checkpoint ... --no_vlm

    # Viewport penceresi kapalı
    .\isaaclab.bat -p vlm_isaac_demo_v9.py --task ... --checkpoint ... --no_viewport

Kontroller:
    SPACE - Hedef değiştir
    R - Reset
    V - Viewport aç/kapat
    ESC - Çıkış
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import sys
import types
import importlib.util
import traceback

# Flash Attention Bypass (Windows için gerekli)
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

# Isaac Lab AppLauncher (CRITICAL - must be before other Isaac imports)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--no_vlm", action="store_true", help="Disable VLM")
parser.add_argument("--no_camera", action="store_true", help="Disable camera")
parser.add_argument("--no_viewport", action="store_true", help="Disable OpenCV viewport window")
parser.add_argument("--disable_fabric", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

# CRITICAL: Enable cameras in AppLauncher
if not args_cli.no_camera:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import other modules (after AppLauncher)
import carb
import omni.appwindow
import omni.usd
from pxr import UsdGeom, Gf, UsdShade, Sdf
import gymnasium as gym

# Isaac Lab task imports (CRITICAL - registers environments)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Replicator imports
import omni.replicator.core as rep

# OpenCV for viewport (optional)
CV2_AVAILABLE = False
CV2_GUI_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
    # Test if GUI functions are available
    try:
        # This will fail if highgui is not compiled with GUI support
        if hasattr(cv2, 'namedWindow'):
            cv2.namedWindow("_test_", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test_")
            CV2_GUI_AVAILABLE = True
            print("[VIEWPORT] OpenCV GUI loaded")
        else:
            print("[VIEWPORT] OpenCV loaded (headless - no GUI)")
    except Exception as e:
        print(f"[VIEWPORT] OpenCV loaded but GUI not available: {e}")
except ImportError:
    print("[VIEWPORT] OpenCV not available")

# Isaac Sim Viewport (alternative to OpenCV)
OMNI_UI_AVAILABLE = False
try:
    import omni.ui as ui
    from omni.kit.viewport.utility import get_active_viewport
    OMNI_UI_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# Robot-Mounted Camera (NEW in v9)
# ============================================================
class RobotMountedCamera:
    """
    Robotun base frame'ine monte edilmiş kamera.
    Robot hareket ettikçe kamera da hareket eder.
    """

    def __init__(
        self,
        robot_prim_path: str = "/World/envs/env_0/Robot",
        width: int = 320,
        height: int = 240,
        offset: tuple = (0.35, 0.0, 0.15),  # Forward, left, up from base
        rotation: tuple = (0.0, 15.0, 0.0),  # Pitch down slightly
    ):
        self.robot_prim_path = robot_prim_path
        self.width = width
        self.height = height
        self.offset = offset
        self.rotation = rotation

        # Camera will be created under robot's base link
        self.camera_path = f"{robot_prim_path}/base/front_camera"
        self.render_product = None
        self.rgb_annotator = None
        self.last_image = None
        self.initialized = False
        self.warmup_counter = 0
        self.warmup_frames = 50

    def setup(self):
        """Setup camera after simulation starts"""
        try:
            stage = omni.usd.get_context().get_stage()

            # First check if robot exists
            robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
            if not robot_prim.IsValid():
                print(f"[CAMERA] Robot not found at {self.robot_prim_path}")
                # Try alternative path
                self.robot_prim_path = "/World/envs/env_0/Robot"
                robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
                if not robot_prim.IsValid():
                    print(f"[CAMERA] Robot still not found, using world camera")
                    return self._setup_world_camera(stage)

            # Check for base link
            base_path = f"{self.robot_prim_path}/base"
            base_prim = stage.GetPrimAtPath(base_path)
            if not base_prim.IsValid():
                print(f"[CAMERA] Base link not found, using world camera")
                return self._setup_world_camera(stage)

            # Create camera under base
            self.camera_path = f"{base_path}/front_camera"
            camera_prim = stage.GetPrimAtPath(self.camera_path)

            if not camera_prim.IsValid():
                print(f"[CAMERA] Creating robot-mounted camera at {self.camera_path}")
                camera = UsdGeom.Camera.Define(stage, self.camera_path)
                camera.GetFocalLengthAttr().Set(18.0)  # Wide angle
                camera.GetHorizontalApertureAttr().Set(20.955)
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

                # Set local transform relative to base
                # Camera default: -Z forward, +Y up
                # We want: +X forward, +Z up
                # Quaternion (w, x, y, z) = (0.5, 0.5, -0.5, -0.5)
                xform = UsdGeom.Xformable(camera.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*self.offset))
                xform.AddOrientOp().Set(Gf.Quatf(0.5, 0.5, -0.5, -0.5))

                print(f"[CAMERA] Offset: {self.offset}, Quaternion: (0.5, 0.5, -0.5, -0.5)")

            return self._setup_replicator()

        except Exception as e:
            print(f"[CAMERA] Setup failed: {e}")
            traceback.print_exc()
            return False

    def _setup_world_camera(self, stage):
        """Fallback: Create world camera if robot mounting fails"""
        self.camera_path = "/World/VLMCamera"
        camera_prim = stage.GetPrimAtPath(self.camera_path)

        if not camera_prim.IsValid():
            print(f"[CAMERA] Creating world camera at {self.camera_path}")
            camera = UsdGeom.Camera.Define(stage, self.camera_path)
            camera.GetFocalLengthAttr().Set(24.0)
            camera.GetHorizontalApertureAttr().Set(20.955)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, -5.0, 3.0))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 0.0))

        return self._setup_replicator()

    def _setup_replicator(self):
        """Setup Replicator render product and annotator"""
        try:
            self.render_product = rep.create.render_product(
                self.camera_path,
                resolution=(self.width, self.height)
            )

            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])

            self.initialized = True
            print(f"[CAMERA] Initialized! Resolution: {self.width}x{self.height}")
            print(f"[CAMERA] Path: {self.camera_path}")
            return True

        except Exception as e:
            print(f"[CAMERA] Replicator setup failed: {e}")
            return False

    def get_image(self) -> np.ndarray:
        """Get current camera image (non-blocking)"""
        if not self.initialized:
            return None

        try:
            data = self.rgb_annotator.get_data()

            if data is not None and len(data) > 0:
                if isinstance(data, np.ndarray):
                    img = data
                else:
                    img = np.array(data)

                if len(img.shape) == 3:
                    if img.shape[2] == 4:  # RGBA -> RGB
                        img = img[:, :, :3]
                    self.last_image = img
                    self.warmup_counter += 1  # Increment on successful capture
                    return img

            return self.last_image

        except Exception as e:
            return self.last_image

    def update(self):
        """Update camera (call every step for warmup)"""
        if not self.initialized:
            return
        # Just get image to increment warmup counter
        self.get_image()

    def is_ready(self) -> bool:
        return self.initialized and self.warmup_counter >= self.warmup_frames


# ============================================================
# Isaac Sim Viewport Window (Alternative to OpenCV)
# ============================================================
class IsaacSimViewport:
    """Isaac Sim UI ile kamera görüntüsü göster"""

    def __init__(self, camera_path: str, width: int = 320, height: int = 240):
        self.camera_path = camera_path
        self.width = width
        self.height = height
        self.window = None
        self.image_provider = None
        self.enabled = False

    def setup(self):
        """Isaac Sim'de viewport window oluştur"""
        if not OMNI_UI_AVAILABLE:
            print("[ISAAC_VIEWPORT] omni.ui not available")
            return False

        try:
            # Create window
            self.window = ui.Window("Robot Camera View", width=400, height=350)

            with self.window.frame:
                with ui.VStack():
                    ui.Label("Go2 Front Camera", height=20, alignment=ui.Alignment.CENTER)

                    # Image widget with byte provider
                    self.image_provider = ui.ByteImageProvider()
                    ui.ImageWithProvider(
                        self.image_provider,
                        width=self.width,
                        height=self.height,
                        alignment=ui.Alignment.CENTER
                    )

                    # Info labels
                    self.target_label = ui.Label("Target: -", height=20)
                    self.status_label = ui.Label("Status: IDLE", height=20)

            self.enabled = True
            print("[ISAAC_VIEWPORT] Window created!")
            return True

        except Exception as e:
            print(f"[ISAAC_VIEWPORT] Setup failed: {e}")
            return False

    def update(self, image: np.ndarray, target: str = "", status: str = "IDLE", bbox: list = None):
        """Viewport'u güncelle"""
        if not self.enabled or image is None:
            return

        try:
            # Draw bbox if available
            if bbox is not None and CV2_AVAILABLE:
                img = image.copy()
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                img = image

            # Convert to RGBA for omni.ui
            if img.shape[2] == 3:
                rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = img
                rgba[:, :, 3] = 255
            else:
                rgba = img

            # Update image provider
            self.image_provider.set_bytes_data(
                rgba.flatten().tolist(),
                [self.width, self.height]
            )

            # Update labels
            if hasattr(self, 'target_label'):
                self.target_label.text = f"Target: {target}"
            if hasattr(self, 'status_label'):
                self.status_label.text = f"Status: {status}"

        except Exception as e:
            pass  # Silently fail to avoid spam

    def close(self):
        if self.window:
            self.window.visible = False


# ============================================================
# OpenCV Viewport Display (NEW in v9)
# ============================================================
class ViewportDisplay:
    """Ayrı pencerede kamera görüntüsü göster veya dosyaya kaydet"""

    def __init__(self, window_name: str = "Go2 Camera - Press V to toggle"):
        self.window_name = window_name
        self.enabled = CV2_GUI_AVAILABLE and not args_cli.no_viewport
        self.window_created = False
        self.setup_failed = False  # Prevent repeated error messages

        # Save to file mode (when GUI not available)
        self.save_mode = CV2_AVAILABLE and not CV2_GUI_AVAILABLE
        self.save_counter = 0
        self.save_interval = 30  # Save every N updates

        # Overlay info
        self.target_name = ""
        self.command = (0.0, 0.0, 0.0)
        self.status = "IDLE"
        self.last_bbox = None

    def setup(self):
        if self.save_mode:
            print("[VIEWPORT] GUI not available - will save images to 'camera_capture.png'")
            return True

        if not CV2_GUI_AVAILABLE:
            if not self.setup_failed:
                print("[VIEWPORT] GUI not available - skipping viewport window")
                self.setup_failed = True
            return False
        if not self.enabled:
            return False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
            self.window_created = True
            print(f"[VIEWPORT] Window created")
            return True
        except Exception as e:
            if not self.setup_failed:
                print(f"[VIEWPORT] Setup failed: {e}")
                self.setup_failed = True
            self.enabled = False
            return False

    def update(self, image: np.ndarray, target: str = None, cmd: tuple = None,
               status: str = None, bbox: list = None):
        if not CV2_AVAILABLE:
            return

        if target:
            self.target_name = target
        if cmd:
            self.command = cmd
        if status:
            self.status = status
        if bbox:
            self.last_bbox = bbox

        # Create display image
        if image is None:
            return

        # RGB to BGR for OpenCV
        display = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

        # Draw bounding box if available
        if self.last_bbox:
            x1, y1, x2, y2 = [int(v) for v in self.last_bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, self.target_name, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw overlay
        h, w = display.shape[:2]

        # Info bar at bottom
        cv2.rectangle(display, (0, h - 50), (w, h), (30, 30, 30), -1)

        # Target
        cv2.putText(display, f"Target: {self.target_name}", (5, h - 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Command
        vx, vy, vyaw = self.command
        cv2.putText(display, f"Cmd: vx={vx:.2f} vyaw={vyaw:.2f}", (5, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Status
        color = (0, 255, 0) if self.status == "FOUND" else (0, 165, 255)
        cv2.putText(display, self.status, (w - 80, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Crosshair
        cx, cy = w // 2, (h - 50) // 2
        cv2.line(display, (cx - 15, cy), (cx + 15, cy), (255, 255, 255), 1)
        cv2.line(display, (cx, cy - 15), (cx, cy + 15), (255, 255, 255), 1)

        # GUI mode
        if self.window_created:
            cv2.imshow(self.window_name, display)
            cv2.waitKey(1)
        # Save mode (headless)
        elif self.save_mode:
            self.save_counter += 1
            if self.save_counter >= self.save_interval:
                cv2.imwrite("camera_capture.png", display)
                self.save_counter = 0

    def toggle(self):
        if self.save_mode:
            print("[VIEWPORT] Save mode - press S to save current frame")
            return

        if not CV2_GUI_AVAILABLE:
            print("[VIEWPORT] GUI not available - cannot toggle")
            return

        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
            self.enabled = False
            print("[VIEWPORT] Window closed")
        else:
            self.enabled = True
            self.setup_failed = False  # Allow retry
            self.setup()

    def close(self):
        if self.window_created and CV2_GUI_AVAILABLE:
            cv2.destroyAllWindows()

    def save_now(self, image: np.ndarray):
        """Manuel olarak şu anki frame'i kaydet"""
        if image is not None and CV2_AVAILABLE:
            display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            filename = f"camera_capture_{self.save_counter}.png"
            cv2.imwrite(filename, display)
            print(f"[VIEWPORT] Saved: {filename}")
            self.save_counter += 1


# ============================================================
# VLM Navigator with Improved Navigation Logic (v9)
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

        # Navigation parameters (NEW in v9)
        self.forward_speed = 0.4
        self.turn_gain = 0.6
        self.approach_threshold = 0.2  # Screen center threshold
        self.reached_distance = 0.25

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

        result = {
            "found": False,
            "target": target,
            "x": 0.0,
            "distance": 1.0,
            "time_ms": (time.time()-t0)*1000,
            "bbox": None
        }

        key = "<CAPTION_TO_PHRASE_GROUNDING>"
        if key in parsed and parsed[key].get("bboxes"):
            bbox = parsed[key]["bboxes"][0]
            x1, y1, x2, y2 = bbox
            cx = (x1+x2)/2
            result["x"] = (cx / w) * 2 - 1  # Normalized [-1, 1]
            area = (x2-x1) * (y2-y1) / (w*h)
            result["distance"] = max(0.1, 1.0 - area * 5)
            result["found"] = True
            result["bbox"] = bbox

        return result

    def compute_navigation_command(self, result: dict, search_dir: float = 1.0):
        """
        Compute navigation command based on VLM detection result.

        Returns:
            (vx, vy, vyaw): Velocity command
            status: "FOUND", "SEARCHING", "REACHED"
        """
        if not result["found"]:
            # Search behavior: rotate in place
            return (0.1, 0.0, 0.4 * search_dir), "SEARCHING"

        x = result["x"]  # [-1, 1], 0 = center
        dist = result["distance"]

        # Check if reached
        if dist < self.reached_distance and abs(x) < self.approach_threshold:
            return (0.0, 0.0, 0.0), "REACHED"

        # Navigation logic
        if abs(x) < self.approach_threshold:
            # Target centered - move forward
            vx = self.forward_speed * (0.5 + dist * 0.5)  # Slow down when close
            vyaw = 0.0
        else:
            # Target off-center - turn towards it
            vx = self.forward_speed * 0.3  # Slow forward while turning
            vyaw = -x * self.turn_gain  # Negative because x>0 means right, need to turn right (negative yaw)

        return (vx, 0.0, vyaw), "FOUND"


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
    print("   VLM Navigation Demo - Go2 (v9)")
    print("   Robot-Mounted Camera + Viewport + Navigation")
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
        traceback.print_exc()
        return

    # Reset env first (this starts simulation)
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Spawn objects
    spawn_target_objects()

    # Get robot prim path for camera mounting
    robot_prim_path = "/World/envs/env_0/Robot"

    # Setup camera AFTER simulation starts
    camera = None
    if not args_cli.no_camera:
        print("\n[CAMERA] Setting up robot-mounted camera...")
        camera = RobotMountedCamera(
            robot_prim_path=robot_prim_path,
            width=320,
            height=240,
            offset=(0.40, 0.0, 0.10),      # Forward, left, up
            rotation=(-90.0, -90.0, 0.0)   # X: -90, Y: -90
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

    # Setup viewport display (NEW in v9)
    viewport = ViewportDisplay()
    isaac_viewport = None

    if camera is not None:
        viewport.setup()

        # Also try Isaac Sim native viewport
        if OMNI_UI_AVAILABLE and not args_cli.no_viewport:
            isaac_viewport = IsaacSimViewport(camera.camera_path, 320, 240)
            isaac_viewport.setup()

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
    # Target objects (English for better VLM detection)
    targets = ["blue box", "red ball", "green box", "yellow cone", "orange box"]
    target_idx = 0
    command = torch.tensor([[0.4, 0.0, 0.0]], device=device, dtype=torch.float32)
    search_dir = 1.0
    nav_status = "IDLE"
    last_bbox = None

    print("\n" + "="*60)
    print("  SPACE - Next target | R - Reset | S - Save image | ESC - Quit")
    print("="*60)
    print(f"\n[START] Target: {targets[target_idx]}")
    print(f"[START] VLM: {vlm is not None}, Camera: {camera is not None}")
    viewport_status = "GUI" if viewport.window_created else ("Save" if viewport.save_mode else "Off")
    isaac_vp_status = "On" if (isaac_viewport is not None and isaac_viewport.enabled) else "Off"
    print(f"[START] Viewport: {viewport_status}, Isaac UI: {isaac_vp_status}\n")

    step = 0
    vlm_interval = 30  # VLM every 30 steps (~0.6s at 50Hz)

    # Main loop
    while simulation_app.is_running():

        # Keyboard
        if keyboard.just_pressed("ESCAPE"):
            print("\n[EXIT] ESC pressed")
            break

        if keyboard.just_pressed("SPACE"):
            target_idx = (target_idx + 1) % len(targets)
            search_dir *= -1  # Alternate search direction
            last_bbox = None
            print(f"\n[TARGET] {targets[target_idx]}")

        if keyboard.just_pressed("R"):
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            last_bbox = None
            print("\n[RESET]")

        if keyboard.just_pressed("V"):
            viewport.toggle()

        # S tuşu ile manuel kamera kaydı
        if keyboard.just_pressed("S"):
            if camera is not None and camera.is_ready():
                img = camera.get_image()
                if img is not None:
                    viewport.save_now(img)

        # Update camera every step (for warmup)
        if camera is not None:
            camera.update()

        # Debug every 100 steps
        if step % 100 == 0:
            print(f"[STEP {step:5d}] cmd=[{command[0,0]:.2f}, {command[0,1]:.2f}, {command[0,2]:.2f}] status={nav_status}")

        # VLM inference (after camera warmup)
        if vlm is not None and camera is not None and step % vlm_interval == 0:
            # Debug camera status
            if step % 100 == 0:
                print(f"[DEBUG] Camera ready: {camera.is_ready()}, warmup: {camera.warmup_counter}/{camera.warmup_frames}")

            if camera.is_ready():
                try:
                    img = camera.get_image()

                    if img is not None:
                        # VLM inference
                        result = vlm.find_object(img, targets[target_idx])

                        # Compute navigation command (NEW in v9)
                        cmd_tuple, nav_status = vlm.compute_navigation_command(result, search_dir)
                        command = torch.tensor([cmd_tuple], device=device, dtype=torch.float32)

                        if result["found"]:
                            last_bbox = result["bbox"]
                            print(f"[VLM] {nav_status} '{result['target']}' x={result['x']:.2f} d={result['distance']:.2f} | {result['time_ms']:.0f}ms")
                            if nav_status == "REACHED":
                                print(f"[VLM] ★ TARGET REACHED ★")
                        else:
                            last_bbox = None
                            print(f"[VLM] SEARCHING '{result['target']}' | {result['time_ms']:.0f}ms")

                        # Update viewport
                        viewport.update(
                            image=img,
                            target=targets[target_idx],
                            cmd=cmd_tuple,
                            status=nav_status,
                            bbox=last_bbox
                        )

                        # Update Isaac Sim viewport
                        if isaac_viewport is not None:
                            isaac_viewport.update(img, targets[target_idx], nav_status, last_bbox)
                    else:
                        if step % 100 == 0:
                            print(f"[DEBUG] Camera returned None image")

                except Exception as e:
                    print(f"[VLM] Error: {e}")
                    import traceback
                    traceback.print_exc()

        # Update viewport even without VLM (for camera preview)
        elif camera is not None and camera.is_ready() and step % 5 == 0:
            img = camera.get_image()
            if img is not None:
                if viewport.window_created or viewport.save_mode:
                    viewport.update(
                        image=img,
                        target=targets[target_idx],
                        cmd=tuple(command[0].cpu().tolist()),
                        status=nav_status
                    )
                if isaac_viewport is not None:
                    isaac_viewport.update(img, targets[target_idx], nav_status)

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

    print("\n[EXIT] Cleaning up...")
    viewport.close()
    if isaac_viewport is not None:
        isaac_viewport.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()