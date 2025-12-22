"""
Semantic Map Integration Demo
==============================

Bu script mevcut vlm_isaac_demo.py ile SemanticMap entegrasyonunu gösterir.

Workflow:
1. Simülasyon başlar
2. İlk VLM detection'da semantic map initialize edilir
3. Sonraki frame'lerde VLM YERİNE semantic map kullanılır
4. Periyodik VLM re-detection ile güncelleme (isteğe bağlı)

Performans Karşılaştırması:
- VLM her frame: ~200ms/frame → 5 Hz
- Semantic Map: ~2ms/frame → 500 Hz
- 100x hızlanma!

Kullanım:
    # Isaac Lab environment'ta çalıştır
    python semantic_map_integration_demo.py --test_mode

    # Veya mevcut demo'ya import et
    from semantic_map_integration_demo import SemanticMapNavigator
"""

import numpy as np
import torch
import time
from typing import Optional, Dict, Tuple, Any

# Import our semantic map module
import sys

sys.path.insert(0, '/home/claude/vlm_rl_general_system/src')

from semantic_map import (
    SemanticMap,
    HeightMapExtractor,
    ObjectTracker,
    CameraIntrinsics,
)


class SemanticMapNavigator:
    """
    VLM-based initialization + Semantic Map tracking ile navigation.

    Mevcut VLMNavigator'ın optimize edilmiş versiyonu.
    VLM'i sadece initialization ve recovery için kullanır.
    """

    def __init__(
            self,
            device: str = "cuda",
            vlm_model: str = "microsoft/Florence-2-base",
            height_map_size: Tuple[int, int] = (64, 64),
            map_range: float = 4.0,  # ±2m in each direction
    ):
        self.device = device

        # Semantic map components
        self.semantic_map = SemanticMap(device=device)
        self.height_extractor = HeightMapExtractor(
            output_size=height_map_size,
            x_range=(-map_range / 2, map_range / 2),
            y_range=(-map_range / 2, map_range / 2),
        )
        self.object_tracker = ObjectTracker()

        # VLM (lazy load)
        self.vlm = None
        self.vlm_model_name = vlm_model

        # Navigation parameters
        self.forward_speed = 0.4
        self.turn_gain = 0.6
        self.approach_threshold = 0.3  # meters
        self.reached_threshold = 0.5  # meters

        # State
        self.initialized = False
        self.current_target_id: Optional[int] = None
        self.frame_count = 0
        self.vlm_redetect_interval = 300  # VLM her 300 frame'de bir (re-verify)

        # Performance stats
        self.init_time_ms = 0.0
        self.last_update_time_ms = 0.0
        self.total_vlm_calls = 0

    def _load_vlm(self):
        """VLM'i lazy load et"""
        if self.vlm is not None:
            return

        print(f"[SemanticMapNav] Loading VLM: {self.vlm_model_name}")
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
        from PIL import Image

        self._Image = Image

        self.processor = AutoProcessor.from_pretrained(
            self.vlm_model_name, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(self.vlm_model_name, trust_remote_code=True)
        config._attn_implementation = "eager"

        self.vlm = AutoModelForCausalLM.from_pretrained(
            self.vlm_model_name,
            config=config,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.vlm.eval()

        print(f"[SemanticMapNav] VLM loaded. GPU: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def initialize_from_image(
            self,
            rgb_image: np.ndarray,
            depth_image: np.ndarray,
            camera_intrinsics: Dict[str, float],
            robot_pose: np.ndarray,
    ) -> bool:
        """
        VLM ile scene'i analiz et ve semantic map'i initialize et.

        Bu fonksiyon başlangıçta BİR KEZ çağrılır.

        Args:
            rgb_image: H x W x 3 RGB image
            depth_image: H x W depth in meters
            camera_intrinsics: {"fx", "fy", "cx", "cy", "width", "height"}
            robot_pose: [x, y, yaw] or [x, y, z, qw, qx, qy, qz]

        Returns:
            success: bool
        """
        t0 = time.time()

        # Load VLM if needed
        self._load_vlm()

        # Run VLM detection to find all objects
        detections = self._detect_all_objects(rgb_image)

        if len(detections) == 0:
            print("[SemanticMapNav] No objects detected!")
            return False

        # Initialize semantic map
        intrinsics = CameraIntrinsics.from_dict(camera_intrinsics)

        success = self.semantic_map.initialize_from_vlm(
            vlm_detections=detections,
            depth_image=depth_image,
            camera_intrinsics=camera_intrinsics,
            robot_pose=robot_pose,
        )

        self.initialized = success
        self.total_vlm_calls += 1
        self.init_time_ms = (time.time() - t0) * 1000

        print(f"[SemanticMapNav] Initialized in {self.init_time_ms:.1f}ms with {len(detections)} objects")

        return success

    def _detect_all_objects(self, rgb_image: np.ndarray) -> list:
        """
        VLM ile tüm objeleri tespit et.

        Returns:
            List of {"label": str, "color": str, "bbox": [x1,y1,x2,y2], "confidence": float}
        """
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1 else rgb_image.astype(np.uint8)

        pil_image = self._Image.fromarray(rgb_image)
        w, h = pil_image.size

        detections = []

        # Define objects to search for
        search_queries = [
            ("blue box", "box", "blue"),
            ("red ball", "ball", "red"),
            ("green box", "box", "green"),
            ("yellow cone", "cone", "yellow"),
            ("orange box", "box", "orange"),
        ]

        for query, label, color in search_queries:
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            inputs = self.processor(
                text=task + query,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.vlm.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )

            text = self.processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(text, task=task, image_size=(w, h))

            key = "<CAPTION_TO_PHRASE_GROUNDING>"
            if key in parsed and parsed[key].get("bboxes"):
                for bbox in parsed[key]["bboxes"]:
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1) / (w * h)

                    # Filter reasonable detections
                    if 0.01 < area < 0.5:
                        detections.append({
                            "label": label,
                            "color": color,
                            "bbox": bbox,
                            "confidence": 1.0 - area,  # Smaller = usually closer = higher confidence
                        })

        return detections

    def update(
            self,
            depth_image: np.ndarray,
            robot_pose: np.ndarray,
            camera_intrinsics: Optional[Dict[str, float]] = None,
    ):
        """
        Her frame'de çağrılır. VLM KULLANMAZ - sadece geometric tracking.

        Hedef: < 2ms

        Args:
            depth_image: Current depth image
            robot_pose: Current robot pose
            camera_intrinsics: Optional (uses cached if not provided)
        """
        if not self.initialized:
            return

        t0 = time.time()

        # Convert depth to height map
        if camera_intrinsics:
            intrinsics = CameraIntrinsics.from_dict(camera_intrinsics)
        else:
            # Use default
            h, w = depth_image.shape[:2]
            intrinsics = CameraIntrinsics.from_fov(w, h, horizontal_fov_deg=70)

        height_map = self.height_extractor.depth_to_heightmap(depth_image, intrinsics)

        # Update semantic map
        self.semantic_map.update_from_depth(height_map, robot_pose)

        self.frame_count += 1
        self.last_update_time_ms = (time.time() - t0) * 1000

    def set_target(self, description: str) -> bool:
        """
        Hedef objesi belirle (Türkçe veya İngilizce).

        Args:
            description: "mavi kutu", "red ball", etc.

        Returns:
            success: Target found in semantic map
        """
        if not self.initialized:
            print("[SemanticMapNav] Not initialized!")
            return False

        obj_id = self.semantic_map.get_object_by_description(description)

        if obj_id is None:
            print(f"[SemanticMapNav] Object '{description}' not found in map")
            return False

        self.current_target_id = obj_id
        obj = self.semantic_map.objects[obj_id]
        print(f"[SemanticMapNav] Target set: [{obj_id}] {obj.color.value} {obj.label.value}")

        return True

    def get_navigation_command(
            self,
            robot_pose: np.ndarray,
            search_direction: float = 1.0,
    ) -> Tuple[Tuple[float, float, float], str]:
        """
        Navigation komutu hesapla.

        Bu fonksiyon VLM KULLANMAZ - semantic map'ten direkt okur.
        Hedef: < 0.1ms

        Args:
            robot_pose: [x, y, yaw]
            search_direction: +1 or -1 for search rotation

        Returns:
            (vx, vy, vyaw): Velocity command
            status: "SEARCHING", "APPROACHING", "REACHED", "LOST"
        """
        if not self.initialized or self.current_target_id is None:
            return (0.0, 0.0, 0.4 * search_direction), "SEARCHING"

        # Get target relative position
        rel_pos = self.semantic_map.get_relative_position(
            self.current_target_id, robot_pose
        )

        if rel_pos is None:
            return (0.1, 0.0, 0.4 * search_direction), "LOST"

        # Compute navigation
        distance = np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)
        angle = np.arctan2(rel_pos[1], rel_pos[0])

        # Check if reached
        if distance < self.reached_threshold:
            return (0.0, 0.0, 0.0), "REACHED"

        # Check if centered
        if abs(angle) < 0.2:  # ~12 degrees
            # Move forward
            vx = min(self.forward_speed, distance * 0.5)
            vyaw = -angle * 0.5  # Small correction
            return (vx, 0.0, vyaw), "APPROACHING"
        else:
            # Turn towards target
            vx = self.forward_speed * 0.2  # Slow forward
            vyaw = -angle * self.turn_gain
            vyaw = np.clip(vyaw, -0.6, 0.6)
            return (vx, 0.0, vyaw), "TURNING"

    def get_target_info(self, robot_pose: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Mevcut hedef hakkında bilgi döndür.

        Returns:
            {"id", "label", "color", "distance", "angle", "world_pos", "relative_pos"}
        """
        if self.current_target_id is None or self.current_target_id not in self.semantic_map.objects:
            return None

        obj = self.semantic_map.objects[self.current_target_id]
        rel_pos = self.semantic_map.get_relative_position(self.current_target_id, robot_pose)

        if rel_pos is None:
            return None

        distance = np.linalg.norm(rel_pos[:2])
        angle = np.arctan2(rel_pos[1], rel_pos[0])

        return {
            "id": self.current_target_id,
            "label": obj.label.value,
            "color": obj.color.value,
            "distance": distance,
            "angle": angle,
            "angle_deg": np.rad2deg(angle),
            "world_pos": obj.world_position.tolist(),
            "relative_pos": rel_pos.tolist(),
            "confidence": obj.confidence,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Performans istatistikleri"""
        return {
            "initialized": self.initialized,
            "num_objects": len(self.semantic_map.objects),
            "total_vlm_calls": self.total_vlm_calls,
            "init_time_ms": self.init_time_ms,
            "last_update_ms": self.last_update_time_ms,
            "frame_count": self.frame_count,
        }


def run_standalone_test():
    """
    Isaac Lab olmadan test.
    Simüle edilmiş data ile semantic map'i test eder.
    """
    print("\n" + "=" * 60)
    print("SemanticMapNavigator Standalone Test")
    print("=" * 60)

    # Create navigator (VLM olmadan - ground truth ile)
    nav = SemanticMapNavigator(device="cpu")

    # Simulate ground truth objects
    ground_truth = [
        {"label": "box", "color": "blue", "position": [2.0, 0.5, 0.2]},
        {"label": "ball", "color": "red", "position": [3.0, -1.0, 0.15]},
        {"label": "cone", "color": "yellow", "position": [1.5, 1.5, 0.3]},
    ]

    nav.semantic_map.initialize_from_ground_truth(ground_truth)
    nav.initialized = True

    print("\nInitialized objects:")
    for obj in nav.semantic_map.objects.values():
        print(f"  [{obj.id}] {obj.color.value} {obj.label.value} at {obj.world_position}")

    # Test target setting
    print("\n--- Target Tests ---")

    tests = ["mavi kutu", "kırmızı top", "yellow cone", "green box"]
    for desc in tests:
        success = nav.set_target(desc)
        print(f"'{desc}' -> {'✓' if success else '✗'}")

    # Test navigation
    print("\n--- Navigation Test ---")
    nav.set_target("mavi kutu")

    robot_pose = np.array([0.0, 0.0, 0.0])  # Origin, facing +X

    for i in range(10):
        cmd, status = nav.get_navigation_command(robot_pose)
        info = nav.get_target_info(robot_pose)

        print(f"Step {i}: {status}")
        print(f"  Command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, vyaw={cmd[2]:.2f}")
        if info:
            print(f"  Target: dist={info['distance']:.2f}m, angle={info['angle_deg']:.1f}°")

        # Simulate robot movement (simple forward)
        robot_pose[0] += cmd[0] * 0.1
        robot_pose[2] += cmd[2] * 0.1

        if status == "REACHED":
            print("  ★ TARGET REACHED ★")
            break

    # Performance test
    print("\n--- Performance Test ---")

    # Fake depth image
    depth = np.full((240, 320), 5.0, dtype=np.float32)

    N = 1000
    t0 = time.time()
    for _ in range(N):
        nav.update(depth, robot_pose)
    dt = (time.time() - t0) * 1000 / N

    print(f"Average update time: {dt:.3f} ms (target: < 2ms)")

    N = 10000
    t0 = time.time()
    for _ in range(N):
        nav.get_navigation_command(robot_pose)
    dt = (time.time() - t0) * 1000 / N

    print(f"Average nav command time: {dt:.4f} ms (target: < 0.1ms)")

    print("\n--- Stats ---")
    print(nav.get_stats())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", action="store_true", help="Run standalone test")
    args = parser.parse_args()

    if args.test_mode:
        run_standalone_test()
    else:
        print("Usage:")
        print("  python semantic_map_integration_demo.py --test_mode")
        print("\nOr import in your Isaac Lab script:")
        print("  from semantic_map_integration_demo import SemanticMapNavigator")