"""
VLM Wrapper using Florence-2
Fast object grounding for robot navigation
Fixed: _supports_sdpa attribute error
"""

import torch
import json
import re
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict

class VLMWrapper:
    """
    Florence-2 wrapper for object grounding.
    """

    COLOR_MAP = {
        "mavi": "blue", "blue": "blue",
        "kırmızı": "red", "red": "red",
        "yeşil": "green", "green": "green",
        "sarı": "yellow", "yellow": "yellow",
        "turuncu": "orange", "orange": "orange",
        "mor": "purple", "purple": "purple",
        "beyaz": "white", "white": "white",
        "siyah": "black", "black": "black",
        "kahverengi": "brown", "brown": "brown",
        "pembe": "pink", "pink": "pink",
        "gri": "gray", "gray": "gray", "grey": "gray",
        "turkuaz": "cyan", "cyan": "cyan",
    }

    OBJECT_MAP = {
        "sandalye": "chair", "chair": "chair",
        "masa": "table", "table": "table",
        "dolap": "cabinet", "cabinet": "cabinet",
        "koltuk": "sofa", "sofa": "sofa", "couch": "sofa",
        "kutu": "box", "box": "box", "cube": "box",
        "top": "ball", "ball": "ball",
        "silindir": "cylinder", "cylinder": "cylinder",
        "koni": "cone", "cone": "cone",
    }

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"[VLM] Loading {model_id}...")
        print(f"[VLM] Device: {device}, Dtype: {dtype}")

        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[VLM] Available VRAM: {total_vram:.1f} GB")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        # FIX: Explicitly set attn_implementation to avoid SDPA check
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",  # FIX: Avoid _supports_sdpa error
        ).to(device)

        self.device = device
        self.model.eval()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"[VLM] GPU Memory - Allocated: {allocated:.2f} GB")

        print(f"[VLM] Model loaded successfully!")

    def parse_command(self, command: str) -> Tuple[str, str]:
        command_lower = command.lower()
        command_clean = command_lower
        for suffix in ["'e", "'a", "ye", "ya", "'ye", "'ya", "e git", "a git", "yi bul", "ı bul", "u bul", "ü bul"]:
            command_clean = command_clean.replace(suffix, "")

        color = "unknown"
        obj = "object"

        for tr, en in self.COLOR_MAP.items():
            if tr in command_clean:
                color = en
                break

        for tr, en in self.OBJECT_MAP.items():
            if tr in command_clean:
                obj = en
                break

        return color, obj

    def ground_object(self, image: np.ndarray, command: str) -> Dict:
        import time
        start = time.time()

        color, obj = self.parse_command(command)
        target_phrase = f"{color} {obj}"
        print(f"[VLM] Looking for: '{target_phrase}'")

        # Convert to PIL
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        img_width, img_height = pil_image.size

        # Try phrase grounding
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = target_phrase

        inputs = self.processor(
            text=task_prompt + text_input,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(img_width, img_height)
        )

        elapsed = time.time() - start
        print(f"[VLM] Inference time: {elapsed*1000:.0f}ms")
        print(f"[VLM] Raw output: {parsed}")

        result = self._parse_grounding_result(parsed, img_width, img_height, color, obj)

        if not result["found"]:
            print("[VLM] Grounding failed, trying object detection...")
            result = self._try_object_detection(pil_image, target_phrase, color, obj)

        return result

    def _parse_grounding_result(self, parsed: dict, img_width: int, img_height: int, color: str, obj: str) -> Dict:
        try:
            if "<CAPTION_TO_PHRASE_GROUNDING>" in parsed:
                grounding_data = parsed["<CAPTION_TO_PHRASE_GROUNDING>"]
                if "bboxes" in grounding_data and len(grounding_data["bboxes"]) > 0:
                    bbox = grounding_data["bboxes"][0]
                    x1, y1, x2, y2 = bbox

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    norm_x = (center_x / img_width) * 2 - 1
                    norm_y = center_y / img_height

                    bbox_area = (x2 - x1) * (y2 - y1)
                    img_area = img_width * img_height
                    size_ratio = bbox_area / img_area
                    distance = 1.0 - min(size_ratio * 5, 0.9)

                    return {
                        "found": True,
                        "x": float(norm_x),
                        "y": float(distance),
                        "confidence": 0.9,
                        "color": color,
                        "object": obj,
                        "bbox": [x1, y1, x2, y2],
                    }
        except Exception as e:
            print(f"[VLM] Parse error: {e}")

        return {"found": False, "x": 0.0, "y": 0.5, "confidence": 0.0, "color": color, "object": obj}

    def _try_object_detection(self, pil_image: Image, target: str, color: str, obj: str) -> Dict:
        img_width, img_height = pil_image.size
        task_prompt = "<OD>"

        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(img_width, img_height)
        )

        print(f"[VLM] Object detection: {parsed}")

        if "<OD>" in parsed:
            od_data = parsed["<OD>"]
            if "bboxes" in od_data and "labels" in od_data:
                for i, label in enumerate(od_data["labels"]):
                    label_lower = label.lower()
                    if obj in label_lower or label_lower in obj:
                        bbox = od_data["bboxes"][i]
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        norm_x = (center_x / img_width) * 2 - 1

                        bbox_area = (x2 - x1) * (y2 - y1)
                        img_area = img_width * img_height
                        size_ratio = bbox_area / img_area
                        distance = 1.0 - min(size_ratio * 5, 0.9)

                        return {
                            "found": True,
                            "x": float(norm_x),
                            "y": float(distance),
                            "confidence": 0.7,
                            "color": color,
                            "object": obj,
                            "bbox": [x1, y1, x2, y2],
                            "label": label,
                        }

        return {"found": False, "x": 0.0, "y": 0.5, "confidence": 0.0, "color": color, "object": obj}

    def describe_image(self, image: np.ndarray) -> str:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        task_prompt = "<DETAILED_CAPTION>"

        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=pil_image.size
        )

        return parsed.get("<DETAILED_CAPTION>", "No caption generated")

    def warmup(self):
        print("[VLM] Warming up...")
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy[100:150, 100:150] = [255, 0, 0]
        self.ground_object(dummy, "red box")
        print("[VLM] Warmup complete!")


class NavigationController:
    def __init__(self, max_linear_vel: float = 1.0, max_angular_vel: float = 1.0, goal_threshold: float = 0.15):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.goal_threshold = goal_threshold

    def target_to_velocity(self, target: Dict) -> np.ndarray:
        if not target.get("found", False):
            return np.array([0.0, 0.0, 0.3])

        x_offset = target["x"]
        y_distance = target["y"]

        if y_distance < self.goal_threshold and abs(x_offset) < 0.2:
            return np.array([0.0, 0.0, 0.0])

        vyaw = -x_offset * self.max_angular_vel

        if y_distance < 0.3:
            vx = y_distance * self.max_linear_vel
        else:
            vx = min(0.3 + y_distance * 0.7, 1.0) * self.max_linear_vel

        if abs(x_offset) > 0.5:
            vx *= 0.5

        return np.array([vx, 0.0, vyaw])

    def is_goal_reached(self, target: Dict) -> bool:
        if not target.get("found", False):
            return False
        return target["y"] < self.goal_threshold and abs(target["x"]) < 0.2


if __name__ == "__main__":
    import sys
    import time

    print("=" * 60)
    print("Florence-2 VLM Test")
    print("=" * 60)

    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        command = sys.argv[2] if len(sys.argv) > 2 else "mavi sandalyeye git"
        print(f"\n[Test] Loading: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        print(f"[Test] Image shape: {image_np.shape}")
    else:
        print("\n[Test] No image, using dummy")
        image_np = np.zeros((512, 512, 3), dtype=np.uint8)
        image_np[200:300, 150:250] = [0, 0, 255]
        image_np[100:200, 300:400] = [255, 0, 0]
        command = "blue box"

    print("\n[Test] Loading Florence-2...")
    start = time.time()
    vlm = VLMWrapper()
    print(f"[Test] Loaded in {time.time()-start:.1f}s")

    vlm.warmup()

    print(f"\n[Test] Command: '{command}'")
    start = time.time()
    result = vlm.ground_object(image_np, command)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"  Time: {elapsed*1000:.0f}ms")
    print(f"  Found: {result['found']}")
    if result['found']:
        print(f"  Position: x={result['x']:.2f}, y={result['y']:.2f}")
        print(f"  Confidence: {result['confidence']:.2f}")
        if 'bbox' in result:
            print(f"  BBox: {result['bbox']}")

    nav = NavigationController()
    vel = nav.target_to_velocity(result)
    print(f"\n  Velocity: vx={vel[0]:.2f}, vy={vel[1]:.2f}, vyaw={vel[2]:.2f}")

    print(f"\n[Test] Image description...")
    desc = vlm.describe_image(image_np)
    print(f"  {desc[:300]}...")

    print(f"\n{'='*60}")
    print("MORE TESTS")
    print('='*60)
    for cmd in ["blue chair", "mavi sandalye", "chair", "sandalye"]:
        start = time.time()
        r = vlm.ground_object(image_np, cmd)
        print(f"  '{cmd}' -> found={r['found']}, x={r['x']:.2f}, y={r['y']:.2f} ({(time.time()-start)*1000:.0f}ms)")

    print("\n" + "=" * 60)
    print("Done!")