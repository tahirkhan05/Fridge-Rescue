"""
Module: ml/detection/predict.py
Purpose: Food Object Detection Inference Pipeline using trained model
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "detection_best.pt"

class FoodDetector:
    def __init__(self, model_path: Path = MODEL_PATH, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        if not model_path.exists():
            print(f"Warning: {model_path} not found. Attempting to fall back to yolo11n.pt or yolov8n.pt")
            try:
                self.model = YOLO("yolo11n.pt")
            except Exception:
                self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(model_path))

    def detect(self, image_input: Any) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Runs object detection on image (filepath, PIL Image, or np.ndarray).
        Returns:
            detections: List of dicts with name, confidence, bbox [x1, y1, x2, y2], and crop
            annotated_image: np.ndarray with bounding boxes drawn
        """
        if isinstance(image_input, (str, Path)):
            image_np = cv2.imread(str(image_input))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image_np = np.array(image_input)
        else:
            image_np = image_input.copy()

        h, w = image_np.shape[:2]
        results = self.model.predict(image_np, conf=self.conf_threshold, verbose=False)

        detections = []
        annotated_np = image_np.copy()

        # Class counts for aggregate quantity
        class_counts = {}

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [int(v.item()) for v in box.xyxy[0]]
                x1, y1, x2, y2 = xyxy

                # Bound check
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                crop = image_np[y1:y2, x1:x2]

                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                detections.append({
                    "name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "crop": crop
                })

                # Draw bounding box
                cv2.rectangle(annotated_np, (x1, y1), (x2, y2), (46, 204, 113), 2)
                label_text = f"{cls_name} {int(conf * 100)}%"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_np, (x1, max(0, y1 - 20)), (x1 + tw, y1), (46, 204, 113), -1)
                cv2.putText(annotated_np, label_text, (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return detections, annotated_np
