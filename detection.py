"""YOLO-based traffic sign detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


DEFAULT_YOLO_CONFIDENCE = 0.25
DEFAULT_MAX_DETECTIONS = 20


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "label": self.label,
        }


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(Path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Sequence[int],
    margin: float = 0.06,
) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    height, width = image_shape[:2]
    pad_x = int((right - left) * margin)
    pad_y = int((bottom - top) * margin)
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def crop_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bbox
    return image[top:bottom, left:right].copy()


class YoloTrafficSignDetector:
    """Thin wrapper around the existing YOLO detector."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = DEFAULT_YOLO_CONFIDENCE,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        force_cpu: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.device = "cpu" if force_cpu else None
        self.model = self._load_model()

    def _load_model(self):
        if YOLO is None:
            print("[WARNING] ultralytics is not installed. Detection will fall back to full-image ROI.")
            return None
        if not self.model_path.exists():
            print(f"[WARNING] YOLO model not found: {self.model_path}. Detection will use full-image ROI.")
            return None
        try:
            model = YOLO(str(self.model_path))
            print(f"[INFO] YOLO detector loaded from: {self.model_path}")
            return model
        except Exception as exc:
            print(f"[WARNING] Failed to initialize YOLO detector: {exc}")
            return None

    def detect(self, image: np.ndarray) -> List[Detection]:
        height, width = image.shape[:2]
        fallback = [Detection(bbox=(0, 0, width, height), confidence=1.0, label="full_image")]

        if self.model is None:
            return fallback

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            print(f"[WARNING] YOLO inference failed: {exc}")
            return fallback

        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                left, top, right, bottom = expand_bbox(xyxy, image.shape)
                if right <= left or bottom <= top:
                    continue
                class_id = int(box.cls[0]) if box.cls is not None else -1
                label = str(class_id)
                if hasattr(result, "names") and class_id in result.names:
                    label = str(result.names[class_id])
                detections.append(
                    Detection(
                        bbox=(left, top, right, bottom),
                        confidence=float(box.conf[0]) if box.conf is not None else 0.0,
                        label=label,
                    )
                )

        if not detections:
            return fallback

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.max_detections]
