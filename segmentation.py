"""Segmentation and OCR-ready ROI refinement for detected traffic signs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torchvision
except ImportError:
    torch = None
    torchvision = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class SegmentationResult:
    mask: np.ndarray
    masked_roi: np.ndarray
    confidence: float
    backend: str
    iou: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask": self.mask,
            "masked_roi": self.masked_roi,
            "confidence": self.confidence,
            "backend": self.backend,
            "iou": self.iou,
        }


@dataclass
class RefinedROIVariant:
    name: str
    image: np.ndarray
    offset: Tuple[int, int]


@dataclass
class RefinedROISet:
    mask: np.ndarray
    tight_bbox: Tuple[int, int, int, int]
    variants: List[RefinedROIVariant]


def clean_binary_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)
    cleaned = np.where(blurred > 80, 255, 0).astype(np.uint8)
    return cleaned


def keep_largest_component(mask: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    total_area = mask.shape[0] * mask.shape[1]
    best_label = 0
    best_area = 0
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label_index

    if best_label == 0 or best_area < max(32, int(total_area * min_area_ratio)):
        return binary
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        height, width = mask.shape[:2]
        return (0, 0, width, height)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return (left, top, right, bottom)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def perspective_crop(image: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    ordered = order_quad_points(np.asarray(box, dtype=np.float32))
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])
    target_width = int(max(width_top, width_bottom))
    target_height = int(max(height_left, height_right))
    if target_width < 12 or target_height < 12:
        return None

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
    return warped if warped.size else None


def refine_segmented_roi(roi: np.ndarray, segmentation: SegmentationResult) -> RefinedROISet:
    cleaned_mask = keep_largest_component(clean_binary_mask(segmentation.mask))
    left, top, right, bottom = mask_to_bbox(cleaned_mask)

    variants: List[RefinedROIVariant] = []
    full_masked = apply_mask(roi, cleaned_mask)
    if full_masked.size:
        variants.append(RefinedROIVariant(name="segmented_masked", image=full_masked, offset=(0, 0)))

    tight_roi = roi[top:bottom, left:right].copy()
    tight_mask = cleaned_mask[top:bottom, left:right]
    if tight_roi.size:
        variants.append(
            RefinedROIVariant(
                name="segmented_tight",
                image=apply_mask(tight_roi, tight_mask),
                offset=(left, top),
            )
        )

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        primary_contour = max(contours, key=cv2.contourArea)
        rectified = perspective_crop(roi, primary_contour)
        if rectified is not None and rectified.size:
            variants.append(RefinedROIVariant(name="segmented_rectified", image=rectified, offset=(0, 0)))

    unique_variants: List[RefinedROIVariant] = []
    seen_signatures = set()
    for variant in variants:
        if variant.image.shape[0] < 8 or variant.image.shape[1] < 8:
            continue
        signature = (variant.name, variant.image.shape, int(variant.image.sum() // max(1, variant.image.size)))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_variants.append(variant)

    if not unique_variants:
        unique_variants.append(RefinedROIVariant(name="segmented_masked", image=roi.copy(), offset=(0, 0)))

    return RefinedROISet(
        mask=cleaned_mask,
        tight_bbox=(left, top, right, bottom),
        variants=unique_variants,
    )


def apply_mask(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        mask_3c = cv2.merge([mask, mask, mask])
    else:
        mask_3c = mask
    return cv2.bitwise_and(roi, mask_3c)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


class SignSegmenter:
    """Segment a sign inside a YOLO ROI using the best available backend."""

    def __init__(
        self,
        backend: str = "auto",
        yolo_seg_model_path: Optional[str] = None,
        mask_threshold: float = 0.5,
        force_cpu: bool = True,
    ) -> None:
        self.backend = backend
        self.mask_threshold = mask_threshold
        self.device = "cpu" if force_cpu else ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.yolo_seg_model_path = Path(yolo_seg_model_path) if yolo_seg_model_path else None
        self.mask_rcnn = None
        self.yolo_seg = None
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        if self.backend in {"auto", "maskrcnn"}:
            self.mask_rcnn = self._load_mask_rcnn()
        if self.backend in {"auto", "yolov8-seg"}:
            self.yolo_seg = self._load_yolo_seg()

    def _load_mask_rcnn(self):
        if torch is None or torchvision is None:
            return None
        try:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
            model.eval()
            model.to(self.device)
            print("[INFO] Torchvision Mask R-CNN segmentation backend initialized.")
            return model
        except Exception as exc:
            print(f"[WARNING] Mask R-CNN backend unavailable: {exc}")
            return None

    def _load_yolo_seg(self):
        if YOLO is None or self.yolo_seg_model_path is None:
            return None
        if not self.yolo_seg_model_path.exists():
            print(f"[WARNING] YOLOv8-seg model not found: {self.yolo_seg_model_path}")
            return None
        try:
            model = YOLO(str(self.yolo_seg_model_path))
            print(f"[INFO] YOLOv8-seg backend initialized from: {self.yolo_seg_model_path}")
            return model
        except Exception as exc:
            print(f"[WARNING] YOLOv8-seg backend unavailable: {exc}")
            return None

    def segment(self, roi: np.ndarray, gt_mask: Optional[np.ndarray] = None) -> SegmentationResult:
        result = (
            self._segment_with_yolo_seg(roi)
            if self.backend == "yolov8-seg"
            else None
        )

        if result is None and self.backend == "classical":
            result = self._segment_with_classical_fallback(roi)

        if result is None and self.backend in {"auto", "maskrcnn"}:
            result = self._segment_with_mask_rcnn(roi)

        if result is None and self.backend in {"auto", "yolov8-seg"}:
            result = self._segment_with_yolo_seg(roi)

        if result is None:
            result = self._segment_with_classical_fallback(roi)

        if gt_mask is not None:
            result.iou = compute_iou(result.mask, gt_mask)
        return result

    def _segment_with_mask_rcnn(self, roi: np.ndarray) -> Optional[SegmentationResult]:
        if self.mask_rcnn is None or torch is None or torchvision is None:
            return None
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor = torchvision.transforms.functional.to_tensor(rgb).to(self.device)
            with torch.no_grad():
                outputs = self.mask_rcnn([tensor])[0]
            if len(outputs["scores"]) == 0:
                return None
            best_index = int(torch.argmax(outputs["scores"]).item())
            best_score = float(outputs["scores"][best_index].item())
            mask_tensor = outputs["masks"][best_index, 0]
            mask = (mask_tensor.detach().cpu().numpy() >= self.mask_threshold).astype(np.uint8) * 255
            if mask.sum() == 0:
                return None
            mask = clean_binary_mask(mask)
            return SegmentationResult(
                mask=mask,
                masked_roi=apply_mask(roi, mask),
                confidence=best_score,
                backend="maskrcnn",
            )
        except Exception as exc:
            print(f"[WARNING] Mask R-CNN segmentation failed: {exc}")
            return None

    def _segment_with_yolo_seg(self, roi: np.ndarray) -> Optional[SegmentationResult]:
        if self.yolo_seg is None:
            return None
        try:
            results = self.yolo_seg.predict(source=roi, verbose=False, device=self.device)
            for result in results:
                masks = getattr(result, "masks", None)
                boxes = getattr(result, "boxes", None)
                if masks is None or boxes is None or masks.data is None or len(masks.data) == 0:
                    continue
                best_index = int(np.argmax(boxes.conf.cpu().numpy()))
                best_score = float(boxes.conf[best_index].item())
                mask = masks.data[best_index].cpu().numpy()
                mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = np.where(mask >= self.mask_threshold, 255, 0).astype(np.uint8)
                mask = clean_binary_mask(mask)
                return SegmentationResult(
                    mask=mask,
                    masked_roi=apply_mask(roi, mask),
                    confidence=best_score,
                    backend="yolov8-seg",
                )
        except Exception as exc:
            print(f"[WARNING] YOLOv8-seg segmentation failed: {exc}")
        return None

    def _segment_with_classical_fallback(self, roi: np.ndarray) -> SegmentationResult:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, 60, 140)
        thresholded = cv2.bitwise_or(adaptive, otsu)
        combined = cv2.bitwise_or(thresholded, edges)

        grabcut_mask = np.zeros(roi.shape[:2], np.uint8)
        margin_x = max(2, roi.shape[1] // 12)
        margin_y = max(2, roi.shape[0] // 12)
        rect = (margin_x, margin_y, max(1, roi.shape[1] - 2 * margin_x), max(1, roi.shape[0] - 2 * margin_y))
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi, grabcut_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            grabcut_mask = np.where(
                (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            combined = cv2.bitwise_or(combined, grabcut_mask)
        except cv2.error:
            pass

        cleaned_mask = clean_binary_mask(combined)
        if cleaned_mask.sum() == 0:
            cleaned_mask = np.full(roi.shape[:2], 255, dtype=np.uint8)

        return SegmentationResult(
            mask=cleaned_mask,
            masked_roi=apply_mask(roi, cleaned_mask),
            confidence=0.5,
            backend="classical_fallback",
        )
