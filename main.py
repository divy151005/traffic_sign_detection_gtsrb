"""OCR-primary traffic guide sign recognition pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from detection import YoloTrafficSignDetector, crop_roi
from ocr import DEFAULT_DET_MODEL_DIR, DEFAULT_REC_MODEL_DIR, PaddleGuideSignOCR, load_image
from segmentation import SignSegmenter, refine_segmented_roi
from text_processing import StructuredTextResult, structure_guide_sign_text

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = BASE_DIR / "data" /  "Test" /"7.png"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "ocr_primary"
DEFAULT_YOLO_MODEL = BASE_DIR / "models" / "yolov8_traffic_sign_board.pt"
DEFAULT_OCR_DET_DIR = DEFAULT_DET_MODEL_DIR
DEFAULT_OCR_REC_DIR = DEFAULT_REC_MODEL_DIR
DEFAULT_OCR_CLS_DIR = BASE_DIR / "models" / "paddleocr" / "cls"
DEFAULT_SEGMENTATION_BACKEND = "classical"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def parse_ocr_langs(value: Optional[Iterable[str] | str]) -> List[str]:
    if value is None:
        return ["en", "hi", "ch"]
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    langs = [str(part).strip().lower() for part in parts if str(part).strip()]
    return langs or ["en", "hi", "ch"]


def compute_fused_confidence(ocr_confidence: float, detection_confidence: Optional[float]) -> float:
    if detection_confidence is None:
        return ocr_confidence
    return round((0.65 * float(ocr_confidence)) + (0.35 * float(detection_confidence)), 4)


def should_segment_detection(roi_image, detection_label: str) -> bool:
    height, width = roi_image.shape[:2]
    if height <= 0 or width <= 0:
        return False

    area = height * width
    aspect_ratio = width / max(height, 1)
    label_text = str(detection_label).lower()

    board_keywords = ("board", "guide", "destination", "text", "overhead", "signboard")
    if any(keyword in label_text for keyword in board_keywords):
        return True

    if area < 12_000:
        return False

    # Destination boards are typically wider than symbol-centric traffic signs.
    if aspect_ratio >= 1.35:
        return True

    # Tall portrait signs sometimes contain stacked text, but squares and circles usually do not.
    if aspect_ratio <= 0.72 and area >= 28_000:
        return True

    return False


def _roi_items_to_dicts(
    roi_items: Iterable[object],
    left: int = 0,
    top: int = 0,
    detection_label: str = "full_image",
    detection_confidence: Optional[float] = None,
    roi_variant: str = "detector_roi",
    segmentation_backend: Optional[str] = None,
    segmentation_confidence: Optional[float] = None,
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for item in roi_items:
        confidence = float(item.confidence)
        output.append(
            {
                "text": item.text,
                "confidence": confidence,
                "fused_confidence": compute_fused_confidence(confidence, detection_confidence),
                "detection_confidence": detection_confidence,
                "detection_label": detection_label,
                "ocr_source": getattr(item, "source", "original"),
                "ocr_source_count": getattr(item, "source_count", 1),
                "ocr_language": getattr(item, "language", "ch"),
                "roi_variant": roi_variant,
                "segmentation_backend": segmentation_backend,
                "segmentation_confidence": segmentation_confidence,
                "box": [[point[0] + left, point[1] + top] for point in item.box],
                "x_center": item.x_center + left,
                "y_center": item.y_center + top,
            }
        )
    return output


def _retag_ocr_items(roi_items: Iterable[object], prefix: str) -> List[object]:
    for item in roi_items:
        source = getattr(item, "source", "original")
        item.source = f"{prefix}:{source}"
    return list(roi_items)


def run_ocr_for_variant(ocr_engine: PaddleGuideSignOCR, image, fast_mode: bool):
    return ocr_engine.run_raw_ocr_fast(image) if fast_mode else ocr_engine.run_raw_ocr(image)


def collect_ocr_candidates(
    image_path: str,
    yolo_model_path: Optional[str],
    yolo_conf: float,
    ocr_engine: PaddleGuideSignOCR,
    use_segmentation: bool = True,
    segmentation_backend: str = DEFAULT_SEGMENTATION_BACKEND,
    yolo_seg_model_path: Optional[str] = None,
    fast_mode: bool = True,
) -> List[Dict[str, object]]:
    image = load_image(image_path)
    raw_items: List[Dict[str, object]] = []
    segmenter: Optional[SignSegmenter] = None

    if yolo_model_path:
        detector = YoloTrafficSignDetector(yolo_model_path, confidence_threshold=yolo_conf)
        detections = detector.detect(image)
    else:
        detections = []

    if not detections:
        detections = []

    if detections:
        for detection in detections:
            roi = crop_roi(image, detection.bbox)
            left, top, _, _ = detection.bbox

            ocr_variants = [("detector_roi", roi, 0, 0, None, None)]
            if use_segmentation and should_segment_detection(roi, detection.label):
                if segmenter is None:
                    segmenter = SignSegmenter(
                        backend=segmentation_backend,
                        yolo_seg_model_path=yolo_seg_model_path,
                    )
                segmentation = segmenter.segment(roi)
                refined = refine_segmented_roi(roi, segmentation)
                for variant in refined.variants:
                    ocr_variants.append(
                        (
                            variant.name,
                            variant.image,
                            variant.offset[0],
                            variant.offset[1],
                            segmentation.backend,
                            segmentation.confidence,
                        )
                    )

            for roi_variant, variant_image, offset_x, offset_y, seg_backend, seg_conf in ocr_variants:
                roi_items = _retag_ocr_items(run_ocr_for_variant(ocr_engine, variant_image, fast_mode), roi_variant)
                raw_items.extend(
                    _roi_items_to_dicts(
                        roi_items,
                        left=left + offset_x,
                        top=top + offset_y,
                        detection_label=detection.label,
                        detection_confidence=detection.confidence,
                        roi_variant=roi_variant,
                        segmentation_backend=seg_backend,
                        segmentation_confidence=seg_conf,
                    )
                )

    if not raw_items:
        full_image_items = _retag_ocr_items(run_ocr_for_variant(ocr_engine, image, fast_mode), "full_image")
        raw_items.extend(_roi_items_to_dicts(full_image_items, roi_variant="full_image"))
        if use_segmentation:
            if segmenter is None:
                segmenter = SignSegmenter(
                    backend=segmentation_backend,
                    yolo_seg_model_path=yolo_seg_model_path,
                )
            segmentation = segmenter.segment(image)
            refined = refine_segmented_roi(image, segmentation)
            for variant in refined.variants:
                roi_items = _retag_ocr_items(run_ocr_for_variant(ocr_engine, variant.image, fast_mode), variant.name)
                raw_items.extend(
                    _roi_items_to_dicts(
                        roi_items,
                        left=variant.offset[0],
                        top=variant.offset[1],
                        roi_variant=variant.name,
                        segmentation_backend=segmentation.backend,
                        segmentation_confidence=segmentation.confidence,
                    )
                )

    return raw_items


def build_output(raw_items: List[Dict[str, object]], extra_terms: Optional[List[str]] = None) -> Dict[str, object]:
    processed: StructuredTextResult = structure_guide_sign_text(raw_items, extra_terms=extra_terms)
    return {
        "raw_ocr": processed.raw_ocr,
        "cleaned_text": processed.cleaned_text,
        "structured_output": processed.structured_output,
        "structured_entries": processed.structured_entries,
        "rejected_ocr_noise": processed.rejected_ocr_noise,
        "ocr_failed": len(processed.raw_ocr) == 0,
    }


def run_pipeline(
    image_path: str,
    output_dir: str,
    yolo_model_path: Optional[str] = None,
    yolo_conf: float = 0.25,
    ocr_det_model_dir: Optional[str] = None,
    ocr_rec_model_dir: Optional[str] = None,
    ocr_cls_model_dir: Optional[str] = None,
    extra_terms: Optional[List[str]] = None,
    ocr_langs: Optional[List[str]] = None,
    use_segmentation: bool = True,
    segmentation_backend: str = DEFAULT_SEGMENTATION_BACKEND,
    yolo_seg_model_path: Optional[str] = None,
    fast_mode: bool = True,
) -> Dict[str, object]:
    output_path = ensure_dir(output_dir)
    ocr_engine = PaddleGuideSignOCR(
        use_angle_cls=False,
        lang=(ocr_langs[0] if ocr_langs else "en"),
        langs=parse_ocr_langs(ocr_langs),
        det_model_dir=ocr_det_model_dir,
        rec_model_dir=ocr_rec_model_dir,
        cls_model_dir=ocr_cls_model_dir,
    )

    raw_items = collect_ocr_candidates(
        image_path,
        yolo_model_path,
        yolo_conf,
        ocr_engine,
        use_segmentation=use_segmentation,
        segmentation_backend=segmentation_backend,
        yolo_seg_model_path=yolo_seg_model_path,
        fast_mode=fast_mode,
    )
    result = build_output(raw_items, extra_terms=extra_terms)

    result_path = output_path / f"{Path(image_path).stem}_ocr_primary.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    result["result_path"] = str(result_path)
    return result


def list_input_images(folder: str | Path) -> List[Path]:
    folder_path = Path(folder)
    return sorted(
        path for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def run_batch_pipeline(
    input_dir: str,
    output_dir: str,
    yolo_model_path: Optional[str] = None,
    yolo_conf: float = 0.25,
    ocr_det_model_dir: Optional[str] = None,
    ocr_rec_model_dir: Optional[str] = None,
    ocr_cls_model_dir: Optional[str] = None,
    extra_terms: Optional[List[str]] = None,
    ocr_langs: Optional[List[str]] = None,
    use_segmentation: bool = True,
    segmentation_backend: str = DEFAULT_SEGMENTATION_BACKEND,
    yolo_seg_model_path: Optional[str] = None,
    fast_mode: bool = True,
) -> Dict[str, object]:
    image_paths = list_input_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_dir}")

    summary_output_dir = ensure_dir(output_dir)
    batch_results: List[Dict[str, object]] = []

    for image_path in image_paths:
        result = run_pipeline(
            image_path=str(image_path),
            output_dir=str(summary_output_dir),
            yolo_model_path=yolo_model_path,
            yolo_conf=yolo_conf,
            ocr_det_model_dir=ocr_det_model_dir,
            ocr_rec_model_dir=ocr_rec_model_dir,
            ocr_cls_model_dir=ocr_cls_model_dir,
            extra_terms=extra_terms,
            ocr_langs=ocr_langs,
            use_segmentation=use_segmentation,
            segmentation_backend=segmentation_backend,
            yolo_seg_model_path=yolo_seg_model_path,
            fast_mode=fast_mode,
        )
        batch_results.append(
            {
                "image": str(image_path),
                "result_path": result["result_path"],
                "ocr_failed": result["ocr_failed"],
                "structured_output": result["structured_output"],
                "structured_entries": result["structured_entries"],
                "raw_item_count": len(result["raw_ocr"]),
            }
        )

    summary = {
        "input_dir": str(Path(input_dir).resolve()),
        "total_images": len(batch_results),
        "successful_images": sum(not item["ocr_failed"] for item in batch_results),
        "failed_images": sum(item["ocr_failed"] for item in batch_results),
        "results": batch_results,
    }

    summary_path = summary_output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR-primary traffic guide sign recognition with PaddleOCR.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE_PATH), help="Path to the input image.")
    parser.add_argument("--input-dir", type=str, default="", help="Optional directory of images for batch processing.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for OCR outputs.")
    parser.add_argument("--yolo-model", type=str, default=(str(DEFAULT_YOLO_MODEL) if DEFAULT_YOLO_MODEL.exists() else ""), help="Optional YOLO detector for ROI cropping.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    parser.add_argument("--ocr-det-model-dir", type=str, default=str(DEFAULT_OCR_DET_DIR), help="Optional PaddleOCR detection model directory.")
    parser.add_argument("--ocr-rec-model-dir", type=str, default="", help="Optional PaddleOCR recognition model directory.")
    parser.add_argument("--ocr-cls-model-dir", type=str, default=str(DEFAULT_OCR_CLS_DIR), help="Optional PaddleOCR classifier model directory.")
    parser.add_argument("--ocr-langs", nargs="*", default=["en"], help="OCR languages to try, e.g. en hi ch pa.")
    parser.add_argument("--extra-terms", nargs="*", default=None, help="Optional extra guide-sign terms for fuzzy correction.")
    parser.add_argument("--disable-segmentation", action="store_true", help="Disable segmentation-assisted ROI refinement before OCR.")
    parser.add_argument("--segmentation-backend", type=str, default=DEFAULT_SEGMENTATION_BACKEND, choices=["classical", "auto", "maskrcnn", "yolov8-seg"], help="Segmentation backend to use for ROI refinement.")
    parser.add_argument("--yolo-seg-model", type=str, default="", help="Optional YOLOv8 segmentation model path.")
    parser.add_argument("--accurate-mode", action="store_true", help="Use slower multi-variant OCR for maximum recall.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_model = args.yolo_model if args.yolo_model else None
    if args.input_dir:
        summary = run_batch_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            yolo_model_path=yolo_model,
            yolo_conf=args.yolo_conf,
            ocr_det_model_dir=args.ocr_det_model_dir,
            ocr_rec_model_dir=(args.ocr_rec_model_dir or None),
            ocr_cls_model_dir=args.ocr_cls_model_dir,
            extra_terms=args.extra_terms,
            ocr_langs=parse_ocr_langs(args.ocr_langs),
            use_segmentation=not args.disable_segmentation,
            segmentation_backend=args.segmentation_backend,
            yolo_seg_model_path=(args.yolo_seg_model or None),
            fast_mode=not args.accurate_mode,
        )
        print("\n[BATCH SUMMARY]")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    result = run_pipeline(
        image_path=args.image,
        output_dir=args.output_dir,
        yolo_model_path=yolo_model,
        yolo_conf=args.yolo_conf,
        ocr_det_model_dir=args.ocr_det_model_dir,
        ocr_rec_model_dir=(args.ocr_rec_model_dir or None),
        ocr_cls_model_dir=args.ocr_cls_model_dir,
        extra_terms=args.extra_terms,
        ocr_langs=parse_ocr_langs(args.ocr_langs),
        use_segmentation=not args.disable_segmentation,
        segmentation_backend=args.segmentation_backend,
        yolo_seg_model_path=(args.yolo_seg_model or None),
        fast_mode=not args.accurate_mode,
    )

    print("\n[RAW OCR]")
    print(json.dumps(result["raw_ocr"], indent=2, ensure_ascii=False))

    print("\n[CLEANED TEXT]")
    print(json.dumps(result["cleaned_text"], indent=2, ensure_ascii=False))

    print("\n[STRUCTURED OUTPUT]")
    print(result["structured_output"])

    print("\n[STRUCTURED ENTRIES]")
    print(json.dumps(result["structured_entries"], indent=2, ensure_ascii=False))

    if result["ocr_failed"]:
        print("\n[INFO] OCR failed completely. No CNN fallback is configured in this OCR-primary pipeline.")


if __name__ == "__main__":
    main()
