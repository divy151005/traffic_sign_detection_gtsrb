"""PaddleOCR-first utilities for traffic guide sign recognition."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
RUNTIME_CACHE_DIR = PROJECT_DIR / "outputs" / "paddlex_cache_fresh"
TEMP_DIR = PROJECT_DIR / "outputs" / "tmp_runtime"
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_CACHE_HOME"] = str(RUNTIME_CACHE_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMPDIR"] = str(TEMP_DIR)
tempfile.tempdir = str(TEMP_DIR)

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


OCR_MIN_CONFIDENCE = 0.60
DEFAULT_DET_MODEL_DIR = RUNTIME_CACHE_DIR / "official_models" / "PP-OCRv5_server_det"
DEFAULT_REC_MODEL_DIR = RUNTIME_CACHE_DIR / "official_models" / "PP-OCRv5_server_rec"
MERGE_DISTANCE_THRESHOLD = 24.0
EN_REC_MODEL_DIR = PROJECT_DIR / "outputs" / "paddlex_cache_runtime" / "official_models" / "en_PP-OCRv5_mobile_rec"
HI_REC_MODEL_DIR = PROJECT_DIR / "outputs" / "paddlex_cache_runtime" / "official_models" / "devanagari_PP-OCRv5_mobile_rec"
SUPPORTED_OCR_LANGS = {
    "ch": "Chinese",
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi",
}


@dataclass
class OCRItem:
    text: str
    confidence: float
    box: List[List[float]]
    x_center: float
    y_center: float
    source: str = "original"
    source_count: int = 1
    language: str = "ch"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": self.box,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "source": self.source,
            "source_count": self.source_count,
            "language": self.language,
        }


@dataclass
class OCRResult:
    raw_ocr: List[Dict[str, Any]]
    cleaned_text: List[str]
    structured_output: str
    ocr_failed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_ocr": self.raw_ocr,
            "cleaned_text": self.cleaned_text,
            "structured_output": self.structured_output,
            "ocr_failed": self.ocr_failed,
        }


class PaddleGuideSignOCR:
    """OCR-primary recognizer for multilingual traffic guide signs."""

    def __init__(
        self,
        lang: str = "ch",
        langs: Optional[Sequence[str]] = None,
        use_angle_cls: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ) -> None:
        self.lang = lang
        self.langs = self._normalize_langs(langs or [lang])
        self.use_angle_cls = use_angle_cls
        self.det_model_dir = self._resolve_model_dir(det_model_dir, DEFAULT_DET_MODEL_DIR)
        self.rec_model_dir = self._resolve_model_dir(rec_model_dir, DEFAULT_REC_MODEL_DIR)
        self.cls_model_dir = self._resolve_model_dir(cls_model_dir, None)
        self.engines = self._initialize_engines()

    @staticmethod
    def _normalize_langs(langs: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        for lang in langs:
            code = str(lang).strip().lower()
            if not code:
                continue
            if code not in normalized:
                normalized.append(code)
        return normalized or ["ch"]

    @staticmethod
    def get_available_ocr_languages() -> Dict[str, str]:
        return dict(SUPPORTED_OCR_LANGS)

    @staticmethod
    def get_default_rec_model_dir(lang: str) -> Optional[Path]:
        if lang == "ch":
            return DEFAULT_REC_MODEL_DIR
        if lang == "en":
            return EN_REC_MODEL_DIR
        if lang == "hi":
            return HI_REC_MODEL_DIR
        return None

    @staticmethod
    def _resolve_model_dir(path_value: Optional[str], default: Optional[Path]) -> Optional[Path]:
        if path_value:
            resolved = Path(path_value).resolve()
            return resolved if PaddleGuideSignOCR._is_readable_model_dir(resolved) else None
        if default is not None and PaddleGuideSignOCR._is_readable_model_dir(default):
            return default.resolve()
        return None

    @staticmethod
    def _is_readable_model_dir(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        config_path = path / "inference.yml"
        model_path = path / "inference.pdmodel"
        params_path = path / "inference.pdiparams"
        required_paths = (config_path, model_path, params_path)
        try:
            for item in required_paths:
                if not item.exists():
                    return False
                with item.open("rb"):
                    pass
        except OSError:
            return False
        return True

    def _initialize_engine_for_lang(self, lang_code: str):
        if PaddleOCR is None:
            print("[WARNING] PaddleOCR is not installed. OCR output will be empty.")
            return None

        kwargs: Dict[str, Any] = {
            "use_textline_orientation": bool(self.use_angle_cls),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "device": "cpu",
            "enable_mkldnn": False,
            "cpu_threads": 1,
        }
        if self.det_model_dir and self.det_model_dir.exists():
            kwargs["text_detection_model_dir"] = str(self.det_model_dir)
        rec_model_dir = self.rec_model_dir
        if rec_model_dir is None:
            rec_model_dir = self._resolve_model_dir(None, self.get_default_rec_model_dir(lang_code))
        if rec_model_dir and rec_model_dir.exists():
            kwargs["text_recognition_model_dir"] = str(rec_model_dir)
        if self.cls_model_dir and self.cls_model_dir.exists():
            kwargs["textline_orientation_model_dir"] = str(self.cls_model_dir)

        if "text_recognition_model_dir" not in kwargs:
            kwargs["lang"] = lang_code

        try:
            engine = PaddleOCR(**kwargs)
            print(f"[INFO] PaddleOCR initialized successfully for lang={lang_code}.")
            return engine
        except Exception as exc:
            print(f"[WARNING] PaddleOCR initialization failed for lang={lang_code}: {exc}")
            return None

    def _initialize_engines(self) -> List[Tuple[str, Any]]:
        engines: List[Tuple[str, Any]] = []
        for lang_code in self.langs:
            engine = self._initialize_engine_for_lang(lang_code)
            if engine is not None:
                engines.append((lang_code, engine))
        if not engines:
            print("[WARNING] No OCR engines were initialized successfully.")
        return engines

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpened = cv2.filter2D(
            enhanced,
            -1,
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        )
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def threshold_image(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def inverted_threshold_image(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def bright_text_variant(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 90, 255))
        yellow_mask = cv2.inRange(hsv, (12, 40, 120), (42, 255, 255))
        text_mask = cv2.bitwise_or(white_mask, yellow_mask)
        text_mask = cv2.morphologyEx(
            text_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        boosted = cv2.bitwise_and(image, image, mask=text_mask)
        gray = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def generate_image_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        enhanced = self.preprocess_image(image)
        thresholded = self.threshold_image(enhanced)
        inverted_thresholded = self.inverted_threshold_image(enhanced)
        bright_text = self.bright_text_variant(enhanced)
        return [
            ("original", image),
            ("enhanced", enhanced),
            ("thresholded", thresholded),
            ("inverted_thresholded", inverted_thresholded),
            ("bright_text", bright_text),
        ]

    def generate_image_variants_fast(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        enhanced = self.preprocess_image(image)
        thresholded = self.threshold_image(enhanced)
        return [
            ("original", image),
            ("thresholded", thresholded),
        ]

    @staticmethod
    def _parse_output(raw_output: Any, source: str = "original", language: str = "ch") -> List[OCRItem]:
        parsed: List[OCRItem] = []
        if not raw_output:
            return parsed

        first_result = raw_output[0]
        if isinstance(first_result, dict):
            boxes = first_result.get("rec_polys") or first_result.get("dt_polys") or []
            texts = first_result.get("rec_texts") or []
            scores = first_result.get("rec_scores") or []
            for box, text, confidence in zip(boxes, texts, scores):
                points = np.array(box, dtype=np.float32).reshape(-1, 2)
                parsed.append(
                    OCRItem(
                        text=str(text),
                        confidence=float(confidence),
                        box=points.tolist(),
                        x_center=float(points[:, 0].mean()),
                        y_center=float(points[:, 1].mean()),
                        source=source,
                        language=language,
                    )
                )
            return parsed

        for line in first_result:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            box = np.array(line[0], dtype=np.float32).reshape(-1, 2)
            text_info = line[1]
            if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                continue
            parsed.append(
                OCRItem(
                    text=str(text_info[0]),
                    confidence=float(text_info[1]),
                    box=box.tolist(),
                    x_center=float(box[:, 0].mean()),
                    y_center=float(box[:, 1].mean()),
                    source=source,
                    language=language,
                )
            )
        return parsed

    def _run_ocr_once(self, image: np.ndarray, source: str, engine: Any, language: str) -> List[OCRItem]:
        if engine is None:
            return []
        try:
            raw_output = engine.predict(
                image,
                use_textline_orientation=bool(self.use_angle_cls),
            )
        except TypeError:
            try:
                raw_output = engine.ocr(
                    image,
                    use_textline_orientation=bool(self.use_angle_cls),
                )
            except TypeError:
                raw_output = engine.ocr(image)
        except Exception as exc:
            print(f"[WARNING] PaddleOCR inference failed for lang={language}: {exc}")
            return []

        parsed = self._parse_output(raw_output, source=source, language=language)
        return [item for item in parsed if item.confidence >= OCR_MIN_CONFIDENCE]

    @staticmethod
    def _should_replace(existing: OCRItem, candidate: OCRItem) -> bool:
        if candidate.confidence > existing.confidence:
            return True
        if candidate.confidence == existing.confidence and len(candidate.text) > len(existing.text):
            return True
        return False

    @staticmethod
    def _is_same_region(first: OCRItem, second: OCRItem) -> bool:
        return (
            abs(first.x_center - second.x_center) <= MERGE_DISTANCE_THRESHOLD
            and abs(first.y_center - second.y_center) <= MERGE_DISTANCE_THRESHOLD
        )

    def merge_ocr_items(self, items: Sequence[OCRItem]) -> List[OCRItem]:
        merged: List[OCRItem] = []
        grouped_sources: Dict[int, set[str]] = defaultdict(set)
        for candidate in sorted(items, key=lambda item: item.confidence, reverse=True):
            duplicate_index: Optional[int] = None
            for index, existing in enumerate(merged):
                if self._is_same_region(existing, candidate):
                    duplicate_index = index
                    break
            if duplicate_index is None:
                candidate.source_count = 1
                merged.append(candidate)
                grouped_sources[len(merged) - 1].add(candidate.source)
            elif self._should_replace(merged[duplicate_index], candidate):
                grouped_sources[duplicate_index].add(candidate.source)
                merged[duplicate_index] = candidate
                merged[duplicate_index].source_count = len(grouped_sources[duplicate_index])
            else:
                grouped_sources[duplicate_index].add(candidate.source)
                merged[duplicate_index].source_count = len(grouped_sources[duplicate_index])
        return merged

    def run_raw_ocr(self, image: np.ndarray) -> List[OCRItem]:
        if not self.engines:
            return []
        collected: List[OCRItem] = []
        for language, engine in self.engines:
            for source, variant in self.generate_image_variants(image):
                collected.extend(self._run_ocr_once(variant, source=source, engine=engine, language=language))
        return self.merge_ocr_items(collected)

    def run_raw_ocr_fast(self, image: np.ndarray) -> List[OCRItem]:
        if not self.engines:
            return []
        collected: List[OCRItem] = []
        for language, engine in self.engines:
            for source, variant in self.generate_image_variants_fast(image):
                collected.extend(self._run_ocr_once(variant, source=source, engine=engine, language=language))
        return self.merge_ocr_items(collected)


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(Path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image
