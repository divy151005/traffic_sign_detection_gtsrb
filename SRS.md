# Software Requirements Specification (SRS) - Traffic Sign Detection GTSRB + Guide Sign OCR

## 1. Introduction

### 1.1 Purpose
This SRS defines requirements for the Traffic Sign Detection and Guide Sign Recognition system. The system detects traffic signs using YOLOv8, extracts ROIs, performs OCR using PaddleOCR (PP-OCRv5), and structures text (e.g., place names, distances).

### 1.2 Scope
- **Backend:** CLI pipeline for image processing.
- **Frontend:** Local animated Gradio UI for interactive inference.
- Input: JPG/PNG/GIF images.
- Output: JSON with raw OCR, cleaned text, structured output, and structured entries.
- Debug output: rejected OCR noise items with rejection reasons.
- Platforms: macOS/Linux with Python 3.13+.

### 1.3 Definitions
- **ROI:** Region of Interest (detected sign bbox).
- **OCR Item:** Text, confidence, box coordinates.
- **Structured Text:** Place names + distances (e.g., "DELHI 25KM").
- **Fused Confidence:** Weighted score combining OCR confidence and YOLO confidence.

## 2. Overall Description

### 2.1 Product Perspective
ML pipeline for Indian/Asian guide sign recognition. Uses GTSRB dataset styles, PaddleOCR for multilingual OCR including English and Hindi, with an optional Punjabi lane.

### 2.2 Product Functions
- Load image.
- YOLO detect signs (yolov8_traffic_sign_board.pt).
- Crop ROIs, OCR each.
- Run OCR across original and enhanced image variants.
- Fallback full-image OCR.
- Clean/structure text (fuzzy match places/distances/directions).
- Process folders in batch mode and emit summary JSON.
- Support browser-based Gradio UI.

### 2.3 User Classes
- ML Engineers: Run pipeline, extend models.
- End-users: CLI via run_all.sh.
- Demo users: UI-based inference via Gradio.

### 2.4 Operating Environment
- Python 3.13, venv.
- Deps: ultralytics, paddlepaddle, paddleocr, opencv-python.
- Hardware: CPU (force_cpu=True), optional GPU.

## 3. Functional Requirements

### 3.1 Pipeline Execution (FR-1)
- Input: Image path (CLI arg).
- Steps:
  1. Load image (cv2.imread).
  2. Detect signs (YOLO conf=0.25).
  3. OCR ROIs using original/enhanced/thresholded variants.
  4. Fuse OCR and detection confidence.
  5. Structure text.
- Output: `*_ocr_primary.json` + console print.

### 3.2 Detection (FR-2)
- Model: yolov8_traffic_sign_board.pt.
- Output: List[Detection] (bbox, conf, label).

### 3.3 OCR (FR-3)
- Engine: PaddleGuideSignOCR (det=PP-OCRv5_server_det, rec=PP-OCRv5_server_rec).
- Input option: `--ocr-langs` for `en`, `hi`, `ch`, and optional `pa`.
- Output: List[OCRItem] (text, conf, box, x_center, y_center, source, language).

### 3.4 Text Processing (FR-4)
- Clean tokens, extract places/distances/directions, and filter low-confidence noise.
- Output: StructuredTextResult (raw_ocr, cleaned_text, structured_output, structured_entries, rejected_ocr_noise).

### 3.5 Batch Processing (FR-5)
- Input: Folder path via `--input-dir`.
- Output: Per-image JSON files + `batch_summary.json`.

### 3.6 UI Inference (FR-6)
- Launch via `./run_all.sh` or `./run_all.sh --ui`.
- Output: Annotated preview, cleaned text, structured result, downloadable JSON.

## 4. Non-Functional Requirements

### 4.1 Performance
- Inference: <5s/image on M1 Mac CPU.
- Accuracy: >80% OCR conf on clear signs.

### 4.2 Reliability
- Graceful fallback: No YOLO → full OCR.
- Cache models in outputs/paddlex_cache_fresh.

### 4.3 Usability
- Single command UI: `./run_all.sh`
- Single command CLI: `./run_all.sh --cli --image img.jpg`
- JSON outputs self-contained.

### 4.4 Portability
- Cross-platform (macOS/Linux).
- Venv isolated.

## 5. Assumptions & Dependencies
- data.rar extracted for Test/7.png.
- models/yolov8_traffic_sign_board.pt exists.
- Internet for first PaddleOCR model download.

## 6. Future Enhancements
- Real-time video.
- Multi-lang support.
- CNN fallback for low-conf OCR.
