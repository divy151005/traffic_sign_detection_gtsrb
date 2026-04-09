"""Animated Gradio UI for the traffic sign detection + OCR pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import gradio as gr
import numpy as np

from main import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_OCR_CLS_DIR,
    DEFAULT_OCR_DET_DIR,
    DEFAULT_SEGMENTATION_BACKEND,
    DEFAULT_YOLO_MODEL,
    build_output,
    collect_ocr_candidates,
    ensure_dir,
    parse_ocr_langs,
)
from ocr import PaddleGuideSignOCR, load_image

BASE_DIR = Path(__file__).resolve().parent
APP_OUTPUT_DIR = ensure_dir(BASE_DIR / "outputs" / "ui_runs")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;700;800&display=swap');

:root {
  --bg: #07111f;
  --panel: rgba(8, 17, 32, 0.72);
  --panel-strong: rgba(9, 22, 41, 0.88);
  --line: rgba(151, 176, 255, 0.18);
  --text: #f8fbff;
  --muted: #9cb2d1;
  --blue: #79a8ff;
  --cyan: #6ee7f2;
  --amber: #ffb95e;
  --green: #7ef0c2;
  --shadow: 0 30px 90px rgba(0, 0, 0, 0.35);
}

body, .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(121, 168, 255, 0.20), transparent 26%),
    radial-gradient(circle at 85% 15%, rgba(110, 231, 242, 0.16), transparent 22%),
    radial-gradient(circle at 50% 100%, rgba(255, 185, 94, 0.14), transparent 25%),
    linear-gradient(145deg, #040913 0%, #08111f 48%, #0b1729 100%);
  color: var(--text);
  font-family: 'Plus Jakarta Sans', sans-serif;
}

.gradio-container {
  max-width: 1320px !important;
}

.app-shell {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(151, 176, 255, 0.16);
  border-radius: 32px;
  padding: 28px;
  backdrop-filter: blur(18px);
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02)),
    rgba(5, 12, 24, 0.56);
  box-shadow: var(--shadow);
}

.app-shell::before,
.app-shell::after {
  content: "";
  position: absolute;
  inset: auto;
  width: 320px;
  height: 320px;
  border-radius: 999px;
  filter: blur(24px);
  opacity: 0.28;
  animation: drift 10s ease-in-out infinite;
  pointer-events: none;
}

.app-shell::before {
  top: -110px;
  right: -80px;
  background: radial-gradient(circle, rgba(121, 168, 255, 0.8), transparent 68%);
}

.app-shell::after {
  left: -100px;
  bottom: -140px;
  background: radial-gradient(circle, rgba(110, 231, 242, 0.7), transparent 68%);
  animation-delay: -4s;
}

.hero {
  position: relative;
  z-index: 1;
  padding: 10px 6px 26px;
}

.eyebrow {
  display: inline-flex;
  gap: 10px;
  align-items: center;
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cyan);
  background: rgba(110, 231, 242, 0.08);
  border: 1px solid rgba(110, 231, 242, 0.18);
}

.hero h1 {
  margin: 18px 0 10px;
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(2.5rem, 4vw, 4.75rem);
  line-height: 0.95;
  letter-spacing: -0.05em;
}

.hero p {
  max-width: 760px;
  margin: 0;
  font-size: 1.02rem;
  line-height: 1.7;
  color: var(--muted);
}

.hero-grid {
  margin-top: 24px;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.metric-card {
  padding: 18px;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(12, 24, 43, 0.8), rgba(7, 15, 28, 0.66));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
  animation: rise 0.8s ease both;
}

.metric-card:nth-child(2) { animation-delay: 0.08s; }
.metric-card:nth-child(3) { animation-delay: 0.16s; }

.metric-label {
  color: var(--muted);
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.13em;
}

.metric-value {
  margin-top: 12px;
  font-size: 1.9rem;
  font-weight: 800;
  font-family: 'Space Grotesk', sans-serif;
}

.metric-copy {
  margin-top: 8px;
  color: var(--muted);
  font-size: 0.95rem;
}

.section-card,
.result-card {
  border: 1px solid var(--line);
  border-radius: 24px;
  background: var(--panel);
  backdrop-filter: blur(14px);
  box-shadow: var(--shadow);
}

.section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.1rem;
  letter-spacing: -0.02em;
}

.status-shell {
  padding: 18px 20px;
  border-radius: 20px;
  border: 1px solid rgba(126, 240, 194, 0.15);
  background: linear-gradient(135deg, rgba(126, 240, 194, 0.08), rgba(121, 168, 255, 0.07));
}

.status-shell[data-state="warning"] {
  border-color: rgba(255, 185, 94, 0.2);
  background: linear-gradient(135deg, rgba(255, 185, 94, 0.10), rgba(121, 168, 255, 0.05));
}

.status-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.2rem;
  margin-bottom: 6px;
}

.status-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
}

.status-chip {
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 0.9rem;
  color: var(--text);
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.results-markdown,
.results-json textarea,
.results-text textarea {
  color: var(--text) !important;
}

.results-json,
.results-text,
.results-markdown {
  border-radius: 20px !important;
}

.primary-button {
  background: linear-gradient(135deg, var(--blue), var(--cyan)) !important;
  border: 0 !important;
  color: #06101d !important;
  font-weight: 800 !important;
  box-shadow: 0 18px 35px rgba(121, 168, 255, 0.28);
}

.secondary-button {
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(255, 255, 255, 0.1) !important;
  color: var(--text) !important;
}

@keyframes drift {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(-12px, 18px, 0) scale(1.05); }
}

@keyframes rise {
  from {
    opacity: 0;
    transform: translateY(16px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 900px) {
  .app-shell {
    padding: 18px;
    border-radius: 24px;
  }

  .hero-grid {
    grid-template-columns: 1fr;
  }
}
"""

HERO_HTML = """
<section class="hero">
  <div class="eyebrow">Animated Vision Console</div>
  <h1>Traffic Sign Intelligence, Reframed for Demo Day.</h1>
  <p>
    Upload a road-sign image and run your YOLO + PaddleOCR pipeline inside a cinematic interface
    with live controls, structured text extraction, and visual overlays built for fast review.
  </p>
  <div class="hero-grid">
    <div class="metric-card">
      <div class="metric-label">Detection + OCR</div>
      <div class="metric-value">Hybrid</div>
      <div class="metric-copy">YOLO-assisted region proposals feeding an OCR-first extraction flow.</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Interface Style</div>
      <div class="metric-value">Animated</div>
      <div class="metric-copy">Glassmorphism panels, motion accents, and presentation-ready output cards.</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Deployment Feel</div>
      <div class="metric-value">One Command</div>
      <div class="metric-copy">Launch locally, upload an image, and inspect structured sign text instantly.</div>
    </div>
  </div>
</section>
"""


def parse_extra_terms(value: str) -> List[str]:
    return [term.strip() for term in value.split(",") if term.strip()]


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def annotate_image(image: np.ndarray, raw_items: Sequence[dict]) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for index, item in enumerate(raw_items, start=1):
        box = item.get("box") or []
        points = np.array(box, dtype=np.int32)
        if points.size == 0:
            continue
        cv2.polylines(canvas, [points], isClosed=True, color=(110, 231, 242), thickness=2)
        x, y = points[0]
        label = str(item.get("text", f"ROI {index}"))
        confidence = float(item.get("confidence", 0.0))
        tag = f"{label} ({confidence:.2f})"
        cv2.putText(
            canvas,
            tag[:44],
            (int(x), max(18, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (121, 168, 255),
            2,
            cv2.LINE_AA,
        )

    return to_rgb(canvas)


def format_cleaned_text(lines: Iterable[str]) -> str:
    cleaned = list(lines)
    if not cleaned:
        return "No cleaned text lines were produced."
    return "\n".join(f"{idx}. {line}" for idx, line in enumerate(cleaned, start=1))


def build_status_html(result: dict) -> str:
    raw_count = len(result.get("raw_ocr", []))
    cleaned_count = len(result.get("cleaned_text", []))
    structured_entries = result.get("structured_entries", [])
    structured = result.get("structured_output") or "No structured route pairs extracted."
    failed = bool(result.get("ocr_failed"))
    partial = (not failed) and cleaned_count > 0 and not any(entry.get("distance") is not None for entry in structured_entries)
    state = "warning" if failed or partial else "success"
    title = "Partial extraction" if partial else ("OCR needs attention" if failed else "Inference completed")
    copy = (
        "The pipeline ran, but OCR did not return confident text. Try a sharper crop or a different image."
        if failed
        else (
            "OCR found candidate text, but it could not be converted into a confident place-distance pair."
            if partial
            else "The pipeline finished successfully and extracted candidate guide-sign text."
        )
    )
    return f"""
    <div class="status-shell" data-state="{state}">
      <div class="status-title">{title}</div>
      <div>{copy}</div>
      <div class="status-meta">
        <span class="status-chip">Raw OCR items: {raw_count}</span>
        <span class="status-chip">Cleaned lines: {cleaned_count}</span>
        <span class="status-chip">Structured output: {structured}</span>
      </div>
    </div>
    """


def run_inference(
    image: str | None,
    yolo_conf: float,
    extra_terms: str,
    use_detector: bool,
    use_segmentation: bool,
    accurate_mode: bool,
    ocr_langs: List[str],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if not image:
        raise gr.Error("Upload an image before running the pipeline.")

    progress(0.12, desc="Loading pipeline assets")
    image_path = Path(image)
    source_image = load_image(str(image_path))
    extra_terms_list = parse_extra_terms(extra_terms)
    yolo_model_path = str(DEFAULT_YOLO_MODEL) if use_detector and DEFAULT_YOLO_MODEL.exists() else None

    ocr_engine = PaddleGuideSignOCR(
        use_angle_cls=False,
        lang=(ocr_langs[0] if ocr_langs else "en"),
        langs=parse_ocr_langs(ocr_langs),
        det_model_dir=str(DEFAULT_OCR_DET_DIR),
        rec_model_dir=None,
        cls_model_dir=str(DEFAULT_OCR_CLS_DIR),
    )

    progress(0.48, desc="Running detection and OCR")
    raw_items = collect_ocr_candidates(
        image_path=str(image_path),
        yolo_model_path=yolo_model_path,
        yolo_conf=yolo_conf,
        ocr_engine=ocr_engine,
        use_segmentation=use_segmentation,
        segmentation_backend=DEFAULT_SEGMENTATION_BACKEND,
        fast_mode=not accurate_mode,
    )
    result = build_output(raw_items, extra_terms=extra_terms_list)

    result_path = APP_OUTPUT_DIR / f"{image_path.stem}_ui_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    progress(0.82, desc="Rendering dashboard")
    annotated = annotate_image(source_image, result["raw_ocr"])
    cleaned_text = format_cleaned_text(result["cleaned_text"])
    structured_output = result["structured_output"] or "No structured route-distance pairs extracted."
    raw_json = json.dumps(result["raw_ocr"], indent=2, ensure_ascii=False)
    status_html = build_status_html(result)
    result_file = str(result_path)

    progress(1.0, desc="Ready")
    return annotated, cleaned_text, structured_output, raw_json, status_html, result_file


def load_demo():
    if DEFAULT_IMAGE_PATH.exists():
        return str(DEFAULT_IMAGE_PATH)
    return None


with gr.Blocks(
    css=CSS,
    title="Traffic Sign Detection UI",
) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(HERO_HTML)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["section-card"]):
                gr.Markdown("### Input Console", elem_classes=["section-title"])
                input_image = gr.Image(
                    type="filepath",
                    label="Road Sign Image",
                    sources=["upload"],
                    value=load_demo(),
                    height=420,
                )
                with gr.Row():
                    yolo_conf = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.25,
                        step=0.05,
                        label="YOLO Confidence Threshold",
                    )
                    use_detector = gr.Checkbox(
                        value=DEFAULT_YOLO_MODEL.exists(),
                        label="Use YOLO region proposals",
                    )
                    use_segmentation = gr.Checkbox(
                        value=True,
                        label="Use segmentation refinement",
                    )
                extra_terms = gr.Textbox(
                    label="Extra vocabulary hints",
                    placeholder="Delhi, Chandigarh, Expressway, NH44",
                    lines=2,
                )
                ocr_langs = gr.CheckboxGroup(
                    choices=[("English", "en"), ("Hindi", "hi"), ("Punjabi", "pa"), ("Chinese", "ch")],
                    value=["en"],
                    label="OCR languages",
                    info="Punjabi is exposed as an optional lane. If no local/official recognizer is available, it will be skipped safely.",
                )
                accurate_mode = gr.Checkbox(
                    value=False,
                    label="Accurate mode (slower)",
                )
                with gr.Row():
                    run_button = gr.Button("Run Animated Inference", elem_classes=["primary-button"])
                    clear_button = gr.Button("Reset", elem_classes=["secondary-button"])

            with gr.Column(scale=6):
                status_html = gr.HTML(
                    value=build_status_html({"raw_ocr": [], "cleaned_text": [], "structured_output": "", "ocr_failed": False}),
                    elem_classes=["result-card"],
                )
                preview_image = gr.Image(
                    label="Annotated OCR Preview",
                    interactive=False,
                    height=420,
                )

        with gr.Row(equal_height=True):
            cleaned_text_box = gr.Textbox(
                label="Cleaned OCR Lines",
                lines=8,
                elem_classes=["results-text", "result-card"],
            )
            structured_output_box = gr.Textbox(
                label="Structured Output",
                lines=8,
                elem_classes=["results-text", "result-card"],
            )

        with gr.Row(equal_height=True):
            raw_json_box = gr.Code(
                label="Raw OCR JSON",
                language="json",
                elem_classes=["results-json", "result-card"],
            )
            result_file = gr.File(label="Saved Result JSON", elem_classes=["result-card"])

        run_button.click(
            fn=run_inference,
            inputs=[input_image, yolo_conf, extra_terms, use_detector, use_segmentation, accurate_mode, ocr_langs],
            outputs=[preview_image, cleaned_text_box, structured_output_box, raw_json_box, status_html, result_file],
        )

        clear_button.click(
            fn=lambda: (None, "", "", "", build_status_html({"raw_ocr": [], "cleaned_text": [], "structured_output": "", "ocr_failed": False}), None),
            outputs=[preview_image, cleaned_text_box, structured_output_box, raw_json_box, status_html, result_file],
        )


if __name__ == "__main__":
    demo.launch()
