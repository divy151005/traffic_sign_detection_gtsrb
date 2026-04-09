**Colab Setup**

Run the following in a Colab cell:

```python
!git clone <your-repo-url>
%cd traffic_sign_detection_gtsrb
!pip install ultralytics easyocr pytesseract opencv-python-headless torch torchvision
```

If you want Tesseract fallback:

```python
!apt-get update -qq
!apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-pan
```

Upload or mount your weights and sample image, then run:

```python
!python main.py \
  --image data/1.jpeg \
  --yolo-model models/yolov8_traffic_sign_board.pt \
  --segmentation-backend auto \
  --yolo-seg-model models/yolov8_traffic_sign_board_seg.pt \
  --languages en hi pa \
  --save-segmented \
  --visualize-masks
```

**Notes**

- `auto` tries Mask R-CNN first, then YOLOv8-seg if provided, then a classical ROI fallback.
- For segmentation IoU evaluation, place masks in a directory and name them `sign_00_mask.png`, `sign_01_mask.png`, etc.
- For OCR accuracy, pass reference strings with `--gt-texts`.
