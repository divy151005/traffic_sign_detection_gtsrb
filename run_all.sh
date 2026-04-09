#!/bin/zsh

# Launcher for the traffic sign detection project.
# Usage:
#   ./run_all.sh                -> launches animated Gradio UI
#   ./run_all.sh --ui           -> launches animated Gradio UI
#   ./run_all.sh --cli ...args  -> runs main.py with CLI arguments
#   ./run_all.sh --cli --input-dir data/Test -> batch mode

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
cd $SCRIPT_DIR

echo "🚀 Setting up virtual environment..."

# Create/activate venv if not exists
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install opencv-python ultralytics paddlepaddle paddleocr numpy pillow torch torchvision

MODE="ui"

if [ "$#" -gt 0 ] && [ "$1" = "--cli" ]; then
  MODE="cli"
  shift
elif [ "$#" -gt 0 ] && [ "$1" = "--ui" ]; then
  shift
fi

if [ "$MODE" = "ui" ]; then
  echo "🎬 Launching animated UI..."
  python app.py "$@"
  echo "✅ UI session ended."
else
  echo "⚙️ Running main pipeline..."
  python main.py "$@"
  echo "✅ Pipeline complete. Check outputs/ocr_primary/ for results."
  echo "💡 Fast single image: ./run_all.sh --cli --image data/Test/7.png --yolo-conf 0.3"
  echo "💡 Higher accuracy:   ./run_all.sh --cli --image data/Test/7.png --accurate-mode --ocr-langs en hi"
  echo "💡 Batch mode:   ./run_all.sh --cli --input-dir data/Test"
fi
