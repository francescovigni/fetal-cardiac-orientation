#!/usr/bin/env bash
# Fine-tune YOLOv5s on FOCUS cardiac/thorax boxes.
# 200 training images is small: heavy augmentation off (ultrasound is not photography),
# rotation ON because fetal lie is arbitrary, mosaic ON to synthesise context.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
python "$R/yolov5/train.py" \
  --img 640 --batch 8 --epochs 150 \
  --data "$R/data/processed/yolo/focus.yaml" \
  --hyp "$R/configs/hyp.focus.yaml" \
  --weights yolov5s.pt \
  --project "$R/runs/yolo" --name focus --exist-ok \
  --patience 40
