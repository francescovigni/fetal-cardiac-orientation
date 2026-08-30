#!/usr/bin/env bash
# FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement
# Zenodo 14597550, CC-BY-4.0. 300 images, oriented boxes + ellipses + masks for cardiac & thorax.
set -euo pipefail
RAW="$(dirname "$0")/../data/raw"
mkdir -p "$RAW"
if [ ! -d "$RAW/FOCUS" ]; then
  echo "downloading FOCUS (58 MB)…"
  curl -L -o "$RAW/FOCUS-dataset.zip" "https://zenodo.org/records/14597550/files/FOCUS-dataset.zip?download=1"
  unzip -q -o "$RAW/FOCUS-dataset.zip" -d "$RAW/FOCUS"
fi
echo "FOCUS ready:"
for s in training validation testing; do
  echo "  $s: $(ls "$RAW/FOCUS/$s/images" | wc -l | tr -d ' ') images"
done
