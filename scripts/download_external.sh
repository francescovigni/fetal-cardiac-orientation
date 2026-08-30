#!/usr/bin/env bash
# FETAL_PLANES_DB — Zenodo 3904280, CC-BY-4.0, 2.1 GB, 12,400 maternal-fetal images.
# Used only as an unlabelled external set: 1,718 images are labelled 'Fetal thorax'
# (the four-chamber plane), and the CSV records the ultrasound machine per image.
set -euo pipefail
RAW="$(dirname "$0")/../data/raw"
mkdir -p "$RAW"
if [ ! -d "$RAW/FETAL_PLANES" ]; then
  echo "downloading FETAL_PLANES_DB (2.1 GB)…"
  curl -L -o "$RAW/FETAL_PLANES_ZENODO.zip" \
    "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
  unzip -q -o "$RAW/FETAL_PLANES_ZENODO.zip" -d "$RAW/FETAL_PLANES"
fi
python3 - "$RAW/FETAL_PLANES/FETAL_PLANES_DB_data.csv" <<'PY'
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1]), delimiter=";"))
th = [r for r in rows if r["Plane"].strip() == "Fetal thorax"]
print(f"FETAL_PLANES ready: {len(rows)} images, {len(th)} labelled 'Fetal thorax'")
print("  machines:", dict(Counter(r["US_Machine"].strip() for r in th)))
PY
