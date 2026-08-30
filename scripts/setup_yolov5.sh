#!/usr/bin/env bash
# YOLOv5 is used as-is from the upstream repo (the v5 line has no pip package of its own).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -d "$R/yolov5" ]; then
  git clone --depth 1 https://github.com/ultralytics/yolov5 "$R/yolov5"
fi
pip install -r "$R/yolov5/requirements.txt"

# YOLOv5 forces torch.use_deterministic_algorithms(True); Apple's MPS backend has no
# deterministic index_put_, so training dies on the first backward pass. warn_only=True
# keeps the seeding and drops the hard failure.
sed -i '' 's/        torch.use_deterministic_algorithms(True)$/        torch.use_deterministic_algorithms(True, warn_only=True)/' "$R/yolov5/utils/general.py" 2>/dev/null || \
sed -i 's/        torch.use_deterministic_algorithms(True)$/        torch.use_deterministic_algorithms(True, warn_only=True)/' "$R/yolov5/utils/general.py"
echo "yolov5 ready at $R/yolov5"
