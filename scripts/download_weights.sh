#!/usr/bin/env bash
# Fetch the trained checkpoints from the latest GitHub release, so the pipeline can
# be run without reproducing ~40 minutes of training.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${FHO_REPO:-francescovigni/fetal-cardiac-orientation}"
TAG="${1:-latest}"

mkdir -p "$R/runs/landmarks" "$R/runs/yolo/focus/weights" "$R/runs/yolo/focus_rot180/weights"

fetch () {   # $1 asset name, $2 destination
  if [ -f "$2" ]; then echo "have $(basename "$2")"; return; fi
  echo "downloading $1 …"
  if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$1" --output "$2" --clobber
  else
    base="https://github.com/$REPO/releases/${TAG/latest/latest/download}"
    [ "$TAG" = "latest" ] || base="https://github.com/$REPO/releases/download/$TAG"
    curl -fL -o "$2" "$base/$1"
  fi
}

fetch landmarks_best.pt        "$R/runs/landmarks/landmarks_best.pt"
fetch detector_rot180_best.pt  "$R/runs/yolo/focus_rot180/weights/best.pt"
fetch detector_rot30_best.pt   "$R/runs/yolo/focus/weights/best.pt"

echo
echo "checkpoints ready:"
ls -lh "$R/runs/landmarks/landmarks_best.pt" \
       "$R/runs/yolo/focus_rot180/weights/best.pt" \
       "$R/runs/yolo/focus/weights/best.pt" | awk '{print "  "$5"\t"$9}'
