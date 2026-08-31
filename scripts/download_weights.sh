#!/usr/bin/env bash
# Fetch the trained checkpoints from a GitHub release, so the pipeline runs without
# reproducing ~40 minutes of training.  Pass a tag to pin a version, e.g. v0.1.0.
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${FHO_REPO:-francescovigni/fetal-cardiac-orientation}"
TAG="${1:-}"

mkdir -p "$R/runs/landmarks" "$R/runs/yolo/focus/weights" "$R/runs/yolo/focus_rot180/weights"

if [ -n "$TAG" ]; then
  URL_BASE="https://github.com/$REPO/releases/download/$TAG"
else
  URL_BASE="https://github.com/$REPO/releases/latest/download"
fi

fetch () {   # $1 asset name, $2 destination
  if [ -s "$2" ]; then echo "have $(basename "$2")"; return; fi
  echo "downloading $1 …"
  if command -v gh >/dev/null 2>&1; then
    # `gh release download` takes no tag when the latest release is wanted
    if [ -n "$TAG" ]; then
      gh release download "$TAG" --repo "$REPO" --pattern "$1" --output "$2" --clobber
    else
      gh release download --repo "$REPO" --pattern "$1" --output "$2" --clobber
    fi
  else
    curl -fL --retry 3 -o "$2" "$URL_BASE/$1"
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
