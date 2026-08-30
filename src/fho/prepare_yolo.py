"""Build a YOLOv5 dataset from FOCUS.

FOCUS ships *oriented* boxes; YOLOv5 consumes axis-aligned ones, so the OBB is
collapsed to its enclosing AABB here.  The orientation is not thrown away — it is
the target of stage 2, and stage 1 only has to find the organ.

    python -m fho.prepare_yolo --raw data/raw/FOCUS --out data/processed/yolo
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .focus import LABELS, load_split

CLASS_ID = {name: i for i, name in enumerate(LABELS)}


def obb_to_aabb(corners: np.ndarray) -> tuple[float, float, float, float]:
    return (float(corners[:, 0].min()), float(corners[:, 1].min()),
            float(corners[:, 0].max()), float(corners[:, 1].max()))


def main(raw: Path, out: Path) -> None:
    out = Path(out)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val", "test"):
        samples = load_split(raw, split)
        n_boxes = 0
        for s in samples:
            w, h = Image.open(s.image_path).size
            dst = out / "images" / split / f"{s.stem}.png"
            if not dst.exists():
                shutil.copy(s.image_path, dst)
            lines = []
            for label, corners in s.obbs.items():
                if label not in CLASS_ID:
                    continue
                x0, y0, x1, y1 = obb_to_aabb(corners)
                # clip: a few thorax boxes extend past the image border
                x0, y0 = max(x0, 0.0), max(y0, 0.0)
                x1, y1 = min(x1, w - 1.0), min(y1, h - 1.0)
                if x1 <= x0 or y1 <= y0:
                    continue
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                bw, bh = (x1 - x0) / w, (y1 - y0) / h
                lines.append(f"{CLASS_ID[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_boxes += 1
            (out / "labels" / split / f"{s.stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] = (len(samples), n_boxes)

    yaml = out / "focus.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for n, i in CLASS_ID.items())
    )
    for split, (n_img, n_box) in counts.items():
        print(f"{split:5s} {n_img:4d} images  {n_box:4d} boxes")
    print("wrote", yaml)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--out", default="data/processed/yolo")
    a = p.parse_args()
    main(Path(a.raw), Path(a.out))
