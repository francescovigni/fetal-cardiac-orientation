"""Measured computational cost of every stage.

Reports parameters, weight size and latency for the learned route, and latency
for the closed-form route, on whatever device is available.  Numbers quoted in
the README come from this module, so they can be re-measured rather than trusted.

The closed-form estimators are reported twice: as implemented here, and against
the OpenCV equivalents.  ``geometry.py`` deliberately imports no cv2 — its PCA
takes probability weights and physical pixel spacing, which ``cv2.moments`` does
not — and its convex hull is a pure-Python monotone chain written for clarity.
Quoting only the local timings would make rotating calipers look inherently
expensive, which it is not.

    python -m fho.bench            # learned + closed-form
    python -m fho.bench --detector runs/yolo/focus/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .no_training import cleanup


def timeit(fn, n: int = 100, warmup: int = 10) -> float:
    """Mean wall-clock milliseconds per call."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def _reference_mask(shape=(663, 961), a=90, b=62, theta=45.0):
    """A cardiac-sized ellipse in a full-resolution frame."""
    img = np.zeros(shape, np.uint8)
    cv2.ellipse(img, (shape[1] // 2, shape[0] // 2), (a, b), theta, 0, 360, 255, -1)
    return (img > 0).astype(np.uint8), float(theta)


def bench_landmark(size: int = 192, width: int = 32, batch: int = 32) -> dict:
    import torch

    from .model import LandmarkNet

    model = LandmarkNet(4, width).eval()
    params = sum(p.numel() for p in model.parameters())
    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    x = torch.randn(1, 1, size, size)
    with torch.no_grad():
        cpu_ms = timeit(lambda: model(x), n=50)
        xb = torch.randn(batch, 1, size, size)
        batch_ms = timeit(lambda: model(xb), n=10, warmup=2)

        device, dev_ms = "cpu", None
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        if device != "cpu":
            m2, x2 = model.to(device), x.to(device)
            dev_ms = timeit(lambda: m2(x2), n=50)
            model.to("cpu")

    return dict(
        params_m=params / 1e6,
        weights_mb=weights_mb,
        crop_px=size,
        cpu_ms=cpu_ms,
        device=device,
        device_ms=dev_ms,
        batch=batch,
        batch_ms=batch_ms,
        crops_per_s=batch / batch_ms * 1000,
    )


def bench_closed_form() -> dict:
    mask, truth = _reference_mask()
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)

    def cv_moments():
        M = cv2.moments(mask, binaryImage=True)
        mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
        return float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % 180.0)

    return dict(
        set_pixels=int(mask.sum()),
        truth_deg=truth,
        pca_ms=timeit(lambda: G.mask_axis(mask), n=200),
        pca_cv2_ms=timeit(cv_moments, n=200),
        pca_error_deg=float(G.angdiff_axial(G.mask_axis(mask)["angle"], truth)),
        cleanup_ms=timeit(lambda: cleanup(mask), n=200),
        minarea_ms=timeit(lambda: G.minarea_axis(pts), n=20, warmup=2),
        minarea_cv2_ms=timeit(lambda: cv2.minAreaRect(pts), n=200),
        minarea_error_deg=float(G.angdiff_axial(G.minarea_axis(pts)["angle"], truth)),
    )


def bench_detector(weights: Path, repo: Path, size: int = 640) -> dict | None:
    import torch

    if not weights.exists():
        return None
    det = torch.hub.load(str(repo), "custom", path=str(weights), source="local", verbose=False)
    n_params = sum(p.numel() for p in det.model.parameters())
    img = (np.random.default_rng(0).random((663, 961, 3)) * 255).astype(np.uint8)
    return dict(params_m=n_params / 1e6, input_px=size, ms=timeit(lambda: det(img), n=20, warmup=5))


def report(land: dict, closed: dict, det: dict | None) -> str:
    out = ["Computational cost", ""]
    if det:
        out += [
            f"  detector       {det['params_m']:6.2f} M params   "
            f"{det['ms']:7.2f} ms / image @ {det['input_px']} px"
        ]
    d = f"{land['device_ms']:7.2f} ms ({land['device']})" if land["device_ms"] else "n/a"
    out += [
        f"  landmark net   {land['params_m']:6.2f} M params   "
        f"{land['cpu_ms']:7.2f} ms (cpu)   {d}   weights {land['weights_mb']:.1f} MB",
        f"                 batch-{land['batch']} cpu {land['batch_ms']:.1f} ms "
        f"-> {land['crops_per_s']:.0f} crops/s",
        "",
        f"  closed form, on a {closed['set_pixels']} px mask:",
        f"    moments (ours)      {closed['pca_ms']:7.3f} ms   error "
        f"{closed['pca_error_deg']:.2f} deg",
        f"    moments (cv2)       {closed['pca_cv2_ms']:7.3f} ms",
        f"    cleanup             {closed['cleanup_ms']:7.3f} ms",
        f"    min-area (ours)     {closed['minarea_ms']:7.3f} ms   error "
        f"{closed['minarea_error_deg']:.2f} deg",
        f"    min-area (cv2)      {closed['minarea_cv2_ms']:7.3f} ms",
    ]
    total = closed["pca_cv2_ms"] + closed["cleanup_ms"]
    out += ["", f"  orientation from an existing mask: {total:.2f} ms end to end"]
    if det and land["device_ms"]:
        out.append(
            f"  orientation from a raw frame:      "
            f"{det['ms'] + land['device_ms']:.1f} ms end to end"
        )
    return "\n".join(out)


def main(a):
    land = bench_landmark(size=a.size, width=a.width, batch=a.batch)
    closed = bench_closed_form()
    det = bench_detector(Path(a.detector), Path(a.repo)) if a.detector else None
    print(report(land, closed, det))
    if a.json:
        Path(a.json).write_text(
            json.dumps(dict(landmark=land, closed_form=closed, detector=det), indent=1)
        )
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--detector", default="")
    p.add_argument("--repo", default="yolov5")
    p.add_argument("--json", default="runs/bench.json")
    main(p.parse_args())
