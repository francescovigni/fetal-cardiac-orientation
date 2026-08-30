"""End-to-end smoke tests on synthetic data — no download, no GPU, no network."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_yolo_label_export_is_normalised_and_in_range(fake_focus, tmp_path):
    from fho.prepare_yolo import main as prepare

    out = tmp_path / "yolo"
    prepare(fake_focus, out)
    for split in ("train", "val", "test"):
        labels = sorted((out / "labels" / split).glob("*.txt"))
        assert len(labels) == 4
        for f in labels:
            for line in f.read_text().strip().splitlines():
                cls, cx, cy, w, h = line.split()
                assert cls in {"0", "1"}
                assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))
    assert (out / "focus.yaml").exists()


def test_training_loop_runs_and_reduces_the_angular_loss(fake_focus, tmp_path):
    """Two epochs on four synthetic images: enough to catch a broken loop."""
    from argparse import Namespace

    from fho.train_landmarks import main as train

    args = Namespace(
        raw=fake_focus,
        out=str(tmp_path / "run"),
        epochs=2,
        batch=2,
        lr=1e-3,
        size=192,
        margin=0.35,
        width=8,
        workers=0,
        eval_every=1,
    )
    train(args)
    ckpt = tmp_path / "run" / "landmarks_best.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert {"model", "spec", "width", "val_median"} <= set(state)
    assert np.isfinite(state["val_median"])


def test_orientation_from_a_box_returns_a_complete_record(fake_focus, tmp_path):
    """predict.orientation_from_box, exercised without the detector."""
    import cv2

    from fho.focus import load_split
    from fho.landmarks import CropSpec
    from fho.model import LandmarkNet
    from fho.predict import orientation_from_box

    torch.manual_seed(0)
    model = LandmarkNet(4, 8)
    model.eval()
    s = load_split(fake_focus, "test")[0]
    img = cv2.imread(str(s.image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    x0, y0, x1, y1 = s.cardiac.aabb()

    r = orientation_from_box(model, CropSpec(), img, (x0, y0, x1, y1))
    assert 0.0 <= r["angle_deg"] < 180.0
    assert len(r["landmarks"]) == 4
    assert isinstance(r["assessable"], bool)
    assert np.isfinite(r["roundness"])
