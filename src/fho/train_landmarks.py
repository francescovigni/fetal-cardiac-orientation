"""Train the landmark model on FOCUS crops.

    python -m fho.train_landmarks --raw data/raw/FOCUS --epochs 200

Crops are taken from the *ground-truth* heart box during training and from the
YOLO detection at inference.  That mismatch is deliberate and is measured in
evaluate.py (``--source yolo`` vs ``--source gt``): the difference between the two
is the error stage 1 contributes to the final angle, and reporting it separately
is the only way to know which stage to work on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import geometry as G
from .focus import load_split
from .landmarks import K, CropSpec, make_example
from .model import LandmarkNet, Loss, axial_angle_from_coords

STRIDE = 4


class FocusCrops(Dataset):
    def __init__(self, raw, split, spec: CropSpec, train: bool, seed: int = 0):
        self.samples = load_split(raw, split)
        self.spec, self.train = spec, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        if self.train:
            ex = make_example(
                s, self.spec, jitter=self.rng,
                rot_deg=float(self.rng.uniform(-180, 180)),
                gain=(float(np.exp(self.rng.normal(0, 0.20))),
                      float(self.rng.normal(0, 0.05))),
                flip=bool(self.rng.random() < 0.5),
            )
        else:
            ex = make_example(s, self.spec)
        return (torch.from_numpy(ex["image"])[None],
                torch.from_numpy(ex["landmarks"] / STRIDE),
                torch.tensor(ex["gt_angle"], dtype=torch.float32))


def angles_from_coords(coords: torch.Tensor) -> np.ndarray:
    z = axial_angle_from_coords(coords).detach().cpu().numpy()
    return G.decode_axial(z)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    errs, preds, gts = [], [], []
    for x, xy_t, gt in loader:
        _, coords = model(x.to(device))
        a = angles_from_coords(coords)
        errs.append(G.angdiff_axial(a, gt.numpy()))
        preds.append(a)
        gts.append(gt.numpy())
    e = np.concatenate(errs)
    return dict(median=float(np.median(e)), mean=float(e.mean()),
                p90=float(np.percentile(e, 90)), max=float(e.max()),
                frac_gt10=float(np.mean(e > 10)),
                preds=np.concatenate(preds).tolist(), gts=np.concatenate(gts).tolist())


def main(a):
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    spec = CropSpec(size=a.size, margin=a.margin)
    tr = DataLoader(FocusCrops(a.raw, "train", spec, True), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers, drop_last=True)
    va = DataLoader(FocusCrops(a.raw, "val", spec, False), batch_size=a.batch,
                    num_workers=a.workers)

    model = LandmarkNet(K, a.width).to(device)
    crit = Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for ep in range(1, a.epochs + 1):
        model.train()
        run = {}
        for x, xy_t, _ in tr:
            x, xy_t = x.to(device), xy_t.to(device)
            axis, coords = model(x)
            loss, parts = crit(axis, coords, xy_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v / len(tr)
        if ep % a.eval_every == 0 or ep == a.epochs:
            m = evaluate(model, va, device)
            history.append(dict(epoch=ep, **{k: m[k] for k in
                                             ("median", "mean", "p90", "frac_gt10")}, **run))
            print(f"ep {ep:4d}  xy {run['xy']:.3f}  ang {run['ang']:.4f}  dir {run['dir']:.4f}  |  "
                  f"val median {m['median']:5.2f}°  p90 {m['p90']:6.2f}°  "
                  f">10° {100*m['frac_gt10']:4.1f}%")
            if m["median"] < best:
                best = m["median"]
                torch.save(dict(model=model.state_dict(), spec=vars(spec),
                                width=a.width, val_median=best), out / "landmarks_best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val median angular error: {best:.2f}°   -> {out/'landmarks_best.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--out", default="runs/landmarks")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--margin", type=float, default=0.35)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    main(p.parse_args())
