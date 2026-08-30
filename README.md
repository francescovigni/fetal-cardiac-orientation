# fetal-cardiac-orientation

Detection and orientation estimation on prenatal four-chamber ultrasound, in two stages: a YOLOv5 detector for the cardiac and thoracic regions, then a landmark-regression model for the heart's long axis. Public data ([FOCUS](https://zenodo.org/records/14597550), CC-BY-4.0), reproducible from a clean clone.

![Predicted heart long axis against the annotation, best / median / worst test cases](docs/figures/qualitative.png)

The reasoning behind the design choices, the three failures that produced them, and how to read the numbers: **[docs/article.md](docs/article.md)**.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data          # download FOCUS (58 MB) into data/raw
make test          # 9 unit tests on the angle algebra
make baselines     # closed-form estimators, no training, no GPU
make yolo          # clone + patch yolov5, prepare labels, train the detector
make train         # train the orientation model  (~25 min on an M-series GPU)
make eval meta     # agreement statistics, then the label-free metamorphic suite
make data-external external   # unlabelled external test on FETAL_PLANES_DB (2.1 GB)
make figures       # regenerate every figure in docs/figures from the checkpoints
```

Device is picked automatically: MPS, CUDA, or CPU.

## Results

Held-out test split, 50 images. Trained on 200.

**Detection** — YOLOv5s, 150 epochs, 640 px, 14 ms/image.

| Class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| cardiac | 0.990 | 1.000 | 0.995 | 0.637 |
| thorax | 0.999 | 1.000 | 0.995 | 0.660 |

Table stakes: one organ, one view, centred, always present.

**Orientation** — landmark regression, 400 epochs.

| | |
|---|---|
| median absolute error | 7.04° (95 % CI 4.84–9.26) |
| p90 | 12.92° |
| Bland-Altman bias | −0.55° |
| 95 % limits of agreement | −18.23° to +17.12° |
| ICC(2,1) | 0.980 |

![Bland-Altman and error distribution](docs/figures/agreement.png)

Unbiased, but the limits of agreement are the number that counts: a single scan can be misplaced by ±18° against a clinical normal band roughly 40° wide. Validation median was 4.41°, so there is a real generalisation gap on 200 training images, and training had not converged.

**Classical baselines** on the ground-truth masks:

| Estimator | median | p90 | >10° |
|---|---|---|---|
| PCA / image moments | 0.28° | 0.45° | 0 % |
| minimum-area rectangle | 4.95° | 88.42° | 44 % |

Minimum-area fails because for an ellipse the enclosing rectangle reaches the same minimum area at both the major and the minor alignment, so the estimator is bimodal and picks one of two optima 90° apart. It is exact for area and unstable for axis. Note the masks are rasterised from the same ellipse annotations, so 0.28° measures geometry-code consistency, not accuracy.

**Metamorphic tests**, no labels used:

| Property | median | p90 |
|---|---|---|
| rotation equivariance ±15° | 2.9–3.6° | 7.4–9.0° |
| rotation equivariance ±30° | 3.7–4.7° | 10.0–12.5° |
| mirror equivariance | 2.55° | 12.00° |
| gain invariance | 0.74° | 3.90° |
| crop-scale invariance | 3.65° | 8.63° |

All fail the tolerances set in the file; the tolerances were not relaxed. Gain invariance is the actionable one: a pure brightness change moves an anatomical measurement by up to 5°.

**Abstention.** None of the model's internal confidence signals buys much accuracy. The best of the candidates is simply *heart size*, not head agreement and not predicted elongation — and predicted elongation is worse than useless, since abstaining by it makes the median error rise.

![Risk-coverage curves for each candidate confidence signal](docs/figures/risk_coverage.png)

## External evaluation, without labels

FOCUS has orientation ground truth; almost no other public fetal dataset does. That does not block an external test, because the properties the estimator must satisfy need no labels at all. The same metamorphic suite runs on [FETAL_PLANES_DB](https://zenodo.org/records/3904280) (Zenodo 3904280, CC-BY-4.0), a different hospital, different operators and four ultrasound machines, using its 1,718 `Fetal thorax` images.

```bash
make external      # 500 images, stratified by ultrasound machine
```

Detection fires on **93 %** of external thorax images at mean confidence 0.75, without ever having seen this dataset.

![Internal versus external self-consistency, and detection by machine](docs/figures/external.png)

| Property | FOCUS median | external median | FOCUS p90 | external p90 |
|---|---|---|---|---|
| rotation ±15° | 2.9–3.6° | 3.9–4.3° | 7.4–9.0° | 20.5–21.9° |
| rotation ±30° | 3.7–4.7° | 5.3–6.5° | 10.0–12.5° | 29.2–39.6° |
| mirror | 2.55° | 6.51° | 12.00° | 44.28° |
| gain | 0.74° | 1.96° | 3.90° | 8.03° |
| crop scale | 3.65° | 11.82° | 8.63° | 43.96° |

**The medians move a little and the tails triple.** That is the shape of a model that still works on typical external images and fails outright on a substantial minority — a distinction an average would erase. Crop-scale sensitivity is the worst of it, 3.65° to 11.82°, which says the model has partly learned the FOCUS crop convention rather than the anatomy.

By machine, the detector fires on 95 % of Voluson E6 images but only **81 % of Aloka** images, at the lowest mean confidence of the four. Aloka is 41 % of the source dataset and is absent from FOCUS.

None of this needed a single annotation.

## Data

FOCUS, 300 prenatal four-chamber images (200/50/50), grayscale, ~961×663. Each image carries three parallel annotations for `cardiac` and `thorax`:

```
annfiles_ellipse/NNN.txt      cx cy a b theta_deg label       (a = semi-major)
annfiles_rectangle/NNN.txt    x1 y1 … x4 y4 label difficulty  (DOTA-style oriented box)
annfiles_mask/NNN-{cardiac,thorax}.png
```

`focus.verify_consistency` cross-checks the ellipse parameters against the independent oriented boxes across all 200 training images before either is trusted: centre agrees to 0.07 px, semi-major to 0.09 px, angle to **0.033°**. That is what licenses using the ellipse angle as ground truth.

Rejected alternatives: FETAL_PLANES_DB (12,400 images, class labels only, no geometry); CAMUS and EchoNet-Dynamic (adult echocardiography).

## Design notes

Oriented boxes are collapsed to axis-aligned for YOLOv5, which only has to find the organ. Orientation is the target of stage two.

Augmentation departs from the YOLOv5 defaults on physical grounds (`configs/hyp.focus.yaml`): hue and saturation off (grayscale), brightness kept (gain varies between machines), rotation ±30° (fetal lie is arbitrary), vertical flip off (would swap near and far field), horizontal flip on (a valid lie), mixup and copy-paste off (blending two fetal hearts produces anatomy that does not exist).

Stage two regresses four landmarks — the endpoints of the cardiac ellipse's axes — rather than the angle directly, so the output is inspectable: a clinician can look at four points and say they are wrong. Three things had to be fixed, each documented at the point in the source where it applies:

- **The 180° endpoint swap.** An ellipse is invariant under 180° rotation, which exchanges both endpoint pairs, so any fixed labelling convention is discontinuous. Solved with a swap-invariant loss (`model.Loss`).
- **Heatmaps are the wrong estimator here.** Axis endpoints have no distinctive local appearance; they are defined by a global property of the shape. Heatmaps plateaued at ~28° against 45° for chance.
- **Global average pooling is nearly orientation-invariant.** It discards the spatial layout that encodes the angle. A 3×3 grid instead of 1×1 was the difference between plateauing and converging.

Angles are treated as **axial** throughout — defined modulo 180°, handled in the doubled-angle representation, aggregated with circular statistics. Direction (apex-left vs apex-right) is levocardia versus dextrocardia, a diagnosis that needs the spine or stomach bubble, so it is not inferred from a cropped heart.

`predict.py` returns `assessable: false` when the predicted shape is too round for an axis to exist (`b/a > 0.93`), when the major and minor axis votes disagree by more than 12°, or when the two heads disagree by more than 15°. Thresholds come from the risk-coverage table on validation. Caveat: that table is nearly flat, so these signals are only weakly informative — reported in the article rather than presented as a working confidence estimate.

## What this does not show

- Not the clinical cardiac axis, which is measured against the spine-to-sternum midline. FOCUS annotates neither spine nor septum. `geometry.cardiac_axis()` is one spine landmark away from it.
- ±18° limits of agreement are not clinically useful.
- No human reader ceiling established, so 7° has no reference point.
- External evaluation covers **self-consistency only**, not accuracy. There is no orientation ground truth outside FOCUS, so external error against a reference remains unmeasured.
- No gestational-age stratification, and no reader study.
- The abstention rule is not calibrated.
- Not a medical device, not validated for clinical use.

## Layout

```
configs/          YOLOv5 data and hyperparameter configs
scripts/          data download, yolov5 setup, detector training
src/fho/
  focus.py        dataset parsing, annotation cross-validation
  geometry.py     axial angle algebra, circular statistics, PCA and min-area baselines
  landmarks.py    crops, augmentation, single-affine warp, canonical landmarks
  model.py        landmark regression network, swap-invariant loss
  train_landmarks.py
  baselines.py    closed-form estimators on the ground-truth masks
  evaluate.py     Bland-Altman, ICC, stratification, risk-coverage, bootstrap CIs
  metamorphic.py  label-free equivariance and invariance tests
  predict.py      end-to-end detection to angle
  prepare_yolo.py
docs/article.md   the write-up
tests/            unit tests for the angle algebra
```

## Licence

Code MIT. FOCUS is CC-BY-4.0 and must be cited: *FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement*, Zenodo, `10.5281/zenodo.14597550`.
