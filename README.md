# Fetal heart detection and orientation estimation

Two stages on prenatal four-chamber ultrasound:

1. **Detection** — a custom **YOLOv5** model that finds the fetal heart (and the thorax) in a four-chamber view.
2. **Orientation** — a **landmark-regression** model that predicts the endpoints of the cardiac ellipse's axes and derives the heart's long axis from them, with calibration-free confidence signals and an abstention rule.

Everything is trained on **prenatal** data. No adult echocardiography is used anywhere in the pipeline.

---

## Data

**[FOCUS — Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement](https://zenodo.org/records/14597550)**, Zenodo `10.5281/zenodo.14597550`, **CC-BY-4.0**, 58 MB.

| Split | Images |
|---|---|
| training | 200 |
| validation | 50 |
| testing | 50 |

Grayscale, ~961×663, five distinct acquisition sizes. Each image carries three parallel annotations for **two structures** (`cardiac`, `thorax`):

```
annfiles_ellipse/NNN.txt      cx cy a b theta_deg label       (a = semi-major)
annfiles_rectangle/NNN.txt    x1 y1 … x4 y4 label difficulty  (DOTA-style oriented box)
annfiles_mask/NNN-{cardiac,thorax}.png
```

The dataset therefore ships **oriented** boxes — which is exactly what an orientation project needs and what almost no public fetal dataset provides.

### The annotations are internally consistent — verified, not assumed

`focus.verify_consistency` cross-checks the ellipse parameters against the independent oriented-box annotation across all 200 training images:

```
centre        median 0.050 px    p95 0.071 px    max 0.071 px
semi-major    median 0.025 px    p95 0.068 px    max 0.085 px
angle         median 0.006°      p95 0.021°      max 0.033°
```

Agreement to a third of a *hundredth* of a degree. That is what licenses using `ellipse.theta` as exact orientation ground truth for the rest of the project.

```bash
./scripts/download_data.sh
```

### Why not the other candidates

* **FETAL_PLANES_DB** ([Zenodo 3904280](https://zenodo.org/records/3904280), 12,400 images) — has a *Thorax* class containing four-chamber views, but only image-level class labels. Useful as extra unlabelled/weakly-labelled data for pretraining; useless as orientation ground truth.
* **CAMUS**, **EchoNet-Dynamic** — adult echocardiography. Different anatomy, different acquisition, different clinical question. Excluded.

---

## Stage 1 — YOLOv5 heart detector

FOCUS ships oriented boxes; YOLOv5 consumes axis-aligned ones, so `prepare_yolo.py` collapses each OBB to its enclosing AABB. The orientation isn't discarded — it is the target of stage 2. Stage 1 only has to find the organ.

```bash
PYTHONPATH=src python -m fho.prepare_yolo          # -> data/processed/yolo (300 images, 600 boxes)
./scripts/setup_yolov5.sh                          # clones ultralytics/yolov5
./scripts/train_yolo.sh
```

Result on the held-out test split, YOLOv5s, 150 epochs, 640 px:

| Class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| all | 0.994 | 1.000 | 0.995 | 0.649 |
| cardiac | 0.990 | 1.000 | 0.995 | 0.637 |
| thorax | 0.999 | 1.000 | 0.995 | 0.660 |

14 ms per image on an M-series GPU. These numbers are unremarkable and should be
read as such: one organ, one view, centred, always present. Detection is not the
hard part of this problem, and a detector that scored anything else would mean
something was wrong with the labels.

Two classes are kept, not one. The thorax box is what a cardiothoracic ratio needs, and it is the reference frame the clinical cardiac axis is measured against.

### Augmentation

`configs/hyp.focus.yaml` departs from the YOLOv5 defaults on physical grounds, not by tuning:

| Setting | Value | Why |
|---|---|---|
| `hsv_h`, `hsv_s` | 0.0 | The images are grayscale. Hue and saturation jitter is meaningless. |
| `hsv_v` | 0.4 | Gain and brightness genuinely vary between machines and operators. |
| `degrees` | 30 | **Fetal lie is arbitrary.** Rotation is physically legitimate here, unlike in chest radiography. |
| `scale` | 0.35 | Depth setting varies. |
| `flipud` | 0.0 | A vertical flip would swap near and far field. No probe does that; it would teach an artifact that never occurs. |
| `fliplr` | 0.5 | A left-right flip is a valid fetal lie — it changes situs semantics only, see §Sign. |
| `mixup`, `copy_paste` | 0.0 | Blending two fetal hearts produces anatomy that does not exist. |

---

## Stage 2 — orientation by landmark regression

Four landmarks: the endpoints of the cardiac ellipse's **major** and **minor** axes, ordered `[major+, major-, minor+, minor-]`. The long axis is recovered from them, with **both axes voting** — the major endpoints directly, the minor endpoints rotated by 90°.

```bash
PYTHONPATH=src python -m fho.train_landmarks --epochs 400
PYTHONPATH=src python -m fho.evaluate --split test
PYTHONPATH=src python -m fho.metamorphic
```

### Why landmarks rather than regressing the angle directly

* The output is **inspectable**. A clinician can look at four points and say they are wrong. Nobody can audit a scalar.
* It **degrades gracefully** — the two axes disagreeing is a confidence signal that costs nothing.
* It matches how the quantity is measured by hand, so the comparison against a human reader is like-for-like.
* In a regulated device, localised evidence is worth more than a slightly better number.

### Three design decisions that were found the hard way

Each of these was a measured failure before it was a design choice. They are documented in the source because they are the transferable part of the work.

**1. The 180° endpoint swap.** An ellipse is invariant under a 180° rotation, which exchanges *both* pairs of endpoints at once. Two labellings are equally correct, so any fixed convention is discontinuous somewhere and the network receives contradictory targets for visually identical crops. The loss scores both assignments and keeps the better one per sample — the landmark analogue of the doubled-angle encoding used for the angle itself.

**2. Heatmaps are the wrong estimator for these landmarks.** The first version used per-landmark Gaussian heatmaps and plateaued at ~28° median error, barely better than the 45° of chance for axial data. The reason is that an ellipse axis endpoint has **no distinctive local appearance** — it is a point on a smooth boundary, defined by a *global* property of the shape. Heatmap regression is right for landmarks with local evidence (an apex, a valve hinge, a vertebral body) and wrong here. The landmarks are regressed globally instead.

**3. Global average pooling is nearly orientation-invariant.** Pooling the final feature map to 1×1 discards exactly the spatial layout that encodes the angle. Replacing it with a 3×3 grid is the difference between plateauing and converging.

A second head predicts the doubled angle `(sin 2θ, cos 2θ)` directly. It is never the reported output — its disagreement with the landmark-derived angle is a free consistency check at inference.

### Angles are axial, and handled as such

Heart orientation is defined **modulo 180°**, not 360°. Every angle in `geometry.py` goes through the doubled-angle representation, so there is no discontinuity at the wrap. Circular mean and circular SD are used for aggregation — the linear mean of 179° and 1° is 90°, which is the answer to no question anyone asked.

### Sign and situs

The septum's axis is a *line*. Turning it into a *direction* — apex-left versus apex-right — is **levocardia versus dextrocardia**, which is a diagnosis, not a convention. It cannot be recovered from the heart crop alone; it needs the spine position, the stomach bubble side, or the descending aorta. This project therefore reports an **axial** angle and leaves the sign to a situs determination that a single cropped frame cannot make.

### Abstention

`predict.py` returns `assessable: false` rather than a number when any of three conditions holds:

| Signal | Meaning |
|---|---|
| predicted roundness `b/a > 0.93` | the shape is too round for an axis to exist |
| major/minor axis votes disagree by > 12° | the four landmarks are not a consistent ellipse |
| the two heads disagree by > 15° | the model is not internally consistent on this image |

The thresholds come from the risk-coverage table in `evaluate.py`, computed on validation — not guessed. The dataset itself demonstrates why the first one is needed: FOCUS's **thorax** ellipses are near-circular (`b/a ≈ 0.99`), so thorax *orientation* is genuinely undefined, even though the thorax centre and radius are perfectly usable.

---

## Classical baselines

Two closed-form estimators run on the ground-truth masks, with no training and no GPU:

```bash
PYTHONPATH=src python -m fho.baselines --split test
```

```
cardiac / test  n=50
  moments   median   0.28°   mean   0.27°   p90   0.45°   max   0.55°   |err|>10°  0.0%
  minarea   median   4.95°   mean  39.32°   p90  88.42°   max  89.70°   |err|>10° 44.0%
```

**Read this carefully — the min-area failures are not noise.** For an ellipse, the area of the enclosing rectangle is `4·√(a²c²+b²s²)·√(a²s²+b²c²)`, which attains the same minimum `4ab` at **both** the major and the minor alignment. The minimum-area rectangle is therefore genuinely bimodal on elliptical shapes, and the rotating-calipers estimator picks one of the two 90°-apart optima essentially at random — hence a cluster of ~89° errors and a 4.95° median. Minimum-area is the exact optimum for *area* and an unstable estimator for *axis*. That distinction is the whole reason the orientation head exists.

**And a caveat that must not be glossed:** the FOCUS masks are rasterisations of the same ellipse annotation, so the 0.28° moment result measures *numerical consistency of the geometry code*, not clinical accuracy. It is a floor, not evidence. The real test is the learned model predicting from pixels, which is what `evaluate.py` reports.

---

## Validation — the part that matters

`evaluate.py` deliberately does not print one accuracy number.

1. **Agreement, not accuracy.** Bland–Altman bias and 95 % limits of agreement, plus ICC(2,1). A mean error hides systematic bias, which is the failure that changes a clinical reading.
2. **Circular statistics** throughout, on the doubled angle.
3. **Stratification** by shape roundness and heart size — roundness is the covariate that provably breaks every orientation estimator, so it is reported, not averaged away.
4. **Risk–coverage.** Error as a function of how much the model abstains. If error does not fall as coverage drops, the confidence estimate is decorative.
5. **Stage attribution.** `--source gt` versus `--source yolo` measures what stage 1 costs the final angle. Without it you cannot tell which stage to work on.
6. **Bootstrap confidence intervals** on the median error — 50 test images is a small number and the interval says so.

### Metamorphic tests — validation with no labels at all

```bash
PYTHONPATH=src python -m fho.metamorphic
```

| Property | Assertion |
|---|---|
| rotation equivariance | rotate the input by δ, the predicted axis must move by exactly δ |
| mirror equivariance | flip left–right, the axis must reflect |
| gain invariance | brightness and contrast must not move the axis at all |
| crop-scale invariance | a wider crop must not move the axis |

These catch coordinate-convention bugs, wrap-around bugs and augmentation leaks that a held-out set will happily hide — and because they need no ground truth, the same code runs as a **deployment monitor** on incoming scans.

---

## Measured results

All numbers below were produced by the code in this repository on the FOCUS
**test** split (50 images), model trained for 400 epochs on the 200 training
images. Reproduce with `make train && make eval && make meta`.

### Orientation, landmark model

```
== landmark model / test / crops from gt ==  n=50

  median |error|     7.04°   95% CI [4.84, 9.26]
  mean   |error|     7.39°
  p90    |error|    12.92°
  max    |error|    22.74°
  fraction >10°     34.0%

  Bland-Altman     bias -0.55°   LoA [-18.23, +17.12]
  ICC(2,1)         0.9803
  circular SD of the signed difference  8.95°
```

Read the four numbers together, because each says something the others hide:

* **Bias −0.55°** — essentially unbiased. There is no systematic rotation error,
  which is the failure that would matter most clinically.
* **ICC 0.980** — the model tracks the reference well *across* cases. On its own
  this number is flattering, because cardiac angles span a wide range and ICC
  rewards that.
* **Limits of agreement ±18°** — this is the honest one. On an individual scan
  the estimate can be 18° off, and the clinical normal band is only about 40°
  wide. A 400-epoch model on 200 images is not yet a measuring instrument.
* **Validation median was 4.41°, test median is 7.04°** — a real generalisation
  gap on 50 images, and the bootstrap CI [4.84, 9.26] says the test estimate
  itself is loose. Both are consequences of dataset size, not of the estimator.

Training had not converged at 400 epochs (train angular loss still falling,
0.047 at the last epoch). Longer schedules and a pretrained backbone are the
obvious next steps, along with more data.

### Stratification and abstention

```
  error by roundness b/a           error by heart semi-major (px)
    0.60–0.75  n=30  median 6.70°     60–90    n= 6  median 8.07°
    0.75–0.90  n=20  median 7.93°     90–120   n=20  median 9.26°
                                      120–400  n=24  median 4.93°
```

Rounder hearts are harder, as expected — but the effect is smaller than the size
effect. Small hearts are markedly worse (9.26° vs 4.93°), which is the resolution
argument showing up in the data: at 90 px semi-major a few pixels of landmark
error is several degrees.

**The abstention signals are weak, and that is a finding, not an omission.** The
risk-coverage table is nearly flat: dropping to 53 % coverage moves the median
from 7.04° to 5.40°, and the p90 barely moves at all. Head disagreement and
major/minor axis disagreement are only weakly informative about error here. A
confidence signal that does not buy error reduction should not be shipped as one —
the honest report is that this model does not yet know when it is wrong.

### Metamorphic tests

```
  rotation equivariance -30°   median  4.74°  p90  9.96°  max 15.41°
  rotation equivariance -15°   median  3.63°  p90  9.03°  max 12.13°
  rotation equivariance +15°   median  2.89°  p90  7.43°  max 21.69°
  rotation equivariance +30°   median  3.73°  p90 12.52°  max 32.94°
  mirror equivariance          median  2.55°  p90 12.00°  max 48.18°
  gain invariance              median  0.74°  p90  3.90°  max  5.31°
  crop-scale invariance        median  3.65°  p90  8.63°  max 26.08°
```

Every one of these is a **FAIL** against the tolerances set in the file, and the
tolerances were not relaxed to make them pass. What they say:

* Rotation self-consistency (≈3–5° median) is of the same order as the model's own
  test error (7°), so the residual is genuine model variance rather than a
  coordinate-convention bug.
* **Gain invariance is violated** — up to 5° of axis movement from a pure
  brightness and contrast change, which is anatomy-free. That is a concrete,
  fixable defect: stronger gain augmentation, or per-image intensity
  normalisation before the network.
* **Crop-scale sensitivity of 3.65° median** predicts part of the degradation
  when crops come from YOLO instead of ground truth, and argues for training with
  wider crop jitter than the current ±12 %.

These tests earned their place immediately: the first run reported errors of
*exactly twice* the applied rotation on every image, which is the unmistakable
signature of a flipped sign — in the test's expected value, as it turned out. A
held-out set would never have surfaced that. The expectation is now derived from
the warp matrix itself, so the test has no convention of its own to get wrong.


---

## What this is *not*, and what would come next

**The clinical cardiac axis is not what is measured here.** Clinically it is the angle between the **interventricular septum** and the **thoracic anteroposterior midline** (spine → sternum), normally ~45° with a normal band around 25–65°, and deviation is an independent screening marker for congenital heart disease. FOCUS annotates the cardiac and thoracic *regions* but **not the spine and not the septum**, so what this project estimates is the **heart's long axis in the image frame**. `geometry.cardiac_axis(heart_angle, ap_midline)` is already there and is a one-line call away from the clinical quantity — it needs a spine landmark.

Concretely, to close that gap:

1. **Annotate one spine point** (and optionally the sternum) on the 300 FOCUS images. A single-point annotation pass over 300 images is a couple of hours, and it converts every number here into the clinical quantity.
2. Add it as a fifth landmark with the same swap-invariant machinery — the AP midline is a *directed* line, so it does not need the axial treatment.
3. Then the error budget becomes `σ²_septum + σ²_midline`, and the midline term is expected to dominate. Measure both before optimising either.
4. **Establish the human ceiling.** Two or three readers measuring the same clips, plus one reader measuring twice. No model beats the annotators' agreement with each other, and that number — not a target picked in advance — is what the model should be held to.
5. Evaluate the **decision**, not only the number: abnormal-axis yes/no at the normative cut-off, since a 3° error is irrelevant mid-band and decisive at the boundary.

---

## Layout

```
configs/          YOLOv5 data + hyperparameter configs
scripts/          data download, yolov5 setup, yolo training
src/fho/
  focus.py        dataset parsing, annotation cross-validation
  geometry.py     axial angle algebra, circular stats, PCA and min-area baselines
  landmarks.py    crops, augmentation, single-affine warp, canonical landmarks
  model.py        landmark regression network and the swap-invariant loss
  train_landmarks.py
  baselines.py    closed-form estimators on the ground-truth masks
  evaluate.py     Bland-Altman, ICC, stratification, risk-coverage, bootstrap CIs
  metamorphic.py  label-free equivariance and invariance tests
  predict.py      end-to-end YOLO -> crop -> landmarks -> angle
  prepare_yolo.py
tests/            unit tests for the angle algebra
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/download_data.sh
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Trains on Apple Silicon (MPS), CUDA, or CPU — the device is picked automatically.

## Licence and citation

Code: MIT. Data: FOCUS is CC-BY-4.0 and must be cited —
*FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement*, Zenodo, `10.5281/zenodo.14597550`.
