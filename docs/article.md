# Estimating fetal cardiac orientation, and knowing when it is wrong

A two-stage pipeline on prenatal four-chamber ultrasound: find the heart, then estimate its long axis. The detection half took an afternoon and is not interesting. The estimation half took the rest of the time, failed three times for reasons worth writing down, and still is not accurate enough to measure anything clinically. This article is mostly about the third part.

**Code:** [github.com/francescovigni/fetal-cardiac-orientation](https://github.com/francescovigni/fetal-cardiac-orientation) · **Data:** [FOCUS](https://zenodo.org/records/14597550), CC-BY-4.0 · **Licence:** MIT

---

## What it does and does not do

| | |
|---|---|
| Detects the fetal heart and thorax | mAP@50 **0.995**, recall **1.00**, 14 ms per image |
| Estimates the heart's long axis in the image frame | median error **7.0°**, 95 % limits of agreement **±18°** |
| Measures the clinical cardiac axis | **No.** That needs a spine landmark this dataset does not have |
| Knows when it is wrong | **Not yet.** The abstention signals barely reduce error |

The clinical normal band for cardiac axis is roughly 40° wide. Limits of agreement of ±18° mean a single scan can be misplaced by nearly half that band. This is a working method with an honest error bar, not an instrument.

---

## Why orientation, and why it is harder than it looks

Cardiac axis is the angle between the interventricular septum and the thoracic anteroposterior midline, measured on the four-chamber view. It sits around 45° normally, and deviation from that band is an independent screening marker for congenital heart disease. It is also abnormal when the heart is pushed rather than malformed, as in diaphragmatic hernia.

Three properties make it awkward for a standard vision pipeline.

**It is a difference of two angles.** The heart's axis alone is not the measurement. The thoracic reference matters just as much, and in practice it is the noisier of the two terms. Any error budget that ignores it is optimistic.

**It is axial, not directional.** An axis is defined modulo 180°, not 360°. Turning it into a direction, apex-left versus apex-right, is levocardia versus dextrocardia. That is a diagnosis, and it cannot be read off a cropped heart. It needs the spine, the stomach bubble, or the descending aorta.

**Fetal lie is arbitrary.** There is no canonical orientation in the image, unlike a chest radiograph. Rotation augmentation is physically legitimate here, and rotation equivariance is a property the model genuinely must have.

---

## The data, and checking it before trusting it

[FOCUS](https://zenodo.org/records/14597550) is 300 prenatal four-chamber images, 200/50/50, CC-BY-4.0. Each image carries three parallel annotations for two structures, cardiac and thorax: an ellipse, an oriented bounding box in DOTA format, and a mask. A public fetal dataset shipping native oriented boxes is unusual and is what made this project possible at all.

Before using the ellipse angle as ground truth, I checked it against the independently stored oriented box across all 200 training images:

```
centre        median 0.050 px    max 0.071 px
semi-major    median 0.025 px    max 0.085 px
angle         median 0.006°      max 0.033°
```

Agreement to three hundredths of a degree. That check is thirty lines of code and it is the difference between "the annotations are consistent" and "I assume the annotations are consistent". It also caught the convention: `a` is the semi-major axis and `theta` is its direction in image coordinates, with y pointing down. Getting that wrong would have silently mirrored every angle in the project.

Two candidate datasets were rejected. FETAL_PLANES_DB has 12,400 images but only image-level class labels, so no geometry. CAMUS and EchoNet-Dynamic are adult echocardiography: different anatomy, different acquisition, different question.

---

## Stage one: detection

The oriented boxes are collapsed to axis-aligned boxes for YOLOv5, which only needs to find the organ. The orientation is not discarded; it is the target of stage two.

| Class | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| cardiac | 0.990 | 1.000 | 0.995 | 0.637 |
| thorax | 0.999 | 1.000 | 0.995 | 0.660 |

Read these as table stakes. One organ, one view, centred, always present. A detector scoring anything else would mean something was wrong with the labels.

The part worth keeping is the augmentation policy, which departs from the YOLOv5 defaults on physical grounds rather than by tuning.

| Setting | Value | Reason |
|---|---|---|
| hue, saturation | off | The images are grayscale |
| brightness | 0.4 | Gain genuinely varies between machines and operators |
| rotation | ±30° | Fetal lie is arbitrary |
| vertical flip | off | It would swap near and far field. No probe does that |
| horizontal flip | 0.5 | A valid fetal lie; it changes situs semantics only |
| mixup, copy-paste | off | Blending two fetal hearts produces anatomy that does not exist |

Two classes are kept rather than one. The thorax box is what a cardiothoracic ratio needs, and it is the reference frame the clinical axis is measured against.

---

## Stage two: three failures, in order

The orientation head regresses four landmarks, the endpoints of the cardiac ellipse's major and minor axes, and derives the axis from them. Landmarks rather than a direct angle regression because the output is inspectable: a clinician can look at four points and say they are wrong, and nobody can audit a scalar. In a regulated device that difference is worth more than a slightly better number.

Three things had to fail first.

### The 180° endpoint swap

An ellipse is invariant under a 180° rotation, and that rotation exchanges both pairs of axis endpoints at once. Two labellings are therefore equally correct. Any fixed convention, such as "the major endpoint with non-negative x comes first", is discontinuous somewhere, and under rotation augmentation the network receives contradictory targets for visually identical crops. It cannot learn.

The fix is a loss that scores both assignments and keeps the better one per sample. It is the landmark analogue of encoding an axial angle as `(sin 2θ, cos 2θ)`, which is the standard cure for the same disease in oriented-box regression.

### Heatmaps are the wrong estimator for these landmarks

The first version used per-landmark Gaussian heatmaps, the default choice for landmark regression. It plateaued at about 28° median error, against 45° for random guessing on axial data.

The reason is specific and, once seen, obvious. An ellipse axis endpoint has no distinctive local appearance. It is a point on a smooth boundary, defined by a global property of the shape: the extremum along a direction. A receptive field centred on it sees the same thing it would see a few pixels along the contour. Heatmap regression works when a landmark has local evidence, such as an apex, a valve hinge, or a vertebral body. These landmarks do not, so they are regressed globally instead.

### Global average pooling is almost orientation-invariant

Pooling the final feature map to 1×1 discards the spatial layout, and the spatial layout is where the angle lives. This is not subtle in hindsight and it was still the last thing I looked at. Replacing global pooling with a 3×3 grid moved the model from plateauing around 21° to converging.

Sequence, measured on the same validation split: heatmaps 28°, global regression with average pooling 21°, global regression with a spatial grid 4.4°.

---

## Results, read honestly

Test split, 50 images, 400 epochs on 200 training images.

```
median |error|     7.04°   95 % CI [4.84, 9.26]
p90    |error|    12.92°
Bland-Altman       bias -0.55°   LoA [-18.23, +17.12]
ICC(2,1)           0.980
```

![Bland-Altman and error distribution on the test split](figures/agreement.png)

Four numbers, four different things.

The **bias of −0.55°** says there is no systematic rotation error, which is the failure that would matter most clinically and the one a mean absolute error would hide.

The **ICC of 0.980** is the flattering number. Cardiac angles span a wide range, and ICC rewards tracking that range. Quoted alone it would be misleading.

The **limits of agreement of ±18°** are the honest number, and they are what a clinician would ask for. On an individual scan the estimate can be that far off.

The gap between **4.41° on validation and 7.04° on test** is a real generalisation gap on 200 training images, and the bootstrap interval [4.84, 9.26] says the test estimate itself is loose. Training had not converged at 400 epochs.

![Predicted axis against annotation: best, median and worst test cases](figures/qualitative.png)

Stratifying is more informative than the headline:

```
by roundness b/a          by heart size (semi-major, px)
  0.60–0.75  median 6.70°    60–90    median 8.07°
  0.75–0.90  median 7.93°    90–120   median 9.26°
                             120–400  median 4.93°
```

Rounder hearts are harder, as the geometry predicts, but the size effect is larger. At 90 px semi-major, a few pixels of landmark error is several degrees. That is the resolution argument appearing in data rather than in a paragraph.

### The abstention signals do not work, and that is a result

The model carries two internal consistency checks: the major and minor axes each vote for the angle, and a second head predicts the angle directly. Both disagreements are free confidence signals.

![Risk-coverage for each candidate confidence signal](figures/risk_coverage.png)

Neither buys much. Dropping to 53 % coverage moves the median from 7.04° to 5.40°, and the p90 barely moves at all. The best signal available is not either of them: it is simply **heart size**. Predicted elongation is worse than nothing, since abstaining by it makes the median error rise. A confidence signal that does not reduce error is not a confidence signal, and reporting it as one would be worse than having none. This model does not yet know when it is wrong.

---

## Validating without labels

Held-out sets are not the only instrument available, and on 50 images they are a blunt one. Four properties can be asserted with no ground truth at all:

- rotate the input by δ, the predicted axis must move by exactly δ;
- mirror the input, the axis must reflect;
- change brightness and contrast, the axis must not move, because neither is anatomy;
- widen the crop, the axis must not move.

```
rotation equivariance ±15°   median 2.9–3.6°   p90  7.4–9.0°
rotation equivariance ±30°   median 3.7–4.7°   p90 10.0–12.5°
mirror equivariance          median 2.6°       p90 12.0°
gain invariance              median 0.7°       p90  3.9°   max 5.3°
crop-scale invariance        median 3.7°       p90  8.6°
```

Every one fails the tolerances set in the file. The tolerances were not relaxed to make them pass.

What they say is useful. Rotation self-consistency, 3° to 5° median, is the same order as the model's own test error, so the residual is genuine model variance rather than a coordinate bug. **Gain invariance is violated by up to 5°**, which is a concrete defect: a pure brightness change moves an anatomical measurement. That points at stronger intensity augmentation or per-image normalisation, and it is not something a test set would ever have told me.

These tests earned their place twice.

**First**, the initial run reported errors of exactly twice the applied rotation on every single image. Errors of exactly 2δ are the signature of a flipped sign, and the flip turned out to be in the test's own expected value rather than in the model. The expectation is now derived from the warp matrix itself, so the test has no convention left to get wrong.

**Second**, running the same suite against a deliberately undertrained checkpoint gives rotation errors of almost exactly δ. That is the signature of a model predicting a near-constant angle regardless of input. Detecting "the model ignores the image" with no labels, on data you have never annotated, is exactly what you want running as a monitor in deployment.

---

## An aside on oriented boxes: PCA versus rotating calipers

Two classical estimators for the axis of a shape. PCA takes the axes from the eigenvectors of the point covariance. Rotating calipers over the convex hull returns the exact minimum-area enclosing rectangle. On the ground-truth masks:

```
moments (PCA)   median 0.28°   p90  0.45°   fraction >10°   0.0 %
min-area rect   median 4.95°   p90 88.42°   fraction >10°  44.0 %
```

The minimum-area failures are not noise. For an ellipse, the enclosing rectangle has area `4·√(a²c²+b²s²)·√(a²s²+b²c²)`, which reaches the same minimum `4ab` at **both** the major and the minor alignment. The estimator is genuinely bimodal on elliptical shapes, so it picks one of two optima 90° apart essentially at random. Minimum-area is exact for area and unstable for axis. They optimise different things, and the tightest box is not the best axis.

One caveat that must not be buried: the FOCUS masks are rasterised from the same ellipse annotations, so the 0.28° figure measures the numerical consistency of the geometry code, not clinical accuracy. It is a floor, not evidence.

---

## Testing it somewhere else, with no labels

The obvious objection to everything above is that it is 300 images from one source. The obvious answer is an external test set, and the obvious obstacle is that orientation ground truth does not exist outside FOCUS.

It turns out not to matter. The four properties in the metamorphic suite hold on any image, annotated or not, so the same code runs unchanged on [FETAL_PLANES_DB](https://zenodo.org/records/3904280): a different hospital, different operators, four ultrasound machines, and 1,718 images labelled as the thorax plane.

The detector fires on **93 %** of them at mean confidence 0.75, having never seen the dataset. Then the orientation model:

| Property | FOCUS median | external median | FOCUS p90 | external p90 |
|---|---|---|---|---|
| rotation ±15° | 2.9–3.6° | 3.9–4.3° | 7.4–9.0° | 20.5–21.9° |
| rotation ±30° | 3.7–4.7° | 5.3–6.5° | 10.0–12.5° | 29.2–39.6° |
| mirror | 2.55° | 6.51° | 12.00° | 44.28° |
| gain | 0.74° | 1.96° | 3.90° | 8.03° |
| crop scale | 3.65° | 11.82° | 8.63° | 43.96° |

![Internal versus external self-consistency, and detection by machine](figures/external.png)

The medians move modestly. The tails triple. That gap is the whole result: the model still works on typical external images and fails outright on a substantial minority, and any summary that reported only a mean would have hidden it.

Crop-scale sensitivity going from 3.65° to 11.82° is the most specific finding. It says the model has partly learned the FOCUS crop convention rather than the anatomy, which is a training-time fix, not a data problem.

Splitting by machine, the detector fires on 95 % of Voluson E6 images and **81 % of Aloka** images, at the lowest mean confidence of the four manufacturers. Aloka is 41 % of the external dataset and appears nowhere in FOCUS.

None of this required a single annotation, and the same code can run as a monitor on incoming scans in production, where labels never arrive at all.

## What this does not show

- **It is not the clinical cardiac axis.** It is the heart's long axis in the image frame. The clinical quantity is measured against the spine-to-sternum midline, and FOCUS annotates neither the spine nor the septum.
- **±18° limits of agreement are not clinically useful** against a normal band roughly 40° wide.
- **No human ceiling was established.** Two readers measuring the same clips disagree by some amount, and no model can beat that. Without it, 7° has no reference point.
- **External evaluation covers self-consistency, not accuracy.** Without orientation labels outside FOCUS, external error against a reference is still unmeasured. No gestational-age stratification, no reader study.
- **The abstention rule is not calibrated**, for the reason given above.
- **Nothing here is a medical device**, and none of it has been validated for clinical use.

## What would close the gap

1. Annotate one spine point on the 300 images. A single-point pass is a couple of hours and converts every number here into the clinical quantity. The function that consumes it is already written.
2. Then measure both error terms separately. The midline term is expected to dominate, and optimising the heart term first would be wasted effort.
3. Establish the reader ceiling: two or three clinicians measuring the same clips, one measuring twice. That number sets the target, not one picked in advance.
4. Evaluate the decision rather than the number: abnormal axis, yes or no, at the normative cut-off. A 3° error is irrelevant mid-band and decisive at the boundary.
5. Fix gain sensitivity, which the metamorphic suite has already localised.

## What this would take on your data

The transferable part is not the model, it is the harness: annotation cross-validation, the swap-invariant loss, agreement statistics instead of accuracy, stratification by the covariate that actually breaks the estimator, risk-coverage for abstention, and label-free equivariance tests that double as a deployment monitor. On a labelled in-house set that is roughly a two-week piece of work, and the first week is spent counting confirmed positives per class before any modelling starts.

## Reproducing

```bash
git clone https://github.com/francescovigni/fetal-cardiac-orientation.git && cd fetal-cardiac-orientation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data && make test && make baselines
make train && make eval && make meta
```

Verified from a clean clone with an empty environment: 9 unit tests pass, the dataset downloads, the annotation cross-check reproduces to the same 0.033°, the baselines reproduce exactly, and training runs on CPU as well as on MPS.

FOCUS must be cited: *FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement*, Zenodo, `10.5281/zenodo.14597550`.
