# Estimating Fetal Cardiac Orientation: Method, Experiments, and Failure Analysis

Technical companion to the [README](../README.md). The README answers *what was built and why it matters*; this document answers *how it was built, why each choice was made, and what the experiments revealed*.

Every experiment below is written as **Question → Method → Result → Interpretation**.

---

## 1. Abstract

Given a four-chamber fetal ultrasound image, the task is to localise the heart and estimate its orientation. The system is two-stage: a YOLOv5 detector for the cardiac and thoracic regions, then a CNN that regresses four geometric landmarks — the endpoints of the cardiac ellipse's major and minor axes — from which the axial angle and an oriented bounding box are reconstructed in closed form.

The project's organising question is whether the orientation must be learned at all. When a segmentation mask is available it need not be: the principal axis follows analytically from second-order image moments, at 0.28° median error on the available masks. The learned model exists for the case where only a detection box is available, and reaches 7.04° median error with 95 % limits of agreement of ±18°.

Both routes are evaluated beyond in-distribution accuracy: agreement statistics rather than accuracy, a cross-dataset run on a second public dataset with no orientation labels, label-free metamorphic testing, and a simulated-segmentation-failure study that characterises where the closed-form estimator breaks. Several results are negative and are retained, because they are what selected the final design.

All data is public and CC-BY-4.0. Nothing here is clinically validated.

---

## 2. Problem formulation

Let `I` be a four-chamber ultrasound image containing a fetal heart. The target is the orientation `θ` of the heart's long axis.

Three properties shape every design decision:

**The quantity is axial, not directional.** An axis is invariant under a 180° rotation, so `θ` and `θ+180°` denote the same thing. Naive regression on `θ ∈ [0, 360)` places a discontinuity inside the label space. The fix used throughout is the doubled-angle encoding

```
z(θ) = (sin 2θ, cos 2θ) ,   θ = ½ atan2(z₁, z₂) mod 180°
```

which is continuous across the wrap. Aggregation uses circular statistics: the circular mean of axial angles is `½ arg E[e^{2iθ}]`, and the linear mean of 179° and 1° — which is 90° — is the answer to no question.

**Direction is a separate, harder problem.** Turning the axis into a direction (apex-left versus apex-right) distinguishes levocardia from dextrocardia. That is a diagnosis requiring the spine, the stomach bubble, or the descending aorta, none of which is inside a cropped heart. The system deliberately reports an axis and abstains from the sign.

**There is no canonical orientation in the image.** Fetal lie is arbitrary, unlike a chest radiograph. Rotation is therefore a physically legitimate augmentation, and rotation *equivariance* is a property the estimator must satisfy — which makes it testable without labels.

The clinical cardiac axis is the angle between the interventricular septum and the thoracic anteroposterior midline. This project measures the heart's long axis in the image frame, which is a component of that quantity but not the quantity itself. See §19.

---

## 3. Dataset and annotations

**FOCUS** (Zenodo `10.5281/zenodo.14597550`, CC-BY-4.0) — 300 prenatal four-chamber images, split 200 / 50 / 50, grayscale, approximately 961×663 across five distinct acquisition sizes. Each image carries three parallel annotations for two structures, `cardiac` and `thorax`:

```
annfiles_ellipse/NNN.txt      cx cy a b theta_deg label       (a = semi-major)
annfiles_rectangle/NNN.txt    x1 y1 … x4 y4 label difficulty  (DOTA-style oriented box)
annfiles_mask/NNN-{cardiac,thorax}.png
```

A public fetal dataset shipping native oriented boxes is unusual, and it is what made the project tractable.

**FETAL_PLANES_DB** (Zenodo `10.5281/zenodo.3904280`, CC-BY-4.0) — 12,400 maternal-fetal images across six plane classes, of which **1,718** are labelled `Fetal thorax`. Metadata records patient, operator and **ultrasound machine** (Voluson E6, Aloka, Voluson S10, Other). Used **only** as an unlabelled external set; never trained on.

### Experiment: are the annotations self-consistent?

> **Question.** The ellipse angle is about to be used as ground truth for the entire project. Is it trustworthy?
>
> **Method.** `focus.verify_consistency` reconstructs centre, semi-major length and long-edge direction from the *independently stored* oriented-box corners and compares them with the ellipse parameters, across all 200 training images.
>
> **Result.**
> ```
> centre        median 0.050 px    p95 0.071 px    max 0.071 px
> semi-major    median 0.025 px    p95 0.068 px    max 0.085 px
> angle         median 0.006°      p95 0.021°      max 0.033°
> ```
>
> **Interpretation.** Agreement to three hundredths of a degree. Two annotation formats generated from a common source would agree exactly; two independent transcriptions would not agree this closely. The check also pinned the convention — `a` is the semi-major axis and `theta` its direction in **image coordinates with y pointing down** — which, had it been assumed instead of verified, would have silently mirrored every angle in the project.

---

## 4. End-to-end system

```
image → YOLOv5 → cardiac box → single affine warp → 192×192 crop
      → LandmarkNet → 4 endpoints → geometric reconstruction → axial angle + oriented box
                                                              → abstention flags
```

Alternative path, when a mask is available rather than a box:

```
mask → largest connected component + opening → second-order moments → axial angle
```

The two paths are independent implementations of the same measurement, which makes each a check on the other.

**Crop construction** deserves a note. The rotation about the heart centre and the crop are composed into a **single 2×3 affine matrix**, applied to the image via `warpAffine` and to the landmark coordinates as points. Image and labels therefore transform by construction, not by two parallel pieces of code that must agree. Out-of-image regions fill with zeros, which is what ultrasound background is anyway. The matrix is returned with each example and is reused later to define metamorphic expectations (§13) — so the test has no coordinate convention of its own to get wrong.

---

## 5. YOLOv5 detection

FOCUS ships oriented boxes; YOLOv5 consumes axis-aligned ones. `prepare_yolo.py` collapses each oriented box to its enclosing axis-aligned box and writes YOLO-format labels for two classes. Thorax is kept alongside cardiac because it is the reference frame a cardiothoracic ratio and a clinical cardiac axis would need.

Training: YOLOv5s, 150 epochs, 640 px, batch 8, with a custom hyperparameter file.

> **Question.** Can the heart be localised reliably enough to feed a downstream geometric estimator?
>
> **Method.** Fine-tune YOLOv5s on 200 images; evaluate on the 50-image held-out test split.
>
> **Result.**
>
> | Class | P | R | mAP@50 | mAP@50-95 |
> |---|---|---|---|---|
> | cardiac | 0.990 | 1.000 | 0.995 | 0.637 |
> | thorax | 0.999 | 1.000 | 0.995 | 0.660 |
>
> 14 ms per image.
>
> **Interpretation.** Table stakes, and reported as such. One organ, one view, centred, always present — a detector scoring anything else would indicate a labelling problem. The number carries no information about the hard part of the task. Its real test is the cross-dataset firing rate in §12.

---

## 6. Geometric orientation baselines

Two closed-form estimators need no training at all.

**Second-order moments (PCA).** Take the set pixels of a mask, weight by probability if the mask is soft, convert to physical units, and take the leading eigenvector of the covariance. Also available for free: the anisotropy `λ₂/λ₁`, and a first-order angular standard error from the eigengap, `Var(θ) ≈ λ₁λ₂/(λ₁−λ₂)²/n`.

**Minimum-area enclosing rectangle.** Rotating calipers over the convex hull. By Toussaint's result the optimal rectangle has a side collinear with a hull edge, so only `h` orientations need testing.

> **Question.** Can orientation be recovered without training an orientation model?
>
> **Method.** Run both estimators on the ground-truth cardiac masks and compare against the annotated ellipse angle.
>
> **Result.**
>
> | Estimator | median | p90 | > 10° |
> |---|---|---|---|
> | second-order moments | **0.28°** | 0.45° | 0 % |
> | minimum-area rectangle | 4.95° | 88.42° | **44 %** |
>
> **Interpretation.** The moments estimator is extremely accurate on clean masks. The minimum-area failures are not noise: for an ellipse the enclosing rectangle has area `4√(a²c²+b²s²)·√(a²s²+b²c²)`, which attains the same minimum `4ab` at **both** the major and the minor alignment. The estimator is genuinely bimodal and selects one of two optima 90° apart. It is exact for *area* and unstable for *axis* — different objectives, and the tightest box is not the best axis.
>
> **Caveat, stated rather than buried.** The FOCUS masks are rasterised from the same ellipse annotations, so 0.28° measures the numerical consistency of the geometry code, not accuracy against a real segmenter. §14 is the experiment that carries information.

---

## 7. Landmark-based orientation model

When only a box is available, the moments have no mask to operate on and the angle must be learned.

**Prediction target.** Four landmarks, ordered `[major+, major−, minor+, minor−]`, being the endpoints of the cardiac ellipse's axes.

**Rationale.** Four points are inspectable — a reviewer can reject an individual landmark, whereas a scalar admits no audit. The oriented box follows in closed form: centre `(p₀+p₁)/2`, half-axes `u=(p₀−p₁)/2`, `v=(p₂−p₃)/2`, corners `c ± u ± v`. And two independent votes for the angle fall out at no cost: the major endpoints give it directly, the minor endpoints give it rotated by 90°, which in doubled-angle space is a negation, so both combine as unit vectors:

```
θ = ½ atan2( sin2θ_major − sin2θ_minor ,  cos2θ_major − cos2θ_minor )
```

**Architecture.**

```
input 192×192×1
  └─ 5 × [stride-2 conv-BN-SiLU, conv-BN-SiLU]  32→64→128→256→256 ch   (/32 → 6×6)
      └─ AdaptiveAvgPool2d(3)                    3×3 spatial grid
          └─ Linear(2304 → 256) + SiLU + Dropout(0.1)
              ├─ coord head  Linear(256 → 8), tanh, mapped into the crop
              └─ axis head   Linear(256 → 2), L2-normalised → (sin 2θ, cos 2θ)
```

The **axis head is never reported**. It exists so that two estimates of the same quantity can disagree, giving a confidence signal at inference.

---

## 8. Training formulation

The loss has three terms.

**Swap-invariant coordinate L1.** An ellipse is invariant under a 180° rotation, which exchanges *both* endpoint pairs simultaneously. Consequently `[major+, major−, minor+, minor−]` and `[major−, major+, minor−, minor+]` are equally correct labellings of the same shape, and any fixed tie-break is discontinuous somewhere in configuration space. Under rotation augmentation the network then receives contradictory targets for visually identical crops. The loss evaluates both permutations, `[0,1,2,3]` and `[1,0,3,2]`, and keeps the smaller per sample. This is the landmark analogue of the doubled-angle encoding used for the angle itself.

**Angular loss on the coordinate-derived axis**, as `1 − cos` between predicted and target unit vectors in doubled-angle space. This ties optimisation to the quantity that is actually reported rather than only to pixel positions, and it is automatically invariant to the same swap, since doubling identifies `v` with `−v`.

**Angular loss on the direct axis head**, identically defined.

Weights `(1.0, 2.0, 1.0)`. AdamW, lr 3e-3, OneCycle schedule, batch 16, gradient clipping at 5.0, 400 epochs on 200 images. Device selection is automatic: MPS, then CUDA, then CPU.

---

## 9. Data augmentation

Augmentation was chosen from acquisition physics rather than from defaults.

| Transformation | Detector | Landmark model | Justification |
|---|---|---|---|
| rotation | ±30° | ±180° | fetal lie is arbitrary; rotation is physically legitimate |
| horizontal flip | p = 0.5 | p = 0.5 | a valid fetal lie; changes situs semantics only |
| vertical flip | off | off | would swap near and far field — no probe does this |
| brightness / gain | on | log-normal gain + offset | gain genuinely varies between machines and operators |
| scale | ±35 % | ±12 % crop jitter, ±6 % centre jitter | depth setting varies |
| hue / saturation | off | off | the images are grayscale |
| mixup / copy-paste | off | off | blending two fetal hearts produces anatomy that does not exist |

The vertical-flip exclusion is the clearest case of the principle: including it would teach the network an artifact that no acquisition can produce.

---

## 10. Experimental protocol

- **Splits** are the dataset's own 200 / 50 / 50. The test split is used once per reported configuration.
- **Metrics are agreement, not accuracy.** Bland–Altman bias and 95 % limits of agreement on the signed axial difference, ICC(2,1) for absolute agreement, and bootstrap confidence intervals on the median error, because 50 images is a small number and the interval should say so.
- **Circular statistics** throughout, on the doubled angle.
- **Stratification** by shape roundness and heart size, since roundness is the covariate that provably degrades any orientation estimator.
- **Risk–coverage** for every candidate confidence signal.
- **Label-free tests** (§13) and **cross-dataset evaluation** (§12) as separate layers.
- All figures are regenerated from the checkpoints by `fho.figures`, so a figure cannot drift away from the number it illustrates.

---

## 11. In-distribution results

> **Question.** How accurate is the learned estimator, and how should that accuracy be described?
>
> **Method.** Evaluate the 400-epoch checkpoint on the 50-image test split, using ground-truth crops so that stage-1 error is excluded.
>
> **Result.**
> ```
> median |error|     7.04°   95 % CI [4.84, 9.26]
> p90    |error|    12.92°
> Bland-Altman       bias -0.55°   LoA [-18.23, +17.12]
> ICC(2,1)           0.980
> ```
>
> ![Bland-Altman and error distribution](figures/agreement.png)
>
> **Interpretation.** Four numbers describing four different things.
>
> *Bias −0.55°* — essentially unbiased. A systematic rotation offset is the failure that would matter most and the one a mean absolute error conceals.
>
> *ICC 0.980* — flattering. Cardiac angles span a wide range and ICC rewards tracking that range; quoted alone it misleads.
>
> *Limits of agreement ±18°* — the number a clinician would ask for, against a clinical normal band roughly 40° wide.
>
> *Validation 4.41° versus test 7.04°* — a real generalisation gap on 200 training images, and the bootstrap interval says the test estimate is itself loose. Training had not converged at 400 epochs.
>
> Relative to §6, the learned route is roughly an order of magnitude worse than the closed-form one. It exists only for the case where no mask is available.

![Best, median and worst test cases](figures/qualitative.png)

**Stratification.** By roundness: 6.70° for `b/a` in 0.60–0.75, 7.93° for 0.75–0.90 — rounder is harder, as the geometry predicts. By size: 9.26° at 90–120 px semi-major, 4.93° above 120 px. The size effect dominates: a few pixels of landmark error is several degrees on a small heart, which is the resolution argument appearing in data rather than in prose.

---

## 12. External validation

> **Question.** Does the model generalise to a different hospital and different ultrasound machines, when no orientation ground truth exists there?
>
> **Method.** Run the detector and the label-free property tests on 500 images sampled from the 1,718 `Fetal thorax` images of FETAL_PLANES_DB, stratified by the recorded ultrasound machine. Nothing from this dataset was ever trained on.
>
> **Result.** The detector fires on **93 %** at mean confidence 0.75.
>
> | Property | median: FOCUS → external | p90: FOCUS → external |
> |---|---|---|
> | rotation ±15° | 2.9–3.6° → 3.9–4.3° | 7.4–9.0° → **20.5–21.9°** |
> | rotation ±30° | 3.7–4.7° → 5.3–6.5° | 10.0–12.5° → **29.2–39.6°** |
> | mirror | 2.6° → 6.5° | 12.0° → **44.3°** |
> | gain | 0.7° → 2.0° | 3.9° → 8.0° |
> | crop scale | 3.7° → **11.8°** | 8.6° → **44.0°** |
>
> By machine: 95 % firing on Voluson E6, **81 % on Aloka** at the lowest mean confidence of the four. Aloka is 41 % of the external dataset and appears nowhere in FOCUS.
>
> ![Internal versus external self-consistency, and detection by machine](figures/external.png)
>
> **Interpretation.** The medians move modestly; the tails triple. The model still works on typical external images and fails outright on a minority — a distinction that any mean-based summary erases. Crop-scale sensitivity rising from 3.7° to 11.8° is the most specific finding: it indicates the model partly learned the FOCUS crop convention rather than the anatomy, which is a training-time fix (wider crop jitter) and not a data problem.
>
> Worth contrasting with §6: the closed-form estimator has no distribution to be out of. It is the same arithmetic on any mask from any machine, and its failure modes are geometric and enumerable. That asymmetry, more than the accuracy gap, is the argument for reading the angle off a segmentation whenever one exists.

---

## 13. Metamorphic testing

> **Question.** Can the estimator be validated where labels do not exist?
>
> **Method.** Assert four properties that must hold on any image: rotate by δ and the axis moves by δ; mirror and the axis reflects; change gain and contrast and the axis does not move; widen the crop and the axis does not move. Expected values are computed from the actual warp matrix, so the test carries no coordinate convention of its own.
>
> **Result.**
> ```
> rotation equivariance ±15°   median 2.9–3.6°   p90  7.4–9.0°
> rotation equivariance ±30°   median 3.7–4.7°   p90 10.0–12.5°
> mirror equivariance          median 2.55°      p90 12.00°
> gain invariance              median 0.74°      p90  3.90°   max 5.31°
> crop-scale invariance        median 3.65°      p90  8.63°
> ```
> All fail the tolerances set in the file. The tolerances were not relaxed.
>
> **Interpretation.** Rotation self-consistency (3–5° median) is the same order as the model's own test error, so the residual is genuine model variance rather than a coordinate bug. **Gain invariance is violated by up to 5°**: a pure brightness change moves an anatomical measurement. That is a concrete defect, localised without any annotation, and it points at intensity normalisation or stronger gain augmentation.

The suite justified itself twice, in ways a held-out set could not have.

**First**, its initial run reported errors of *exactly* twice the applied rotation on every image. Errors of exactly 2δ are the signature of a flipped sign, and the flip was in the test's own expected value. Deriving the expectation from the warp matrix removed the possibility.

**Second**, run against a deliberately undertrained checkpoint it returns rotation errors of almost exactly δ — the signature of a model predicting a near-constant angle regardless of input. Detecting "the model ignores the image" without ground truth is precisely what a deployment monitor must do, since labels never arrive in production.

---

## 14. Robustness to segmentation failure

This is the experiment that makes the closed-form route usable as an answer rather than a curiosity.

> **Question.** If an existing pipeline supplies the mask, how good does that mask have to be?
>
> **Method.** `fho.no_training` degrades the ground-truth cardiac masks in four parameterised ways that model how segmenters actually fail — morphological erosion and dilation (systematic under- and over-segmentation), a smooth random field perturbing the contour (ragged boundary), a wedge removed (a chamber lost to shadowing), and an added component (neighbouring tissue included). Each corrupted mask is scored by Dice against the true mask, and the resulting angle error is measured with and without a two-line cleanup: keep the largest connected component, then morphologically open.
>
> **Result.**
>
> | Failure mode | Dice | raw mask | after cleanup |
> |---|---|---|---|
> | erosion | 0.77 | 0.22° | **0.22°** |
> | dilation | 0.82 | 0.40° | **0.40°** |
> | ragged contour | 0.66 | 40.20° | **1.72°** |
> | chunk missing | 0.83 | 20.87° | 21.13° |
> | adjacent tissue included | 0.87 | 46.21° | 44.73° |
>
> ![Angle error against mask quality and against failure mode](figures/no_training.png)
>
> **Interpretation.** Four conclusions, none of which follows from the Dice score alone.
>
> **Symmetric error is free.** Eroding or dilating to Dice 0.77 costs 0.22°. Second moments depend on how mass is *distributed*, not on how thick the mask is, so the most common complaint about a segmenter is irrelevant to this measurement.
>
> **Cleanup is not optional.** Largest connected component plus opening takes a ragged contour from 40.2° to **1.7°**.
>
> **Asymmetric mass survives cleanup.** A missing chunk (21°) or tissue leaking across a contiguous boundary (45°) cannot be removed by connected components, because the spurious mass is attached. This is the failure mode to look for in any real segmenter, and it is what would determine whether the approach works on a given pipeline.
>
> **Dice does not predict the angle error.** A mask at Dice 0.87 gives 46°; a mask at Dice 0.77 gives 0.22°. The left panel of the figure is a cloud, not a curve. "Our segmenter scores 0.9" is not an answer to "will the orientation be right".

---

## 15. Oriented bounding box analysis

An axis-aligned box has four degrees of freedom; an oriented box has five, or eight if the corners are stored directly, which is what FOCUS does. The fifth number is the measurement.

> **Question.** What does collapsing the oriented annotation to axis-aligned cost, and what does the orientation head recover?
>
> **Method.** Compute the area ratio between each oriented box's enclosing axis-aligned box and the oriented box itself, across all 300 images, and plot against the box's own orientation. Separately, compute rotated IoU (via convex-polygon intersection) between the annotation and both the predicted oriented box and the axis-aligned box.
>
> **Result.** Median area ratio **×1.97**, p90 ×2.05, above ×1.5 on 83 % of cases. Median rotated IoU: **0.83** for the predicted oriented box, **0.51** for the axis-aligned box.
>
> ![Cost of dropping the angle, and what the orientation head recovers](figures/obb_cost.png)
> ![Oriented box versus the axis-aligned box the detector is given](figures/oriented_boxes.png)
>
> **Interpretation.** The empirical points follow the analytic curve `(|cos θ| + k|sin θ|)(|sin θ| + k|cos θ|)/k` for the mean aspect ratio `k = 0.73`, which is zero-cost at 0° and 90° and peaks at 45°. **The annotations cluster at 45° and 135°** — where a fetal heart sits in a correctly obtained four-chamber view — so the near-worst case is the ordinary case. The axis-aligned box sits at 0.51 IoU, right on the threshold at which most detection benchmarks stop counting a box as correct.

![Annotated and predicted oriented boxes across the angle range](figures/obb_gallery.png)

**A consequence for metric choice.** Rotated IoU decays with angle error at a rate governed entirely by aspect ratio:

![Rotated IoU against angle error, by aspect ratio](figures/obb_iou_sensitivity.png)

At 1:1 the angle is meaningless and IoU barely moves; at 4:1 a 10° error costs a quarter of the IoU and 20° breaks the 0.5 threshold; at 8:1 a 10° error is nearly fatal. A fetal heart is around 1.4:1, so its mAP looks forgiving and hides exactly the quantity of interest. **An oriented-box metric is an angle metric**, and on a near-square object the angle error should be reported directly.

### Why not a rotated detector

`mmrotate`, YOLO-OBB and similar predict θ inside the detector and would collapse the two stages into one. That is a legitimate design, not chosen here for three reasons.

**Angle periodicity plus dataset size.** θ modulo 180° and the near-square degeneracy require dedicated machinery — doubled-angle encodings, circular smooth labels, Gaussian-Wasserstein or KLD losses treating the box as a 2-D Gaussian. Those exist because the problem is real, and they are more than 200 training images support well.

**Interpretability.** The landmark head returns four points a reviewer can reject individually. A rotated detector returns a number.

**Separable failure modes.** Detection transfers to a second hospital at 93 %; orientation degrades sharply out of distribution. A single combined metric would have hidden that, and since the angle *is* the measurement, it deserves its own number rather than being folded into mAP.

---

## 16. Ablation and failed approaches

Two architectural choices were selected by failure rather than by prior belief. Both are retained here because they are the transferable part.

> **Question.** Should the landmarks be predicted as heatmaps, the standard formulation for landmark localisation?
>
> **Method.** Per-landmark Gaussian heatmaps at stride 4, trained with a heatmap loss and a soft-argmax coordinate head.
>
> **Result.** Plateaued at approximately **28° median error**, against 45° for random guessing on axial data.
>
> **Interpretation.** An ellipse axis endpoint has **no distinctive local appearance**. It is a point on a smooth boundary, defined by a *global* property of the shape — the extremum along a direction — so a receptive field centred on it sees what it would see a few pixels along the contour. Heatmap regression is the right tool where a landmark carries local evidence (an apex, a valve hinge, a vertebral body) and the wrong tool here. A second, subtler failure compounded it: a soft-argmax over heatmaps trained at MSE scale produces a nearly uniform distribution, so the coordinate output collapses toward the crop centre while the heatmap loss continues to fall — a silent failure that looks like training progress.

> **Question.** How should the convolutional features be pooled before the regression heads?
>
> **Method.** Compare global average pooling to a 3×3 adaptive spatial grid, all else equal.
>
> **Result.** Global average pooling plateaued around **21°**; the 3×3 grid converged to **4.4°** on validation.
>
> **Interpretation.** Global average pooling is very nearly orientation-invariant: it discards exactly the spatial layout that encodes the angle. Obvious in hindsight, and the last thing checked.

Sequence on the same validation split: **heatmaps 28° → global pooling 21° → spatial grid 4.4°.**

---

## 17. Confidence and abstention

Three candidate confidence signals were tested rather than assumed.

> **Question.** Does the eigengap standard error of the PCA estimator identify unreliable predictions?
>
> **Method.** Correlate the first-order standard error `Var(θ) ≈ λ₁λ₂/(λ₁−λ₂)²/n` against actual angle error across every corruption in §14.
>
> **Result.** **r = +0.03.**
>
> **Interpretation.** It fails, and it fails for a specific and instructive reason: spurious attached mass *increases* the spread along one direction, which **widens** the eigengap. The estimator therefore becomes more confident as it becomes wrong. A theoretically motivated uncertainty is still a hypothesis until it is measured.

> **Question.** Do the learned model's internal consistency signals support abstention?
>
> **Method.** Risk–coverage curves for head disagreement, major/minor axis disagreement, predicted elongation and heart size, against an oracle.
>
> **Result.** Dropping to 53 % coverage moves the median from 7.04° to 5.40°, and the p90 barely moves. The best signal is simply **heart size**; predicted elongation is *worse than nothing*, in that abstaining by it raises the median error.
>
> ![Risk-coverage for each candidate confidence signal](figures/risk_coverage.png)
>
> **Interpretation.** A confidence signal that does not reduce error is not a confidence signal, and shipping it as one would be worse than shipping none. `predict.py` still exposes `assessable` with thresholds on roundness, axis disagreement and head disagreement — because they are the right *shape* of rule and the roundness condition is geometrically sound — but the risk–coverage result is reported alongside, and the honest summary is that this model does not yet know when it is wrong.

---

## 18. Failure analysis

Consolidated, with what each finding implies for the next iteration.

| Finding | Evidence | Implication |
|---|---|---|
| Heatmaps unsuitable for globally-defined landmarks | 28° plateau | Choose the localisation formulation from the landmark's evidence structure, not from convention |
| Global average pooling harms orientation | 21° → 4.4° | Preserve spatial layout when the target is geometric |
| Minimum-area rectangle unstable | 44 % beyond 10° | Optimising area is not optimising axis |
| Eigengap confidence uninformative | r = +0.03 | Validate uncertainty empirically |
| Model confidence weakly informative | flat risk–coverage | Do not ship an unvalidated abstention rule |
| Crop-scale sensitivity | 3.65° → 11.82° external | Widen crop jitter; the model learned a convention |
| Gain invariance violated | up to 5° | Intensity normalisation before the network |
| External tails triple | p90 10–12° → 29–40° | Report tails and covariates, not means |
| Asymmetric mask error unfixable | 21–45° after cleanup | Characterise the segmenter's failure mode, not its Dice |
| Sign error in the test itself | errors of exactly 2δ | Derive test expectations from the transform, not from a convention |

---

## 19. Limitations

- **This is not the clinical cardiac axis.** That quantity is measured against the thoracic spine-to-sternum midline; FOCUS annotates neither spine nor septum. What is measured is the heart's long axis in the image frame. `geometry.cardiac_axis()` combines the two and needs one additional landmark.
- **±18° limits of agreement are not clinically useful** against a normal band roughly 40° wide.
- **External evaluation covers self-consistency, not accuracy.** No orientation labels exist outside FOCUS, so external error against a reference is unmeasured.
- **The segmentation-failure study is simulated.** The corruptions are plausible and parameterised, but they model failure rather than sample a real segmenter's output.
- **No human reader ceiling.** Two clinicians measuring the same clip disagree by some amount, and no model beats that. Without it, 7° has no reference point.
- **300 training images from a single source**, no gestational-age stratification, no multi-site training.
- **The 0.28° moments figure is a floor**, since the masks derive from the same ellipse annotations.
- **Not a medical device. No clinical validation. Not for clinical use.**

---

## 20. Conclusions

The central result is that **orientation does not have to be learned**. Where a segmentation exists, second-order moments recover the axis analytically, two orders of magnitude more accurately than the trained model, with no distribution to be out of and with failure modes that are geometric and enumerable rather than empirical. The learned landmark model earns its place only in the case where a detector returns a box and no mask.

The second result is methodological. Three of the most useful findings here — the gain-invariance violation, the crop-scale shortcut, and the external tail degradation — came from tests that used **no labels at all**, and the same tests transfer to any dataset and keep working after deployment. Two more — the heatmap failure and the pooling failure — came from experiments that produced negative results and thereby selected the architecture.

The third is a caution. Several quantities that ought to have worked did not: a theoretically motivated uncertainty estimate correlated with error at r = +0.03, and the model's own consistency signals bought almost no accuracy through abstention. Both are reported rather than dropped, because a pipeline that cannot tell when it is wrong is a different product from one that can, and the difference is only visible if it is measured.

### Next steps

1. Annotate one spine landmark on the 300 images, converting every number here into the clinical quantity; measure both error terms separately afterwards, since the midline term is expected to dominate.
2. Establish the reader ceiling with two or three clinicians measuring the same clips and one measuring twice.
3. Repeat §14 against a real segmenter's masks rather than simulated corruptions.
4. Fix the two localised defects: crop-scale sensitivity and gain sensitivity.
5. Evaluate the decision rather than the angle — abnormal axis at the normative cut-off — since a 3° error is irrelevant mid-band and decisive at the boundary.

---

## Reproducing

```bash
git clone https://github.com/francescovigni/fetal-cardiac-orientation.git
cd fetal-cardiac-orientation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

make data test baselines no-training   # closed-form route, no GPU needed
make yolo train eval meta figures      # learned route
make data-external external            # cross-dataset run
```

Verified from a clean clone in an empty environment: unit tests pass, the datasets download, the annotation cross-check reproduces to the same 0.033°, the baselines reproduce exactly, and training runs on CPU as well as MPS.

Data: *FOCUS*, Zenodo `10.5281/zenodo.14597550`; Burgos-Artizzu et al., *FETAL_PLANES_DB*, Zenodo `10.5281/zenodo.3904280`. Both CC-BY-4.0. Code MIT.
