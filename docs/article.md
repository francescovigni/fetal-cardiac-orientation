# Estimating fetal cardiac orientation, and knowing when it is wrong

There is a question that comes up whenever an imaging pipeline already exists and someone wants a new geometric measurement out of it:

> **If the pipeline already finds the organ, can the orientation be read off directly — without training a network for it?**

For the fetal heart in a four-chamber ultrasound, the answer is **yes, if you have a mask, and only if you know how that mask fails**. This is the measured version of that answer, on public data, with the failure modes characterised rather than assumed.

**Code:** [github.com/francescovigni/fetal-cardiac-orientation](https://github.com/francescovigni/fetal-cardiac-orientation) · **Data:** [FOCUS](https://zenodo.org/records/14597550) and [FETAL_PLANES_DB](https://zenodo.org/records/3904280), CC-BY-4.0

| Route | Input needed | Median error | Fails when |
|---|---|---|---|
| Second-order moments | a mask | **0.28°** clean, 0.2–1.7° under symmetric or ragged error | mass is added or removed **asymmetrically** — 21–45° |
| Minimum-area rectangle | a mask | 4.95°, **44 % beyond 10°** | always, on near-elliptical shapes: it is bimodal |
| Learned landmark model | only a box | **7.0°**, limits of agreement ±18° | out of distribution: p90 triples on a second hospital |

## Why the geometry is awkward before any of that

Cardiac axis is the angle between the interventricular septum and the thoracic anteroposterior midline, near 45° normally, and deviation is an independent screening marker for congenital heart disease.

It is **a difference of two angles**, and the thoracic reference is usually the noisier term. It is **axial, not directional** — defined modulo 180°, and turning it into a direction is levocardia versus dextrocardia, a diagnosis needing the spine or stomach bubble rather than a cropped heart. And **fetal lie is arbitrary**, so there is no canonical orientation to lean on, which makes rotation equivariance a property the estimator genuinely must have.

## The closed-form route, and where it breaks

Given a mask, the axis is the principal eigenvector of the covariance of the set pixels — probability-weighted if the mask is soft, and computed in physical units, because anisotropic pixel spacing skews the axis by the pixel aspect ratio. It is closed form, differentiable, and takes microseconds.

On the undamaged FOCUS masks it recovers the annotated angle to **0.28° median**. That number on its own would be dishonest: those masks are rasterised from the same ellipse annotations, so it measures the numerical consistency of the geometry code, not accuracy on a real segmenter's output.

The informative experiment is what happens when the mask is wrong. `fho.no_training` degrades the ground-truth masks the way segmenters actually fail — systematic under- and over-segmentation, a ragged contour, a chunk lost to shadowing, neighbouring tissue leaking in — scores each corruption by Dice, and measures the resulting angle error.

![Angle error against mask quality and against failure mode](figures/no_training.png)

| Failure mode | Dice | raw mask | after cleanup |
|---|---|---|---|
| erosion | 0.77 | 0.22° | **0.22°** |
| dilation | 0.82 | 0.40° | **0.40°** |
| ragged contour | 0.66 | 40.20° | **1.72°** |
| chunk missing | 0.83 | 20.87° | 21.13° |
| adjacent tissue included | 0.87 | 46.21° | 44.73° |

**Symmetric error is free.** Eroding or dilating until Dice falls to 0.77 costs 0.22°. Second moments do not care how thick the mask is, only how the mass is distributed — so the most common complaint about a segmenter turns out to be irrelevant to this measurement.

**Two lines of cleanup are not optional.** Keeping the largest connected component and morphologically opening it takes a ragged contour from **40.2° to 1.7°**. Skip it and a noisy boundary destroys the estimate — and a detached blob does something worse than destroy it, because the extra mass *widens* the eigengap, so the estimator becomes more confident as it becomes wrong.

**Asymmetric mass is what survives cleanup.** A missing chunk (21°) or tissue leaking across a contiguous boundary (45°) cannot be removed by connected components, because the spurious mass is attached. This is the failure to look for in any real segmenter, and it is what would decide whether the approach works on a given pipeline.

**Dice does not predict the angle error.** A mask at Dice 0.87 gives 46°; a mask at Dice 0.77 gives 0.22°. The scatter on the left of the figure is a cloud, not a curve. "Our segmenter scores 0.9" does not answer "will the orientation be right".

Two cautions that cut against the method:

The **obvious confidence signal does not work**. PCA offers a first-order standard error from the eigengap, `Var(θ) ≈ λ₁λ₂/(λ₁−λ₂)²/n`, which ought to flag ill-conditioned cases. Its correlation with actual error across all corruptions is **r = +0.03**. It is fooled by precisely the failure that matters. Abstention has to come from elsewhere — a second estimator, or temporal consistency across frames of the same exam.

And the **other closed-form estimator is a trap**. The minimum-area enclosing rectangle, from rotating calipers on the convex hull, is exact for area and unstable for axis: for an ellipse the enclosing rectangle reaches the same minimum `4ab` at both the major and the minor alignment, so it is genuinely bimodal and picks one of two optima 90° apart. Median 4.95°, and 44 % of cases beyond 10°, on undamaged masks. Picking the wrong closed-form estimator costs far more than not training one.

## When there is no mask, only a box

If the pipeline returns a box, the moments have nothing to work on and the angle must be learned. That is the rest of the repository: YOLOv5 for the cardiac and thoracic regions, then a landmark head predicting the endpoints of the cardiac ellipse's axes.

Landmarks rather than a scalar angle because the output is inspectable — a clinician can look at four points and say they are wrong, and nobody can audit a number. Three things had to fail before it worked.

**The 180° endpoint swap.** An ellipse is invariant under a 180° rotation, which exchanges both pairs of endpoints at once, so two labellings are equally correct and any fixed convention is discontinuous somewhere. Under rotation augmentation the network then receives contradictory targets for visually identical crops. The fix is a loss scoring both assignments and keeping the better one — the landmark analogue of the `(sin 2θ, cos 2θ)` encoding that cures the same disease in oriented-box regression.

**Heatmaps are the wrong estimator for these landmarks.** Per-landmark Gaussian heatmaps, the default choice, plateaued at ~28° median against 45° for chance. An ellipse axis endpoint has no distinctive local appearance: it is a point on a smooth boundary defined by a global property of the shape. Heatmaps are right for an apex or a valve hinge and wrong here.

**Global average pooling is almost orientation-invariant.** Pooling to 1×1 discards the spatial layout, which is where the angle lives.

Measured on the same validation split: heatmaps 28°, global regression with average pooling 21°, global regression with a 3×3 spatial grid 4.4°.

### Results, read honestly

```
median |error|     7.04°   95 % CI [4.84, 9.26]
p90    |error|    12.92°
Bland-Altman       bias -0.55°   LoA [-18.23, +17.12]
ICC(2,1)           0.980
```

![Bland-Altman and error distribution](figures/agreement.png)

Four numbers saying four different things. The **bias of −0.55°** means no systematic rotation error, the failure that would matter most clinically and the one a mean absolute error hides. The **ICC of 0.980** is the flattering one — cardiac angles span a wide range and ICC rewards tracking it. The **limits of agreement, ±18°**, are what a clinician would ask for, against a clinical normal band roughly 40° wide. And the gap between **4.41° on validation and 7.04° on test**, with a bootstrap interval of [4.84, 9.26], says both that there is a real generalisation gap on 200 training images and that the test estimate is itself loose. Training had not converged.

So the learned route is an order of magnitude worse than the closed-form one, and it exists only for the case where no mask is available.

Stratifying says more than the headline. By roundness, 6.70° for `b/a` in 0.60–0.75 against 7.93° for 0.75–0.90. By size, 9.26° at 90–120 px semi-major against 4.93° above 120 px — a few pixels of landmark error is several degrees on a small heart, which is the resolution argument appearing in data rather than in a paragraph.

![Risk-coverage for each candidate confidence signal](figures/risk_coverage.png)

The model's internal confidence signals do not work either. Dropping to 53 % coverage moves the median from 7.04° to 5.40°, and the p90 barely moves. The best signal available is not the two heads agreeing — it is simply **heart size**. Predicted elongation is worse than nothing: abstaining by it makes the median error rise. A confidence signal that does not reduce error is not a confidence signal.

## Validating without labels

A held-out set is not the only instrument, and on 50 images it is a blunt one. Four properties can be asserted with no ground truth at all: rotate the input by δ and the axis must move by δ; mirror it and the axis must reflect; change brightness and contrast and it must not move, because neither is anatomy; widen the crop and it must not move.

```
rotation equivariance ±15°   median 2.9–3.6°   p90  7.4–9.0°
rotation equivariance ±30°   median 3.7–4.7°   p90 10.0–12.5°
mirror equivariance          median 2.55°      p90 12.00°
gain invariance              median 0.74°      p90  3.90°   max 5.31°
crop-scale invariance        median 3.65°      p90  8.63°
```

Every one fails the tolerances set in the file, and the tolerances were not relaxed to make them pass. Rotation self-consistency, 3° to 5°, is the same order as the model's own test error, so the residual is model variance rather than a coordinate bug. **Gain invariance is violated by up to 5°**: a pure brightness change moves an anatomical measurement, which is a concrete defect a test set would never have surfaced.

The suite earned its place twice. Its first run reported errors of exactly twice the applied rotation on every image — errors of exactly 2δ being the signature of a flipped sign, which was in the test's own expected value. The expectation is now derived from the warp matrix itself, so the test has no convention left to get wrong. And run against a deliberately undertrained checkpoint it gives rotation errors of almost exactly δ: the signature of a model predicting a near-constant angle regardless of input. Detecting "the model ignores the image" with no labels is what you want running as a monitor in production, where labels never arrive.

## Testing it somewhere else, still with no labels

The obvious objection is 300 images from one source. The obvious obstacle to an external set is that orientation ground truth does not exist outside FOCUS. It does not matter, because those four properties hold on any image, so the same code runs unchanged on [FETAL_PLANES_DB](https://zenodo.org/records/3904280): a different hospital, different operators, four ultrasound machines, 1,718 images of the thorax plane.

The detector fires on **93 %** of them at mean confidence 0.75, having never seen the dataset.

| Property | median: FOCUS → external | p90: FOCUS → external |
|---|---|---|
| rotation ±15° | 2.9–3.6° → 3.9–4.3° | 7.4–9.0° → **20.5–21.9°** |
| rotation ±30° | 3.7–4.7° → 5.3–6.5° | 10.0–12.5° → **29.2–39.6°** |
| mirror | 2.6° → 6.5° | 12.0° → **44.3°** |
| gain | 0.7° → 2.0° | 3.9° → 8.0° |
| crop scale | 3.7° → **11.8°** | 8.6° → **44.0°** |

![Internal versus external self-consistency, and detection by machine](figures/external.png)

The medians move modestly. The tails triple. That gap is the result: the learned model still works on typical external images and fails outright on a minority, and a summary reporting only a mean would have hidden it. Crop-scale sensitivity going from 3.7° to 11.8° is the most specific finding — it says the model partly learned the FOCUS crop convention rather than the anatomy, which is a training-time fix. By machine, the detector fires on 95 % of Voluson E6 images and **81 % of Aloka** images at the lowest mean confidence of the four; Aloka is 41 % of the external dataset and appears nowhere in FOCUS.

It is worth putting the two routes side by side here. The closed-form estimator has no distribution to be out of: it is the same arithmetic on any mask from any machine, and its failure modes are geometric and enumerable. The learned model has to be re-validated on every new source. That asymmetry, more than the accuracy gap, is the argument for reading the angle off an existing segmentation whenever one exists.

## What this does not show

- **Not the clinical cardiac axis.** It is the heart's long axis in the image frame; the clinical quantity is measured against the spine-to-sternum midline, and FOCUS annotates neither spine nor septum. The function that would combine them is written and needs one spine landmark.
- **The closed-form results are against simulated segmentation failure**, not against a real segmenter's output. The corruptions are plausible and parameterised, but they are a model of failure, not a sample of one.
- **±18° limits of agreement are not clinically useful** for the learned route.
- **External evaluation covers self-consistency, not accuracy** — there are no orientation labels outside FOCUS.
- **No human reader ceiling**, so 7° has no reference point. No gestational-age stratification.
- **Not a medical device**, and not validated for clinical use.

## What would close the gap

Annotate one spine point on the 300 images — a couple of hours, and it converts every number here into the clinical quantity. Measure both error terms separately afterwards, because the midline is expected to dominate and optimising the heart term first would be wasted effort. Establish the reader ceiling with two or three clinicians measuring the same clips and one measuring twice, so the target is a number rather than an aspiration. Evaluate the decision rather than the angle — abnormal axis, yes or no, at the normative cut-off, since a 3° error is irrelevant mid-band and decisive at the boundary. And run the degradation study against a real segmenter's masks rather than simulated ones, which is a day's work once such a segmenter exists.

## What this would take on your data

If a segmentation already exists, the closed-form route is an afternoon: moments, the two-line cleanup, and the degradation study run against your own segmenter's failures rather than simulated ones, which tells you immediately whether it is good enough. If only a detection box exists, it is a two-week piece of work, and the first week goes on counting confirmed positives per class before any modelling starts.

The transferable part is not the model. It is the harness: annotation cross-validation before trusting labels, a swap-invariant loss for the axial ambiguity, agreement statistics instead of accuracy, stratification by the covariate that actually breaks the estimator, risk-coverage for abstention, and label-free equivariance tests that double as a deployment monitor and as a cross-dataset generalisation probe.

## Reproducing

```bash
git clone https://github.com/francescovigni/fetal-cardiac-orientation.git
cd fetal-cardiac-orientation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data test baselines no-training      # the closed-form route, no GPU needed
make yolo train eval meta figures         # the learned route
make data-external external               # the cross-dataset run
```

Verified from a clean clone in an empty environment: unit tests pass, the dataset downloads, the annotation cross-check reproduces to the same 0.033°, the baselines reproduce exactly, and training runs on CPU as well as MPS.

Both datasets are CC-BY-4.0 and must be cited: *FOCUS*, Zenodo `10.5281/zenodo.14597550`; Burgos-Artizzu et al., *FETAL_PLANES_DB*, Zenodo `10.5281/zenodo.3904280`.
