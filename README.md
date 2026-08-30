# fetal-cardiac-orientation

Finding the fetal heart in a four-chamber ultrasound and estimating its long axis, on public data ([FOCUS](https://zenodo.org/records/14597550), CC-BY-4.0), reproducible from a clean clone.

The model is ordinary. The interesting part is what it takes to know whether it works — including on a second hospital's data where no orientation labels exist at all.

![Predicted heart long axis against the annotation: best, median and worst test cases](docs/figures/qualitative.png)

Full write-up: **[docs/article.md](docs/article.md)**

## The result worth reading

The metamorphic properties an orientation estimator must satisfy — rotate the input, the axis rotates with it; change the brightness, it does not move — hold on **any** image, annotated or not. So the same suite runs unchanged on [FETAL_PLANES_DB](https://zenodo.org/records/3904280): a different hospital, four ultrasound machines, 1,718 thorax images, zero labels.

| Property | median: FOCUS → external | p90: FOCUS → external |
|---|---|---|
| rotation ±15° | 2.9–3.6° → 3.9–4.3° | 7.4–9.0° → **20.5–21.9°** |
| rotation ±30° | 3.7–4.7° → 5.3–6.5° | 10.0–12.5° → **29.2–39.6°** |
| mirror | 2.6° → 6.5° | 12.0° → **44.3°** |
| gain | 0.7° → 2.0° | 3.9° → 8.0° |
| crop scale | 3.7° → **11.8°** | 8.6° → **44.0°** |

![Internal versus external self-consistency, and detection by machine](docs/figures/external.png)

**The medians move a little. The tails triple.** The model still works on typical external images and fails outright on a minority — a distinction any average would erase. Crop-scale sensitivity going from 3.7° to 11.8° says it partly learned the FOCUS crop convention rather than the anatomy, which is a training fix, not a data problem.

Detection transfers better: it fires on 93 % of external images at mean confidence 0.75, but on **81 % of Aloka** images against 95 % for Voluson E6. Aloka is 41 % of that dataset and appears nowhere in FOCUS.

## In-distribution numbers

Held-out test split, 50 images, trained on 200.

| Detection (YOLOv5s) | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| cardiac | 0.990 | 1.000 | 0.995 | 0.637 |
| thorax | 0.999 | 1.000 | 0.995 | 0.660 |

Table stakes: one organ, one view, centred, always present.

| Orientation | |
|---|---|
| median absolute error | 7.04° (95 % CI 4.84–9.26) |
| Bland-Altman bias | −0.55° |
| 95 % limits of agreement | −18.2° to +17.1° |
| ICC(2,1) | 0.980 |

Unbiased, and the limits of agreement are the number that counts: a single scan can be misplaced by ±18° against a clinical normal band roughly 40° wide. Validation median was 4.41°, so there is a real generalisation gap on 200 images, and training had not converged.

![Bland-Altman and error distribution](docs/figures/agreement.png)

**Abstention does not work yet, and that is reported rather than hidden.** No internal confidence signal buys much accuracy. The best candidate is simply *heart size*; predicted elongation is worse than nothing, since abstaining by it raises the median error.

![Risk-coverage for each candidate confidence signal](docs/figures/risk_coverage.png)

**Classical baselines** on the ground-truth masks: PCA moments 0.28° median, minimum-area rectangle 4.95° median with 44 % of cases beyond 10°. Minimum-area fails because an ellipse's enclosing rectangle hits the same minimum area at both the major and the minor alignment, so the estimator is bimodal. Exact for area, unstable for axis. (The masks are rasterised from the same ellipses, so 0.28° measures geometry-code consistency, not accuracy.)

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make data test baselines      # download, 9 unit tests, closed-form estimators
make yolo train               # detector, then the orientation model (~25 min, M-series GPU)
make eval meta figures        # agreement statistics, metamorphic suite, figures
make data-external external   # the external run above (2.1 GB download)
```

Device is chosen automatically: MPS, CUDA, or CPU.

## Design notes

- **Annotations are cross-checked before being trusted.** FOCUS stores each structure as an ellipse *and* an independent oriented box; they agree to 0.07 px and **0.033°** across all 200 training images. That is what licenses using the angle as ground truth.
- **The 180° endpoint swap.** An ellipse is invariant under 180° rotation, which exchanges both endpoint pairs, so any fixed labelling convention is discontinuous and the network gets contradictory targets. Solved with a swap-invariant loss.
- **Heatmaps are the wrong estimator here.** Axis endpoints have no distinctive local appearance; they are defined by a global property of the shape. Heatmaps plateaued at ~28°, against 45° for chance.
- **Global average pooling is nearly orientation-invariant.** It discards the spatial layout that encodes the angle. A 3×3 grid instead of 1×1 was the difference between plateauing and converging.
- **Angles are axial**, defined modulo 180°, handled in the doubled-angle representation and aggregated with circular statistics. Direction (apex-left vs apex-right) is levocardia versus dextrocardia — a diagnosis needing the spine or stomach bubble, not something to infer from a cropped heart.
- **Augmentation follows the physics**, not the defaults: rotation ±30° because fetal lie is arbitrary, no vertical flip because no probe swaps near and far field, no hue because the images are grayscale, no mixup because blending two fetal hearts produces anatomy that does not exist.

## What this does not show

- **Not the clinical cardiac axis**, which is measured against the spine-to-sternum midline. FOCUS annotates neither spine nor septum. `geometry.cardiac_axis()` is one spine landmark away.
- **±18° limits of agreement are not clinically useful.**
- **External evaluation covers self-consistency, not accuracy** — there are no orientation labels outside FOCUS.
- **No reader ceiling**, so 7° has no reference point. No gestational-age stratification.
- **Not a medical device.** Not validated for clinical use.

## Layout

`src/fho/` — `focus.py` parsing and annotation cross-validation · `geometry.py` axial angle algebra, circular statistics, PCA and min-area baselines · `landmarks.py` crops and augmentation · `model.py` network and swap-invariant loss · `evaluate.py` Bland-Altman, ICC, stratification, risk-coverage · `metamorphic.py` label-free tests · `external.py` the cross-dataset run · `figures.py` · `predict.py` end to end.

## Licence

Code MIT. FOCUS and FETAL_PLANES_DB are CC-BY-4.0 and must be cited: *FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric Measurement*, Zenodo `10.5281/zenodo.14597550`; Burgos-Artizzu et al., *FETAL_PLANES_DB*, Zenodo `10.5281/zenodo.3904280`.
