# Sample image

`four_chamber_002.png` is image `002` of the FOCUS testing split, redistributed here so
that `make demo` runs from a clean clone without downloading the dataset.

**It was chosen as a median case, not a best case.** A showcase image would have been
the 0.3° one, and that would have misrepresented the model.

| | angle | error |
|---|---|---|
| annotation | 138.6° | — |
| model, from the **ground-truth crop** | 131.7° | 6.9° (test-set median is 7.04°) |
| model, **end to end through the detector** | 127.2° | 11.4° |

The 4.5° gap between the last two rows is what stage 1 costs stage 2 on this image:
the detector's box is not the annotated box, so the crop differs and the orientation
moves with it. That is the same effect the round-trip control quantifies across the
whole split.

FOCUS is licensed **CC-BY-4.0**, which permits redistribution with attribution:

> *FOCUS: Four-chamber Ultrasound Image Dataset for Fetal Cardiac Biometric
> Measurement*, Zenodo, doi:10.5281/zenodo.14597550
