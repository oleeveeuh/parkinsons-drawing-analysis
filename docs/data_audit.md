# Data audit

Verified inventory of the two public datasets used in this project, the
subject leakage found in them, and the consequences for evaluation.
All counts below were produced by `scripts/audit_data.py` from filename
parsing and byte-level comparison — regenerate with:

```bash
python scripts/audit_data.py \
  --spiral-root KaggleSpiral/drawings \
  --temporal-root "Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/alldata" \
  --output results/data_audit.json
```

## Spatial dataset — Kaggle "Parkinson's Drawings" (CC BY-NC-ND 4.0)

### Filename parsing rule

Drawing filenames encode participant, class group and drawing number:

| Filename | Participant | Group | Drawing | Class-aware subject id |
|---|---|---|---|---|
| `V01PE02.png` | `V01` | `PE` (Parkinson) | 2nd | `parkinson:V01` |
| `V01HE02.png` | `V01` | `HE` (Healthy) | 2nd | `healthy:V01` |
| `V55HO15.png` | `V55` | `HO` (Healthy) | 15th | `healthy:V55` |

Group codes observed: `PE`/`PO` (Parkinson's) and `HE`/`HO` (Healthy).
Numeric codes are **namespaced by class** (`parkinson:V01` ≠ `healthy:V01`);
they are different people. All grouping in this project uses the
class-aware id.

### Verified inventory (one row per unique image)

| | Spiral | Wave |
|---|---|---|
| Images | 102 (51 PD / 51 HC) | 102 (51 PD / 51 HC) |
| Unique participants | **28** (15 PD / 13 HC) | **28** (15 PD / 13 HC) |
| Participants shared between tasks | **28 of 28** — the same 28 people drew both | — |
| Drawings per participant | 3 (one PD participant has 9; one HC participant has 15) | same |

Two participants contribute far more drawings than the rest:
`parkinson:V03` (9 drawings/task) and `healthy:V55` (15 drawings/task).
Image-level class counts (51/51) therefore *overstate* participant-level
balance (15/13) and give those two participants disproportionate weight
in any image-level analysis.

### Duplication

The dataset ships each task twice: `KaggleSpiral/spiral|wave/` and
`KaggleSpiral/drawings/spiral|wave/` are byte-identical (verified by
MD5, 204 files each way). The loaders read one copy; the duplicate tree
was removed from Git tracking.

### Supplied train/test split is subject-leaky

| Task | Train images | Test images | Train subjects | Test subjects | Subjects in **both** |
|---|---|---|---|---|---|
| Spiral | 72 | 30 | 28 | 25 | **25 of 25 (100%)** |
| Wave | 72 | 30 | 28 | 24 | **24 of 24 (100%)** |

Every drawing in the supplied testing directories belongs to a
participant who also appears in the supplied training directories.
Consequences:

* The supplied split is **not** an independent estimate of anything. A
  model can score well by memorising participant-specific stroke
  characteristics (pen pressure, size, tremor baseline) rather than
  learning class-level structure.
* Image-level cross-validation is also invalid: each participant
  contributes ~3 correlated drawings per task, so an image-level fold
  split treats those drawings as independent patients and inflates both
  accuracy and effective sample size (~28 participants, not 102).
* All evaluation in this repository therefore uses participant-grouped
  folds (`StratifiedGroupKFold`, outer **and** inner), with the
  supplied split used only as an audit target.

## Temporal dataset — UCI "Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet" (CC BY 4.0)

### What the repository contains

* 40 semicolon-separated tablet recordings: 25 `PWP (N).txt` +
  15 `Healthy (N).txt`, columns `X;Y;Z;Pressure;GripAngle;Timestamp[;TestID]`
  (one recording per file, thousands of samples per recording).
* Class counts: 25 PD / 15 HC — majority-class accuracy 62.5%.

### Provenance gaps (documented, not resolved)

* The UCI record (DOI [10.24432/C5Q01S](https://doi.org/10.24432/C5Q01S))
  lists **77 instances: 62 PWP + 15 healthy**. This repository uses all
  15 healthy and **25 of 62** PWP recordings — the 37 remaining PWP
  files from the UCI archive are absent. The 25/15 composition matches
  the cohort described in the dataset paper (Isenkul et al.), so the
  subset appears to be the paper cohort, but this cannot be verified
  from the files alone.
* Original UCI filenames (e.g. `P_27110003.txt`) were replaced by
  generic `PWP (N).txt` / `Healthy (N).txt`, destroying the original
  participant identifiers. Each file is *treated as* one unique
  participant (one continuous recording), which is plausible but
  unverifiable after renaming.
* The `data/train` (29 files) and `data/test` (11 files) directories are
  byte-identical copies of `alldata` files (verified), i.e. a locally
  created split with no documented provenance. Because participant
  identity was destroyed by renaming, even its apparent participant
  disjointness cannot be verified. It is **not** used for evaluation.
* Timestamp units differ between files (epoch-seconds in some, raw
  counters in others). Features are computed on per-recording
  normalised time (divided by the median positive sample interval), a
  per-recording transformation that removes this acquisition artifact
  without using any dataset-level statistics.

### The z-channel investigation, stated honestly

The previous README described removing z-based features after finding
that `z_range` alone separated the classes almost perfectly, and framed
that as domain-informed artifact removal. The honest description is:
**exploratory post-hoc data-quality analysis** — the feature was removed
*after observing evaluation results*, which is a form of selection on
outcomes. This repository:

1. excludes z features by default (documented suspicion of a recording
   artifact: z values from different tablet configurations are not
   comparable);
2. re-evaluates them transparently via
   `python scripts/run_temporal_evaluation.py --include-z`;
3. reports both runs. Under the corrected protocol (per-recording time
   normalisation, participant-grouped nested CV), including z features
   changed participant-level accuracy from 92.5% to 92.5% (AUC 0.947 →
   0.957) — i.e. no dramatic inflation — supporting (but not proving)
   the artifact hypothesis. See `results/real/temporal_real*.json`.

## Consequences for evaluation (summary)

1. No use of supplied train/test directories as evaluation data.
2. Participant-level grouping everywhere; `parkinson:V01` and
   `healthy:V01` remain distinct.
3. One prediction per participant (mean out-of-fold probability,
   thresholded) before primary metrics.
4. Spiral and wave evaluated on **identical** participant folds (all 28
   participants completed both tasks); task comparisons are paired.
5. The dummy majority baseline anchors interpretation: 46.4% (spatial
   participant level) and 62.5% (temporal).
