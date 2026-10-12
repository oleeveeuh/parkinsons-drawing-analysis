# Parkinson's Drawing Analysis — Subject-Aware Evaluation Benchmark

An exploratory, research-only benchmark that classifies labels in public
Parkinson's drawing datasets **and** shows why the obvious way of
evaluating them is wrong: the datasets' participants are repeated across
images, so image-level splits silently leak subjects and inflate
accuracy. This project rebuilds the evaluation around participants, not
images.

> **Research only.** Not intended for diagnosis, screening, or any
> clinical use. "PD" / "healthy" here are dataset labels, not verified
> clinical statuses. Results require subject-independent external
> validation before any real-world interpretation.

## The problem in 30 seconds

The Kaggle drawings dataset looks like *102 samples per task* — but its
filenames (`V01PE02.png`, `V01HE02.png`, …) reveal only **28
participants per task**, and **every participant in the supplied test
set also appears in the supplied training set**:

| | Spiral | Wave |
|---|---|---|
| Images | 102 (51/51) | 102 (51/51) |
| Actual participants | **28** (15 PD / 13 HC) | the **same 28** |
| Supplied test participants also in training | **25 of 25** | **24 of 24** |

Three correlated drawings of one person are not three independent
patients. A model that memorises one participant's stroke style scores
well on that person's other drawings. The temporal (UCI) dataset has the
same class of problem: its local train/test split has no verifiable
provenance. Every evaluation in this repo therefore uses
**participant-grouped, nested cross-validation** with hard
subject-disjointness assertions. Details and verified counts:
[docs/data_audit.md](docs/data_audit.md).

## Datasets

| | Temporal (UCI) | Spatial (Kaggle) |
|---|---|---|
| Content | Tablet recordings (X;Y;Z;pressure;grip;time) | Static spiral / wave PNGs |
| Participants | 40 files (25 PD / 15 HC)¹ | 28 per task (15 PD / 13 HC), same 28 in both tasks |
| Observations | 1 recording per participant | ~3 drawings per participant per task (3–15) |
| License | CC BY 4.0 | CC BY-NC-ND 4.0 (attribution, non-commercial, no derivatives) |
| Majority-class accuracy | 62.5% | 46.4% (participant level) |

¹ The UCI record lists 77 files (62 PWP + 15 HC); this subset uses 25
PWP + all 15 HC. Original participant IDs were lost to renaming — a
documented provenance gap. See [docs/data_sources.md](docs/data_sources.md)
for citations, attribution and download instructions.

**These are two different datasets with different participants.** The
temporal and spatial pipelines are separate case studies; nothing here
is multimodal fusion, and no claim is made that one modality is more
diagnostic than the other.

## Pipelines

**Temporal:** per-recording handcrafted features (25: velocity,
acceleration, jerk, pressure statistics, pause ratio, 4–8 Hz tremor-band
power, …). Timestamps are normalised per recording (mixed units across
the raw files). The z-channel features are excluded by default — they
were removed post hoc after showing near-perfect class separation; the
exclusion is re-evaluated transparently with `--include-z`
([docs/data_audit.md](docs/data_audit.md#the-z-channel-investigation-stated-honestly)).

**Spatial:** frozen per-image features — HOG (9 orientations, 8×8 cells,
2×2 blocks on 128×128 grayscale) + 12 geometric shape descriptors
(Otsu ink threshold). All per-image, so no fitted statistics and no
leakage from caching. An optional frozen ResNet-18 (torchvision weights
API, eval mode, hashed cache keys) is available via `--cnn` but is never
used by tests or CI.

## Validation protocol (the core of this project)

```
Participants ──► StratifiedGroupKFold (5 outer folds, group = participant)
                  │  fold boundary: assert(train subjects ∩ eval subjects = ∅)
                  ├─► inner StratifiedGroupKFold (3 folds) = hyperparameter selection
                  │    └─ Pipeline [VarianceThreshold → StandardScaler → PCA(95%) → model]
                  │       fitted on inner-training only
                  └─► fit best config on outer-training, predict outer-eval once
Drawing-level OOF predictions ──► aggregate per participant (mean probability,
                                  threshold 0.5) ──► participant-level metrics
```

* All fitted preprocessing (variance, scaling, PCA) lives inside the
  sklearn `Pipeline` — validation folds can never leak into it.
* Class-weighted models instead of SMOTE: the data is roughly balanced
  and contains repeated drawings per participant; oversampling would
  synthesise near-duplicates of inner-validation neighbours.
* Identical participant folds are shared between spiral and wave, so the
  task comparison is paired.
* Baselines: majority-class `DummyClassifier`, class-weighted logistic
  regression, shallow decision tree. XGBoost is optional and only
  reported next to them.

**Uncertainty:** confidence intervals come from a deterministic
subject-level bootstrap (2,000 resamples, participants resampled with
replacement within class, seed 42). Fold-to-fold standard deviation is
reported separately in the JSON artifacts and is **not** a confidence
interval. Spiral-vs-wave differences use a paired sign-flip permutation
test (10,000 permutations, seed 42) and are exploratory.

## Multimodal & multi-feature fusion — architecture and testing

A fusion module ([src/pd_drawing_analysis/fusion.py](src/pd_drawing_analysis/fusion.py))
implements and tests three architectures under the same subject-grouped
nested CV, with all mixing/selection confined to inner folds:

| Strategy | Mechanism |
|---|---|
| Early fusion | Concatenate per-participant feature vectors → one classifier |
| Late fusion | One classifier per modality → probability average, mixing weight selected on inner folds only |
| Stacked fusion | Per-modality inner-fold out-of-fold probabilities → logistic meta-learner (never sees outer-evaluation participants) |

**Scope, stated plainly:** the two real datasets here **cannot be
fused** — they contain different participants. Fusion is exercised on:

1. **Synthetic paired modalities** — the same synthetic participant
   produces both a recording and a drawing, with the class signal split
   across modalities (half the impaired participants are separable only
   in recordings, half only in drawings), so fusion is genuinely
   required. Result: late fusion 0.917 vs 0.667 / 0.750 for the single
   modalities; early fusion dilutes under the 8,000-dimension HOG
   (0.750) and stacked meta-learning underperforms at n=12 — both
   instructive, reported as found.
2. **Real-data feature-family ablation** — HOG vs shape descriptors of
   the same drawings (`--ablation-features`): HOG alone 0.821, shape
   alone 0.643, fused 0.750. Fusion did **not** beat the stronger single
   family — an honest negative ablation.

No result here supports a cross-dataset multimodal claim; the architecture
exists so the fusion mechanics and their leakage tests are real.

## Mixture of experts — scaffolded and tested

[src/pd_drawing_analysis/moe.py](src/pd_drawing_analysis/moe.py): classical
softmax-gated mixture of experts with EM training, k-means gate
initialisation (uniform init empirically collapses onto one expert), and
per-expert feature slices — evaluated under the same subject-grouped
nested CV (`scripts/run_moe_evaluation.py`).

| Setting | MoE | Best single model | Finding |
|---|---|---|---|
| Real: per-task experts, same 28 participants | 0.786 | 0.786 (spiral expert alone, same AUC) | gate ≈ uniform (0.57/0.43); mixture adds nothing |
| Synthetic two-regime, n=24 | 0.708 | 0.750 | no gain despite near-perfect gate localisation |
| Mechanics validation, n=1200 rows | 0.700 | 0.850 | beats its weak expert (0.633); architecture is data-hungry |

The gate recovers latent regimes in every setting (purity ≥ 0.99), so the
implementation is correct — the honest conclusion is that a classical MoE
does **not** beat a single regularised model at these sample sizes, and no
gain is claimed. (The repo name's historical "MoE" referred to a removed
v1 codebase; this module is an independent rebuild.)

## Results — reproduced under the corrected protocol

Participant-level out-of-fold results, generated by the commands below
(the artifacts live in the git-ignored `results/real/` — regenerate them
any time and compare):

**Temporal** — UCI, 40 participants, 5×3 nested subject-grouped CV:

| Model | Accuracy | 95% CI¹ | Bal. acc. | F1 | ROC-AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|---|
| Majority dummy | 0.625 | — | 0.500 | — | n/a | 1.000 | 0.000 |
| LogReg (balanced) | 0.925 | [0.825, 1.000] | 0.913 | 0.941 | 0.947 | 0.960 | 0.867 |
| Decision tree (d 2–4) | 0.725 | [0.575, 0.850] | 0.713 | 0.776 | 0.688 | 0.760 | 0.667 |

**Spatial** — Kaggle, 28 participants per task, identical folds:

| Model | Task | Accuracy | 95% CI¹ | Bal. acc. | F1 | ROC-AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|---|---|
| Majority dummy | — | 0.464 | — | 0.469 | — | n/a | — | — |
| LogReg (balanced) | spiral | 0.821 | [0.679, 0.964] | 0.818 | 0.839 | 0.867 | 0.867 | 0.769 |
| LogReg (balanced) | wave | 0.821 | [0.679, 0.964] | 0.818 | 0.839 | 0.826 | 0.867 | 0.769 |

¹ Deterministic subject-bootstrap percentile intervals — *not* fold
standard deviation. Confusion matrices (LogReg, participants): temporal
`TN=13 FP=2 FN=1 TP=24`; spatial spiral `TN=10 FP=3 FN=2 TP=13`.

**What the leakage fix changed** (measured apples-to-apples — same model,
features and participant-level aggregation, only the split differs,
30 repeated partitions; regenerate with
`python scripts/measure_split_impact.py`):

| Comparison | Leaky image-level split | Participant-grouped split |
|---|---|---|
| Spiral participant accuracy (mean ± std over seeds) | 0.860 ± 0.033 | 0.833 ± 0.028 |
| Advertised sample size | 102 images | 28 participants |
| Test images whose subject was in training | **97–100%** | 0% |

The accuracy delta alone is small and seed-unstable at n=28 (+2.6pp mean;
the leaky split "wins" only 16/30 seeds) — the reliable damage was
elsewhere: a **3.6× sample-size inflation** (102 "patients" were 28 people)
and the erasure of a claimed finding. The previous pipeline reported
*spiral beats wave by 9.6pp* (72.4% vs 62.8%, p=0.212) as a
task-complexity result; under participant-grouped paired evaluation the
difference is **exactly zero (p = 1.0)** — no evidence one task is more
diagnostic. The old headline numbers (72.4%/62.8%/90.0%) are **not
comparable** to the tables above (different features, model, protocol).

**The spurious feature, before and after diagnosis:** on the temporal
dataset, `z_range` alone appeared to separate the classes at **95.0%**
with a naive in-sample threshold — reproducing the previous README's
claim and triggering the artifact investigation. Under participant-grouped
CV on that single feature the honest score is **0.800 accuracy / 0.925
AUC**, and adding all z-features to the full model moves its AUC by only
+0.01 (0.947 → 0.957) — a pipeline artifact, not signal (details:
[docs/data_audit.md](docs/data_audit.md)).

## Limitations

* Tiny participant counts (28–40) → wide intervals; point estimates are
  unstable and no family of models was "selected" on outer folds.
* Post-hoc analytic decisions exist (z-feature exclusion, per-recording
  time normalisation); both are documented and re-evaluated within the
  corrected protocol.
* No demographic/clinical metadata (age, stage, medication) → confounders
  unmeasurable; labels unverified.
* Handcrafted features and classical models only; no deep-learning
  baseline was trained (the optional CNN is a frozen feature extractor).
* Single-dataset each; no external validation. Nothing here is evidence
  of clinical utility — these are dataset-label classification results.

## Setup

Requires Python 3.11 or 3.12.

```bash
git clone git@github.com:oleeveeuh/parkinsons-drawing-analysis.git
cd parkinsons-drawing-analysis
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test dependencies
```

Optional extras: `.[xgboost]` (comparison model; macOS also needs
`brew install libomp`) and `.[cnn]` (torch/torchvision for optional
frozen CNN features — downloads weights only when you pass `--cnn`).

### 5-minute offline verification (no data, no downloads)

```bash
python scripts/make_synthetic_fixtures.py          # writes data/synthetic/
python scripts/run_temporal_evaluation.py --synthetic
python scripts/run_spatial_evaluation.py --synthetic
python scripts/run_fusion_evaluation.py --synthetic   # paired-modality fusion demo
pytest tests -q                                    # 56 offline tests
```

### Real data (authorized download required)

Follow [docs/data_sources.md](docs/data_sources.md) (UCI + Kaggle,
licenses above). Then:

```bash
python scripts/audit_data.py \
  --spiral-root KaggleSpiral/drawings \
  --temporal-root "Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/alldata" \
  --output results/data_audit.json
python scripts/run_temporal_evaluation.py \
  --data-root "Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/alldata"
python scripts/run_spatial_evaluation.py --data-root KaggleSpiral/drawings
# transparent re-evaluation of the excluded z features:
python scripts/run_temporal_evaluation.py --data-root "…" --include-z
```

Datasets stay on disk, git-ignored; nothing derived from them is
committed except the numbers you choose to quote with a link to your
`results/` artifacts.

### Tests and verification

```bash
pytest tests -q                    # leakage guards, fold-local fitting, aggregation, e2e
python scripts/env_info.py         # environment + versions for reproducibility
```

The tests fail if: a subject crosses any fold boundary; drawings of one
participant are treated as independent; predictions misalign with
labels/subjects; fitted preprocessing ever sees non-training rows; or a
supplied train/test directory split is accepted as subject-independent.

## Repository structure

```
src/pd_drawing_analysis/   subject_ids.py data.py features.py evaluation.py
                           metrics.py reporting.py fusion.py moe.py synthetic.py
scripts/                   audit_data.py run_temporal_evaluation.py
                           run_spatial_evaluation.py run_fusion_evaluation.py
                           run_moe_evaluation.py measure_split_impact.py
                           make_synthetic_fixtures.py env_info.py
tests/                     offline pytest suite (no network, no weights)
configs/                   temporal.yaml spatial.yaml
docs/                      data_audit.md data_sources.md
data/, cache/, results/    git-ignored local data, caches, artifacts
```

The previous MoE-era code was removed before this rewrite; the repository
was renamed from `MoE-MultiSDAR-PD-Classification` to
`parkinsons-drawing-analysis` to reflect current scope (GitHub redirects
the old name).

## Reproducibility notes

* Seeds fixed at 42 everywhere (splits, bootstrap, permutations,
  estimators); `StratifiedGroupKFold` with fixed ordering is
  deterministic; feature extraction is deterministic and unit-tested.
* Every run writes a JSON artifact with the environment, the
  fold-to-subject assignment, and out-of-fold predictions by original
  index — README numbers are copied from these artifacts.
* `python scripts/env_info.py` prints versions; CI runs the full offline
  suite on Python 3.11 and 3.12.

## Realistic future work

* External validation on an independent drawing dataset with subject
  disjointness across datasets.
* *Paired* multimodal data (same participants, tablet + image) to move the
  fusion module from synthetic demonstration to real analysis.
* Larger cohorts with demographic/clinical metadata; reliability
  analysis per participant.
* Conformal prediction intervals; SHAP on participant-level features.

## Citations and licenses

* Isenkul, M., Sakar, B., & Kursun, O. (2014). *Parkinson Disease Spiral
  Drawings Using Digitized Graphics Tablet* [Dataset]. UCI ML Repository.
  DOI: [10.24432/C5Q01S](https://doi.org/10.24432/C5Q01S) — **CC BY 4.0**.
* Kaggle: [Parkinson's Drawings](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)
  — **CC BY-NC-ND 4.0**; dataset images are not redistributed here.
* He et al. (2016), *Deep Residual Learning* (only if using `--cnn`).
* Code license: **none yet** — pending confirmation of ownership /
  institutional (NSF REU) terms. Dataset licenses are unaffected by any
  future code license.

Developed as part of the University of Wyoming HUMANS MOVE NSF REU.
Contact: [oliau@usc.edu](mailto:oliau@usc.edu) ·
[linkedin.com/in/olivialiau](https://linkedin.com/in/olivialiau)
