# Dataset cards

Both datasets are public research datasets. Neither ships with this
repository; `data/` and the original download directories are git-ignored.

## 1. UCI Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet
*(temporal pipeline)*

| Field | Value |
|---|---|
| Canonical URL | https://archive.ics.uci.edu/dataset/395 |
| DOI | [10.24432/C5Q01S](https://doi.org/10.24432/C5Q01S) |
| Creators / citation | Isenkul, M., Sakar, B., & Kursun, O. (2014). *Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet* [Dataset]. UCI Machine Learning Repository. |
| License | CC BY 4.0 (attribution required; commercial use and adaptation allowed) |
| Cohort | UCI record: 77 instances — 62 PWP + 15 healthy. This project uses 25 PWP + 15 healthy files (see provenance gaps in [data_audit.md](data_audit.md)). |
| Unit of observation | One `.txt` file per recording (~10⁴ samples); treated as one participant |
| Label definition | Directory/filename prefix: `PWP` = Parkinson's, `Healthy` = control |
| Required attribution | "Isenkul, M., Sakar, B., & Kursun, O. (2014). Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5Q01S. CC BY 4.0." |
| Known issues | Original participant ids lost to renaming; locally created train/test split of unknown provenance; mixed timestamp units; z-channel recording artifact (see audit). |

### Download

```bash
# manual: open the URL above, download the zip, unzip, and place the
# per-subject text files in one directory, e.g.
#   "Improved Spiral Test Using Digitized Graphics Tablet for Monitoring Parkinson's Disease/data/alldata/"
```

Expected directory counts: 40 `.txt` files in this project's subset
(25 `PWP (N).txt`, 15 `Healthy (N).txt`). `scripts/audit_data.py
--temporal-root <dir>` verifies counts and parseability.

## 2. Kaggle Parkinson's Drawings
*(spatial pipeline)*

| Field | Value |
|---|---|
| Canonical URL | https://www.kaggle.com/datasets/kmader/parkinsons-drawings |
| Creator | Kaggle user `kmader`, compiling drawings used in Parkinson's drawing-test studies |
| License | **CC BY-NC-ND 4.0** |
| Cohort | 28 participants per task (15 PD / 13 HC); same 28 participants completed both tasks |
| Unit of observation | 102 spiral + 102 wave PNG images (the vendor repeats participants across `training`/`testing`) |
| Label definition | Directory name `healthy` / `parkinson`, cross-checked against filename group code (`HE`/`HO` vs `PE`/`PO`) |
| Known issues | Supplied train/test split is 100% subject-leaky (see [data_audit.md](data_audit.md)); dataset tree duplicated on disk; two participants contribute disproportionate drawing counts. |

### License implications (important)

* **BY** — attribution must be given (citation above).
* **NC** — non-commercial use only.
* **ND** — no derivatives: modified copies of the images may not be
  redistributed unless permission is established. This repository
  therefore ships **no image files** and derived feature caches are not
  distributed either.
* The datasets' license does not change with this repository's code
  license: whatever code license is later chosen grants **no** rights to
  the datasets themselves.

### Download

Requires a (free) Kaggle account:

```bash
kaggle datasets download -d kmader/parkinsons-drawings
unzip parkinsons-drawings.zip -d KaggleSpiral
```

Expected layout under `KaggleSpiral/`: `spiral/{training,testing}` and
`wave/{training,testing}` (plus a duplicate `drawings/` tree). Verified
by `scripts/audit_data.py --spiral-root KaggleSpiral/drawings`: 204
images, 28 participants per task.

## Privacy and ethical limitations (both datasets)

* No participant names or direct identifiers are present; filenames are
  pseudonymous codes. This repository retains only those pseudonymous
  identifiers, for grouping.
* Health data was collected from possibly identifiable individuals and
  published historically without modern consent standards. Treat it with
  corresponding care.
* **Prohibited use:** diagnosis, screening, or any clinical decision
  support. The datasets lack demographic and clinical metadata (age,
  disease stage, medication status), so no clinical conclusion can be
  drawn from them.
