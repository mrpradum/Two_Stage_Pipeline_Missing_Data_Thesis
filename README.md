# Mosquito Count Imputation Pipelines (2015 — Site 3 & Site 8)

Two-stage pipelines for retrospective time-series imputation of daily mosquito counts:

* **Stage 1**: MICE (IterativeImputer) with **XGBoost Poisson** as the conditional model
* **Stage 2**: Per-species **XGBoost Poisson** using engineered features + the Stage-1 imputed *other* species (same day)

**Metrics**: we report **MAE** and **RMSE** only (computed on *originally known* test rows).

---

## Sites (S3 vs S8)

| Code   | Site               | Short context                                             |
| ------ | ------------------ | --------------------------------------------------------- |
| **S3** | **Kouribabougou**  | More shade; rainy season = running/edge water             |
| **S8** | **Nanguilabougou** | More direct sun; rainy season = persistent stagnant water |

You’ll see `Final_2015_Data_Site3.csv` and `Final_2015_Data_Site8.csv` used accordingly.

---
## Data & citation

* The **2015 Site 3 & Site 8 datasets** used here are from the paper **“abc”** (the paper PDF/data are in the repository root).
* Please **cite the source paper** when using this code or the derived figures/tables.

Citation:

```
M. Gildenhard, E. K. Rono, A. Diarra, A. Boissière, P. Bascunan, P. Carrillo-Bustamante, D. Camara, H. Krüger, M. Mariko, R. Mariko, P. Mireji, S. E. Nsango, J. Pompon, Y. Reis, M. K. Rono, P. B. Seda, J. Thailayil, A. Traoré, C. V. Yapto, P. Awono-Ambene, R. K. Dabiré, A. Diabaté, D. Masiga, F. Catteruccia, I. Morlais, M. Diallo, D. Sangaré, and E. A. Levashina. “Mosquito microevolution drives Plasmodium falciparum dynamics”. In: Nature Microbiology 4.6 (June 2019). PMCID: PMC7613322, pp. 941–947. +)": 10.1038/s41564- 019-0414-9. pmid: 30911126. /%0: https://doi.org/10.1038/s41564-019-0414-9.
```

```bibtex
@article{gildenhard2019mosquito,
  title={Mosquito microevolution drives Plasmodium falciparum dynamics},
  author={Gildenhard, Markus and Rono, Evans K and Diarra, Assetou and Boissiere, Anne and Bascunan, Priscila and Carrillo-Bustamante, Paola and Camara, Djeneba and Kr{\"u}ger, Hanne and Mariko, Modibo and Mariko, Ramata and others},
  journal={Nature microbiology},
  volume={4},
  number={6},
  pages={941--947},
  year={2019},
  publisher={Nature Publishing Group UK London}
}
```

> Note: In the repo, the CSVs are named `Final_2015_Data_Site3.csv` and `Final_2015_Data_Site8.csv`. If you relocate/rename them, update the pipeline `--file` argument accordingly.

## Analysis windows

* **Main window (mandatory)**: `2015-07-01 : 2015-10-31`
* **Short window (optional)**: `2015-07-01 : 2015-10-21` (use when you explicitly want both spans)

> All examples below assume the **main** window by default. Use the short window only if you need the secondary analysis.

---

## Features & settings (quick reference)

**Lag settings** (`--lag`)

* `no_lag`: overnight t0 stats + `precip_t0` (retrospective)
* `precip_lags`: overnight t0 + `precip_d1`, `precip_sum3`, `precip_sum7` (causal)
* `both_lags`: `precip_lags` + env `median_d1`, `median_ma3` (causal)

**Zero policies** (`--zero-policy`)

* `standard`: 0→NaN in train **and** test
* `train_keeps_zeros`: keep zeros in train; test zeros→NaN

**Tuning**

* **Untuned** = fixed moderate defaults
* **Tuned** = nested K-fold random search
  (`--tuned --cv-folds 5 --imputer-trials 48 --stage2-trials 96`)

**Protocols**

* `80/20` random split (seed=42)
* `LOO` leave-one-out (per day)

---

## All possible combinations (dimension grid)

Dimensions:

* Site ∈ {**S3**, **S8**}
* Protocol ∈ {**80/20**, **LOO**}
* Tuning ∈ {**untuned**, **tuned**}
* Lag ∈ {**no_lag**, **precip_lags**, **both_lags**}
* Zero policy ∈ {**standard**, **train_keeps_zeros**}
* Window ∈ {**main (2015-10-31)**, optional **short (2015-10-21)**}

Per *window*, the **Lag × Zero** pairs are the 6 canonical settings:

1. `no_lag` + `standard`
2. `no_lag` + `train_keeps_zeros`
3. `precip_lags` + `standard`
4. `precip_lags` + `train_keeps_zeros`
5. `both_lags` + `standard`
6. `both_lags` + `train_keeps_zeros`

**Counts (per site):**

* Per protocol, per tuning, **per window**: **6** runs
* If you do **both** protocols (80/20 + LOO) and **both** tuning modes → **6 × 2 × 2 = 24** runs per window
* If you also include the **optional** short window → another **24** runs
* Across **both sites**: double those numbers

---

## Mandatory vs Optional

* ✅ **Mandatory**: For **both sites (S3, S8)**, **both protocols (80/20, LOO)**, **both tuning modes (untuned, tuned)**, run **all 6 Lag×Zero** combinations on the **main window** `2015-07-01 : 2015-10-31`.
  → **24 runs per site** (main window only).

* 🟨 **Optional**: Repeat the same set on the **short window** `2015-07-01 : 2015-10-21` if you need side-by-side window sensitivity.
  → **+24 runs per site** (optional window).

---

## Top-level repository layout

```
.
├── site3_Data_Info.ipynb         # visualization for Site 3 (2015)
├── site8_Data_Info.ipynb         # visualization for Site 8 (2015)
├── all_pots.ipynb                # Notebook wrapper to generate all figs/ from results/
├── pipeline_80_20_NoTuned/
│   ├── pipeline_80_20_2015_S3_NoTuned/
│   └── pipeline_80_20_2015_S8_NoTuned/
├── pipeline_80_20_Tuned/
│   ├── pipeline_80_20_2015_S3_Tuned/
│   └── pipeline_80_20_2015_S8_Tuned/
├── pipeline_loo_NoTuned/
│   ├── pipeline_loo_2015_S3_NoTuned/
│   └── pipeline_loo_2015_S8_NoTuned/
├── pipeline_loo_Tuned/
│   ├── pipeline_loo_2015_S3_Tuned/
│   └── pipeline_loo_2015_S8_Tuned/
├── scripts/                  # (optional helpers; some folders also carry their own scripts/)
└── figs/                     # auto-created by the figure generator (see “Figures”)
```

### Per-protocol, per-site leaf folder layout

Each leaf folder (e.g., `pipeline_80_20_NoTuned/pipeline_80_20_2015_S3_NoTuned/`) contains:

```
Final_2015_Data_Site3.csv          # or Site8, depending on the folder
Final_2015_Data_Site8.csv
pipeline_80_20.py  or  pipeline_loo.py
*_Runs.ipynb                       # batch all 6 (main) or 12 (main+short) combos
results/                           # auto-created; holds __summary__ and __preds__ CSVs
scripts/                           # LaTeX table builders for that folder
```

---

## How to run (CLI)

> Run **inside** the target leaf folder (each has its own `pipeline_*.py` and `results/`).
> `run-all` executes the **6 mandatory combos** for the **main** window by default (add `--window 2015-07-01:2015-10-21` for the short window).

### 80/20 — Site 3

**Untuned — all 6 (main window)**

```bash
cd pipeline_80_20_NoTuned/pipeline_80_20_2015_S3_NoTuned
python pipeline_80_20.py --cmd run-all --file Final_2015_Data_Site3.csv
```

**Tuned — all 6 (main window)**

```bash
cd pipeline_80_20_Tuned/pipeline_80_20_2015_S3_Tuned
python pipeline_80_20.py --cmd run-all \
  --file Final_2015_Data_Site3.csv \
  --tuned --cv-folds 5 --imputer-trials 48 --stage2-trials 96
```

### 80/20 — Site 8

**Untuned — all 6 (main window)**

```bash
cd pipeline_80_20_NoTuned/pipeline_80_20_2015_S8_NoTuned
python pipeline_80_20.py --cmd run-all --file Final_2015_Data_Site8.csv
```

**Tuned — all 6 (main window)**

```bash
cd pipeline_80_20_Tuned/pipeline_80_20_2015_S8_Tuned
python pipeline_80_20.py --cmd run-all \
  --file Final_2015_Data_Site8.csv \
  --tuned --cv-folds 5 --imputer-trials 48 --stage2-trials 96
```

### LOO — Site 3

**Untuned — all 6 (main window)**

```bash
cd pipeline_loo_NoTuned/pipeline_loo_2015_S3_NoTuned
python pipeline_loo.py --cmd run-all --file Final_2015_Data_Site3.csv
```

**Tuned — all 6 (main window)**

```bash
cd pipeline_loo_Tuned/pipeline_loo_2015_S3_Tuned
python pipeline_loo.py --cmd run-all \
  --file Final_2015_Data_Site3.csv \
  --tuned --cv-folds 5 --imputer-trials 48 --stage2-trials 96
```

### LOO — Site 8

**Untuned — all 6 (main window)**

```bash
cd pipeline_loo_NoTuned/pipeline_loo_2015_S8_NoTuned
python pipeline_loo.py --cmd run-all --file Final_2015_Data_Site8.csv
```

**Tuned — all 6 (main window)**

```bash
cd pipeline_loo_Tuned/pipeline_loo_2015_S8_Tuned
python pipeline_loo.py --cmd run-all \
  --file Final_2015_Data_Site8.csv \
  --tuned --cv-folds 5 --imputer-trials 48 --stage2-trials 96
```

**Single-run example**

```bash
python pipeline_80_20.py --cmd run \
  --file   Final_2015_Data_Site8.csv \
  --lag    both_lags \
  --zero-policy train_keeps_zeros \
  --window 2015-07-01:2015-10-31 \
  --tuned --cv-folds 5 --imputer-trials 48 --stage2-trials 96
```

---

## How to run (Notebook)

Each leaf folder has a batching notebook like `*_Runs.ipynb`:

```python
from pipeline_80_20 import run_once     # or: from pipeline_loo import run_once

lags = ["no_lag", "precip_lags", "both_lags"]
zps  = ["standard", "train_keeps_zeros"]
ws, we = "2015-07-01", "2015-10-31"     # main window (mandatory)

for lag in lags:
    for zp in zps:
        run_once(
            file="Final_2015_Data_Site3.csv",   # or Site8
            window_start=ws, window_end=we,
            lag_setting=lag, zero_policy=zp,
            tuned=False                         # set True and pass cv_folds/imputer_trials/stage2_trials for tuned runs
        )
```

To also include the **optional** short window, add a second loop with `we = "2015-10-21"`.

---

## Outputs

Each run writes to `results/`:

* **Summary CSV**
  `8020__<lag>__<zero>__<start>_<end>__<tuned/untuned>__summary__<timestamp>.csv`
  or `loo__...__summary__...csv`
  Columns:
  `protocol, run_tag, file, window_start, window_end, lag_setting, zero_policy, tuned, cv_folds, imputer_trials, stage2_trials, target, known_rows, MAE, RMSE`

* **Per-target predictions**
  `...__<Target>__preds__<timestamp>.csv` with
  `Date, True, Predicted, was_missing, Residual`

> **Scoring rule**: MAE/RMSE are computed **only** on test rows that were *originally known* (not NaN and not 0 before zero-handling). Residuals for originally 0/NaN rows are reported as NaN.

---

## Figures (auto-generated)

You can auto-plot **every run** discovered under any `results/` folder (across all protocols/sites/tuning) and write publication-ready PDFs.

**What gets generated per run**

* `__time_series.pdf` — True vs Predicted over time (with monthly ticks)
* `__residuals_time.pdf` — Residuals (True − Predicted) over time
* `__residual_hist.pdf` — Residual distribution
* `__true_vs_pred_scatter.pdf` — Scatter with y=x reference
* `__ALL.pdf` — a single multi-page PDF containing all four plots

**Where they go**
Figures are grouped into a repo-level tree (auto-created):

```
figs/
└── <Protocol>/                 # 80_20 or LOO
    └── <tuned|untuned>/
        └── Site X/            # “Site 3” or “Site 8”
            └── <window>/      # e.g., 2015-07-01_2015-10-31
                ├── <run_prefix>__time_series.pdf
                ├── <run_prefix>__residuals_time.pdf
                ├── <run_prefix>__residual_hist.pdf
                ├── <run_prefix>__true_vs_pred_scatter.pdf
                └── <run_prefix>__ALL.pdf
```

**Titles**
Two-line titles to avoid cropping:

* Line 1: `Site X, <species>` — `Protocol — Lag — Zero policy — tuned/untuned — window`
* Line 2: the plot name (e.g., *True vs Predicted over time*)

**How to generate**
Place the auto-plot script (or notebook cell) at the repo root and run:

```bash
python scripts/plot_all_runs.py
# or inside a notebook:
from plot_all_runs import plot_everything
plot_everything()  # discovers all results/ folders recursively
```

> The plotter is robust to small filename differences. It infers missing tokens when possible and falls back to reading the Date range from the CSV if the window token isn’t present.

---

## LaTeX tables

Table builders live under `scripts/`. Typical usage:

* **80/20 untuned by window** → writes `results/_8020_untuned_bywindow.tex`
* **Flat “all runs” summary** → single table listing Target / Protocol / Lag / Window / Zero policy / Tuned / n_known / MAE / RMSE

If your filenames follow the default pattern (they do), the scripts will auto-parse.

---

## Reproducibility

* Seed fixed at **42** (splits + random search)
* MICE deterministic (`sample_posterior=False`, fixed order)
* **Tuned** runs use **nested K-fold** CV; selection by validation **RMSE**
* **Evaluation**: final **MAE** and **RMSE** only (no R²)

---

## Environment

Python ≥ 3.10 with:

```
numpy, pandas, scikit-learn, xgboost, matplotlib
```

Quick setup:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
