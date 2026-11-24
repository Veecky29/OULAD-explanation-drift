# Accurate but Misleading: Explanation Drift in Cross-Course Transfer of At-Risk Prediction Models

Evidence from OULAD and an external Moodle dataset.

Code, figures, and tabulated results for a learning-analytics study on the [Open University Learning Analytics Dataset (OULAD)](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset).

**Research question.** After a model is transferred to an unseen course or presentation, predictive accuracy often still looks acceptable — but SHAP explanations can drift and would misdirect interventions. The work is an evaluation framework plus an empirical finding, not a claim of state-of-the-art accuracy. Logistic regression and XGBoost are supporting tools.

**中文摘要：** 跨课程迁移后，AUC 往往看起来还可以，但 SHAP 解释会漂移，按解释去做干预可能指错方向。本文提供一套评估框架和实证结果，不追求精度 SOTA。完整数字、判断和局限见 `results.md`。

> This repository holds the **experiment code and results**. The paper manuscript is written separately and is not included here.

## Results at a glance

The full write-up (Chinese), with tables, citations, and caveats, is in **[`results.md`](results.md)**. Figures are under `figures/` (pilot), `figures/phase2/`, and `figures/phase3/`.

A short, honest summary of the current evidence:

- The main line holds most clearly for **unseen modules (leave-one-module-out) at day 28, at the feature-family level**: transfer AUC drops only modestly, but explanations can point to uninformative features (especially EEE/GGG, which have almost no assignments in the first four weeks).
- After **B = 100** bootstrap resampling of both source and target, **no contrast passes Benjamini–Hochberg q < 0.05**. B = 100 is a self-chosen budget (the common convention is often 1,000) and is noted as such in the write-up.
- A self-designed pre-deployment “trust flag” has limited coverage; the source-relative v2 rule did not flag the Riestra external setting, where drift is not driven by feature degeneracy.
- External validation on the public Riestra-González et al. (2021) Moodle data is **consistent in direction**, with deviations from the OULAD setup documented in `results.md`.

## Data

Raw data files are **not** in this repository. See [`data/README.md`](data/README.md) for download commands.

| Source | What we use | Where to get it |
|---|---|---|
| OULAD (Kuzilek et al., 2017) | VLE clicks, assessments, student info, registrations | [UCI #349](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) (preferred mirror; used by Wu et al., 2026) |
| Riestra-González et al. (2021) | Clean supervised Moodle tables at 10/25/33/50% course length | [GitHub: moisesriestra/moodle-early-performance-prediction](https://github.com/moisesriestra/moodle-early-performance-prediction) |

At-risk label: Fail/Withdrawn vs Pass/Distinction. Students who unregistered before the cutoff are excluded (Wu; Lim).

## Reproducing the pipeline

Python 3.13, CPU-only (XGBoost here does not benefit from a consumer GPU in this setup). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pilot (d28, d56): features, models, figures, noise floor, leakage checks
# ~45 min on 8 CPU cores
bash run_all.sh

# Phase 2: all cutoffs, family-level drift, null control, trust flag (B=30)
# ~1–2 h with the default 3 workers
bash run_phase2.sh

# Phase 3: remaining nulls, B=100 source+target bootstrap, trust v2, Riestra external
# Requires the Riestra clone (see data/README.md). Per-pair jobs are checkpointed.
bash run_phase3.sh
```

Main scripts:

| Script | Role |
|---|---|
| `src/features.py` | Build cutoff-specific feature tables |
| `src/experiment.py`, `src/figures.py`, `src/noise_floor.py`, `src/leakage_checks.py` | Pilot (H1–H3) |
| `src/phase2.py`, `src/phase2_null.py`, `src/phase2_analysis.py` | Phase 2 transfer + null + summaries |
| `src/phase3.py`, `src/phase3_jobs.py`, `src/phase3_trust_oulad.py` | Phase 3 OULAD bootstrap and trust-flag eval |
| `src/phase3_riestra.py`, `src/phase3_riestra_jobs.py` | External Moodle replication |
| `src/phase3_analysis.py`, `src/phase3_md_tables.py` | Phase 3 aggregates and markdown tables |
| `src/trust.py` | Trust-flag definitions (self-designed) |

## Settings and literature citations

Hyperparameters and design choices are taken from published work where possible; anything self-designed is labelled in `results.md`. The table below is the core design (see `results.md` §2, P2.1, and P3.1 for the full list).

| Setting | Value | Source |
|---|---|---|
| Label | Fail/Withdrawn = at-risk | Wu, Jia & Pham (2026); Lim (2026); Zhang & Ahn (2023) |
| Cutoffs | d7, d14 (Lim 2026); d28 (Wu 2026, Lim 2026); d56 / week 8 and d112 / week 16 (Zhang & Ahn 2023) | as cited |
| Early withdrawals | Drop if `date_unregistration` < cutoff | Wu (2026); Lim (2026) |
| Time window | Events with `date` < cutoff, including pre-course (negative days) | Wu (2026) |
| Features | Activity-type clicks/visits, weekly clicks, submissions/scores, demographics, etc. | Zhang & Ahn (2023) Table 6; Wu (2026); Lim (2026) |
| Preprocessing | Train-only median imputation, log1p on counts, scaling; one-hot region | Lim (2026); Wu (2026) |
| Pooled split | Student-grouped 80/20, 10 seeds | Wu (2026) |
| Cross-year | 2013 → 2014 | Wu (2026) |
| SSD / SCD | Zhang & Ahn (2023) | Zhang & Ahn (2023) |
| LOMO | Leave-one-module-out, students disjoint | Lim (2026); Wu (2026) |
| Logistic regression | L2, C=1, `class_weight=balanced` | L2 + scaling from Lim (2026); C is the sklearn default; balanced is a task choice (not specified in the cited papers) |
| XGBoost | 180 trees, learning rate 0.08, 31 leaves (lossguide), λ=1; **fixed, no tuning** | Translated from Lim’s (2026) HistGB configuration, for the same reason: avoid adapting the benchmark to a held-out target |
| Calibration | Platt scaling fit in the source domain | Lim (2026) |
| Bootstrap B | 100 in phase 3 (phase-2 used 30) | **Self-chosen**; often 1,000 in the literature — flagged in the paper |

Self-designed pieces (must be labelled as such): within-presentation grouped 5-fold CV as an optimistic ceiling; target-domain “native” models; split-half / within-module null for explanation noise; feature families; the pre-deployment trust-flag rules (v1, v2); and the “days since last activity” feature.

## Layout

```
run_all.sh  run_phase2.sh  run_phase3.sh
requirements.txt
results.md                 # narrative results (Chinese) and citations
src/                       # experiment code
results/                   # CSV / logs (pilot, phase2, phase3)
figures/                   # pilot, phase2, and phase3 plots
data/                      # download here; raw files are gitignored
```

## Key references

- Kuzilek, J., et al. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
- Wu, M., Jia, J., & Pham, T. T. (2026). Early warning of academic risk from the first four weeks. *Frontiers in Education*. https://doi.org/10.3389/feduc.2026.1917092
- Zhang, C., & Ahn, H. (2023). E-Learning at-risk group prediction considering the semester and realistic factors. *Education Sciences*, 13(11), 1130. https://doi.org/10.3390/educsci13111130
- Lim (2026). Student-disjoint cross-module benchmarking for calibrated early warning in online learning. *IJACSA*, 17(9), Paper 58.
- Riestra-González, M., et al. (2021). Moodle early performance prediction data and code: https://github.com/moisesriestra/moodle-early-performance-prediction
