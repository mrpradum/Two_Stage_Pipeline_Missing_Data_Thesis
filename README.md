# Malaria Mosquito Count Imputation (2015) — Sites 3 & 8

Two-stage pipeline for retrospective imputation of daily mosquito counts using:
- **Stage 1:** MICE with XGBoost–Poisson (multivariate completion; no leakage)
- **Stage 2:** per-species XGBoost–Poisson (features + Stage-1 imputed other species)

We evaluate **80/20 random split** and **Leave-One-Out (LOO)** protocols under **3 lag settings** and **2 zero policies**, with **tuned** and **untuned** variants. Outputs include per-run CSVs and ready-to-`\input{}` LaTeX tables.



