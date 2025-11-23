#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
80/20 pipeline — Nested K-fold tuning (optional) — XGB-Poisson in Stage-1 (MICE) and Stage-2

Supports:
- lag_setting: {"no_lag", "precip_lags", "both_lags"}
- zero_policy: {"standard", "train_keeps_zeros"}
- windows: 2015-07-01..2015-10-21  or  2015-07-01..2015-10-31 (inclusive)
- tuned: if True, nested K-fold tuning with budgets (cv_folds, imputer_trials, stage2_trials)

Outputs per target:
- predictions CSV: Date, True, Predicted, was_missing, Residual
- summary CSV row with MAE, RMSE
"""

import ast, json, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# defining constants
RANDOM_STATE = 42
DATE_COL   = "Date"
PRECIP_COL = "Precipitation"
ENV_VEC_COLS = ["Temperature", "Humidity", "Dew_Point", "Water_Temperature"]
TARGETS = ["Adults_3_Col", "Adults_3_Gam"]

MICE_PARAMS = dict(
    max_iter=20, sample_posterior=False, random_state=RANDOM_STATE,
    imputation_order="ascending", skip_complete=True,
    initial_strategy="median", tol=1e-3,
)

# default (untuned) XGB params
XGB_DEFAULT = dict(
    objective="count:poisson", tree_method="hist",
    n_estimators=400, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)

# utils
def _merge_params(*ds):
    out = {}
    for d in ds:
        if d:
            out.update(d)
    return out

def rmse(y_true, y_pred):
    a = np.asarray(y_true, float); b = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((a - b)**2)))

def _to_vec24(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x, float)
    elif isinstance(x, str):
        try:
            arr = np.array(ast.literal_eval(x), float)
        except Exception:
            arr = np.full(24, np.nan, float)
    else:
        arr = np.full(24, np.nan, float)
    if arr.size < 24: arr = np.pad(arr, (0, 24 - arr.size), constant_values=np.nan)
    elif arr.size > 24: arr = arr[:24]
    return arr

def _stats(a):
    a = np.asarray(a, float)
    return np.nanmin(a), np.nanmax(a), np.nanmedian(a), np.nanstd(a, ddof=0)

def _overnight_features(df_full):
    """prev 18–23 + curr 00–05 -> min/max/median/std per env var (t0 only)."""
    n = len(df_full)
    vecs = {c: df_full[c].apply(_to_vec24).to_list() for c in ENV_VEC_COLS}
    feat = {}
    for c in ENV_VEC_COLS:
        vmin, vmax, vmed, vsd = [], [], [], []
        for i in range(n):
            prev = vecs[c][i - 1] if i - 1 >= 0 else np.full(24, np.nan)
            curr = vecs[c][i]
            win = np.concatenate([prev[18:24], curr[0:6]])
            mn, mx, med, sd = _stats(win)
            vmin.append(mn); vmax.append(mx); vmed.append(med); vsd.append(sd)
        base = f"{c}_overnight"
        feat[f"{base}_min_t0"] = vmin
        feat[f"{base}_max_t0"] = vmax
        feat[f"{base}_median_t0"] = vmed
        feat[f"{base}_std_t0"] = vsd
    return feat

def build_features(file_df: pd.DataFrame, lag_setting: str) -> pd.DataFrame:
    """
    lag_setting:
      - 'no_lag'      : overnight stats + precip_t0 (NaNs->0)
      - 'precip_lags' : overnight stats + precip_d1,sum3,sum7  (causal)
      - 'both_lags'   : overnight stats + env median_d1,median_ma3 + precip_d1,sum3,sum7 (causal)
    """
    feat = _overnight_features(file_df)

    if lag_setting == "no_lag":
        p0 = pd.to_numeric(file_df[PRECIP_COL], errors="coerce").fillna(0.0)
        feat["precip_t0"] = p0.values

    elif lag_setting in {"precip_lags", "both_lags"}:
        p = pd.to_numeric(file_df[PRECIP_COL], errors="coerce").fillna(0.0)
        feat["precip_d1"]   = p.shift(1)
        feat["precip_sum3"] = p.shift(1).rolling(3, min_periods=3).sum()
        feat["precip_sum7"] = p.shift(1).rolling(7, min_periods=7).sum()

        if lag_setting == "both_lags":
            for c in ENV_VEC_COLS:
                med_t0 = pd.Series(feat[f"{c}_overnight_median_t0"])
                feat[f"{c}_overnight_median_d1"]  = med_t0.shift(1)
                feat[f"{c}_overnight_median_ma3"] = med_t0.shift(1).rolling(3, min_periods=3).mean()

    else:
        raise ValueError("lag_setting must be one of {'no_lag','precip_lags','both_lags'}")

    out = pd.DataFrame(feat, index=file_df.index)
    out.insert(0, DATE_COL, pd.to_datetime(file_df[DATE_COL], errors="coerce"))
    return out

def _rand_param_from_grid(rng, grid):
    return {k: grid[k][int(rng.integers(0, len(grid[k])))] for k in grid}

def _xgb_search_grid():
    return {
        "n_estimators":     [300, 500, 700, 900, 1200],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.02, 0.03, 0.05, 0.07, 0.1],
        "subsample":        [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "min_child_weight": [1, 2, 5, 8],
        "reg_alpha":        [0.0, 0.2, 0.5, 1.0, 2.0],
        "reg_lambda":       [0.8, 1.0, 1.3, 1.6, 2.0],
        "max_bin":          [256, 384, 512],
        "gamma":            [0.0, 0.1, 0.2],
    }

def tune_imputer_on_train(Z_train, X_cols, was_missing_train, df_train_truth,
                          n_trials=48, n_splits=5, rng_seed=RANDOM_STATE):
    """Nested K-fold tuning for Stage-1 imputer (XGB inside MICE). Score: mean RMSE across targets."""
    n_splits_eff = max(2, min(n_splits, len(Z_train)))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=rng_seed)
    rng = np.random.default_rng(rng_seed)

    fixed_stage2 = dict(XGB_DEFAULT)  # temporary moderate model for scoring
    fixed_stage2.update(dict(n_estimators=700, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
                             reg_alpha=0.2, reg_lambda=1.3, max_bin=384, gamma=0.1))

    best_score, best_params = np.inf, None
    grid = _xgb_search_grid()
    labels = Z_train.index.to_numpy()

    for _ in range(n_trials):
        params = _rand_param_from_grid(rng, grid)
        fold_scores = []

        for tr_pos, va_pos in kf.split(labels):
            Z_tr, Z_va = Z_train.iloc[tr_pos], Z_train.iloc[va_pos]

            imp = IterativeImputer(
                estimator=XGBRegressor(**_merge_params(XGB_DEFAULT, params)),
                **MICE_PARAMS
            )
            imp.fit(Z_tr)
            Z_imp_tr = pd.DataFrame(imp.transform(Z_tr), columns=Z_train.columns, index=Z_tr.index)
            Z_imp_va = pd.DataFrame(imp.transform(Z_va), columns=Z_train.columns, index=Z_va.index)

            X_imp_tr, X_imp_va = Z_imp_tr[X_cols], Z_imp_va[X_cols]
            Yc_tr = Z_imp_tr[TARGETS].round().clip(lower=0).astype(int)

            tgt_scores = []
            for t in TARGETS:
                other = [c for c in TARGETS if c != t][0]
                Xt_tr = pd.concat([X_imp_tr, Yc_tr[[other]]], axis=1)
                Xt_va = pd.concat([X_imp_va, Z_imp_va[[other]].round().clip(lower=0).astype(int)], axis=1)

                model = XGBRegressor(**fixed_stage2)
                model.fit(Xt_tr, Yc_tr[t])
                yhat = np.rint(np.clip(model.predict(Xt_va), 0, None)).astype(float)

                known_mask = ~was_missing_train[t].iloc[va_pos]
                if known_mask.sum() == 0:
                    continue
                y_true = df_train_truth.iloc[va_pos][t].astype(float).to_numpy()
                tgt_scores.append(rmse(y_true[known_mask.values], yhat[known_mask.values]))

            if tgt_scores:
                fold_scores.append(float(np.mean(tgt_scores)))

        mean_score = float(np.mean(fold_scores)) if fold_scores else np.inf
        if mean_score < best_score:
            best_score, best_params = mean_score, params

    return best_params if best_params is not None else dict()

def tune_stage2_per_target(X_imp_train, Yc_train, was_missing_train, df_train_truth,
                           target_name, other_name, n_trials=96, n_splits=5, rng_seed=RANDOM_STATE+7):
    n_splits_eff = max(2, min(n_splits, len(X_imp_train)))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=rng_seed)
    rng = np.random.default_rng(rng_seed)
    labels = X_imp_train.index.to_numpy()

    grid = _xgb_search_grid()
    best_score, best_params = np.inf, None

    for _ in range(n_trials):
        params = _rand_param_from_grid(rng, grid)
        fold_scores = []

        for tr_pos, va_pos in kf.split(labels):
            Xt_tr = pd.concat([X_imp_train.iloc[tr_pos], Yc_train.iloc[tr_pos][[other_name]]], axis=1)
            Xt_va = pd.concat([X_imp_train.iloc[va_pos], Yc_train.iloc[va_pos][[other_name]]], axis=1)

            model = XGBRegressor(**_merge_params(XGB_DEFAULT, params))
            model.fit(Xt_tr, Yc_train.iloc[tr_pos][target_name])
            yhat = np.rint(np.clip(model.predict(Xt_va), 0, None)).astype(float)

            known_mask = ~was_missing_train[target_name].iloc[va_pos]
            if known_mask.sum() == 0:
                continue
            y_true = df_train_truth.iloc[va_pos][target_name].astype(float).to_numpy()
            fold_scores.append(rmse(y_true[known_mask.values], yhat[known_mask.values]))

        mean_score = float(np.mean(fold_scores)) if fold_scores else np.inf
        if mean_score < best_score:
            best_score, best_params = mean_score, dict(params)

    return best_params if best_params is not None else dict()

def _features_desc(lag_setting):
    if lag_setting == "no_lag":
        return "Overnight t0 stats; Precip same-day (precip_t0)"
    if lag_setting == "precip_lags":
        return "Overnight t0 stats; Precip d1 + sum3 + sum7 (causal)"
    return "Overnight t0 stats + env median_d1 + median_ma3; Precip d1 + sum3 + sum7 (causal)"

def _zero_desc(zp):
    return "TRAIN & TEST: 0→NaN" if zp == "standard" else "TRAIN keeps zeros; TEST 0→NaN"

def run_once(file, window_start, window_end, lag_setting, zero_policy,
             tuned=False, cv_folds=5, imputer_trials=48, stage2_trials=96,
             output_dir="results"):
    """
    Single 80/20 experiment with optional nested K-fold tuning (cv_folds/imputer_trials/stage2_trials).
    """
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"8020__{lag_setting}__{zero_policy}__{window_start}_{window_end}__{'tuned' if tuned else 'untuned'}"

    # Load + sort
    raw = pd.read_csv(file)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw = raw.sort_values(DATE_COL).reset_index(drop=True)

    # Build features on full, then slice window
    X_full = build_features(raw, lag_setting=lag_setting)
    mask = (X_full[DATE_COL] >= pd.Timestamp(window_start)) & (X_full[DATE_COL] <= pd.Timestamp(window_end))
    X_win = X_full.loc[mask].reset_index(drop=True)
    dates = X_win[DATE_COL]
    X = X_win.drop(columns=[DATE_COL]); X_cols = X.columns.tolist()

    Y_all = raw.loc[mask, TARGETS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    was_missing_orig = {t: (Y_all[t].isna() | (Y_all[t] == 0)) for t in TARGETS}

    # 80/20 split
    n = len(X)
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

    X_tr, X_te = X.iloc[idx_tr].reset_index(drop=True), X.iloc[idx_te].reset_index(drop=True)
    Y_tr, Y_te = Y_all.iloc[idx_tr].reset_index(drop=True), Y_all.iloc[idx_te].reset_index(drop=True)
    dates_te = dates.iloc[idx_te].reset_index(drop=True)
    was_missing_te = {t: was_missing_orig[t].iloc[idx_te].reset_index(drop=True) for t in TARGETS}
    was_missing_tr = {t: was_missing_orig[t].iloc[idx_tr].reset_index(drop=True) for t in TARGETS}

    # zero handling AFTER split
    Z_tr = pd.concat([X_tr, Y_tr], axis=1).copy()
    Z_te = pd.concat([X_te, Y_te], axis=1).copy()
    if zero_policy == "standard":
        for t in TARGETS:
            Z_tr.loc[Z_tr[t] == 0, t] = np.nan
            Z_te.loc[Z_te[t] == 0, t] = np.nan
    elif zero_policy == "train_keeps_zeros":
        for t in TARGETS:
            Z_te.loc[Z_te[t] == 0, t] = np.nan
    else:
        raise ValueError("zero_policy must be {'standard','train_keeps_zeros'}")

    # Stage-1 tuning (nested K-fold on TRAINING set)
    if tuned:
        best_mice_xgb = tune_imputer_on_train(
            Z_train=Z_tr, X_cols=X_cols,
            was_missing_train=was_missing_tr,
            df_train_truth=Y_tr,
            n_trials=imputer_trials, n_splits=cv_folds, rng_seed=RANDOM_STATE
        )
    else:
        best_mice_xgb = dict()

    # Stage-1 fit on TRAINING; transform TRAINING & TEST
    imp = IterativeImputer(estimator=XGBRegressor(**_merge_params(XGB_DEFAULT, best_mice_xgb)), **MICE_PARAMS)
    imp.fit(Z_tr)
    Z_imp_tr = pd.DataFrame(imp.transform(Z_tr), columns=Z_tr.columns, index=Z_tr.index)
    Z_imp_te = pd.DataFrame(imp.transform(Z_te), columns=Z_te.columns, index=Z_te.index)

    X_imp_tr, X_imp_te = Z_imp_tr[X_cols], Z_imp_te[X_cols]
    Yc_tr = Z_imp_tr[TARGETS].round().clip(lower=0).astype(int)
    Yc_te = Z_imp_te[TARGETS].round().clip(lower=0).astype(int)

    # Stage-2 tuning per target (nested K-fold on TRAINING)
    best_stage2 = {}
    for t in TARGETS:
        other = [c for c in TARGETS if c != t][0]
        if tuned:
            best_s2 = tune_stage2_per_target(
                X_imp_train=X_imp_tr, Yc_train=Yc_tr,
                was_missing_train=was_missing_tr, df_train_truth=Y_tr,
                target_name=t, other_name=other,
                n_trials=stage2_trials, n_splits=cv_folds, rng_seed=RANDOM_STATE+7
            )
        else:
            best_s2 = dict()
        best_stage2[t] = best_s2

    # final Stage-2 fit on TRAINING set and predicting on TEST set
    preds = {}
    for t in TARGETS:
        other = [c for c in TARGETS if c != t][0]
        Xt_tr = pd.concat([X_imp_tr, Yc_tr[[other]]], axis=1)
        Xt_te = pd.concat([X_imp_te, Yc_te[[other]]], axis=1)
        model = XGBRegressor(**_merge_params(XGB_DEFAULT, best_stage2[t]))
        model.fit(Xt_tr, Yc_tr[t])
        yhat = np.rint(np.clip(model.predict(Xt_te), 0, None)).astype(int)
        preds[t] = yhat

    # Metrics + CSVs
    features_desc = _features_desc(lag_setting)
    zero_desc = _zero_desc(zero_policy)
    metrics_rows = []

    for t in TARGETS:
        mask_known = ~was_missing_te[t]
        if mask_known.sum() > 0:
            y_true = Y_te[t].iloc[mask_known.to_numpy()].astype(float).to_numpy()
            y_pred = pd.Series(preds[t]).iloc[mask_known.to_numpy()].astype(float).to_numpy()
            mae_val = mean_absolute_error(y_true, y_pred)
            rmse_val = rmse(y_true, y_pred)
        else:
            mae_val = rmse_val = None

        out = pd.DataFrame({
            DATE_COL: dates_te,
            "True": Y_te[t].values,
            "Predicted": preds[t],
            "was_missing": was_missing_te[t].astype(bool).values
        }).sort_values(DATE_COL).reset_index(drop=True)
        out["Residual"] = np.where(~out["was_missing"],
                                   out["True"].astype(float) - out["Predicted"].astype(float),
                                   np.nan)

        pred_path = out_dir / f"{run_tag}__{t}__preds__{run_ts}.csv"
        out.to_csv(pred_path, index=False)

        metrics_rows.append({
            "protocol": "80/20",
            "run_tag": run_tag,
            "timestamp": run_ts,
            "file": Path(file).name,
            "window_start": window_start,
            "window_end": window_end,
            "lag_setting": lag_setting,
            "features_desc": features_desc,
            "zero_policy": zero_desc,
            "tuned": tuned,
            "cv_folds": cv_folds,
            "imputer_trials": imputer_trials,
            "stage2_trials": stage2_trials,
            "target": t,
            "known_rows": int(mask_known.sum()),
            "MAE": None if mae_val is None else float(mae_val),
            "RMSE": None if rmse_val is None else float(rmse_val),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = out_dir / f"{run_tag}__summary__{run_ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # console summary
    print("=" * 110)
    print(f"[80/20] {features_desc} | Zero: {zero_desc} | Window: {window_start}..{window_end}")
    print(f"Tuned: {tuned} | cv_folds={cv_folds} | imputer_trials={imputer_trials} | stage2_trials={stage2_trials}")
    if tuned:
        print("Best Stage-1 (imputer) XGB params:", best_mice_xgb)
        for t in TARGETS:
            print(f"Best Stage-2 ({t}) XGB params:", best_stage2[t])
    for row in metrics_rows:
        print(f"{row['target']}: known={row['known_rows']} | MAE={row['MAE']} | RMSE={row['RMSE']}")
    print(f"Saved: {metrics_path}")
    print("=" * 110)

# CLI
def _parse_window(win_str):
    try:
        s, e = win_str.split(":")
        return s, e
    except Exception:
        raise argparse.ArgumentTypeError("window must be 'YYYY-MM-DD:YYYY-MM-DD'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", choices=["run", "run-all"], default="run")
    ap.add_argument("--file", required=True)
    ap.add_argument("--lag", choices=["no_lag", "precip_lags", "both_lags"])
    ap.add_argument("--zero-policy", choices=["standard", "train_keeps_zeros"], dest="zero_policy")
    ap.add_argument("--window", type=_parse_window)
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--imputer-trials", type=int, default=48)
    ap.add_argument("--stage2-trials", type=int, default=96)
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    if args.cmd == "run":
        run_once(
            file=args.file,
            window_start=args.window[0], window_end=args.window[1],
            lag_setting=args.lag, zero_policy=args.zero_policy,
            tuned=args.tuned, cv_folds=args.cv_folds,
            imputer_trials=args.imputer_trials, stage2_trials=args.stage2_trials,
            output_dir=args.output_dir
        )
    else:
        # all combos as per our methodology (3 lags × 2 windows × 2 zero policies)
        lags = ["no_lag", "precip_lags", "both_lags"]
        wins = [("2015-07-01","2015-10-21"), ("2015-07-01","2015-10-31")]
        zps  = ["standard", "train_keeps_zeros"]
        for lag in lags:
            for (ws, we) in wins:
                for zp in zps:
                    run_once(
                        file=args.file, window_start=ws, window_end=we,
                        lag_setting=lag, zero_policy=zp,
                        tuned=args.tuned, cv_folds=args.cv_folds,
                        imputer_trials=args.imputer_trials, stage2_trials=args.stage2_trials,
                        output_dir=args.output_dir
                    )

if __name__ == "__main__":
    main()
