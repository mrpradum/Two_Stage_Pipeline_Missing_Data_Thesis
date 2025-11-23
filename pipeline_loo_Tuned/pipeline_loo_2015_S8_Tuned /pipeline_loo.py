#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOO pipeline — Nested K-fold tuning (inside each outer fold) — XGB-Poisson in Stage-1/2

Supports:
- lag_setting: {"no_lag", "precip_lags", "both_lags"}
- zero_policy: {"standard", "train_keeps_zeros"}
- windows: 2015-07-01..2015-10-21  or  2015-07-01..2015-10-31 (inclusive)
- tuned: if True, nested K-fold tuning with budgets (cv_folds, imputer_trials, stage2_trials)
"""

import ast, json, argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# constants
RANDOM_STATE = 42
DATE_COL   = "Date"
PRECIP_COL = "Precipitation"
ENV_VEC_COLS = ["Temperature", "Humidity", "Dew_Point", "Water_Temperature"]
TARGETS = ["Adults_8_Col", "Adults_8_Gam0"]

MICE_PARAMS = dict(
    max_iter=25, sample_posterior=False, random_state=RANDOM_STATE,
    imputation_order="ascending", skip_complete=True,
    initial_strategy="median", tol=1e-3
)

XGB_DEFAULT = dict(
    objective="count:poisson", tree_method="hist",
    n_estimators=400, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
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

def _rand_choice(rng, arr):
    return arr[int(rng.integers(0, len(arr)))]

def _rand_params(rng, grid):
    return {k: _rand_choice(rng, v) for k, v in grid.items()}

def tune_imputer_on_train(Z_train, X_cols, was_missing_train, df_train_truth,
                          n_trials=48, n_splits=5, rng_seed=RANDOM_STATE):
    n_splits_eff = max(2, min(n_splits, len(Z_train)))
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=rng_seed)
    rng = np.random.default_rng(rng_seed)
    grid = _xgb_search_grid()

    # temporary moderate Stage-2 for scoring
    fixed_stage2 = dict(XGB_DEFAULT)
    fixed_stage2.update(dict(n_estimators=700, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
                             reg_alpha=0.2, reg_lambda=1.3, max_bin=384, gamma=0.1))

    best_score, best_params = np.inf, None
    labels = Z_train.index.to_numpy()

    for _ in range(n_trials):
        params = _rand_params(rng, grid)
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
            best_score, best_params = mean_score, dict(params)

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
        params = _rand_params(rng, grid)
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
    Single LOO experiment with optional nested K-fold tuning (per outer fold).
    """
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"LOO__{lag_setting}__{zero_policy}__{window_start}_{window_end}__{'tuned' if tuned else 'untuned'}"

    raw = pd.read_csv(file)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw = raw.sort_values(DATE_COL).reset_index(drop=True)

    X_full = build_features(raw, lag_setting=lag_setting)
    mask = (X_full[DATE_COL] >= pd.Timestamp(window_start)) & (X_full[DATE_COL] <= pd.Timestamp(window_end))
    X_win = X_full.loc[mask].reset_index(drop=True)
    dates = X_win[DATE_COL]
    X = X_win.drop(columns=[DATE_COL]); X_cols = X.columns.tolist()

    Y_all = raw.loc[mask, TARGETS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    was_missing_orig = {t: (Y_all[t].isna() | (Y_all[t] == 0)) for t in TARGETS}

    n = len(X); idx_all = np.arange(n)
    preds = {t: np.full(n, np.nan, float) for t in TARGETS}

    chosen_imp_params = []
    chosen_stage2_params = {t: [] for t in TARGETS}

    Z_all_base = pd.concat([X, Y_all], axis=1)
    features_desc = _features_desc(lag_setting)
    zero_desc = _zero_desc(zero_policy)

    for i in range(n):
        train_idx = idx_all[idx_all != i]; test_idx = np.array([i])

        Z_train = Z_all_base.iloc[train_idx].copy()
        Z_test  = Z_all_base.iloc[test_idx].copy()

        # zero handling per fold
        if zero_policy == "standard":
            for t in TARGETS:
                Z_train.loc[Z_train[t] == 0, t] = np.nan
                Z_test .loc[Z_test [t] == 0, t] = np.nan
        elif zero_policy == "train_keeps_zeros":
            for t in TARGETS:
                Z_test.loc[Z_test[t] == 0, t] = np.nan
        else:
            raise ValueError("zero_policy must be {'standard','train_keeps_zeros'}")

        was_missing_train = {t: was_missing_orig[t].iloc[train_idx] for t in TARGETS}
        df_train_truth = Y_all.iloc[train_idx]

        # tune imputer on this outer-train
        if tuned:
            best_imp = tune_imputer_on_train(
                Z_train, X_cols,
                was_missing_train=was_missing_train, df_train_truth=df_train_truth,
                n_trials=imputer_trials, n_splits=cv_folds, rng_seed=RANDOM_STATE
            )
        else:
            best_imp = dict()
        chosen_imp_params.append(best_imp)

        # fit imputer -> complete train & test
        imp = IterativeImputer(estimator=XGBRegressor(**_merge_params(XGB_DEFAULT, best_imp)), **MICE_PARAMS)
        imp.fit(Z_train)
        Z_imp_tr = pd.DataFrame(imp.transform(Z_train), columns=Z_train.columns, index=Z_train.index)
        Z_imp_te = pd.DataFrame(imp.transform(Z_test),  columns=Z_test.columns,  index=Z_test.index)

        X_imp_tr, X_imp_te = Z_imp_tr[X_cols], Z_imp_te[X_cols]
        Yc_tr = Z_imp_tr[TARGETS].round().clip(lower=0).astype(int)
        Yc_te = Z_imp_te[TARGETS].round().clip(lower=0).astype(int)

        # tune and predict per target
        for t in TARGETS:
            other = [c for c in TARGETS if c != t][0]
            if tuned:
                best_s2 = tune_stage2_per_target(
                    X_imp_train=X_imp_tr, Yc_train=Yc_tr,
                    was_missing_train=was_missing_train, df_train_truth=df_train_truth,
                    target_name=t, other_name=other,
                    n_trials=stage2_trials, n_splits=cv_folds, rng_seed=RANDOM_STATE+7
                )
            else:
                best_s2 = dict()
            chosen_stage2_params[t].append(best_s2)

            Xt_tr = pd.concat([X_imp_tr, Yc_tr[[other]]], axis=1)
            Xt_te = pd.concat([X_imp_te, Yc_te[[other]]], axis=1)
            model = XGBRegressor(**_merge_params(XGB_DEFAULT, best_s2))
            model.fit(Xt_tr, Yc_tr[t])
            preds[t][i] = float(np.clip(model.predict(Xt_te)[0], 0, None))

    # aggregate across outer folds
    metrics_rows = []
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    for t in TARGETS:
        y_pred_all = np.rint(preds[t]).astype(int)
        mask_known = ~was_missing_orig[t]
        if mask_known.sum() > 0:
            y_true = Y_all.loc[mask_known, t].astype(float).to_numpy()
            y_pred = y_pred_all[mask_known.to_numpy()].astype(float)
            mae_val = mean_absolute_error(y_true, y_pred)
            rmse_val = rmse(y_true, y_pred)
        else:
            mae_val = rmse_val = None

        out = pd.DataFrame({
            DATE_COL: dates,
            "True": Y_all[t].values,
            "Predicted": y_pred_all,
            "was_missing": was_missing_orig[t].astype(bool).values
        }).sort_values(DATE_COL).reset_index(drop=True)
        out["Residual"] = np.where(~out["was_missing"],
                                   out["True"].astype(float) - out["Predicted"].astype(float),
                                   np.nan)
        preds_path = Path(output_dir) / f"{run_tag}__{t}__preds__{ts_now}.csv"
        out.to_csv(preds_path, index=False)

        # modal params info
        imp_mode_key, imp_mode_cnt = Counter(tuple(sorted(p.items())) for p in chosen_imp_params).most_common(1)[0] if chosen_imp_params else ((), 0)
        s2_mode_key, s2_mode_cnt = Counter(tuple(sorted(p.items())) for p in chosen_stage2_params[t]).most_common(1)[0] if chosen_stage2_params[t] else ((), 0)

        metrics_rows.append({
            "protocol": "LOO",
            "run_tag": run_tag,
            "timestamp": ts_now,
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
            "imputer_params_mode": json.dumps(dict(imp_mode_key)) if imp_mode_key else "",
            "stage2_params_mode": json.dumps(dict(s2_mode_key)) if s2_mode_key else "",
            "imputer_mode_count": int(imp_mode_cnt),
            "stage2_mode_count": int(s2_mode_cnt),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = Path(output_dir) / f"{run_tag}__summary__{ts_now}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # console summary
    print("=" * 110)
    print(f"[LOO] {_features_desc(lag_setting)} | Zero: {_zero_desc(zero_policy)} | Window: {window_start}..{window_end}")
    print(f"Tuned: {tuned} | cv_folds={cv_folds} | imputer_trials={imputer_trials} | stage2_trials={stage2_trials}")
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
