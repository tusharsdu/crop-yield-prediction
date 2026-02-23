"""
src/models/traditional_models.py
──────────────────────────────────
Random Forest and XGBoost / GradientBoosting regressors
for crop yield prediction.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    _XGB = True
except ImportError:
    _XGB = False
    print("XGBoost not installed — using GradientBoostingRegressor as fallback.")


def train_traditional_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Train Random Forest and XGBoost/GBM models.

    Returns
    -------
    results   : dict  {model_name: {model, preds, rmse, mae, r2, scaler}}
    y_train   : training labels
    y_test    : test labels
    """
    train_split = cfg["data"]["train_split"]
    split_idx   = int(len(df) * train_split)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n► Training Random Forest …")
    rf_cfg = cfg["random_forest"]
    rf = RandomForestRegressor(
        n_estimators    = rf_cfg["n_estimators"],
        max_depth       = rf_cfg["max_depth"],
        min_samples_leaf= rf_cfg["min_samples_leaf"],
        max_features    = rf_cfg["max_features"],
        random_state    = rf_cfg["random_state"],
        n_jobs          = rf_cfg["n_jobs"],
    )
    rf.fit(X_train_sc, y_train)
    rf_preds = rf.predict(X_test_sc)
    results["Random Forest"] = _pack(rf, rf_preds, y_test, scaler)
    _print_metrics("Random Forest", results["Random Forest"])

    # ── XGBoost / GBM ────────────────────────────────────────────────────────
    print("\n► Training XGBoost …")
    xb_cfg = cfg["xgboost"]
    if _XGB:
        model = xgb.XGBRegressor(
            n_estimators     = xb_cfg["n_estimators"],
            learning_rate    = xb_cfg["learning_rate"],
            max_depth        = xb_cfg["max_depth"],
            subsample        = xb_cfg["subsample"],
            colsample_bytree = xb_cfg["colsample_bytree"],
            reg_alpha        = xb_cfg["reg_alpha"],
            reg_lambda       = xb_cfg["reg_lambda"],
            random_state     = xb_cfg["random_state"],
            verbosity        = 0,
        )
        name = "XGBoost"
    else:
        model = GradientBoostingRegressor(
            n_estimators  = xb_cfg["n_estimators"],
            learning_rate = xb_cfg["learning_rate"],
            max_depth     = xb_cfg["max_depth"],
            subsample     = xb_cfg["subsample"],
            random_state  = xb_cfg["random_state"],
        )
        name = "GradientBoosting"

    model.fit(X_train_sc, y_train)
    xb_preds = model.predict(X_test_sc)
    results[name] = _pack(model, xb_preds, y_test, scaler)
    _print_metrics(name, results[name])

    # ── Save models ───────────────────────────────────────────────────────────
    _save_models(results, cfg)

    return results, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pack(model, preds, y_test, scaler) -> dict:
    return {
        "model":  model,
        "preds":  preds,
        "scaler": scaler,
        "rmse":   float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae":    float(mean_absolute_error(y_test, preds)),
        "r2":     float(r2_score(y_test, preds)),
    }


def _print_metrics(name: str, res: dict) -> None:
    print(f"  {name:<22}  RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  R²={res['r2']:.4f}")


def _save_models(results: dict, cfg: dict) -> None:
    out_dir = cfg["output"]["models_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name, res in results.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        path  = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            pickle.dump({"model": res["model"], "scaler": res["scaler"]}, f)
        print(f"  ✓ Saved {path}")
