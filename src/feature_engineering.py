"""
src/feature_engineering.py
───────────────────────────
Create lag features, rolling statistics, and climate event flags
from the cleaned climate + yield DataFrame.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Add all derived features to the DataFrame.

    Parameters
    ----------
    df  : cleaned DataFrame from data_loader.load_data()
    cfg : full config dict (features section used)

    Returns
    -------
    df_feat       : DataFrame with all feature columns added
    feature_cols  : list of column names to use as model inputs (X)
    """
    feat_cfg = cfg["features"]

    df = df.copy().sort_values("year").reset_index(drop=True)

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in feat_cfg["lag_years"]:
        df[f"temp_lag{lag}"]       = df["temperature"].shift(lag)
        df[f"rain_lag{lag}"]       = df["rainfall"].shift(lag)
        df[f"yield_rice_lag{lag}"] = df["yield_rice"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in feat_cfg["rolling_windows"]:
        df[f"temp_roll{w}"] = df["temperature"].rolling(w).mean()
        df[f"rain_roll{w}"] = df["rainfall"].rolling(w).mean()
    df["rain_std3"] = df["rainfall"].rolling(3).std()

    # ── Climate event flags ───────────────────────────────────────────────────
    rain_mean = df["rainfall"].mean()
    temp_stress_thresh = feat_cfg.get("heat_stress_temp", 28.0)

    df["drought_flag"] = (df["rainfall"] < rain_mean * feat_cfg["drought_threshold"]).astype(int)
    df["flood_flag"]   = (df["rainfall"] > rain_mean * feat_cfg["flood_threshold"]).astype(int)
    df["temp_stress"]  = (df["temperature"] > temp_stress_thresh).astype(int)

    # ── Normalised time index ─────────────────────────────────────────────────
    df["year_norm"] = (df["year"] - df["year"].min()) / (df["year"].max() - df["year"].min())

    # Drop NaN rows introduced by lagging / rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Build feature column list ─────────────────────────────────────────────
    feature_cols = (
        ["temperature", "rainfall", "temp_anomaly", "rain_anomaly"]
        + [f"temp_lag{l}" for l in feat_cfg["lag_years"]]
        + [f"rain_lag{l}" for l in feat_cfg["lag_years"]]
        + [f"yield_rice_lag{l}" for l in feat_cfg["lag_years"]]
        + [f"temp_roll{w}" for w in feat_cfg["rolling_windows"]]
        + [f"rain_roll{w}" for w in feat_cfg["rolling_windows"]]
        + ["rain_std3", "drought_flag", "flood_flag", "temp_stress", "year_norm"]
    )
    # Keep only columns that exist (safety check)
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"✓ Feature engineering complete — {len(feature_cols)} features, {len(df)} samples")
    return df, feature_cols
