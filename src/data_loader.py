"""
src/data_loader.py
──────────────────
Load, validate, and clean crop yield + climate datasets.

Real data:   Place CSV files in data/raw/ (see README for download links)
Synthetic:   Auto-generated if real data files are not found (demo mode)
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load and return a cleaned, merged DataFrame.

    Priority:
      1. Real FAOSTAT + World Bank CSVs (if found in data/raw/)
      2. Synthetic demo data (if CSVs are missing)

    Parameters
    ----------
    cfg : dict
        Config dict loaded from configs/config.yaml  (data section)

    Returns
    -------
    pd.DataFrame
        Columns: year, temperature, rainfall, temp_anomaly, rain_anomaly,
                 yield_rice, yield_wheat, area_rice, area_wheat
    """
    fao_path  = cfg["data"]["faostat_path"]
    clim_path = cfg["data"]["climate_path"]

    if os.path.exists(fao_path) and os.path.exists(clim_path):
        print("✓ Real datasets found — loading from CSV …")
        df = _load_real_data(fao_path, clim_path)
    else:
        print("⚠  Real datasets not found — running in DEMO MODE with synthetic data.")
        print(f"   Expected:\n     {fao_path}\n     {clim_path}")
        df = _generate_synthetic_data(
            n_years=40,
            start_year=cfg["data"].get("start_year", 1983)
        )

    df = _clean(df)
    _print_summary(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_data(fao_path: str, clim_path: str) -> pd.DataFrame:
    """
    Load and merge FAOSTAT and World Bank climate CSV files.

    Expected FAOSTAT columns  : Year, Item, Element, Value
    Expected Climate columns  : Year, avg_temp, total_rainfall
    (Column names are flexible — the function normalises them)
    """
    fao  = pd.read_csv(fao_path)
    clim = pd.read_csv(clim_path)

    # ── Normalise column names ────────────────────────────────────────────────
    fao.columns  = [c.strip().lower().replace(" ", "_") for c in fao.columns]
    clim.columns = [c.strip().lower().replace(" ", "_") for c in clim.columns]

    # ── Pivot FAOSTAT: wide format (one row per year) ─────────────────────────
    if "element" in fao.columns and "item" in fao.columns:
        yield_df = (
            fao[fao["element"].str.lower().str.contains("yield")]
            .pivot_table(index="year", columns="item", values="value", aggfunc="mean")
        )
        yield_df.columns = [f"yield_{c.lower().replace(' ', '_')}" for c in yield_df.columns]
        yield_df.reset_index(inplace=True)
    else:
        yield_df = fao  # already wide

    # ── Standardise climate column names ─────────────────────────────────────
    rename_map = {}
    for col in clim.columns:
        if "temp" in col:
            rename_map[col] = "temperature"
        elif "rain" in col or "precip" in col:
            rename_map[col] = "rainfall"
    clim.rename(columns=rename_map, inplace=True)

    # ── Merge on year ─────────────────────────────────────────────────────────
    df = pd.merge(yield_df, clim, on="year", how="inner")

    # ── Derived anomaly columns ───────────────────────────────────────────────
    baseline_mask = df["year"] <= df["year"].min() + 9
    df["temp_anomaly"] = df["temperature"] - df.loc[baseline_mask, "temperature"].mean()
    df["rain_anomaly"] = df["rainfall"]    - df.loc[baseline_mask, "rainfall"].mean()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (demo / testing)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_data(n_years: int = 40, start_year: int = 1983) -> pd.DataFrame:
    """
    Generate realistic synthetic climate + crop yield data.
    Mirrors the column structure of FAOSTAT + World Bank datasets.
    """
    rng   = np.random.default_rng(42)
    years = np.arange(start_year, start_year + n_years)
    t     = np.arange(n_years)

    # ── Climate ───────────────────────────────────────────────────────────────
    temperature = 25.0 + 0.04 * t + rng.normal(0, 0.6, n_years)
    rainfall    = 900.0 - 1.5 * t / n_years * 20 + rng.normal(0, 60, n_years)

    drought_idx = rng.choice(n_years, size=5, replace=False)
    flood_idx   = rng.choice(n_years, size=3, replace=False)
    rainfall[drought_idx] *= 0.55
    rainfall[flood_idx]   *= 1.45

    baseline_temp = temperature[:10].mean()
    baseline_rain = rainfall[:10].mean()

    # ── Crop yields (tons / hectare) ──────────────────────────────────────────
    rain_eff  = np.clip((rainfall - 600) / 500, -0.4,  0.3)
    temp_eff  = np.clip(-(temperature - 26) * 0.15, -0.5, 0.1)

    yield_rice  = 2.5 + 0.05 * t + rain_eff + temp_eff + rng.normal(0, 0.12, n_years)
    yield_wheat = 2.8 + 0.04 * t + rain_eff * 0.7 + np.clip(-(temperature - 24) * 0.2, -0.6, 0.1) \
                  + rng.normal(0, 0.10, n_years)

    area_rice   = 40.0 + 0.1 * t + rng.normal(0, 1.5, n_years)
    area_wheat  = 30.0 + 0.08 * t + rng.normal(0, 1.2, n_years)

    return pd.DataFrame({
        "year":         years,
        "temperature":  temperature,
        "rainfall":     rainfall,
        "temp_anomaly": temperature - baseline_temp,
        "rain_anomaly": rainfall    - baseline_rain,
        "yield_rice":   yield_rice,
        "yield_wheat":  yield_wheat,
        "area_rice":    area_rice,
        "area_wheat":   area_wheat,
    })


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fill or drop missing values, clip physiological limits."""
    df = df.drop_duplicates(subset=["year"]).sort_values("year").reset_index(drop=True)

    # Fill minor gaps with linear interpolation
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Clip impossible yields
    for col in df.columns:
        if col.startswith("yield_"):
            df[col] = df[col].clip(lower=0.3)

    df.dropna(inplace=True)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'─'*50}")
    print(f"  Dataset loaded  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Year range      : {df['year'].min()} – {df['year'].max()}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print(f"{'─'*50}")
