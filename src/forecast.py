"""
src/forecast.py
───────────────
10-year forward projection using the best traditional ML model.
Generates two scenarios:
  1. Business-as-Usual  — moderate warming & drying
  2. Climate Stress     — accelerated warming & drying
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "primary":  "#2C5F2D",
    "accent":   "#F96167",
    "neutral":  "#365663",
}
SCENARIO_COLORS = {
    "Business-as-Usual": "#97BC62",
    "Climate Stress":    "#F96167",
}


def run_forecast(
    df:           pd.DataFrame,
    feature_cols: list[str],
    trad_results: dict,
    cfg:          dict,
) -> pd.DataFrame:
    """
    Project crop yield for the next N years under two climate scenarios.

    Returns
    -------
    pd.DataFrame with columns: year, scenario, yield_pred
    """
    fcast_cfg   = cfg["forecast"]
    n_future    = fcast_cfg["n_future_years"]
    target_col  = cfg["data"]["target_col"]

    best_name   = min(trad_results, key=lambda k: trad_results[k]["rmse"])
    best_model  = trad_results[best_name]["model"]
    scaler      = trad_results[best_name]["scaler"]

    last_row    = df.iloc[-1].copy()
    future_yrs  = np.arange(df["year"].max() + 1, df["year"].max() + 1 + n_future)

    scenarios = {
        "Business-as-Usual": fcast_cfg["scenarios"]["business_as_usual"],
        "Climate Stress":    fcast_cfg["scenarios"]["climate_stress"],
    }

    records = []
    for i, yr in enumerate(future_yrs):
        for sc_name, sc_params in scenarios.items():
            row = last_row.copy()
            row["year"]         = yr
            row["temperature"] += sc_params["temp_delta_per_year"] * i
            row["rainfall"]    += sc_params["rain_delta_per_year"] * i
            row["year_norm"]    = (yr - df["year"].min()) / (df["year"].max() - df["year"].min())

            feat_vec = np.array([[row[c] for c in feature_cols if c in row.index]])
            # Pad if shape mismatch (safety)
            if feat_vec.shape[1] < scaler.n_features_in_:
                pad = np.zeros((1, scaler.n_features_in_ - feat_vec.shape[1]))
                feat_vec = np.hstack([feat_vec, pad])
            feat_sc  = scaler.transform(feat_vec[:, :scaler.n_features_in_])
            pred     = float(best_model.predict(feat_sc)[0])

            records.append({"year": yr, "scenario": sc_name, "yield_pred": max(pred, 0.3)})

    fcast_df = pd.DataFrame(records)

    _plot(df, target_col, fcast_df, best_name, cfg)
    _print_table(fcast_df)
    return fcast_df


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _plot(df, target_col, fcast_df, model_name, cfg):
    out_dir = cfg["output"]["plots_dir"]
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    ax.plot(df["year"], df[target_col], "k-o", linewidth=2, markersize=4,
            label="Historical", zorder=5)

    for sc_name, sc_df in fcast_df.groupby("scenario"):
        color = SCENARIO_COLORS[sc_name]
        ax.plot(sc_df["year"], sc_df["yield_pred"], "--",
                color=color, linewidth=2.5, label=f"Forecast: {sc_name}")
        ax.fill_between(sc_df["year"],
                        sc_df["yield_pred"] * 0.92,
                        sc_df["yield_pred"] * 1.08,
                        color=color, alpha=0.12)

    ax.axvline(df["year"].max(), color="gray", linestyle=":", linewidth=1.5,
               label="Forecast start")
    ax.set_title(
        f"{cfg['forecast']['n_future_years']}-Year Rice Yield Forecast — {model_name}\n"
        "Scenario Comparison: Business-as-Usual vs Climate Stress",
        fontweight="bold", color=COLORS["primary"],
    )
    ax.set_xlabel("Year"); ax.set_ylabel("Predicted Yield (t/ha)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "yield_forecast.png")
    plt.savefig(save_path, dpi=cfg["output"]["dpi"], bbox_inches="tight")
    plt.close()
    print(f"✓ Forecast plot saved → {save_path}")


def _print_table(fcast_df: pd.DataFrame) -> None:
    pivot = fcast_df.pivot(index="year", columns="scenario", values="yield_pred").round(3)
    print("\n── 10-Year Yield Forecast (t/ha) ──")
    print(pivot.to_string())
