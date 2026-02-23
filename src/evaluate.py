"""
src/evaluate.py
───────────────
Generate model evaluation plots:
- Predictions vs Actuals
- Residuals
- Feature Importance (Random Forest)
- Model Comparison Bar Chart
- LSTM Training Curve
- Actual vs Predicted Scatter
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    "primary":   "#2C5F2D",
    "secondary": "#97BC62",
    "accent":    "#F96167",
    "neutral":   "#365663",
}
MODEL_COLORS = {
    "Random Forest":      "#97BC62",
    "XGBoost":            "#365663",
    "GradientBoosting":   "#365663",
    "LSTM":               "#F96167",
}


def plot_results(
    trad_results: dict,
    lstm_result:  dict | None,
    df:           pd.DataFrame,
    y_test:       np.ndarray,
    cfg:          dict,
) -> None:
    """
    Create and save the 6-panel model results figure.
    """
    out_dir     = cfg["output"]["plots_dir"]
    train_split = cfg["data"]["train_split"]
    os.makedirs(out_dir, exist_ok=True)

    split_idx  = int(len(df) * train_split)
    test_years = df["year"].values[split_idx:]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        "Model Evaluation — Climate-Resilient Crop Yield Prediction",
        fontsize=16, fontweight="bold", color=COLORS["primary"],
    )

    # ── 1. Predictions vs Actual ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(test_years, y_test, "k-o", linewidth=2, markersize=5,
            label="Actual", zorder=5)
    for name, res in trad_results.items():
        ax.plot(test_years, res["preds"], "--",
                color=MODEL_COLORS.get(name, "blue"), linewidth=1.8, label=name)
    if lstm_result:
        n = len(lstm_result["preds"])
        ax.plot(test_years[-n:], lstm_result["preds"], ":",
                color=MODEL_COLORS["LSTM"], linewidth=1.8, label="LSTM")
    ax.set_title("Predicted vs Actual Rice Yield (Test Set)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Yield (t/ha)")
    ax.legend(fontsize=9)

    # ── 2. Residuals ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for name, res in trad_results.items():
        ax.plot(test_years, y_test - res["preds"],
                color=MODEL_COLORS.get(name, "blue"), linewidth=1.5, label=name)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Residuals (Actual − Predicted)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Residual (t/ha)")
    ax.legend(fontsize=9)

    # ── 3. Feature Importance ────────────────────────────────────────────────
    ax = axes[1, 0]
    if "Random Forest" in trad_results:
        rf      = trad_results["Random Forest"]["model"]
        fi_cols = _get_feature_cols(df, cfg)
        imp = pd.Series(rf.feature_importances_, index=fi_cols[:len(rf.feature_importances_)])
        imp.nlargest(12).sort_values().plot.barh(
            ax=ax, color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.set_title("Top 12 Feature Importances (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance Score")

    # ── 4. Model Comparison ───────────────────────────────────────────────────
    ax = axes[1, 1]
    names     = list(trad_results.keys()) + (["LSTM"] if lstm_result else [])
    rmse_vals = [trad_results[m]["rmse"] for m in trad_results] + \
                ([lstm_result["rmse"]] if lstm_result else [])
    r2_vals   = [trad_results[m]["r2"]  for m in trad_results] + \
                ([lstm_result["r2"]]  if lstm_result else [])

    x = np.arange(len(names)); w = 0.35
    b1 = ax.bar(x - w/2, rmse_vals, w, label="RMSE",
                color=COLORS["accent"],   alpha=0.85, edgecolor="white")
    ax2b = ax.twinx()
    b2 = ax2b.bar(x + w/2, r2_vals, w, label="R²",
                  color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.set_title("Model Comparison: RMSE vs R²", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("RMSE (↓ better)", color=COLORS["accent"])
    ax2b.set_ylabel("R² (↑ better)",  color=COLORS["primary"])
    ax.legend([b1, b2], ["RMSE", "R²"], loc="upper right", fontsize=9)

    # ── 5. LSTM Training Curve ────────────────────────────────────────────────
    ax = axes[2, 0]
    if lstm_result:
        hist = lstm_result["history"].history
        ax.plot(hist["loss"],     color=COLORS["accent"],  linewidth=2, label="Train Loss")
        ax.plot(hist["val_loss"], color=COLORS["neutral"], linewidth=2,
                linestyle="--", label="Val Loss")
        ax.set_title("LSTM Training & Validation Loss", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "LSTM not available\n(install TensorFlow to enable)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        ax.set_title("LSTM Training Curve", fontweight="bold")

    # ── 6. Actual vs Predicted Scatter ────────────────────────────────────────
    ax = axes[2, 1]
    best = min(trad_results, key=lambda k: trad_results[k]["rmse"])
    bp   = trad_results[best]["preds"]
    ax.scatter(y_test, bp, color=COLORS["primary"], alpha=0.8,
               s=60, edgecolors="white", linewidth=0.5)
    lo, hi = min(y_test.min(), bp.min()), max(y_test.max(), bp.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted — {best}", fontweight="bold")
    ax.set_xlabel("Actual Yield (t/ha)"); ax.set_ylabel("Predicted Yield (t/ha)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "model_results.png")
    plt.savefig(save_path, dpi=cfg["output"]["dpi"], bbox_inches="tight")
    plt.close()
    print(f"✓ Model results plot saved → {save_path}")


def print_summary_table(trad_results: dict, lstm_result: dict | None) -> None:
    print("\n" + "=" * 56)
    print("  FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 56)
    print(f"  {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>7}")
    print("  " + "-" * 50)
    for name, res in trad_results.items():
        print(f"  {name:<22} {res['rmse']:>7.4f} {res['mae']:>7.4f} {res['r2']:>7.4f}")
    if lstm_result:
        print(f"  {'LSTM':<22} {lstm_result['rmse']:>7.4f} "
              f"{lstm_result['mae']:>7.4f} {lstm_result['r2']:>7.4f}")
    print("=" * 56)


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame, cfg: dict) -> list[str]:
    feat = cfg["features"]
    cols = (
        ["temperature", "rainfall", "temp_anomaly", "rain_anomaly"]
        + [f"temp_lag{l}" for l in feat["lag_years"]]
        + [f"rain_lag{l}" for l in feat["lag_years"]]
        + [f"yield_rice_lag{l}" for l in feat["lag_years"]]
        + [f"temp_roll{w}" for w in feat["rolling_windows"]]
        + [f"rain_roll{w}" for w in feat["rolling_windows"]]
        + ["rain_std3", "drought_flag", "flood_flag", "temp_stress", "year_norm"]
    )
    return [c for c in cols if c in df.columns]
