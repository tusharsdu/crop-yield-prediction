"""
src/eda.py
──────────
Exploratory Data Analysis — generates a 9-panel visualization
covering climate trends, yield trends, correlations, and extreme events.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

COLORS = {
    "primary":   "#2C5F2D",
    "secondary": "#97BC62",
    "accent":    "#F96167",
    "neutral":   "#365663",
    "light":     "#F5F5F5",
}


def run_eda(df: pd.DataFrame, cfg: dict) -> None:
    """
    Generate and save the EDA figure.

    Parameters
    ----------
    df  : feature-engineered DataFrame
    cfg : full config dict (output.plots_dir used)
    """
    out_dir = cfg["output"]["plots_dir"]
    os.makedirs(out_dir, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Climate-Resilient Crop Yield — Exploratory Data Analysis",
        fontsize=16, fontweight="bold", color=COLORS["primary"], y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Temperature trend ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(df["year"], df["temperature"], color=COLORS["accent"], linewidth=2)
    ax.fill_between(df["year"], df["temperature"], df["temperature"].min(),
                    alpha=0.15, color=COLORS["accent"])
    ax.set_title("Mean Temperature Over Time", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("°C")

    # ── 2. Rainfall bar chart ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(df["year"], df["rainfall"], color=COLORS["neutral"], alpha=0.7, width=0.8)
    drought_mask = df["drought_flag"] == 1
    ax.bar(df["year"][drought_mask], df["rainfall"][drought_mask],
           color=COLORS["accent"], alpha=0.9, width=0.8, label="Drought year")
    ax.set_title("Annual Rainfall", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("mm")
    ax.legend(fontsize=8)

    # ── 3. Crop yield trends ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(df["year"], df["yield_rice"],  color=COLORS["primary"],   linewidth=2, label="Rice")
    ax.plot(df["year"], df["yield_wheat"], color=COLORS["secondary"], linewidth=2, label="Wheat")
    ax.set_title("Crop Yield Over Time", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("tons/hectare")
    ax.legend()

    # ── 4. Scatter: Temp vs Rice yield ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(df["temperature"], df["yield_rice"],
                    c=df["year"], cmap="YlGn", alpha=0.8, s=50)
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_title("Temperature vs Rice Yield", fontweight="bold")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Rice Yield (t/ha)")

    # ── 5. Scatter: Rainfall vs Rice yield ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    point_colors = [COLORS["accent"] if d else COLORS["neutral"] for d in df["drought_flag"]]
    ax.scatter(df["rainfall"], df["yield_rice"], c=point_colors, alpha=0.8, s=50)
    ax.set_title("Rainfall vs Rice Yield\n(red = drought year)", fontweight="bold")
    ax.set_xlabel("Rainfall (mm)"); ax.set_ylabel("Rice Yield (t/ha)")

    # ── 6. Correlation heatmap ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    corr_cols = ["temperature", "rainfall", "temp_anomaly", "rain_anomaly",
                 "yield_rice", "yield_wheat", "drought_flag"]
    corr = df[[c for c in corr_cols if c in df.columns]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)

    # ── 7. Yield distribution ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(df["yield_rice"],  bins=15, color=COLORS["primary"],
            alpha=0.7, label="Rice",  edgecolor="white")
    ax.hist(df["yield_wheat"], bins=15, color=COLORS["secondary"],
            alpha=0.7, label="Wheat", edgecolor="white")
    ax.set_title("Yield Distribution", fontweight="bold")
    ax.set_xlabel("Yield (t/ha)"); ax.set_ylabel("Frequency")
    ax.legend()

    # ── 8. 5-year rolling averages ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax_twin = ax.twinx()
    if "temp_roll5" in df.columns:
        ax.plot(df["year"], df["temp_roll5"], color=COLORS["accent"],
                linewidth=2, label="Temp 5yr avg")
        ax_twin.plot(df["year"], df["rain_roll5"], color=COLORS["neutral"],
                     linewidth=2, linestyle="--", label="Rain 5yr avg")
    ax.set_title("5-Year Rolling Averages", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temp (°C)",  color=COLORS["accent"])
    ax_twin.set_ylabel("Rain (mm)", color=COLORS["neutral"])
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax_twin.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=7)

    # ── 9. Extreme events timeline ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.bar(df["year"], df["drought_flag"], color=COLORS["accent"],  alpha=0.8, label="Drought")
    ax.bar(df["year"], df["flood_flag"],   color=COLORS["neutral"], alpha=0.8,
           bottom=df["drought_flag"], label="Flood")
    ax.set_title("Extreme Climate Events", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Event flag")
    ax.legend()

    save_path = os.path.join(out_dir, "eda_plots.png")
    plt.savefig(save_path, dpi=cfg["output"]["dpi"], bbox_inches="tight")
    plt.close()
    print(f"✓ EDA plot saved → {save_path}")
