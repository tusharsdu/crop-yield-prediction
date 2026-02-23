"""
main.py
────────
Entry point for the Climate-Resilient Crop Yield Prediction System.
Runs the full ML pipeline end-to-end.

Usage
─────
  python main.py                          # Full pipeline with default config
  python main.py --step eda              # EDA only
  python main.py --step train            # Train models only
  python main.py --step forecast         # Forecast only
  python main.py --config path/to/cfg   # Custom config file
"""

import argparse
import sys
import yaml
import os

# ── Allow running from repo root without installing package ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader         import load_data
from src.feature_engineering import engineer_features
from src.eda                 import run_eda
from src.models.traditional_models import train_traditional_models
from src.models.lstm_model   import train_lstm
from src.evaluate            import plot_results, print_summary_table
from src.forecast            import run_forecast


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = "configs/config.yaml"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step_load(cfg):
    df = load_data(cfg)
    df, feature_cols = engineer_features(df, cfg)
    return df, feature_cols


def step_eda(df, cfg):
    print("\n[STEP] Exploratory Data Analysis …")
    run_eda(df, cfg)


def step_train(df, feature_cols, cfg):
    target_col = cfg["data"]["target_col"]

    print("\n[STEP] Training Traditional ML Models …")
    trad_results, y_train, y_test = train_traditional_models(
        df, feature_cols, target_col, cfg
    )

    print("\n[STEP] Training LSTM Model …")
    lstm_result = train_lstm(df, feature_cols, target_col, cfg)

    return trad_results, lstm_result, y_test


def step_evaluate(df, trad_results, lstm_result, y_test, cfg):
    print("\n[STEP] Generating Evaluation Plots …")
    plot_results(trad_results, lstm_result, df, y_test, cfg)
    print_summary_table(trad_results, lstm_result)


def step_forecast(df, feature_cols, trad_results, cfg):
    print("\n[STEP] Generating 10-Year Forecast …")
    return run_forecast(df, feature_cols, trad_results, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Climate-Resilient Crop Yield Prediction System"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--step",
        choices=["all", "eda", "train", "forecast"],
        default="all",
        help="Which pipeline step to run (default: all)"
    )
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  CLIMATE-RESILIENT CROP YIELD PREDICTION SYSTEM")
    print("  AAI-530 Group 4  |  IoT + Machine Learning")
    print("█" * 60)

    cfg = load_config(args.config)

    # Always load data first
    df, feature_cols = step_load(cfg)

    if args.step in ("all", "eda"):
        step_eda(df, cfg)

    if args.step in ("all", "train"):
        trad_results, lstm_result, y_test = step_train(df, feature_cols, cfg)
        step_evaluate(df, trad_results, lstm_result, y_test, cfg)

    if args.step in ("all", "forecast"):
        if args.step == "forecast":
            # Need models — train quickly if not cached
            trad_results, lstm_result, y_test = step_train(df, feature_cols, cfg)
        step_forecast(df, feature_cols, trad_results, cfg)

    print("\n✅  Pipeline complete!")
    print(f"    Plots  → {cfg['output']['plots_dir']}/")
    print(f"    Models → {cfg['output']['models_dir']}/")


if __name__ == "__main__":
    main()
