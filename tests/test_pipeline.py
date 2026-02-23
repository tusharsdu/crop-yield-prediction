"""
tests/test_pipeline.py
───────────────────────
Basic unit tests for the crop yield prediction pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Load default config ──────────────────────────────────────────────────────
@pytest.fixture
def cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


# ── Data Loader ──────────────────────────────────────────────────────────────
class TestDataLoader:
    def test_synthetic_data_shape(self, cfg):
        from src.data_loader import load_data
        df = load_data(cfg)
        assert df.shape[0] > 0,      "DataFrame should not be empty"
        assert "year"        in df.columns
        assert "temperature" in df.columns
        assert "rainfall"    in df.columns
        assert "yield_rice"  in df.columns

    def test_no_missing_values(self, cfg):
        from src.data_loader import load_data
        df = load_data(cfg)
        assert df.isnull().sum().sum() == 0, "No missing values after cleaning"

    def test_yield_non_negative(self, cfg):
        from src.data_loader import load_data
        df = load_data(cfg)
        assert (df["yield_rice"] >= 0).all(),  "Rice yield must be non-negative"
        assert (df["yield_wheat"] >= 0).all(), "Wheat yield must be non-negative"


# ── Feature Engineering ──────────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_feature_count(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        df = load_data(cfg)
        df_feat, feat_cols = engineer_features(df, cfg)
        assert len(feat_cols) > 10, "Should have at least 10 feature columns"

    def test_lag_columns_exist(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        df = load_data(cfg)
        df_feat, _ = engineer_features(df, cfg)
        for lag in cfg["features"]["lag_years"]:
            assert f"temp_lag{lag}" in df_feat.columns
            assert f"rain_lag{lag}" in df_feat.columns

    def test_drought_flag_binary(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        df = load_data(cfg)
        df_feat, _ = engineer_features(df, cfg)
        assert set(df_feat["drought_flag"].unique()).issubset({0, 1})

    def test_no_nan_after_engineering(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        df = load_data(cfg)
        df_feat, feat_cols = engineer_features(df, cfg)
        assert df_feat[feat_cols].isnull().sum().sum() == 0


# ── Traditional Models ────────────────────────────────────────────────────────
class TestTraditionalModels:
    def test_models_train_and_predict(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from src.models.traditional_models import train_traditional_models

        df = load_data(cfg)
        df_feat, feat_cols = engineer_features(df, cfg)
        results, y_train, y_test = train_traditional_models(
            df_feat, feat_cols, cfg["data"]["target_col"], cfg
        )
        assert len(results) >= 2, "Should train at least 2 models"
        for name, res in results.items():
            assert "preds" in res
            assert "rmse"  in res
            assert res["rmse"] > 0

    def test_predictions_reasonable_range(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from src.models.traditional_models import train_traditional_models

        df = load_data(cfg)
        df_feat, feat_cols = engineer_features(df, cfg)
        results, _, y_test = train_traditional_models(
            df_feat, feat_cols, cfg["data"]["target_col"], cfg
        )
        for name, res in results.items():
            preds = res["preds"]
            assert preds.min() > -1,  f"{name}: predictions too low"
            assert preds.max() < 20,  f"{name}: predictions unreasonably high"


# ── Forecast ─────────────────────────────────────────────────────────────────
class TestForecast:
    def test_forecast_shape(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from src.models.traditional_models import train_traditional_models
        from src.forecast import run_forecast

        df = load_data(cfg)
        df_feat, feat_cols = engineer_features(df, cfg)
        results, _, _ = train_traditional_models(
            df_feat, feat_cols, cfg["data"]["target_col"], cfg
        )
        fcast = run_forecast(df_feat, feat_cols, results, cfg)

        n_future    = cfg["forecast"]["n_future_years"]
        n_scenarios = len(cfg["forecast"]["scenarios"])
        assert len(fcast) == n_future * n_scenarios

    def test_forecast_non_negative(self, cfg):
        from src.data_loader import load_data
        from src.feature_engineering import engineer_features
        from src.models.traditional_models import train_traditional_models
        from src.forecast import run_forecast

        df = load_data(cfg)
        df_feat, feat_cols = engineer_features(df, cfg)
        results, _, _ = train_traditional_models(
            df_feat, feat_cols, cfg["data"]["target_col"], cfg
        )
        fcast = run_forecast(df_feat, feat_cols, results, cfg)
        assert (fcast["yield_pred"] >= 0).all(), "Forecasted yields must be non-negative"
