# 🌾 Climate-Resilient Crop Yield Prediction System
### Using IoT Sensor Data & Machine Learning | AAI-530 Group 4

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project builds an end-to-end **climate-resilient crop yield prediction system** by combining:
- Real-world IoT-derived climate data (World Bank / NOAA weather stations)
- Historical crop production data (FAOSTAT)
- Deep Learning (LSTM) and Traditional ML (Random Forest, XGBoost)
- Two-scenario forecasting: Business-as-Usual vs. Climate Stress

The system is designed as a **decision-support tool** for agricultural planners, policy makers, and agri-insurance analysts.

---

## 🗂️ Project Structure

```
crop-yield-prediction/
│
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── setup.py                    ← Package setup
├── .gitignore
│
├── configs/
│   └── config.yaml             ← Model hyperparameters & paths
│
├── data/
│   ├── raw/                    ← Original downloaded datasets (not committed)
│   │   ├── faostat_crop_yield.csv
│   │   └── india_climate_data.csv
│   └── processed/              ← Cleaned & feature-engineered data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← Load & clean FAOSTAT + World Bank data
│   ├── feature_engineering.py  ← Lag features, rolling stats, climate flags
│   ├── eda.py                  ← Exploratory data analysis & plots
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py       ← LSTM deep learning model (TensorFlow)
│   │   └── traditional_models.py ← Random Forest + XGBoost
│   ├── evaluate.py             ← Metrics, comparison plots
│   └── forecast.py             ← Future yield projection (2 scenarios)
│
├── notebooks/
│   └── 01_full_pipeline.ipynb  ← End-to-end Jupyter walkthrough
│
├── outputs/
│   ├── plots/                  ← EDA, model results, forecast charts
│   └── models/                 ← Saved model weights (.h5, .pkl)
│
├── tests/
│   └── test_pipeline.py        ← Basic unit tests
│
└── main.py                     ← 🚀 Entry point — run the full pipeline
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/tusharsdu/crop-yield-prediction.git
cd crop-yield-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📥 Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| FAOSTAT Crop Yield | [fao.org/faostat](https://www.fao.org/faostat/en/#data/QCL) | Annual crop yield (tons/ha) by country |
| India Climate Data | [World Bank Data Catalog](https://datacatalog.worldbank.org) | Monthly temperature & rainfall from NOAA/CRU stations |

Download the CSV files and place them in `data/raw/`. The pipeline will auto-detect and use them. If no real data is found, it runs with synthetic data for demonstration.

---

## 🚀 Running the Pipeline

### Full pipeline (all steps)
```bash
python main.py
```

### Individual steps
```bash
python main.py --step eda           # Only run EDA
python main.py --step train         # Train ML models only
python main.py --step forecast      # Generate forecasts only
```

### With custom config
```bash
python main.py --config configs/config.yaml
```

---

## 🤖 Models

### 1. LSTM (Deep Learning)
- 3-layer stacked LSTM with Dropout + BatchNormalization
- Trained on 5-year sliding windows of climate + yield sequences
- Early stopping + learning rate scheduling
- Framework: TensorFlow / Keras

### 2. Random Forest Regressor
- 300 estimators, max_depth=8
- Feature importance ranking
- Framework: scikit-learn

### 3. XGBoost / Gradient Boosting
- 300 estimators, learning_rate=0.05
- L1/L2 regularization
- Framework: XGBoost (falls back to scikit-learn GBM)

---

## 📊 Outputs

After running `main.py`, outputs are saved to `outputs/plots/`:

| File | Description |
|------|-------------|
| `eda_plots.png` | 9-panel EDA: trends, distributions, correlations |
| `model_results.png` | Predictions vs actuals, residuals, feature importance |
| `yield_forecast.png` | 10-year forecast: 2 climate scenarios |

Model weights saved to `outputs/models/`:
- `lstm_model.h5`
- `random_forest.pkl`
- `xgboost_model.pkl`

---

## 📈 IoT System Architecture

```
[Weather Stations / Satellite Sensors]
         │  (temperature, rainfall, humidity)
         ▼
  [Edge Processing Layer]
  Regional data hubs — anomaly detection, aggregation
         │  (MQTT / publish-subscribe)
         ▼
  [Cloud Storage]
  Time-series database (AWS S3 / GCP BigQuery)
         │
         ▼
  [ML Prediction Layer]     ◄── This repository
  LSTM + Random Forest + XGBoost
         │
         ▼
  [Tableau Public Dashboard]
  Interactive visualizations for planners & policy makers
```

---

## 🧪 Running Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## 👥 Team
**AAI-530 Group 4**

- Tushar Gorad
- Uhana Jyothi
- Bharath TS



---
