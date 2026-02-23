"""
src/models/lstm_model.py
─────────────────────────
Stacked LSTM deep learning model for time-series crop yield prediction.
Requires TensorFlow >= 2.11.  Gracefully skipped if TensorFlow is absent.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    _TF = True
except ImportError:
    _TF = False


def train_lstm(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: dict,
) -> dict | None:
    """
    Prepare sequences, build, and train the LSTM model.

    Returns
    -------
    dict with keys: model, preds, actuals, history, rmse, mae, r2
    or None if TensorFlow is not available.
    """
    if not _TF:
        print("⚠  TensorFlow not installed — LSTM model skipped.")
        return None

    lstm_cfg = cfg["lstm"]
    seq_len  = lstm_cfg["sequence_length"]

    # ── Prepare sequences ─────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te, scaler_X, scaler_y = _make_sequences(
        df, feature_cols, target_col, seq_len, cfg["data"]["train_split"]
    )
    print(f"  LSTM sequences — train: {X_tr.shape}, test: {X_te.shape}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = _build(seq_len, X_tr.shape[2], lstm_cfg)
    model.summary()

    # ── Train ─────────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(patience=lstm_cfg["early_stopping_patience"],
                      restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=lstm_cfg["reduce_lr_factor"],
                          patience=lstm_cfg["reduce_lr_patience"],
                          min_lr=lstm_cfg["min_lr"], monitor="val_loss"),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    preds_sc  = model.predict(X_te).flatten()
    preds     = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    actuals   = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()

    result = {
        "model":   model,
        "preds":   preds,
        "actuals": actuals,
        "history": history,
        "rmse":    float(np.sqrt(mean_squared_error(actuals, preds))),
        "mae":     float(mean_absolute_error(actuals, preds)),
        "r2":      float(r2_score(actuals, preds)),
    }
    print(f"  LSTM  RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R²={result['r2']:.4f}")

    # ── Save weights ──────────────────────────────────────────────────────────
    out_dir = cfg["output"]["models_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "lstm_model.h5"))
    print(f"  ✓ LSTM weights saved → {out_dir}/lstm_model.h5")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_sequences(df, feature_cols, target_col, seq_len, train_split):
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X_raw)
    y_sc = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    Xs, ys = [], []
    for i in range(seq_len, len(X_sc)):
        Xs.append(X_sc[i - seq_len:i])
        ys.append(y_sc[i])

    Xs = np.array(Xs)
    ys = np.array(ys)

    split = int(len(Xs) * train_split)
    return Xs[:split], Xs[split:], ys[:split], ys[split:], scaler_X, scaler_y


def _build(seq_len: int, n_features: int, cfg: dict) -> "Sequential":
    units   = cfg["units"]
    dropout = cfg["dropout"]
    model   = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(dropout[0]),
        BatchNormalization(),

        LSTM(units[1], return_sequences=True),
        Dropout(dropout[1]),

        LSTM(units[2], return_sequences=False),
        Dropout(dropout[2]),

        Dense(cfg["dense_units"][0], activation="relu"),
        Dense(cfg["dense_units"][1], activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss=cfg["loss"],
        metrics=["mae"],
    )
    return model
