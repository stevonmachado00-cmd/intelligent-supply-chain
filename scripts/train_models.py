"""
Phase 2 — Train All ML Models for the Supply Chain Control Tower.

Trains 6 models end-to-end:
  1. XGBoost Demand Forecasting
  2. LSTM Demand Forecasting
  3. Calibrated XGBoost Stockout Prediction
  4. Calibrated XGBoost Supplier Risk Prediction
  5. XGBoost Shipment ETA Prediction
  6. PyTorch Autoencoder Anomaly Detection

All models include SHAP explainability (tree models) and are saved
to mlflow/model_artifacts/.

Run from project root:
    python scripts/train_models.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import xgboost as xgb
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("mlflow/model_artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# =================================================================
#  UTILITIES
# =================================================================

def temporal_split(df: pd.DataFrame, date_col: str = "date",
                   train_frac: float = 0.70, val_frac: float = 0.15):
    """Split by date preserving time order."""
    if date_col in df.columns:
        df = df.sort_values(date_col)
    n = len(df)
    t1, t2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (better than MAPE for zeros)."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total)


def save_metrics(metrics: dict, model_name: str):
    """Save metrics JSON alongside model artifact."""
    out = ARTIFACTS_DIR / model_name / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved to {out}")


def save_shap_summary(model, X_sample: pd.DataFrame, model_name: str):
    """Compute and save SHAP top feature importances."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        mean_abs = np.abs(shap_values).mean(axis=0)
        features = list(X_sample.columns)
        top_idx = np.argsort(mean_abs)[::-1][:10]
        top = [{"feature": features[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
               for i in top_idx]
        out = ARTIFACTS_DIR / model_name / "shap_top_features.json"
        with open(out, "w") as f:
            json.dump(top, f, indent=2)
        logger.info(f"  SHAP top features saved to {out}")
        return top
    except Exception as e:
        logger.warning(f"  SHAP failed: {e}")
        return []


# =================================================================
#  MODEL 1: XGBoost Demand Forecasting
# =================================================================

def train_demand_xgboost():
    logger.info("=" * 65)
    logger.info("MODEL 1: XGBoost Demand Forecasting")
    logger.info("=" * 65)

    df = pd.read_parquet(PROCESSED_DIR / "demand_features.parquet")
    drop_cols = ["date", "StockCode", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    date_col = "date" if "date" in df.columns else None

    train_df, val_df, test_df = temporal_split(df, date_col or "date")
    X_train, y_train = train_df[feature_cols].fillna(0), train_df["target"]
    X_val, y_val = val_df[feature_cols].fillna(0), val_df["target"]
    X_test, y_test = test_df[feature_cols].fillna(0), test_df["target"]

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # Evaluate
    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "wape": round(wape(y_test.values, y_pred), 4),
        "best_iteration": int(model.best_iteration),
    }
    logger.success(f"  Test MAE={metrics['mae']}, RMSE={metrics['rmse']}, WAPE={metrics['wape']}")

    # Save
    model_dir = ARTIFACTS_DIR / "demand_xgb"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    save_metrics(metrics, "demand_xgb")

    # SHAP
    shap_top = save_shap_summary(model, X_test.head(200), "demand_xgb")
    logger.info("  Top SHAP features:")
    for item in shap_top[:5]:
        logger.info(f"    {item['feature']}: {item['mean_abs_shap']}")

    logger.success("  XGBoost demand model saved.")
    return model, feature_cols


# =================================================================
#  MODEL 2: LSTM Demand Forecasting
# =================================================================

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_demand_lstm():
    logger.info("=" * 65)
    logger.info("MODEL 2: LSTM Demand Forecasting")
    logger.info("=" * 65)

    df = pd.read_parquet(PROCESSED_DIR / "demand_features.parquet")
    drop_cols = ["date", "StockCode", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_df, val_df, test_df = temporal_split(df, "date")
    X_train_raw = train_df[feature_cols].fillna(0).values
    y_train_raw = train_df["target"].values
    X_val_raw = val_df[feature_cols].fillna(0).values
    y_val_raw = val_df["target"].values
    X_test_raw = test_df[feature_cols].fillna(0).values
    y_test_raw = test_df["target"].values

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_val_s = scaler.transform(X_val_raw)
    X_test_s = scaler.transform(X_test_raw)

    seq_len = 14  # Shorter for small dataset
    X_tr, y_tr = build_sequences(X_train_s, y_train_raw, seq_len)
    X_vl, y_vl = build_sequences(X_val_s, y_val_raw, seq_len)
    X_te, y_te = build_sequences(X_test_s, y_test_raw, seq_len)

    logger.info(f"  Sequences: Train={X_tr.shape}, Val={X_vl.shape}, Test={X_te.shape}")

    if len(X_tr) == 0 or len(X_vl) == 0:
        logger.warning("  Not enough data for LSTM sequences. Skipping.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_vl), torch.tensor(y_vl))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = LSTMNet(input_size=X_tr.shape[2], hidden_size=64, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    epochs = 80

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.2f} | val_loss={val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    test_tensor = torch.tensor(X_te).to(device)
    with torch.no_grad():
        y_pred = model(test_tensor).cpu().numpy()
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "mae": round(float(mean_absolute_error(y_te, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 2),
        "wape": round(wape(y_te, y_pred), 4),
        "best_val_loss": round(best_val_loss, 4),
    }
    logger.success(f"  Test MAE={metrics['mae']}, RMSE={metrics['rmse']}, WAPE={metrics['wape']}")

    # Save
    model_dir = ARTIFACTS_DIR / "demand_lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "config": {"input_size": X_tr.shape[2], "hidden_size": 64, "num_layers": 2,
                   "dropout": 0.2, "seq_len": seq_len},
        "feature_cols": feature_cols,
    }, model_dir / "model.pt")
    save_metrics(metrics, "demand_lstm")

    logger.success("  LSTM demand model saved.")
    return model, feature_cols


# =================================================================
#  MODEL 3: Calibrated XGBoost Stockout Prediction
# =================================================================

def train_stockout_model():
    logger.info("=" * 65)
    logger.info("MODEL 3: Calibrated XGBoost Stockout Prediction")
    logger.info("=" * 65)

    df = pd.read_parquet(PROCESSED_DIR / "stockout_features.parquet")
    feature_cols = [c for c in df.columns if c != "target"]
    n = len(df)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    X_train, y_train = df[feature_cols].iloc[:t1], df["target"].iloc[:t1]
    X_val, y_val = df[feature_cols].iloc[t1:t2], df["target"].iloc[t1:t2]
    X_test, y_test = df[feature_cols].iloc[t2:], df["target"].iloc[t2:]

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"  Positive class rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # Calculate class weight
    pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    logger.info(f"  Computed scale_pos_weight: {pos_weight:.1f}")

    base_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(pos_weight, 10),
        random_state=SEED, n_jobs=-1, eval_metric="aucpr",
    )
    base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calibrate using cv='prefit'
    calibrated = CalibratedClassifierCV(base_model, cv="prefit", method="isotonic")
    calibrated.fit(X_val, y_val)

    # Evaluate
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4) if len(set(y_test)) > 1 else 0.0,
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4) if len(set(y_test)) > 1 else 0.0,
    }
    logger.success(f"  Test F1={metrics['f1']}, ROC-AUC={metrics['roc_auc']}, PR-AUC={metrics['pr_auc']}")
    logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Save
    model_dir = ARTIFACTS_DIR / "stockout"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"base_model": base_model, "calibrated_model": calibrated}, model_dir / "model.joblib")
    save_metrics(metrics, "stockout")
    save_shap_summary(base_model, X_test.head(200), "stockout")

    logger.success("  Stockout model saved.")
    return calibrated


# =================================================================
#  MODEL 4: Calibrated XGBoost Supplier Risk Prediction
# =================================================================

def train_supplier_risk_model():
    logger.info("=" * 65)
    logger.info("MODEL 4: Calibrated XGBoost Supplier Risk Prediction")
    logger.info("=" * 65)

    df = pd.read_parquet(PROCESSED_DIR / "supplier_risk_features.parquet")
    feature_cols = [c for c in df.columns if c != "target"]

    logger.info(f"  Dataset: {df.shape[0]} rows, high-risk rate: {df['target'].mean():.2%}")

    # Use stratified split so both classes (0 and 1) are guaranteed in train, val, and test
    from sklearn.model_selection import StratifiedShuffleSplit
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(df[feature_cols], df["target"]))
    temp_df = df.iloc[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_sub_idx, test_sub_idx = next(sss2.split(temp_df[feature_cols], temp_df["target"]))

    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    X_train, y_train = df[feature_cols].iloc[train_idx], df["target"].iloc[train_idx]
    X_val, y_val = df[feature_cols].iloc[val_idx], df["target"].iloc[val_idx]
    X_test, y_test = df[feature_cols].iloc[test_idx], df["target"].iloc[test_idx]

    pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))

    base_model = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(pos_weight, 5),
        random_state=SEED, n_jobs=-1, eval_metric="aucpr",
    )
    base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    calibrated = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
    calibrated.fit(X_val, y_val)

    y_prob = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_test, y_prob)), 4)
    except Exception:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    logger.success(f"  Test F1={metrics['f1']}, ROC-AUC={metrics.get('roc_auc', 'N/A')}")

    model_dir = ARTIFACTS_DIR / "supplier_risk"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"base_model": base_model, "calibrated_model": calibrated}, model_dir / "model.joblib")
    save_metrics(metrics, "supplier_risk")
    save_shap_summary(base_model, X_train.head(min(len(X_train), 50)), "supplier_risk")

    logger.success("  Supplier risk model saved.")
    return calibrated


# =================================================================
#  MODEL 5: XGBoost Shipment ETA Prediction
# =================================================================

def train_eta_model():
    logger.info("=" * 65)
    logger.info("MODEL 5: XGBoost Shipment ETA Prediction")
    logger.info("=" * 65)

    df = pd.read_parquet(PROCESSED_DIR / "eta_features.parquet")
    feature_cols = [c for c in df.columns if c != "target"]
    n = len(df)
    t1, t2 = int(n * 0.70), int(n * 0.85)

    X_train, y_train = df[feature_cols].iloc[:t1], df["target"].iloc[:t1]
    X_val, y_val = df[feature_cols].iloc[t1:t2], df["target"].iloc[t1:t2]
    X_test, y_test = df[feature_cols].iloc[t2:], df["target"].iloc[t2:]

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
        early_stopping_rounds=40,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = {
        "mae_hours": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse_hours": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
    }
    logger.success(f"  Test MAE={metrics['mae_hours']} hrs, RMSE={metrics['rmse_hours']} hrs")

    model_dir = ARTIFACTS_DIR / "eta"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    save_metrics(metrics, "eta")
    save_shap_summary(model, X_test.head(200), "eta")

    logger.success("  ETA model saved.")
    return model


# =================================================================
#  MODEL 6: PyTorch Autoencoder Anomaly Detection
# =================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), latent_dim=8, dropout=0.1):
        super().__init__()
        # Encoder
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_anomaly_model():
    logger.info("=" * 65)
    logger.info("MODEL 6: PyTorch Autoencoder Anomaly Detection")
    logger.info("=" * 65)

    df = pd.read_parquet(PROCESSED_DIR / "anomaly_features.parquet")
    logger.info(f"  Dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    n = len(X)
    t1 = int(n * 0.80)  # More train data for unsupervised
    X_train, X_test = X[:t1], X[t1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=128, shuffle=True)

    input_dim = X.shape[1]
    model = Autoencoder(input_dim, hidden_dims=[64, 32, 16], latent_dim=8, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    epochs = 100

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for Xb, _ in train_loader:
            Xb = Xb.to(device)
            recon = model(Xb)
            loss = criterion(recon, Xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Compute anomaly scores
    model.eval()
    with torch.no_grad():
        train_recon = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        test_recon = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    train_errors = np.mean((X_train - train_recon) ** 2, axis=1)
    test_errors = np.mean((X_test - test_recon) ** 2, axis=1)

    # Threshold at 95th percentile of training errors
    threshold = np.percentile(train_errors, 95)
    test_anomalies = (test_errors > threshold).astype(int)

    # Normalize errors to 0-1 anomaly scores
    max_error = max(train_errors.max(), test_errors.max())
    anomaly_scores = np.clip(test_errors / max_error, 0, 1)

    metrics = {
        "threshold": round(float(threshold), 6),
        "train_mean_error": round(float(train_errors.mean()), 6),
        "test_mean_error": round(float(test_errors.mean()), 6),
        "test_anomaly_rate": round(float(test_anomalies.mean()), 4),
        "test_anomaly_count": int(test_anomalies.sum()),
        "test_total": int(len(test_anomalies)),
        "mean_anomaly_score": round(float(anomaly_scores.mean()), 4),
    }
    logger.success(f"  Threshold: {metrics['threshold']:.6f}")
    logger.success(f"  Test anomaly rate: {metrics['test_anomaly_rate']:.2%} ({metrics['test_anomaly_count']}/{metrics['test_total']})")

    # Save
    model_dir = ARTIFACTS_DIR / "anomaly"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "threshold": threshold,
        "config": {"input_dim": input_dim, "hidden_dims": [64, 32, 16], "latent_dim": 8, "dropout": 0.1},
        "feature_cols": list(df.columns),
    }, model_dir / "model.pt")
    save_metrics(metrics, "anomaly")

    logger.success("  Anomaly detection model saved.")
    return model


# =================================================================
#  MAIN: Train All Models
# =================================================================

def main():
    logger.info("=" * 65)
    logger.info("INTELLIGENT SUPPLY CHAIN CONTROL TOWER")
    logger.info("Phase 2: Training All ML Models")
    logger.info("=" * 65)

    results = {}

    # 1. XGBoost Demand
    try:
        train_demand_xgboost()
        results["demand_xgb"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Demand XGBoost failed: {e}")
        results["demand_xgb"] = f"FAILED: {e}"

    # 2. LSTM Demand
    try:
        train_demand_lstm()
        results["demand_lstm"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Demand LSTM failed: {e}")
        results["demand_lstm"] = f"FAILED: {e}"

    # 3. Stockout
    try:
        train_stockout_model()
        results["stockout"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Stockout model failed: {e}")
        results["stockout"] = f"FAILED: {e}"

    # 4. Supplier Risk
    try:
        train_supplier_risk_model()
        results["supplier_risk"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Supplier Risk failed: {e}")
        results["supplier_risk"] = f"FAILED: {e}"

    # 5. ETA
    try:
        train_eta_model()
        results["eta"] = "SUCCESS"
    except Exception as e:
        logger.error(f"ETA model failed: {e}")
        results["eta"] = f"FAILED: {e}"

    # 6. Anomaly Detection
    try:
        train_anomaly_model()
        results["anomaly"] = "SUCCESS"
    except Exception as e:
        logger.error(f"Anomaly model failed: {e}")
        results["anomaly"] = f"FAILED: {e}"

    # Summary
    logger.info("=" * 65)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("=" * 65)
    for model_name, status in results.items():
        icon = "✓" if status == "SUCCESS" else "✗"
        logger.info(f"  [{icon}] {model_name}: {status}")

    # Save summary
    with open(ARTIFACTS_DIR / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    succeeded = sum(1 for v in results.values() if v == "SUCCESS")
    logger.success(f"\n  {succeeded}/{len(results)} models trained successfully.")
    logger.success(f"  Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
