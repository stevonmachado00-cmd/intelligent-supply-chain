"""
Shared dependencies, model artifact registry, and caching for FastAPI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import torch
import torch.nn as nn
from loguru import logger

from src.risk_engine.decision_engine import DecisionEngine
from src.risk_engine.scorer import RiskEngine
from src.simulation.simulator import SupplyChainSimulator

ARTIFACTS_DIR = Path("mlflow/model_artifacts")


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=(64, 32, 16), latent_dim=8, dropout=0.1):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

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


class ModelRegistry:
    """Singleton container loading and serving trained model artifacts."""

    def __init__(self) -> None:
        self.loaded_models: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.shap_top_features: dict[str, list[dict[str, Any]]] = {}
        self._load_all()

    def _load_all(self) -> None:
        logger.info("Initializing Model Registry from mlflow/model_artifacts/...")

        # 1. Demand XGBoost
        demand_xgb_path = ARTIFACTS_DIR / "demand_xgb" / "model.joblib"
        if demand_xgb_path.exists():
            try:
                self.loaded_models["demand_xgb"] = joblib.load(demand_xgb_path)
                logger.info("  ✓ Loaded demand_xgb")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load demand_xgb: {e}")

        # 2. Stockout Classifier
        stockout_path = ARTIFACTS_DIR / "stockout" / "model.joblib"
        if stockout_path.exists():
            try:
                self.loaded_models["stockout"] = joblib.load(stockout_path)
                logger.info("  ✓ Loaded stockout model")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load stockout: {e}")

        # 3. Supplier Risk Classifier
        supplier_risk_path = ARTIFACTS_DIR / "supplier_risk" / "model.joblib"
        if supplier_risk_path.exists():
            try:
                self.loaded_models["supplier_risk"] = joblib.load(supplier_risk_path)
                logger.info("  ✓ Loaded supplier_risk model")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load supplier_risk: {e}")

        # 4. ETA Regressor
        eta_path = ARTIFACTS_DIR / "eta" / "model.joblib"
        if eta_path.exists():
            try:
                self.loaded_models["eta"] = joblib.load(eta_path)
                logger.info("  ✓ Loaded eta model")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load eta: {e}")

        # 5. Anomaly Autoencoder
        anomaly_path = ARTIFACTS_DIR / "anomaly" / "model.pt"
        if anomaly_path.exists():
            try:
                checkpoint = torch.load(anomaly_path, map_location="cpu", weights_only=False)
                cfg = checkpoint.get("config", {})
                model = AutoencoderModel(
                    input_dim=cfg.get("input_dim", 13),
                    hidden_dims=cfg.get("hidden_dims", [64, 32, 16]),
                    latent_dim=cfg.get("latent_dim", 8),
                    dropout=cfg.get("dropout", 0.1),
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self.loaded_models["anomaly"] = {
                    "model": model,
                    "threshold": checkpoint.get("threshold", 0.695),
                    "scaler_mean": checkpoint.get("scaler_mean"),
                    "scaler_scale": checkpoint.get("scaler_scale"),
                }
                logger.info("  ✓ Loaded anomaly autoencoder")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load anomaly autoencoder: {e}")

        # Load SHAP summaries & metrics for loaded models
        for m in ["demand_xgb", "stockout", "supplier_risk", "eta"]:
            shap_file = ARTIFACTS_DIR / m / "shap_top_features.json"
            if shap_file.exists():
                with open(shap_file) as f:
                    self.shap_top_features[m] = json.load(f)

            met_file = ARTIFACTS_DIR / m / "metrics.json"
            if met_file.exists():
                with open(met_file) as f:
                    self.metrics[m] = json.load(f)


# Singleton instances
_registry = ModelRegistry()
_risk_engine = RiskEngine()
_decision_engine = DecisionEngine()
_simulator = SupplyChainSimulator(risk_engine=_risk_engine, decision_engine=_decision_engine)


def get_model_registry() -> ModelRegistry:
    return _registry


def get_risk_engine() -> RiskEngine:
    return _risk_engine


def get_decision_engine() -> DecisionEngine:
    return _decision_engine


def get_simulator() -> SupplyChainSimulator:
    return _simulator
