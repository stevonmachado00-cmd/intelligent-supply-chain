"""
Automated Model Retraining Trigger Pipeline.
Monitors drift signals and orchestrates retraining runs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from src.monitoring.drift_detector import DataDriftDetector

# Drift threshold triggering automated retrain recommendation
DRIFT_RETRAIN_THRESHOLD = 0.30  # If >30% features show statistically significant drift


class ModelRetrainingService:
    """Manages drift-driven retraining workflows."""

    def __init__(self, drift_threshold: float = DRIFT_RETRAIN_THRESHOLD) -> None:
        self.drift_threshold = drift_threshold
        self.history: list[dict[str, Any]] = []

    def evaluate_retraining_need(
        self,
        model_name: Literal["demand_xgb", "stockout", "eta"] = "demand_xgb",
        drift_ratio: float = 0.0,
    ) -> dict[str, Any]:
        """
        Runs drift detection and checks if retraining criteria are met.
        """
        detector = DataDriftDetector(model_name=model_name)
        analysis = detector.run_drift_analysis(drift_ratio=drift_ratio, save_html=True)

        drift_share = analysis["drift_share"]
        needs_retraining = drift_share >= self.drift_threshold

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "drift_share": drift_share,
            "drift_threshold": self.drift_threshold,
            "dataset_drift_detected": analysis["dataset_drift_detected"],
            "drifted_features_count": analysis["drifted_features_count"],
            "total_features_count": analysis["total_features_count"],
            "retraining_recommended": needs_retraining,
            "urgency": "CRITICAL" if drift_share >= 0.50 else "ELEVATED" if needs_retraining else "NORMAL",
            "html_report_path": analysis["html_report_path"],
        }
        self.history.append(result)
        logger.info(
            f"Model {model_name} drift check: {drift_share:.1%} drifted features. Retraining recommended: {needs_retraining}"
        )
        return result

    def trigger_retraining_pipeline(self, model_name: str) -> dict[str, Any]:
        """
        Executes automated model retraining and artifact refresh.
        """
        logger.info(f"Triggering pipeline retraining for {model_name}...")
        start_time = datetime.utcnow()

        if model_name in ["demand_xgb", "all"]:
            from scripts.train_models import train_demand_xgboost
            train_demand_xgboost()

        if model_name in ["stockout", "all"]:
            from scripts.train_models import train_stockout_model
            train_stockout_model()

        if model_name in ["eta", "all"]:
            from scripts.train_models import train_eta_model
            train_eta_model()

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.success(f"Retraining for {model_name} completed in {duration:.1f}s")

        return {
            "status": "COMPLETED",
            "model_name": model_name,
            "retrained_at": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration, 2),
            "artifacts_updated": True,
        }
