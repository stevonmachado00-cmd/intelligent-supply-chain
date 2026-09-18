"""
MLOps Data and Prediction Drift Detection using Evidently AI.
Monitors feature distribution shifts and prediction drift across time horizons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from loguru import logger

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("mlflow/drift_reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class DataDriftDetector:
    """Detects statistical data drift and distribution shifts against reference baseline."""

    def __init__(self, model_name: Literal["demand_xgb", "stockout", "eta"] = "demand_xgb") -> None:
        self.model_name = model_name
        self.reference_data = self._load_reference_data()

    def _load_reference_data(self) -> pd.DataFrame:
        """Load baseline reference training dataset."""
        if self.model_name == "demand_xgb":
            file_path = DATA_DIR / "demand_features.parquet"
        elif self.model_name == "stockout":
            file_path = DATA_DIR / "stockout_features.parquet"
        elif self.model_name == "eta":
            file_path = DATA_DIR / "eta_features.parquet"
        else:
            raise ValueError(f"Unsupported model for drift detection: {self.model_name}")

        if not file_path.exists():
            raise FileNotFoundError(f"Reference data not found at {file_path}")

        df = pd.read_parquet(file_path)
        # Use first 70% as reference baseline (matches training split)
        ref_len = int(len(df) * 0.70)
        return df.iloc[:ref_len].copy()

    def generate_current_sample(self, drift_ratio: float = 0.0, n_samples: int = 500) -> pd.DataFrame:
        """
        Generate a production inference sample.
        If drift_ratio > 0, introduces synthetic distribution shock to simulate real-world drift.
        """
        # Take sample from validation/test portion or sample from reference
        sample = self.reference_data.sample(min(n_samples, len(self.reference_data)), random_state=42).copy()

        if drift_ratio > 0:
            numeric_cols = sample.select_dtypes(include=[np.number]).columns
            drift_cols = numeric_cols[: max(1, int(len(numeric_cols) * drift_ratio))]
            for col in drift_cols:
                # Perturb distribution mean and variance to simulate shift
                shift_factor = 1.0 + np.random.uniform(0.3, 0.8)
                sample[col] = sample[col] * shift_factor + np.random.normal(0, np.std(sample[col]) * 0.5, size=len(sample))

        return sample

    def run_drift_analysis(
        self,
        current_data: pd.DataFrame | None = None,
        drift_ratio: float = 0.0,
        save_html: bool = True,
    ) -> dict[str, Any]:
        """
        Execute Evidently drift report comparing reference vs current data.
        Returns a structured dictionary of drift metrics.
        """
        if current_data is None:
            current_data = self.generate_current_sample(drift_ratio=drift_ratio)

        # Only evaluate non-identifier, continuous/ordinal features
        exclude = ["date", "StockCode", "product_id", "supplier_id", "shipment_id", "target"]
        numeric_cols = [
            c for c in self.reference_data.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        # Keep features that have non-zero variance
        valid_cols = [c for c in numeric_cols if self.reference_data[c].std() > 1e-4]
        ref_subset = self.reference_data[valid_cols].dropna()
        cur_subset = current_data[valid_cols].dropna()

        # Run Evidently DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_subset, current_data=cur_subset)

        report_dict = report.as_dict()

        # Extract drift summary
        metrics = report_dict.get("metrics", [])
        drift_summary = {}
        for m in metrics:
            metric_type = m.get("metric")
            result = m.get("result", {})
            if metric_type == "DatasetDriftMetric":
                drift_summary["dataset_drift"] = result.get("dataset_drift", False)
                # Evidently returns drift_share threshold (e.g. 0.5) and share_of_drifted_columns (actual drifted ratio)
                actual_drift_share = result.get("share_of_drifted_columns", result.get("drift_share", 0.0))
                drift_summary["drift_share"] = round(float(actual_drift_share), 4)
                drift_summary["drift_threshold"] = round(float(result.get("drift_share", 0.5)), 4)
                drift_summary["number_of_drifted_columns"] = result.get("number_of_drifted_columns", 0)
                drift_summary["number_of_columns"] = result.get("number_of_columns", 0)
            elif metric_type == "DataDriftTable":
                drift_by_col = {}
                for col_name, col_stats in result.get("drift_by_columns", {}).items():
                    drift_by_col[col_name] = {
                        "drift_detected": col_stats.get("drift_detected", False),
                        "drift_score": round(float(col_stats.get("drift_score", 0.0)), 4),
                        "stat_test": col_stats.get("stat_test_name", "unknown"),
                    }
                drift_summary["drift_by_columns"] = drift_by_col

        html_path = None
        if save_html:
            report_file = REPORTS_DIR / f"{self.model_name}_drift_report.html"
            report.save_html(str(report_file))
            html_path = str(report_file)
            logger.info(f"Evidently HTML report saved to {report_file}")

        # Save latest JSON metrics
        json_file = REPORTS_DIR / f"{self.model_name}_drift_summary.json"
        with open(json_file, "w") as f:
            json.dump(drift_summary, f, indent=2)

        return {
            "model_name": self.model_name,
            "html_report_path": html_path,
            "dataset_drift_detected": drift_summary.get("dataset_drift", False),
            "drift_share": drift_summary.get("drift_share", 0.0),
            "drifted_features_count": drift_summary.get("number_of_drifted_columns", 0),
            "total_features_count": drift_summary.get("number_of_columns", 0),
            "drift_by_columns": drift_summary.get("drift_by_columns", {}),
        }
