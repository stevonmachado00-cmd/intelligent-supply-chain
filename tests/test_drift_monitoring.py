"""
Unit and integration tests for MLOps drift monitoring and retraining triggers (Phase 7).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.retraining import ModelRetrainingService

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "sc-tower-secret-key-2026"}


def test_drift_detector_initialization():
    """Verify detector loads reference data properly."""
    detector = DataDriftDetector(model_name="demand_xgb")
    assert detector.reference_data is not None
    assert len(detector.reference_data) > 0
    assert "units_sold" in detector.reference_data.columns


def test_drift_analysis_baseline_no_drift():
    """Reference sample without perturbation should produce low or zero drift."""
    detector = DataDriftDetector(model_name="demand_xgb")
    sample = detector.generate_current_sample(drift_ratio=0.0, n_samples=800)
    res = detector.run_drift_analysis(current_data=sample, save_html=False)

    assert res["model_name"] == "demand_xgb"
    assert "drift_share" in res
    assert res["dataset_drift_detected"] is False


def test_drift_analysis_perturbed_shock():
    """Perturbed distribution should trigger drift detection."""
    detector = DataDriftDetector(model_name="demand_xgb")
    shocked = detector.generate_current_sample(drift_ratio=0.8, n_samples=300)
    res = detector.run_drift_analysis(current_data=shocked, save_html=True)

    assert res["html_report_path"] is not None
    assert res["drifted_features_count"] > 0


def test_retraining_service_evaluation():
    """Test automated decision logic for retraining trigger."""
    service = ModelRetrainingService(drift_threshold=0.25)
    result = service.evaluate_retraining_need(model_name="demand_xgb", drift_ratio=0.0)

    assert "retraining_recommended" in result
    assert "drift_share" in result
    assert result["urgency"] in ["NORMAL", "ELEVATED", "CRITICAL"]


def test_monitoring_drift_endpoint():
    """Test /monitoring/drift/check API endpoint."""
    payload = {
        "model_name": "demand_xgb",
        "drift_ratio": 0.0,
        "save_html_report": True,
    }
    response = client.post("/monitoring/drift/check", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "demand_xgb"
    assert "drift_share" in data
    assert "retraining_recommended" in data


def test_monitoring_drift_history_endpoint():
    """Test /monitoring/drift/history audit endpoint."""
    response = client.get("/monitoring/drift/history", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
