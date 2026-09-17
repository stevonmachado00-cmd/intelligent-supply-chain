"""
Integration and functional test suite for the FastAPI Serving Layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "sc-tower-secret-key-2026"}


# =================================================================
# Health & Root Tests
# =================================================================

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "documentation" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_readiness_probe():
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True


# =================================================================
# Authentication Middleware Test
# =================================================================

def test_missing_api_key_unauthorized():
    response = client.post("/risk/score", json={
        "stockout_prob": 0.5,
        "supplier_risk": 0.3,
        "demand_surge": 0.2,
        "shipment_delay_risk": 0.1,
        "anomaly_score": 0.05,
    })
    assert response.status_code == 401
    assert "Missing X-API-Key" in response.json()["detail"]


# =================================================================
# Risk Engine Tests
# =================================================================

def test_risk_score_calculation():
    payload = {
        "stockout_prob": 0.85,
        "supplier_risk": 0.60,
        "demand_surge": 0.40,
        "shipment_delay_risk": 0.30,
        "anomaly_score": 0.10,
        "metadata": {"product_id": "PROD_TEST"},
    }
    response = client.post("/risk/score", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "health_score" in data
    assert data["risk_score"] + data["health_score"] == pytest.approx(100.0, 0.1)
    assert data["status"] in ["GREEN", "AMBER", "RED"]
    assert "primary_driver" in data


def test_batch_risk_scores():
    payload = [
        {"stockout_prob": 0.10, "supplier_risk": 0.10, "demand_surge": 0.10, "shipment_delay_risk": 0.05, "anomaly_score": 0.05},
        {"stockout_prob": 0.90, "supplier_risk": 0.80, "demand_surge": 0.70, "shipment_delay_risk": 0.50, "anomaly_score": 0.30},
    ]
    response = client.post("/risk/batch", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["status"] == "GREEN"
    assert data[1]["status"] == "RED"


# =================================================================
# Decision Engine Tests
# =================================================================

def test_inventory_policy_calculation():
    payload = {
        "product_id": "PROD_0001",
        "current_stock": 350,
        "daily_demand_mean": 45.0,
        "daily_demand_std": 8.0,
        "lead_time_days": 5.0,
        "lead_time_std": 1.0,
        "unit_cost": 15.0,
    }
    response = client.post("/decisions/inventory-policy", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["economic_order_quantity"] > 0
    assert data["safety_stock"] > 0
    assert data["reorder_point"] > data["safety_stock"]
    assert data["days_of_supply"] > 0


def test_prescriptive_recommendations_reorder_trigger():
    payload = {
        "product_id": "PROD_CRITICAL",
        "current_stock": 30,
        "daily_demand_mean": 50.0,
        "daily_demand_std": 10.0,
        "lead_time_days": 6.0,
        "stockout_prob": 0.88,
        "supplier_risk": 0.20,
        "demand_surge": 0.40,
        "shipment_delay_risk": 0.20,
        "anomaly_score": 0.05,
        "primary_supplier_id": "SUP_001",
    }
    response = client.post("/decisions/recommendations", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert len(data["actions"]) >= 1
    action = data["actions"][0]
    assert action["urgency"] in ["CRITICAL", "HIGH"]
    assert action["recommended_order_quantity"] > 0
    assert action["net_benefit"] > 0


# =================================================================
# Model Prediction Tests
# =================================================================

def test_demand_prediction_endpoint():
    payload = {
        "product_id": "PROD_0001",
        "features": {
            "demand_roll_mean_7d": 120.0,
            "units_sold": 110.0,
            "demand_lag_7d": 105.0,
            "demand_lag_1d": 115.0,
            "avg_price": 2.50,
        },
        "model_type": "xgboost",
        "return_shap": True,
    }
    response = client.post("/predict/demand", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_demand_7d"] >= 0.0
    assert data["confidence_interval_upper"] >= data["predicted_demand_7d"]


def test_stockout_prediction_endpoint():
    payload = {
        "product_id": "PROD_0001",
        "current_stock": 100,
        "reorder_point": 250,
        "incoming_po_units": 0,
        "lead_time_days": 7.0,
        "daily_demand": 30.0,
        "avg_demand_7d": 32.0,
        "avg_demand_30d": 28.0,
        "std_demand_7d": 6.0,
        "std_demand_30d": 7.0,
        "days_of_supply": 3.1,
        "demand_volatility": 0.22,
    }
    response = client.post("/predict/stockout", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["calibrated_stockout_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MODERATE", "CRITICAL"]


def test_supplier_risk_endpoint():
    payload = {
        "supplier_id": "SUP_TEST",
        "on_time_delivery_rate": 0.95,
        "avg_delay_days": 0.4,
        "std_delay_days": 0.2,
        "defect_rate": 0.01,
        "cancellation_rate": 0.005,
        "capacity_units_per_month": 6000,
        "price_per_unit": 10.0,
    }
    response = client.post("/predict/supplier-risk", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["calibrated_risk_score"] <= 1.0
    assert data["classification"] in ["RELIABLE", "MONITOR", "HIGH_RISK"]


def test_eta_prediction_endpoint():
    payload = {
        "distance_km": 320.0,
        "weight_kg": 1500.0,
        "vehicle_type_encoded": 1,
        "day_of_week": 3,
        "hour_of_day": 10,
        "month": 8,
        "is_weekend": 0,
        "origin_encoded": 2,
        "dest_encoded": 4,
        "carrier_on_time_rate": 0.94,
        "hist_avg_dur": 7.2,
        "hist_std_dur": 1.1,
    }
    response = client.post("/predict/eta", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_duration_hours"] > 0.0


def test_anomaly_detection_endpoint():
    payload = {
        "features": [300.0, 1200.0, 1, 2, 10, 1, 3, 2.0, 7.5, 6.0, 1.0, 0.15, 0.92]
    }
    response = client.post("/predict/anomaly", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "reconstruction_loss" in data
    assert "is_anomaly" in data
    assert data["anomaly_severity"] in ["NORMAL", "ELEVATED", "CRITICAL"]


# =================================================================
# Simulation Tests
# =================================================================

def test_simulation_presets_endpoint():
    response = client.get("/simulation/presets", headers=AUTH_HEADERS)
    assert response.status_code == 200
    presets = response.json()
    assert len(presets) >= 5
    assert any(p["key"] == "PORT_STRIKE" for p in presets)


def test_simulation_run_endpoint():
    payload = {
        "product_id": "PROD_SIM_01",
        "preset_key": "PORT_STRIKE",
        "current_stock": 400,
        "daily_demand_base": 50.0,
        "daily_demand_std": 10.0,
        "lead_time_days": 6.0,
    }
    response = client.post("/simulation/run", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "simulated_risk_score" in data
    assert "first_stockout_day" in data
    assert len(data["daily_trajectory_sample"]) > 0
    assert "mitigation_actions" in data
