"""
Machine Learning and Deep Learning prediction endpoints (Phase 2 models).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from api.dependencies import ModelRegistry, get_model_registry
from api.middleware.auth import verify_api_key
from api.schemas import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    DemandPredictionRequest,
    DemandPredictionResponse,
    ETAPredictionRequest,
    ETAPredictionResponse,
    StockoutPredictionRequest,
    StockoutPredictionResponse,
    SupplierRiskRequest,
    SupplierRiskResponse,
)

router = APIRouter(prefix="/predict", tags=["Model Predictions"])


# =================================================================
# 1. Demand Prediction
# =================================================================

@router.post("/demand", response_model=DemandPredictionResponse)
async def predict_demand(
    req: DemandPredictionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _auth: str = Depends(verify_api_key),
) -> DemandPredictionResponse:
    """Forecast 7-day forward demand using trained XGBoost model."""
    model = registry.loaded_models.get("demand_xgb")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demand forecasting model is not loaded.",
        )

    try:
        # Align features to expected model input columns
        features_dict = req.features
        expected_cols = getattr(model, "feature_names_in_", None)
        if expected_cols is not None:
            input_df = pd.DataFrame([{col: features_dict.get(col, 0.0) for col in expected_cols}])
        else:
            input_df = pd.DataFrame([features_dict])

        pred = float(model.predict(input_df)[0])
        pred_clamped = max(0.0, pred)

        # 90% confidence intervals (approximate standard deviation from metrics)
        std_est = registry.metrics.get("demand_xgb", {}).get("rmse", 250.0) * 0.5
        ci_lower = max(0.0, round(pred_clamped - 1.645 * std_est, 1))
        ci_upper = round(pred_clamped + 1.645 * std_est, 1)

        shap_top = registry.shap_top_features.get("demand_xgb", []) if req.return_shap else []

        return DemandPredictionResponse(
            product_id=req.product_id,
            model_used="XGBoost Regressor (Trained)",
            predicted_demand_7d=round(pred_clamped, 1),
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            top_shap_drivers=shap_top[:5],
        )
    except Exception as e:
        logger.error(f"Demand prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


# =================================================================
# 2. Stockout Prediction
# =================================================================

@router.post("/stockout", response_model=StockoutPredictionResponse)
async def predict_stockout(
    req: StockoutPredictionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _auth: str = Depends(verify_api_key),
) -> StockoutPredictionResponse:
    """Predict stockout probability within the replenishment lead-time horizon."""
    container = registry.loaded_models.get("stockout")
    if not container:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stockout prediction model is not loaded.",
        )

    try:
        calibrated_model = container.get("calibrated_model") or container.get("base_model")

        feat_dict = {
            "current_stock": req.current_stock,
            "reorder_point": req.reorder_point,
            "incoming_po_units": req.incoming_po_units,
            "lead_time_days": req.lead_time_days,
            "daily_demand": req.daily_demand,
            "avg_demand_7d": req.avg_demand_7d,
            "avg_demand_30d": req.avg_demand_30d,
            "std_demand_7d": req.std_demand_7d,
            "std_demand_30d": req.std_demand_30d,
            "days_of_supply": req.days_of_supply,
            "demand_volatility": req.demand_volatility,
        }
        input_df = pd.DataFrame([feat_dict])

        # Get calibrated probabilities
        probs = calibrated_model.predict_proba(input_df)[0]
        pos_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        is_stockout = pos_prob >= 0.50

        if pos_prob < 0.30:
            level = "LOW"
        elif pos_prob < 0.70:
            level = "MODERATE"
        else:
            level = "CRITICAL"

        shap_top = registry.shap_top_features.get("stockout", [])

        return StockoutPredictionResponse(
            product_id=req.product_id,
            calibrated_stockout_probability=round(pos_prob, 4),
            predicted_stockout=is_stockout,
            risk_level=level,
            top_shap_drivers=shap_top[:5],
        )
    except Exception as e:
        logger.error(f"Stockout prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


# =================================================================
# 3. Supplier Risk Prediction
# =================================================================

@router.post("/supplier-risk", response_model=SupplierRiskResponse)
async def predict_supplier_risk(
    req: SupplierRiskRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _auth: str = Depends(verify_api_key),
) -> SupplierRiskResponse:
    """Evaluate supplier default and delivery failure probability."""
    container = registry.loaded_models.get("supplier_risk")
    if not container:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supplier risk model is not loaded.",
        )

    try:
        calibrated_model = container.get("calibrated_model") or container.get("base_model")

        feat_dict = {
            "on_time_delivery_rate": req.on_time_delivery_rate,
            "avg_delay_days": req.avg_delay_days,
            "std_delay_days": req.std_delay_days,
            "defect_rate": req.defect_rate,
            "cancellation_rate": req.cancellation_rate,
            "capacity_units_per_month": req.capacity_units_per_month,
            "price_per_unit": req.price_per_unit,
            "region_encoded": req.region_encoded,
            "category_encoded": req.category_encoded,
            "avg_delay_hours": req.avg_delay_hours,
            "std_delay_hours": req.std_delay_hours,
            "ship_on_time_rate": req.ship_on_time_rate,
            "shipment_count": req.shipment_count,
        }
        input_df = pd.DataFrame([feat_dict])

        probs = calibrated_model.predict_proba(input_df)[0]
        pos_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        is_high = pos_prob >= 0.50

        if pos_prob < 0.25:
            classification = "RELIABLE"
        elif pos_prob < 0.60:
            classification = "MONITOR"
        else:
            classification = "HIGH_RISK"

        shap_top = registry.shap_top_features.get("supplier_risk", [])

        return SupplierRiskResponse(
            supplier_id=req.supplier_id,
            calibrated_risk_score=round(pos_prob, 4),
            is_high_risk=is_high,
            classification=classification,
            top_shap_drivers=shap_top[:5],
        )
    except Exception as e:
        logger.error(f"Supplier risk prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


# =================================================================
# 4. Shipment ETA Prediction
# =================================================================

@router.post("/eta", response_model=ETAPredictionResponse)
async def predict_eta(
    req: ETAPredictionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _auth: str = Depends(verify_api_key),
) -> ETAPredictionResponse:
    """Predict shipment transit duration in hours."""
    model = registry.loaded_models.get("eta")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shipment ETA model is not loaded.",
        )

    try:
        input_df = pd.DataFrame([req.model_dump()])
        pred_duration = float(model.predict(input_df)[0])
        pred_clamped = max(0.5, pred_duration)

        rmse = registry.metrics.get("eta", {}).get("rmse_hours", 1.54)
        is_delayed = pred_clamped > (req.hist_avg_dur + rmse)
        shap_top = registry.shap_top_features.get("eta", [])

        return ETAPredictionResponse(
            predicted_duration_hours=round(pred_clamped, 2),
            estimated_arrival_variance_hours=round(rmse, 2),
            is_delayed=is_delayed,
            top_shap_drivers=shap_top[:5],
        )
    except Exception as e:
        logger.error(f"ETA prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


# =================================================================
# 5. Anomaly Detection (PyTorch Autoencoder)
# =================================================================

@router.post("/anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomaly(
    req: AnomalyDetectionRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    _auth: str = Depends(verify_api_key),
) -> AnomalyDetectionResponse:
    """Detect transit and operational anomalies via PyTorch Autoencoder reconstruction loss."""
    container = registry.loaded_models.get("anomaly")
    if not container:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Anomaly autoencoder model is not loaded.",
        )

    try:
        model = container["model"]
        threshold = container["threshold"]
        scaler_mean = container.get("scaler_mean")
        scaler_scale = container.get("scaler_scale")

        x_raw = np.array(req.features, dtype=np.float32)

        # Standardize if scaler available
        if scaler_mean is not None and scaler_scale is not None:
            mean_arr = np.array(scaler_mean, dtype=np.float32)
            scale_arr = np.array(scaler_scale, dtype=np.float32)
            x_norm = (x_raw - mean_arr) / (scale_arr + 1e-6)
        else:
            x_norm = x_raw

        tensor_x = torch.tensor(x_norm).unsqueeze(0)
        with torch.no_grad():
            recon = model(tensor_x).squeeze(0).numpy()

        mse = float(np.mean((x_norm - recon) ** 2))
        is_anom = mse > threshold

        if mse > threshold * 1.5:
            sev = "CRITICAL"
        elif is_anom:
            sev = "ELEVATED"
        else:
            sev = "NORMAL"

        return AnomalyDetectionResponse(
            reconstruction_loss=round(mse, 6),
            threshold=round(threshold, 6),
            is_anomaly=is_anom,
            anomaly_severity=sev,
        )
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
