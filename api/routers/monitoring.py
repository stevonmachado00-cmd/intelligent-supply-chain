"""
MLOps Drift Monitoring and Model Retraining endpoints (Phase 7).
"""

from __future__ import annotations

from typing import Any, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.middleware.auth import verify_api_key
from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.retraining import ModelRetrainingService

router = APIRouter(prefix="/monitoring", tags=["MLOps Monitoring & Drift"])

retraining_service = ModelRetrainingService()


class DriftCheckRequest(BaseModel):
    model_name: Literal["demand_xgb", "stockout", "eta"] = Field(default="demand_xgb")
    drift_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional synthetic shock perturbation ratio to stress-test drift detection (0.0 = normal baseline)",
        json_schema_extra={"example": 0.35},
    )
    save_html_report: bool = Field(default=True)


class DriftCheckResponse(BaseModel):
    timestamp: str
    model_name: str
    drift_share: float
    drift_threshold: float
    dataset_drift_detected: bool
    drifted_features_count: int
    total_features_count: int
    retraining_recommended: bool
    urgency: Literal["NORMAL", "ELEVATED", "CRITICAL"]
    html_report_path: str | None


class RetrainTriggerRequest(BaseModel):
    model_name: Literal["demand_xgb", "stockout", "eta", "all"] = Field(default="demand_xgb")


class RetrainTriggerResponse(BaseModel):
    status: str
    model_name: str
    retrained_at: str
    duration_seconds: float
    artifacts_updated: bool


@router.post("/drift/check", response_model=DriftCheckResponse)
async def check_data_drift(
    req: DriftCheckRequest,
    _auth: str = Depends(verify_api_key),
) -> DriftCheckResponse:
    """Run Evidently AI statistical drift detection against reference training data."""
    try:
        res = retraining_service.evaluate_retraining_need(
            model_name=req.model_name,
            drift_ratio=req.drift_ratio,
        )
        return DriftCheckResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift analysis failed: {str(e)}",
        )


@router.get("/drift/history")
async def get_drift_history(
    _auth: str = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return historical drift check audits."""
    return retraining_service.history


@router.post("/retrain/trigger", response_model=RetrainTriggerResponse)
async def trigger_model_retraining(
    req: RetrainTriggerRequest,
    _auth: str = Depends(verify_api_key),
) -> RetrainTriggerResponse:
    """Trigger automated model retraining and artifact refresh."""
    try:
        res = retraining_service.trigger_retraining_pipeline(model_name=req.model_name)
        return RetrainTriggerResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}",
        )
