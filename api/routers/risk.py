"""
Risk Engine scoring and health endpoints (Phase 3).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_risk_engine
from api.middleware.auth import verify_api_key
from api.schemas import RiskScoreRequest, RiskScoreResponse
from src.risk_engine.scorer import RiskEngine

router = APIRouter(prefix="/risk", tags=["Risk Engine"])


@router.post("/score", response_model=RiskScoreResponse)
async def calculate_risk_score(
    req: RiskScoreRequest,
    risk_engine: RiskEngine = Depends(get_risk_engine),
    _auth: str = Depends(verify_api_key),
) -> RiskScoreResponse:
    """Compute unified 0-100 Supply Chain Risk Score and Health Score."""
    assessment = risk_engine.calculate_risk(
        stockout_prob=req.stockout_prob,
        supplier_risk=req.supplier_risk,
        demand_surge=req.demand_surge,
        shipment_delay_risk=req.shipment_delay_risk,
        anomaly_score=req.anomaly_score,
        metadata=req.metadata,
    )

    return RiskScoreResponse(
        risk_score=assessment.risk_score,
        health_score=assessment.health_score,
        status=assessment.status,
        primary_driver=assessment.primary_driver,
        component_contributions=assessment.component_contributions,
        metadata=assessment.metadata,
    )


@router.post("/batch", response_model=list[RiskScoreResponse])
async def calculate_batch_risks(
    requests: list[RiskScoreRequest],
    risk_engine: RiskEngine = Depends(get_risk_engine),
    _auth: str = Depends(verify_api_key),
) -> list[RiskScoreResponse]:
    """Batch calculation of risk scores across multiple SKUs or transit corridors."""
    results = []
    for req in requests:
        assessment = risk_engine.calculate_risk(
            stockout_prob=req.stockout_prob,
            supplier_risk=req.supplier_risk,
            demand_surge=req.demand_surge,
            shipment_delay_risk=req.shipment_delay_risk,
            anomaly_score=req.anomaly_score,
            metadata=req.metadata,
        )
        results.append(
            RiskScoreResponse(
                risk_score=assessment.risk_score,
                health_score=assessment.health_score,
                status=assessment.status,
                primary_driver=assessment.primary_driver,
                component_contributions=assessment.component_contributions,
                metadata=assessment.metadata,
            )
        )
    return results
