"""
Health and readiness check endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import ModelRegistry, get_model_registry
from api.schemas import HealthResponse

router = APIRouter(prefix="", tags=["Health & System"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    registry: ModelRegistry = Depends(get_model_registry),
) -> HealthResponse:
    """Check API operational health and model registry readiness."""
    models_status = {
        name: name in registry.loaded_models
        for name in ["demand_xgb", "stockout", "supplier_risk", "eta", "anomaly"]
    }
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="intelligent-supply-chain-api",
        models_loaded=models_status,
        environment="production",
    )


@router.get("/health/ready")
async def readiness_probe(
    registry: ModelRegistry = Depends(get_model_registry),
) -> dict[str, Any]:
    """Readiness probe verifying models are loaded."""
    ready = len(registry.loaded_models) > 0
    return {"ready": ready, "loaded_count": len(registry.loaded_models)}
