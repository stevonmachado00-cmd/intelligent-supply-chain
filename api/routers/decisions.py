"""
Prescriptive Decision Engine endpoints (Phase 3).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_decision_engine, get_risk_engine
from api.middleware.auth import verify_api_key
from api.schemas import (
    InventoryPolicyRequest,
    InventoryPolicyResponse,
    PrescriptiveActionItem,
    PrescriptiveActionsRequest,
    PrescriptiveActionsResponse,
    RiskScoreResponse,
)
from src.risk_engine.decision_engine import DecisionEngine
from src.risk_engine.scorer import RiskEngine

router = APIRouter(prefix="/decisions", tags=["Prescriptive Decisions"])


@router.post("/inventory-policy", response_model=InventoryPolicyResponse)
async def compute_inventory_policy(
    req: InventoryPolicyRequest,
    decision_engine: DecisionEngine = Depends(get_decision_engine),
    _auth: str = Depends(verify_api_key),
) -> InventoryPolicyResponse:
    """Compute EOQ, Dynamic Safety Stock (95% service level), and Reorder Point (ROP)."""
    policy = decision_engine.compute_inventory_policy(
        product_id=req.product_id,
        current_stock=req.current_stock,
        daily_demand_mean=req.daily_demand_mean,
        daily_demand_std=req.daily_demand_std,
        lead_time_days=req.lead_time_days,
        lead_time_std=req.lead_time_std,
        unit_cost=req.unit_cost,
    )

    return InventoryPolicyResponse(
        product_id=policy.product_id,
        current_stock=policy.current_stock,
        daily_demand=policy.daily_demand,
        lead_time_days=policy.lead_time_days,
        safety_stock=policy.safety_stock,
        reorder_point=policy.reorder_point,
        economic_order_quantity=policy.economic_order_quantity,
        stock_status=policy.stock_status,
        days_of_supply=policy.days_of_supply,
    )


@router.post("/recommendations", response_model=PrescriptiveActionsResponse)
async def generate_prescriptive_actions(
    req: PrescriptiveActionsRequest,
    decision_engine: DecisionEngine = Depends(get_decision_engine),
    risk_engine: RiskEngine = Depends(get_risk_engine),
    _auth: str = Depends(verify_api_key),
) -> PrescriptiveActionsResponse:
    """Evaluate full multi-echelon risk state and generate concrete, prioritized mitigation actions."""
    # 1. Compute multi-model risk
    risk = risk_engine.calculate_risk(
        stockout_prob=req.stockout_prob,
        supplier_risk=req.supplier_risk,
        demand_surge=req.demand_surge,
        shipment_delay_risk=req.shipment_delay_risk,
        anomaly_score=req.anomaly_score,
        metadata={"product_id": req.product_id},
    )

    # 2. Compute inventory policy
    policy = decision_engine.compute_inventory_policy(
        product_id=req.product_id,
        current_stock=req.current_stock,
        daily_demand_mean=req.daily_demand_mean,
        daily_demand_std=req.daily_demand_std,
        lead_time_days=req.lead_time_days,
        lead_time_std=req.lead_time_std,
        unit_cost=req.unit_cost,
    )

    # 3. Generate prescriptive recommendations
    raw_actions = decision_engine.generate_recommendations(
        product_id=req.product_id,
        risk=risk,
        inventory_policy=policy,
        unit_cost=req.unit_cost,
        primary_supplier_id=req.primary_supplier_id,
        alternative_suppliers=req.alternative_suppliers,
    )

    action_items = [
        PrescriptiveActionItem(
            action_type=a.action_type,
            urgency=a.urgency,
            product_id=a.product_id,
            headline=a.headline,
            recommendation=a.recommendation,
            recommended_order_quantity=a.recommended_order_quantity,
            primary_supplier_id=a.primary_supplier_id,
            alternative_supplier_id=a.alternative_supplier_id,
            estimated_cost=a.estimated_cost,
            estimated_stockout_loss_avoided=a.estimated_stockout_loss_avoided,
            net_benefit=a.net_benefit,
        )
        for a in raw_actions
    ]

    return PrescriptiveActionsResponse(
        product_id=req.product_id,
        risk_summary=RiskScoreResponse(
            risk_score=risk.risk_score,
            health_score=risk.health_score,
            status=risk.status,
            primary_driver=risk.primary_driver,
            component_contributions=risk.component_contributions,
            metadata=risk.metadata,
        ),
        inventory_policy=InventoryPolicyResponse(
            product_id=policy.product_id,
            current_stock=policy.current_stock,
            daily_demand=policy.daily_demand,
            lead_time_days=policy.lead_time_days,
            safety_stock=policy.safety_stock,
            reorder_point=policy.reorder_point,
            economic_order_quantity=policy.economic_order_quantity,
            stock_status=policy.stock_status,
            days_of_supply=policy.days_of_supply,
        ),
        actions=action_items,
    )
