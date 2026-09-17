"""
What-If Scenario Simulation endpoints (Phase 4).
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_simulator
from api.middleware.auth import verify_api_key
from api.schemas import SimulationRunRequest, SimulationRunResponse
from src.simulation.presets import get_preset_scenario, list_available_presets
from src.simulation.scenarios import DisruptionScenario
from src.simulation.simulator import SupplyChainSimulator

router = APIRouter(prefix="/simulation", tags=["What-If Simulation"])


@router.get("/presets")
async def get_simulation_presets(
    _auth: str = Depends(verify_api_key),
) -> list[dict[str, str]]:
    """List all available pre-configured disruption scenarios."""
    return list_available_presets()


@router.post("/run", response_model=SimulationRunResponse)
async def run_scenario_simulation(
    req: SimulationRunRequest,
    simulator: SupplyChainSimulator = Depends(get_simulator),
    _auth: str = Depends(verify_api_key),
) -> SimulationRunResponse:
    """Execute a forward What-If disruption simulation and generate financial exposure and mitigation plan."""
    # 1. Resolve Scenario: Preset or Custom
    if req.preset_key:
        try:
            scenario = get_preset_scenario(req.preset_key)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Preset key '{req.preset_key}' is invalid.",
            )
    else:
        scenario = DisruptionScenario(
            name=req.custom_scenario_name or "Custom Simulation",
            description="User-defined stress test shock.",
            demand_shock_pct=req.demand_shock_pct,
            supplier_delay_days=req.supplier_delay_days,
            supplier_capacity_loss_pct=req.supplier_capacity_loss_pct,
            warehouse_capacity_loss_pct=req.warehouse_capacity_loss_pct,
            transport_cost_multiplier=req.transport_cost_multiplier,
            projection_horizon_days=req.projection_horizon_days,
        )

    # 2. Run forward simulation
    res = simulator.run_simulation(
        scenario=scenario,
        product_id=req.product_id,
        current_stock=req.current_stock,
        daily_demand_base=req.daily_demand_base,
        daily_demand_std=req.daily_demand_std,
        lead_time_days=req.lead_time_days,
        unit_cost=req.unit_cost,
        unit_price=req.unit_price,
        scheduled_inbound=req.scheduled_inbound,
        primary_supplier_id=req.primary_supplier_id,
        alternative_suppliers=req.alternative_suppliers,
    )

    # Return sample of daily trajectory (first 7 days) to keep response light
    daily_sample = [s.to_dict() if hasattr(s, "to_dict") else vars(s) for s in res.daily_trajectory[:7]]

    return SimulationRunResponse(
        scenario_name=res.scenario_name,
        product_id=res.product_id,
        projection_horizon_days=res.projection_horizon_days,
        baseline_risk_score=res.baseline_risk_score,
        baseline_health_score=res.baseline_health_score,
        baseline_days_of_supply=res.baseline_days_of_supply,
        baseline_stockout_prob=res.baseline_stockout_prob,
        simulated_risk_score=res.simulated_risk_score,
        simulated_health_score=res.simulated_health_score,
        simulated_days_of_supply=res.simulated_days_of_supply,
        simulated_stockout_prob=res.simulated_stockout_prob,
        risk_delta=res.risk_delta,
        first_stockout_day=res.first_stockout_day,
        total_stockout_days=res.total_stockout_days,
        total_unfulfilled_units=res.total_unfulfilled_units,
        estimated_revenue_loss=res.estimated_revenue_loss,
        daily_trajectory_sample=daily_sample,
        mitigation_actions=res.mitigation_actions,
        mitigation_cost=res.mitigation_cost,
        mitigation_net_savings=res.mitigation_net_savings,
    )
