"""
Pydantic schemas for the Intelligent Supply Chain Control Tower API.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


# =================================================================
# Health & Status
# =================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "intelligent-supply-chain-api"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    environment: str = "production"


# =================================================================
# Model Predictions (Phase 2)
# =================================================================

class DemandPredictionRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "85123A"})
    features: dict[str, float] = Field(
        ...,
        description="Feature dictionary matching trained demand features (lags, rolling stats, price, calendar)",
        json_schema_extra={
            "example": {
                "units_sold": 120.0,
                "avg_price": 2.55,
                "day_of_week": 2,
                "month": 11,
                "demand_lag_1d": 115.0,
                "demand_lag_7d": 140.0,
                "demand_roll_mean_7d": 125.0,
                "demand_roll_std_7d": 18.5,
                "demand_cv_28d": 0.15,
            }
        },
    )
    model_type: Literal["xgboost", "lstm"] = "xgboost"
    return_shap: bool = True


class DemandPredictionResponse(BaseModel):
    product_id: str
    model_used: str
    predicted_demand_7d: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    top_shap_drivers: list[dict[str, Any]] = Field(default_factory=list)


class StockoutPredictionRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "85123A"})
    current_stock: int = Field(..., json_schema_extra={"example": 150})
    reorder_point: int = Field(..., json_schema_extra={"example": 300})
    incoming_po_units: int = Field(default=0, json_schema_extra={"example": 50})
    lead_time_days: float = Field(default=7.0, json_schema_extra={"example": 7.0})
    daily_demand: float = Field(default=25.0, json_schema_extra={"example": 25.0})
    avg_demand_7d: float = Field(default=28.0, json_schema_extra={"example": 28.0})
    avg_demand_30d: float = Field(default=24.0, json_schema_extra={"example": 24.0})
    std_demand_7d: float = Field(default=5.0, json_schema_extra={"example": 5.0})
    std_demand_30d: float = Field(default=6.0, json_schema_extra={"example": 6.0})
    days_of_supply: float = Field(default=5.3, json_schema_extra={"example": 5.3})
    demand_volatility: float = Field(default=0.25, json_schema_extra={"example": 0.25})


class StockoutPredictionResponse(BaseModel):
    product_id: str
    calibrated_stockout_probability: float
    predicted_stockout: bool
    risk_level: Literal["LOW", "MODERATE", "CRITICAL"]
    top_shap_drivers: list[dict[str, Any]] = Field(default_factory=list)


class SupplierRiskRequest(BaseModel):
    supplier_id: str = Field(..., json_schema_extra={"example": "SUP_001"})
    on_time_delivery_rate: float = Field(..., json_schema_extra={"example": 0.88})
    avg_delay_days: float = Field(..., json_schema_extra={"example": 1.5})
    std_delay_days: float = Field(default=0.8, json_schema_extra={"example": 0.8})
    defect_rate: float = Field(default=0.015, json_schema_extra={"example": 0.015})
    cancellation_rate: float = Field(default=0.01, json_schema_extra={"example": 0.01})
    capacity_units_per_month: int = Field(default=5000, json_schema_extra={"example": 5000})
    price_per_unit: float = Field(default=12.0, json_schema_extra={"example": 12.0})
    region_encoded: int = Field(default=0)
    category_encoded: int = Field(default=0)
    avg_delay_hours: float = Field(default=36.0)
    std_delay_hours: float = Field(default=12.0)
    ship_on_time_rate: float = Field(default=0.85)
    shipment_count: int = Field(default=250)


class SupplierRiskResponse(BaseModel):
    supplier_id: str
    calibrated_risk_score: float
    is_high_risk: bool
    classification: Literal["RELIABLE", "MONITOR", "HIGH_RISK"]
    top_shap_drivers: list[dict[str, Any]] = Field(default_factory=list)


class ETAPredictionRequest(BaseModel):
    distance_km: float = Field(..., json_schema_extra={"example": 280.0})
    weight_kg: float = Field(..., json_schema_extra={"example": 1200.0})
    vehicle_type_encoded: int = Field(default=0, json_schema_extra={"example": 0})
    day_of_week: int = Field(default=2, json_schema_extra={"example": 2})
    hour_of_day: int = Field(default=9, json_schema_extra={"example": 9})
    month: int = Field(default=10, json_schema_extra={"example": 10})
    is_weekend: int = Field(default=0, json_schema_extra={"example": 0})
    origin_encoded: int = Field(default=1, json_schema_extra={"example": 1})
    dest_encoded: int = Field(default=3, json_schema_extra={"example": 3})
    carrier_on_time_rate: float = Field(default=0.92, json_schema_extra={"example": 0.92})
    hist_avg_dur: float = Field(default=6.5, json_schema_extra={"example": 6.5})
    hist_std_dur: float = Field(default=1.2, json_schema_extra={"example": 1.2})


class ETAPredictionResponse(BaseModel):
    predicted_duration_hours: float
    estimated_arrival_variance_hours: float
    is_delayed: bool
    top_shap_drivers: list[dict[str, Any]] = Field(default_factory=list)


class AnomalyDetectionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="13 scaled feature values for transit evaluation",
        json_schema_extra={"example": [250.0, 800.0, 0, 2, 10, 1, 3, 5.5, 9.5, 5.0, 1.2, 0.41, 0.90]},
    )


class AnomalyDetectionResponse(BaseModel):
    reconstruction_loss: float
    threshold: float
    is_anomaly: bool
    anomaly_severity: Literal["NORMAL", "ELEVATED", "CRITICAL"]


# =================================================================
# Risk Engine (Phase 3)
# =================================================================

class RiskScoreRequest(BaseModel):
    stockout_prob: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.75})
    supplier_risk: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.40})
    demand_surge: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.30})
    shipment_delay_risk: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.20})
    anomaly_score: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.15})
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskScoreResponse(BaseModel):
    risk_score: float
    health_score: float
    status: Literal["GREEN", "AMBER", "RED"]
    primary_driver: str
    component_contributions: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


# =================================================================
# Decision Engine (Phase 3)
# =================================================================

class InventoryPolicyRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "PROD_0001"})
    current_stock: int = Field(..., json_schema_extra={"example": 350})
    daily_demand_mean: float = Field(..., json_schema_extra={"example": 45.0})
    daily_demand_std: float = Field(..., json_schema_extra={"example": 8.0})
    lead_time_days: float = Field(..., json_schema_extra={"example": 5.0})
    lead_time_std: float = Field(default=1.0, json_schema_extra={"example": 1.0})
    unit_cost: float = Field(default=15.0, json_schema_extra={"example": 15.0})


class InventoryPolicyResponse(BaseModel):
    product_id: str
    current_stock: int
    daily_demand: float
    lead_time_days: float
    safety_stock: int
    reorder_point: int
    economic_order_quantity: int
    stock_status: Literal["HEALTHY", "REORDER_SOON", "CRITICAL_STOCKOUT", "OVERSTOCKED"]
    days_of_supply: float


class PrescriptiveActionsRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "PROD_0001"})
    current_stock: int = Field(..., json_schema_extra={"example": 80})
    daily_demand_mean: float = Field(..., json_schema_extra={"example": 60.0})
    daily_demand_std: float = Field(..., json_schema_extra={"example": 12.0})
    lead_time_days: float = Field(..., json_schema_extra={"example": 7.0})
    lead_time_std: float = Field(default=2.0, json_schema_extra={"example": 2.0})
    unit_cost: float = Field(default=20.0, json_schema_extra={"example": 20.0})
    stockout_prob: float = Field(..., json_schema_extra={"example": 0.85})
    supplier_risk: float = Field(..., json_schema_extra={"example": 0.70})
    demand_surge: float = Field(default=0.40, json_schema_extra={"example": 0.40})
    shipment_delay_risk: float = Field(default=0.35, json_schema_extra={"example": 0.35})
    anomaly_score: float = Field(default=0.10, json_schema_extra={"example": 0.10})
    primary_supplier_id: str = Field(default="SUP_001", json_schema_extra={"example": "SUP_001"})
    alternative_suppliers: list[dict[str, Any]] = Field(
        default_factory=lambda: [{"supplier_id": "SUP_BACKUP_A", "risk_score": 0.12}]
    )


class PrescriptiveActionItem(BaseModel):
    action_type: str
    urgency: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    product_id: str
    headline: str
    recommendation: str
    recommended_order_quantity: int
    primary_supplier_id: str
    alternative_supplier_id: str | None
    estimated_cost: float
    estimated_stockout_loss_avoided: float
    net_benefit: float


class PrescriptiveActionsResponse(BaseModel):
    product_id: str
    risk_summary: RiskScoreResponse
    inventory_policy: InventoryPolicyResponse
    actions: list[PrescriptiveActionItem]


# =================================================================
# Simulation (Phase 4)
# =================================================================

class SimulationRunRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "PROD_0001"})
    preset_key: str | None = Field(
        default=None,
        description="Optional pre-configured preset key (e.g. 'PORT_STRIKE', 'FLASH_SALE_SURGE')",
        json_schema_extra={"example": "PORT_STRIKE"},
    )
    custom_scenario_name: str | None = Field(default="Custom Simulation")
    demand_shock_pct: float = Field(default=30.0, json_schema_extra={"example": 30.0})
    supplier_delay_days: float = Field(default=5.0, json_schema_extra={"example": 5.0})
    supplier_capacity_loss_pct: float = Field(default=15.0, json_schema_extra={"example": 15.0})
    warehouse_capacity_loss_pct: float = Field(default=0.0, json_schema_extra={"example": 0.0})
    transport_cost_multiplier: float = Field(default=1.25, json_schema_extra={"example": 1.25})
    projection_horizon_days: int = Field(default=14, json_schema_extra={"example": 14})

    # State parameters
    current_stock: int = Field(default=500, json_schema_extra={"example": 500})
    daily_demand_base: float = Field(default=50.0, json_schema_extra={"example": 50.0})
    daily_demand_std: float = Field(default=10.0, json_schema_extra={"example": 10.0})
    lead_time_days: float = Field(default=6.0, json_schema_extra={"example": 6.0})
    unit_cost: float = Field(default=15.0, json_schema_extra={"example": 15.0})
    unit_price: float = Field(default=28.0, json_schema_extra={"example": 28.0})
    scheduled_inbound: dict[int, int] | None = Field(
        default_factory=lambda: {4: 400},
        description="Mapping of simulation day index to scheduled units arriving",
    )
    primary_supplier_id: str = Field(default="SUP_001")
    alternative_suppliers: list[dict[str, Any]] = Field(
        default_factory=lambda: [{"supplier_id": "SUP_BACKUP_A", "risk_score": 0.12}]
    )


class SimulationRunResponse(BaseModel):
    scenario_name: str
    product_id: str
    projection_horizon_days: int
    baseline_risk_score: float
    baseline_health_score: float
    baseline_days_of_supply: float
    baseline_stockout_prob: float
    simulated_risk_score: float
    simulated_health_score: float
    simulated_days_of_supply: float
    simulated_stockout_prob: float
    risk_delta: float
    first_stockout_day: int | None
    total_stockout_days: int
    total_unfulfilled_units: float
    estimated_revenue_loss: float
    daily_trajectory_sample: list[dict[str, Any]]
    mitigation_actions: list[dict[str, Any]]
    mitigation_cost: float
    mitigation_net_savings: float
