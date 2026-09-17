"""
Scenario definitions and dataclasses for the What-If Supply Chain Simulation Engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DisruptionScenario:
    """Parameters representing a supply chain shock or perturbation."""
    name: str
    description: str
    demand_shock_pct: float = 0.0  # e.g. +30.0 for +30% demand surge, -20.0 for slump
    supplier_delay_days: float = 0.0  # e.g. +5.0 days delay in inbound transit
    supplier_capacity_loss_pct: float = 0.0  # e.g. 40.0 for 40% vendor output cut
    warehouse_capacity_loss_pct: float = 0.0  # e.g. 25.0 for 25% warehouse floor closure
    transport_cost_multiplier: float = 1.0  # e.g. 1.5 for 50% surge in freight rates
    projection_horizon_days: int = 14  # simulation window (days)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DailyInventoryStep:
    """Snapshot of inventory state on day t of simulation."""
    day: int
    date: str
    beginning_stock: int
    daily_demand: float
    inbound_receipts: int
    ending_stock: int
    unfulfilled_demand: float
    stockout_occurred: bool
    stockout_loss: float


@dataclass
class SimulationResult:
    """Comprehensive comparison of baseline vs. simulated post-shock supply chain."""
    scenario_name: str
    product_id: str
    projection_horizon_days: int

    # Baseline state
    baseline_risk_score: float
    baseline_health_score: float
    baseline_days_of_supply: float
    baseline_stockout_prob: float

    # Simulated post-shock state
    simulated_risk_score: float
    simulated_health_score: float
    simulated_days_of_supply: float
    simulated_stockout_prob: float
    risk_delta: float

    # Disruption consequences
    first_stockout_day: int | None  # None if no stockout occurs
    total_stockout_days: int
    total_unfulfilled_units: float
    estimated_revenue_loss: float

    # Daily trajectory
    daily_trajectory: list[DailyInventoryStep] = field(default_factory=list)

    # Mitigation Plan
    mitigation_actions: list[dict[str, Any]] = field(default_factory=list)
    mitigation_cost: float = 0.0
    mitigation_net_savings: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["daily_trajectory"] = [asdict(s) for s in self.daily_trajectory]
        return res
