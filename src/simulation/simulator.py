"""
Core What-If Simulation Engine for the Intelligent Supply Chain Control Tower.

Simulates the operational and financial impact of supply chain disruptions
over forward projection horizons and generates targeted mitigation plans.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

from src.risk_engine.decision_engine import DecisionEngine
from src.risk_engine.scorer import RiskAssessment, RiskEngine
from src.simulation.scenarios import DailyInventoryStep, DisruptionScenario, SimulationResult


class SupplyChainSimulator:
    """
    Simulation engine capable of stress-testing multi-echelon supply chain states
    under arbitrary or preset disruption scenarios.
    """

    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        decision_engine: DecisionEngine | None = None,
    ) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self.decision_engine = decision_engine or DecisionEngine()

    def run_simulation(
        self,
        scenario: DisruptionScenario,
        product_id: str,
        current_stock: int,
        daily_demand_base: float,
        daily_demand_std: float,
        lead_time_days: float,
        unit_cost: float = 15.0,
        unit_price: float = 28.0,
        scheduled_inbound: dict[int, int] | None = None,  # day_idx -> units
        primary_supplier_id: str = "SUP_001",
        alternative_suppliers: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
    ) -> SimulationResult:
        """
        Execute forward day-by-day inventory simulation and impact evaluation.
        """
        horizon = scenario.projection_horizon_days
        base_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now()

        # Inbound deliveries schedule
        inbound_map = scheduled_inbound or {}

        # -------------------------------------------------------------
        # 1. Baseline Evaluation
        # -------------------------------------------------------------
        base_policy = self.decision_engine.compute_inventory_policy(
            product_id=product_id,
            current_stock=current_stock,
            daily_demand_mean=daily_demand_base,
            daily_demand_std=daily_demand_std,
            lead_time_days=lead_time_days,
            unit_cost=unit_cost,
        )

        base_stockout_prob = 0.85 if base_policy.current_stock < base_policy.reorder_point else 0.05
        base_risk = self.risk_engine.calculate_risk(
            stockout_prob=base_stockout_prob,
            supplier_risk=0.15,
            demand_surge=0.10,
            shipment_delay_risk=0.08,
            anomaly_score=0.05,
            metadata={"product_id": product_id},
        )

        # -------------------------------------------------------------
        # 2. Forward Day-by-Day Simulation under Shock
        # -------------------------------------------------------------
        daily_steps: list[DailyInventoryStep] = []
        stock = current_stock
        first_stockout_day = None
        total_stockout_days = 0
        total_unfulfilled_units = 0.0
        total_revenue_loss = 0.0

        # Demand multiplier
        demand_mult = max(0.1, 1.0 + (scenario.demand_shock_pct / 100.0))
        sim_daily_demand = daily_demand_base * demand_mult

        # Inbound shift due to supplier delay
        delay_shift = int(round(scenario.supplier_delay_days))
        delayed_inbound: dict[int, int] = {}
        for day_idx, qty in inbound_map.items():
            # Apply supplier capacity loss
            effective_qty = int(round(qty * (1.0 - (scenario.supplier_capacity_loss_pct / 100.0))))
            target_day = day_idx + delay_shift
            delayed_inbound[target_day] = delayed_inbound.get(target_day, 0) + effective_qty

        for t in range(1, horizon + 1):
            cur_date_str = (base_date + timedelta(days=t - 1)).strftime("%Y-%m-%d")
            beg_stock = stock
            inbound_qty = delayed_inbound.get(t, 0)

            # Daily demand with minor random perturbation
            d_t = max(0.0, float(np.random.normal(sim_daily_demand, daily_demand_std * 0.5)))
            d_t = round(d_t, 1)

            # Net inventory balance
            available = beg_stock + inbound_qty
            if available >= d_t:
                ending_stock = int(round(available - d_t))
                unfulfilled = 0.0
                is_stockout = False
                loss = 0.0
            else:
                ending_stock = 0
                unfulfilled = round(d_t - available, 1)
                is_stockout = True
                total_stockout_days += 1
                if first_stockout_day is None:
                    first_stockout_day = t
                loss = round(unfulfilled * unit_price, 2)
                total_unfulfilled_units += unfulfilled
                total_revenue_loss += loss

            stock = ending_stock

            daily_steps.append(
                DailyInventoryStep(
                    day=t,
                    date=cur_date_str,
                    beginning_stock=beg_stock,
                    daily_demand=d_t,
                    inbound_receipts=inbound_qty,
                    ending_stock=ending_stock,
                    unfulfilled_demand=unfulfilled,
                    stockout_occurred=is_stockout,
                    stockout_loss=loss,
                )
            )

        # -------------------------------------------------------------
        # 3. Recompute Post-Shock Risk & Health Score
        # -------------------------------------------------------------
        sim_dos = round(stock / max(0.01, sim_daily_demand), 1)

        # Dynamic stockout prob calculation based on projected shortfall
        if total_stockout_days > 0:
            sim_stockout_prob = min(0.99, 0.70 + (total_stockout_days / horizon) * 0.29)
        elif sim_dos < (lead_time_days + delay_shift):
            sim_stockout_prob = 0.65
        else:
            sim_stockout_prob = 0.10

        sim_supplier_risk = min(
            1.0,
            0.15 + (scenario.supplier_delay_days * 0.07) + (scenario.supplier_capacity_loss_pct * 0.008),
        )
        sim_demand_surge = min(1.0, max(0.0, scenario.demand_shock_pct / 100.0))
        sim_delay_risk = min(1.0, scenario.supplier_delay_days / 14.0)
        sim_anomaly = min(
            1.0,
            0.10 + (scenario.supplier_delay_days * 0.04) + (scenario.demand_shock_pct * 0.003),
        )

        sim_risk = self.risk_engine.calculate_risk(
            stockout_prob=sim_stockout_prob,
            supplier_risk=sim_supplier_risk,
            demand_surge=sim_demand_surge,
            shipment_delay_risk=sim_delay_risk,
            anomaly_score=sim_anomaly,
            metadata={"product_id": product_id, "scenario": scenario.name},
        )

        # -------------------------------------------------------------
        # 4. Formulate Actionable Mitigation Plan
        # -------------------------------------------------------------
        sim_policy = self.decision_engine.compute_inventory_policy(
            product_id=product_id,
            current_stock=stock,
            daily_demand_mean=sim_daily_demand,
            daily_demand_std=daily_demand_std * demand_mult,
            lead_time_days=lead_time_days + delay_shift,
            unit_cost=unit_cost * scenario.transport_cost_multiplier,
        )

        raw_actions = self.decision_engine.generate_recommendations(
            product_id=product_id,
            risk=sim_risk,
            inventory_policy=sim_policy,
            unit_cost=unit_cost * scenario.transport_cost_multiplier,
            primary_supplier_id=primary_supplier_id,
            alternative_suppliers=alternative_suppliers,
        )

        mitigation_actions = [a.to_dict() for a in raw_actions]
        mitigation_cost = sum(a.get("estimated_cost", 0.0) for a in mitigation_actions)
        avoided_loss = sum(a.get("estimated_stockout_loss_avoided", 0.0) for a in mitigation_actions)
        net_savings = max(0.0, avoided_loss - mitigation_cost)

        return SimulationResult(
            scenario_name=scenario.name,
            product_id=product_id,
            projection_horizon_days=horizon,
            baseline_risk_score=base_risk.risk_score,
            baseline_health_score=base_risk.health_score,
            baseline_days_of_supply=base_policy.days_of_supply,
            baseline_stockout_prob=base_stockout_prob,
            simulated_risk_score=sim_risk.risk_score,
            simulated_health_score=sim_risk.health_score,
            simulated_days_of_supply=sim_dos,
            simulated_stockout_prob=sim_stockout_prob,
            risk_delta=round(sim_risk.risk_score - base_risk.risk_score, 1),
            first_stockout_day=first_stockout_day,
            total_stockout_days=total_stockout_days,
            total_unfulfilled_units=round(total_unfulfilled_units, 1),
            estimated_revenue_loss=round(total_revenue_loss, 2),
            daily_trajectory=daily_steps,
            mitigation_actions=mitigation_actions,
            mitigation_cost=round(mitigation_cost, 2),
            mitigation_net_savings=round(net_savings, 2),
        )
