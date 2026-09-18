"""
Unit and functional tests for the Multi-Model Risk Engine and Prescriptive Decision Engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.risk_engine.scorer import RiskEngine
from src.risk_engine.decision_engine import DecisionEngine
from src.simulation.scenarios import DisruptionScenario
from src.simulation.simulator import SupplyChainSimulator


def test_risk_engine_healthy_state():
    engine = RiskEngine()
    assessment = engine.calculate_risk(
        stockout_prob=0.05,
        supplier_risk=0.10,
        demand_surge=0.05,
        shipment_delay_risk=0.02,
        anomaly_score=0.01,
    )
    assert assessment.status == "GREEN"
    assert assessment.risk_score < 25.0
    assert assessment.health_score > 75.0


def test_risk_engine_critical_stockout():
    engine = RiskEngine()
    assessment = engine.calculate_risk(
        stockout_prob=1.0,
        supplier_risk=0.80,
        demand_surge=0.80,
        shipment_delay_risk=0.50,
        anomaly_score=0.40,
    )
    assert assessment.status == "RED"
    assert assessment.risk_score > 60.0
    assert assessment.primary_driver == "stockout"


def test_decision_engine_inventory_policy():
    engine = DecisionEngine()
    policy = engine.compute_inventory_policy(
        product_id="TEST_PROD",
        current_stock=100,
        daily_demand_mean=40.0,
        daily_demand_std=8.0,
        lead_time_days=6.0,
        lead_time_std=1.0,
        unit_cost=20.0,
    )
    assert policy.product_id == "TEST_PROD"
    assert policy.safety_stock > 0
    assert policy.reorder_point > policy.safety_stock
    assert policy.economic_order_quantity > 0
    assert policy.days_of_supply == round(100 / 40.0, 1)


def test_decision_engine_recommendations_emergency_replenishment():
    risk_engine = RiskEngine()
    decision_engine = DecisionEngine()

    risk = risk_engine.calculate_risk(
        stockout_prob=0.90,
        supplier_risk=0.20,
        demand_surge=0.30,
        shipment_delay_risk=0.10,
        anomaly_score=0.05,
    )
    policy = decision_engine.compute_inventory_policy(
        product_id="TEST_PROD",
        current_stock=20,
        daily_demand_mean=50.0,
        daily_demand_std=10.0,
        lead_time_days=5.0,
    )
    actions = decision_engine.generate_recommendations(
        product_id="TEST_PROD",
        risk=risk,
        inventory_policy=policy,
        unit_cost=15.0,
    )
    assert len(actions) > 0
    action_types = [a.action_type for a in actions]
    assert "EXPEDITE_ORDER" in action_types
    assert actions[0].urgency == "CRITICAL"


def test_simulation_forward_rollout():
    risk_engine = RiskEngine()
    decision_engine = DecisionEngine()
    simulator = SupplyChainSimulator(risk_engine=risk_engine, decision_engine=decision_engine)

    scenario = DisruptionScenario(
        name="Test Port Shock",
        description="Port strike simulation test",
        demand_shock_pct=40.0,
        supplier_delay_days=6.0,
        projection_horizon_days=10,
    )
    result = simulator.run_simulation(
        scenario=scenario,
        product_id="TEST_PROD",
        current_stock=200,
        daily_demand_base=40.0,
        daily_demand_std=5.0,
        lead_time_days=5.0,
        unit_cost=15.0,
        unit_price=30.0,
    )
    assert result.scenario_name == "Test Port Shock"
    assert result.projection_horizon_days == 10
    assert len(result.daily_trajectory) == 10
    assert result.simulated_risk_score >= result.baseline_risk_score

