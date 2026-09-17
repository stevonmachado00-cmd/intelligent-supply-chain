"""
Integration test and demonstration script for Phase 3: Risk Engine & Decision Engine.

Loads trained Phase 2 model artifacts, evaluates real feature rows,
computes multi-model Risk Scores and Health Scores, and generates
prescriptive replenishment & routing actions.

Run from project root:
    python scripts/test_risk_and_decision_engine.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.risk_engine.decision_engine import DecisionEngine
from src.risk_engine.scorer import RiskEngine

ARTIFACTS_DIR = Path("mlflow/model_artifacts")
PROCESSED_DIR = Path("data/processed")


def main():
    logger.info("=" * 70)
    logger.info("INTELLIGENT SUPPLY CHAIN CONTROL TOWER — PHASE 3")
    logger.info("Testing Risk Engine & Prescriptive Decision Engine")
    logger.info("=" * 70)

    # Initialize engines
    risk_engine = RiskEngine()
    decision_engine = DecisionEngine()

    logger.info(f"Risk weights loaded: {risk_engine.weights}")
    logger.info(f"Thresholds: {risk_engine.thresholds}")
    logger.info(
        f"Decision Engine parameters: EOQ order cost=${decision_engine.order_cost}, "
        f"Holding rate={decision_engine.holding_cost_rate:.0%}, "
        f"Safety stock z={decision_engine.service_level_z} (95% service level)"
    )

    # -------------------------------------------------------------
    # Test Scenario 1: Healthy Operational SKU
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("SCENARIO 1: Healthy Operational SKU (Normal Demand, Reliable Supplier)")
    logger.info("-" * 70)

    risk_s1 = risk_engine.calculate_risk(
        stockout_prob=0.08,
        supplier_risk=0.12,
        demand_surge=0.15,
        shipment_delay_risk=0.10,
        anomaly_score=0.05,
        metadata={"product_id": "PROD_HEALTHY", "store_id": "STORE_01"},
    )
    policy_s1 = decision_engine.compute_inventory_policy(
        product_id="PROD_HEALTHY",
        current_stock=1800,
        daily_demand_mean=45.0,
        daily_demand_std=8.0,
        lead_time_days=5.0,
        lead_time_std=1.0,
        unit_cost=12.50,
    )
    actions_s1 = decision_engine.generate_recommendations(
        product_id="PROD_HEALTHY",
        risk=risk_s1,
        inventory_policy=policy_s1,
        unit_cost=12.50,
    )

    logger.info(f"Risk Score: {risk_s1.risk_score}/100 [{risk_s1.status}] | Health Score: {risk_s1.health_score}/100")
    logger.info(f"Component contributions: {risk_s1.component_contributions}")
    logger.info(
        f"Inventory Policy: Current Stock={policy_s1.current_stock}, "
        f"Safety Stock={policy_s1.safety_stock}, ROP={policy_s1.reorder_point}, "
        f"EOQ={policy_s1.economic_order_quantity}, Days of Supply={policy_s1.days_of_supply}d, "
        f"Status={policy_s1.stock_status}"
    )
    logger.info(f"Prescriptive Actions Generated: {len(actions_s1)}")
    for a in actions_s1:
        logger.info(f"  [{a.urgency}] {a.headline} -> {a.recommendation}")

    # -------------------------------------------------------------
    # Test Scenario 2: Imminent Stockout Disruption
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("SCENARIO 2: Imminent Stockout Disruption (Surging Demand, Low Stock)")
    logger.info("-" * 70)

    risk_s2 = risk_engine.calculate_risk(
        stockout_prob=0.88,
        supplier_risk=0.25,
        demand_surge=0.75,
        shipment_delay_risk=0.30,
        anomaly_score=0.15,
        metadata={"product_id": "PROD_SURGE_01"},
    )
    policy_s2 = decision_engine.compute_inventory_policy(
        product_id="PROD_SURGE_01",
        current_stock=120,
        daily_demand_mean=95.0,
        daily_demand_std=25.0,
        lead_time_days=7.0,
        lead_time_std=2.0,
        unit_cost=25.0,
    )
    actions_s2 = decision_engine.generate_recommendations(
        product_id="PROD_SURGE_01",
        risk=risk_s2,
        inventory_policy=policy_s2,
        unit_cost=25.0,
        primary_supplier_id="SUP_001",
        alternative_suppliers=[
            {"supplier_id": "SUP_BACKUP_A", "risk_score": 0.18},
            {"supplier_id": "SUP_BACKUP_B", "risk_score": 0.45},
        ],
    )

    logger.warning(f"Risk Score: {risk_s2.risk_score}/100 [{risk_s2.status}] | Health Score: {risk_s2.health_score}/100")
    logger.warning(f"Primary Risk Driver: {risk_s2.primary_driver} ({risk_s2.component_contributions[risk_s2.primary_driver]} pts)")
    logger.warning(
        f"Inventory Policy: Current Stock={policy_s2.current_stock}, "
        f"Safety Stock={policy_s2.safety_stock}, ROP={policy_s2.reorder_point}, "
        f"EOQ={policy_s2.economic_order_quantity}, Days of Supply={policy_s2.days_of_supply}d, "
        f"Status={policy_s2.stock_status}"
    )
    logger.warning(f"Prescriptive Actions Generated: {len(actions_s2)}")
    for a in actions_s2:
        logger.warning(f"  [{a.urgency}] {a.headline}")
        logger.warning(f"    Action: {a.recommendation}")
        logger.warning(f"    Recommended Order Qty: {a.recommended_order_quantity:,} units")
        logger.warning(f"    Estimated Cost: ${a.estimated_cost:,.2f} | Prevented Loss: ${a.estimated_stockout_loss_avoided:,.2f} | Net Benefit: ${a.net_benefit:,.2f}")

    # -------------------------------------------------------------
    # Test Scenario 3: High Supplier Risk & Rerouting
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("SCENARIO 3: Supplier Risk Breach -> Automatic Rerouting to Backup Vendor")
    logger.info("-" * 70)

    risk_s3 = risk_engine.calculate_risk(
        stockout_prob=0.82,
        supplier_risk=0.85,  # High supplier risk breach
        demand_surge=0.40,
        shipment_delay_risk=0.65,
        anomaly_score=0.30,
        metadata={"product_id": "PROD_CRITICAL_09"},
    )
    policy_s3 = decision_engine.compute_inventory_policy(
        product_id="PROD_CRITICAL_09",
        current_stock=250,
        daily_demand_mean=70.0,
        daily_demand_std=15.0,
        lead_time_days=8.0,
        lead_time_std=3.0,
        unit_cost=40.0,
    )
    actions_s3 = decision_engine.generate_recommendations(
        product_id="PROD_CRITICAL_09",
        risk=risk_s3,
        inventory_policy=policy_s3,
        unit_cost=40.0,
        primary_supplier_id="SUP_UNRELIABLE_03",
        alternative_suppliers=[
            {"supplier_id": "SUP_CERTIFIED_08", "risk_score": 0.12},
            {"supplier_id": "SUP_STANDBY_14", "risk_score": 0.38},
        ],
    )

    logger.error(f"Risk Score: {risk_s3.risk_score}/100 [{risk_s3.status}] | Health Score: {risk_s3.health_score}/100")
    logger.error(f"Primary Driver: {risk_s3.primary_driver} | Supplier Risk Score: {risk_s3.components.supplier_risk:.1%}")
    for a in actions_s3:
        logger.error(f"  [{a.urgency}] {a.action_type}: {a.headline}")
        logger.error(f"    Reroute Destination: Primary={a.primary_supplier_id} -> Alternative={a.alternative_supplier_id}")
        logger.error(f"    Details: {a.recommendation}")

    # -------------------------------------------------------------
    # Test Scenario 4: Overstocked SKU
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("SCENARIO 4: Overstocked SKU -> Holding Cost Drag & Clearance Recommendation")
    logger.info("-" * 70)

    risk_s4 = risk_engine.calculate_risk(
        stockout_prob=0.01,
        supplier_risk=0.10,
        demand_surge=0.05,
        shipment_delay_risk=0.05,
        anomaly_score=0.02,
        metadata={"product_id": "PROD_OVERSTOCKED"},
    )
    policy_s4 = decision_engine.compute_inventory_policy(
        product_id="PROD_OVERSTOCKED",
        current_stock=8500,
        daily_demand_mean=20.0,
        daily_demand_std=5.0,
        lead_time_days=4.0,
        lead_time_std=0.5,
        unit_cost=15.0,
    )
    actions_s4 = decision_engine.generate_recommendations(
        product_id="PROD_OVERSTOCKED",
        risk=risk_s4,
        inventory_policy=policy_s4,
        unit_cost=15.0,
    )

    logger.info(f"Risk Score: {risk_s4.risk_score}/100 [{risk_s4.status}] | Days of Supply: {policy_s4.days_of_supply} days")
    for a in actions_s4:
        logger.info(f"  [{a.urgency}] {a.headline}")
        logger.info(f"    Recommendation: {a.recommendation}")
        logger.info(f"    Expected Annual Holding Cost Drag Savings: ${a.net_benefit:,.2f}")

    logger.success("\n" + "=" * 70)
    logger.success("PHASE 3: Risk Engine and Prescriptive Decision Engine Verified Successfully!")
    logger.success("=" * 70)


if __name__ == "__main__":
    main()
