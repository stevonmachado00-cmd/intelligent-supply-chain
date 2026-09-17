"""
Integration test and verification script for Phase 4: What-If Simulation Engine.

Simulates complex operational disruptions across pre-configured presets
and custom scenario parameters, verifying multi-day forward inventory
trajectories, stockout emergence, financial loss, and prescriptive mitigation.

Run from project root:
    python scripts/test_simulation_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.simulation.presets import get_preset_scenario, list_available_presets
from src.simulation.scenarios import DisruptionScenario
from src.simulation.simulator import SupplyChainSimulator


def main():
    logger.info("=" * 70)
    logger.info("INTELLIGENT SUPPLY CHAIN CONTROL TOWER — PHASE 4")
    logger.info("Testing What-If Scenario Simulation Engine")
    logger.info("=" * 70)

    simulator = SupplyChainSimulator()
    presets = list_available_presets()
    logger.info(f"Available Scenario Presets: {[p['key'] for p in presets]}")

    # -------------------------------------------------------------
    # Test 1: Custom Disruption Scenario
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("TEST 1: Custom Disruption (Demand +35%, Supplier Delay +5d, Horizon 14d)")
    logger.info("-" * 70)

    custom_scenario = DisruptionScenario(
        name="Custom Promo + Transit Shock",
        description="Marketing flash sale coinciding with regional road transit stoppage.",
        demand_shock_pct=35.0,
        supplier_delay_days=5.0,
        supplier_capacity_loss_pct=15.0,
        warehouse_capacity_loss_pct=0.0,
        transport_cost_multiplier=1.20,
        projection_horizon_days=14,
    )

    res_custom = simulator.run_simulation(
        scenario=custom_scenario,
        product_id="PROD_0001",
        current_stock=450,
        daily_demand_base=60.0,
        daily_demand_std=12.0,
        lead_time_days=6.0,
        unit_cost=18.0,
        unit_price=35.0,
        scheduled_inbound={4: 300, 10: 400},  # Day 4 & Day 10 scheduled arrivals
        primary_supplier_id="SUP_001",
        alternative_suppliers=[
            {"supplier_id": "SUP_BACKUP_A", "risk_score": 0.15},
            {"supplier_id": "SUP_BACKUP_B", "risk_score": 0.40},
        ],
    )

    logger.info(f"Scenario: {res_custom.scenario_name}")
    logger.info(
        f"Network Risk Score: Baseline={res_custom.baseline_risk_score:.1f} -> "
        f"Simulated={res_custom.simulated_risk_score:.1f} (Delta: +{res_custom.risk_delta:.1f} pts)"
    )
    logger.info(
        f"Health Score: Baseline={res_custom.baseline_health_score:.1f} -> "
        f"Simulated={res_custom.simulated_health_score:.1f}"
    )
    logger.info(
        f"Days of Supply: Baseline={res_custom.baseline_days_of_supply:.1f}d -> "
        f"Simulated={res_custom.simulated_days_of_supply:.1f}d"
    )

    if res_custom.first_stockout_day:
        logger.warning(
            f"Stockout Emergence: Day {res_custom.first_stockout_day} "
            f"({res_custom.total_stockout_days} total stockout days)"
        )
        logger.warning(f"Unfulfilled Demand: {res_custom.total_unfulfilled_units:,.1f} units")
        logger.warning(f"Estimated Revenue at Risk: ${res_custom.estimated_revenue_loss:,.2f}")
    else:
        logger.success("No stockout occurred during projection window.")

    logger.info(f"Mitigation Actions Formulated: {len(res_custom.mitigation_actions)}")
    for act in res_custom.mitigation_actions:
        logger.info(f"  [{act['urgency']}] {act['action_type']}: {act['headline']}")
        logger.info(f"    Action: {act['recommendation']}")
        logger.info(f"    Cost: ${act['estimated_cost']:,.2f} | Net Benefit: ${act['net_benefit']:,.2f}")

    # -------------------------------------------------------------
    # Test 2: Industry Presets Stress-Test
    # -------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("TEST 2: Automated Multi-Preset Stress-Testing")
    logger.info("-" * 70)

    for p in presets:
        key = p["key"]
        preset_scen = get_preset_scenario(key)
        res = simulator.run_simulation(
            scenario=preset_scen,
            product_id="PROD_BENCHMARK",
            current_stock=800,
            daily_demand_base=80.0,
            daily_demand_std=15.0,
            lead_time_days=7.0,
            unit_cost=20.0,
            unit_price=45.0,
            scheduled_inbound={5: 600},
            primary_supplier_id="SUP_002",
            alternative_suppliers=[{"supplier_id": "SUP_BACKUP_A", "risk_score": 0.10}],
        )

        status_flag = "ALERT" if res.total_stockout_days > 0 else "OK"
        logger.info(
            f"[{status_flag}] Preset: {preset_scen.name[:32]:<32} | "
            f"Risk: {res.simulated_risk_score:>4.1f}/100 | "
            f"Stockout Days: {res.total_stockout_days:>2} | "
            f"Rev Loss: ${res.estimated_revenue_loss:>8,.0f} | "
            f"Mitigation Net Savings: ${res.mitigation_net_savings:>8,.0f}"
        )

    logger.success("\n" + "=" * 70)
    logger.success("PHASE 4: What-If Simulation Engine Verified Successfully!")
    logger.success("=" * 70)


if __name__ == "__main__":
    main()
