"""
Industry standard pre-configured disruption scenario presets.
"""

from __future__ import annotations

from src.simulation.scenarios import DisruptionScenario


def get_preset_scenario(preset_key: str) -> DisruptionScenario:
    """Retrieve a pre-configured disruption preset by key."""
    presets = {
        "PORT_STRIKE": DisruptionScenario(
            name="Major Port & Logistics Bottleneck",
            description="Labor strike and customs congestion at major entry port; transit delays surge.",
            demand_shock_pct=0.0,
            supplier_delay_days=8.0,
            supplier_capacity_loss_pct=30.0,
            warehouse_capacity_loss_pct=0.0,
            transport_cost_multiplier=1.40,
            projection_horizon_days=21,
        ),
        "FLASH_SALE_SURGE": DisruptionScenario(
            name="Viral Campaign / Flash Sale Surge",
            description="Unexpected viral demand surge driven by marketing campaign or holiday event.",
            demand_shock_pct=65.0,
            supplier_delay_days=1.0,
            supplier_capacity_loss_pct=0.0,
            warehouse_capacity_loss_pct=0.0,
            transport_cost_multiplier=1.10,
            projection_horizon_days=14,
        ),
        "SUPPLIER_INSOLVENCY": DisruptionScenario(
            name="Primary Vendor Production Outage",
            description="Critical tier-1 supplier suffers plant shutdown or financial insolvency.",
            demand_shock_pct=0.0,
            supplier_delay_days=14.0,
            supplier_capacity_loss_pct=80.0,
            warehouse_capacity_loss_pct=0.0,
            transport_cost_multiplier=1.25,
            projection_horizon_days=30,
        ),
        "WAREHOUSE_FACILITY_DISRUPTION": DisruptionScenario(
            name="Central Fulfillment Hub Damage",
            description="Severe weather or facility damage shuts down 50% of central warehouse floor.",
            demand_shock_pct=-5.0,
            supplier_delay_days=3.0,
            supplier_capacity_loss_pct=10.0,
            warehouse_capacity_loss_pct=50.0,
            transport_cost_multiplier=1.35,
            projection_horizon_days=14,
        ),
        "COMPOUND_CASCADE": DisruptionScenario(
            name="Compound Cascade (Black Swan Shock)",
            description="Simultaneous demand surge coinciding with freight lane delays and supplier output cuts.",
            demand_shock_pct=40.0,
            supplier_delay_days=6.0,
            supplier_capacity_loss_pct=35.0,
            warehouse_capacity_loss_pct=15.0,
            transport_cost_multiplier=1.50,
            projection_horizon_days=21,
        ),
    }

    key = preset_key.upper()
    if key not in presets:
        raise KeyError(
            f"Unknown preset '{preset_key}'. Available presets: {list(presets.keys())}"
        )
    return presets[key]


def list_available_presets() -> list[dict[str, str]]:
    """Return list of available scenario presets for UI dropdowns."""
    return [
        {"key": "PORT_STRIKE", "name": "Major Port & Logistics Bottleneck"},
        {"key": "FLASH_SALE_SURGE", "name": "Viral Campaign / Flash Sale Surge"},
        {"key": "SUPPLIER_INSOLVENCY", "name": "Primary Vendor Production Outage"},
        {"key": "WAREHOUSE_FACILITY_DISRUPTION", "name": "Central Fulfillment Hub Damage"},
        {"key": "COMPOUND_CASCADE", "name": "Compound Cascade (Black Swan Shock)"},
    ]
