"""
Prescriptive Decision Engine for the Intelligent Supply Chain Control Tower.

Translates risks and predictions into optimal operational decisions:
  - Economic Order Quantity (EOQ) optimization
  - Dynamic Safety Stock (SS) with service-level guarantees (95% / z=1.65)
  - Dynamic Reorder Point (ROP)
  - Supplier Rerouting and Allocation Recommendations
  - Estimated cost impact and stockout prevention ROI
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from src.risk_engine.scorer import RiskAssessment

CONFIG_PATH = Path("configs/model_config.yaml")


@dataclass
class InventoryPolicy:
    product_id: str
    current_stock: int
    daily_demand: float
    lead_time_days: float
    safety_stock: int
    reorder_point: int
    economic_order_quantity: int
    stock_status: str  # "HEALTHY", "REORDER_SOON", "CRITICAL_STOCKOUT", "OVERSTOCKED"
    days_of_supply: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrescriptiveAction:
    action_type: str  # "EXPEDITE_ORDER", "REORDER", "REROUTE_SUPPLIER", "INVENTORY_TRANSFER", "CLEARANCE"
    urgency: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    product_id: str
    headline: str
    recommendation: str
    recommended_order_quantity: int
    primary_supplier_id: str
    alternative_supplier_id: str | None
    estimated_cost: float
    estimated_stockout_loss_avoided: float
    net_benefit: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionEngine:
    """
    Prescriptive decision-support engine.
    Calculates inventory replenishment policies and supplier mitigation recommendations.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        path = Path(config_path) if config_path else CONFIG_PATH
        if path.exists():
            with open(path, encoding="utf-8-sig") as f:
                cfg = yaml.safe_load(f)
            dec_cfg = cfg.get("decision_engine", {})
            self.order_cost = float(dec_cfg.get("eoq", {}).get("ordering_cost", 500.0))
            self.holding_cost_rate = float(dec_cfg.get("eoq", {}).get("holding_cost_rate", 0.25))
            self.service_level_z = float(dec_cfg.get("safety_stock", {}).get("service_level_z", 1.65))
            self.stockout_trigger = float(dec_cfg.get("stockout_trigger", 0.70))
            self.supplier_risk_trigger = float(dec_cfg.get("supplier_risk_trigger", 0.60))
        else:
            self.order_cost = 500.0
            self.holding_cost_rate = 0.25
            self.service_level_z = 1.65
            self.stockout_trigger = 0.70
            self.supplier_risk_trigger = 0.60

    def compute_eoq(
        self,
        annual_demand: float,
        unit_cost: float,
        order_cost: float | None = None,
        holding_rate: float | None = None,
    ) -> int:
        """
        Calculate Economic Order Quantity (EOQ):
        EOQ = sqrt((2 * D * S) / H)
        where:
          D = Annual Demand in units
          S = Fixed cost per purchase order
          H = Annual holding cost per unit (unit_cost * holding_rate)
        """
        s = order_cost or self.order_cost
        h_rate = holding_rate or self.holding_cost_rate
        h = max(0.01, unit_cost * h_rate)
        d = max(1.0, annual_demand)

        eoq = math.sqrt((2.0 * d * s) / h)
        return max(1, int(round(eoq)))

    def compute_safety_stock(
        self,
        daily_demand_mean: float,
        daily_demand_std: float,
        lead_time_days_mean: float,
        lead_time_days_std: float = 1.0,
        z: float | None = None,
    ) -> int:
        """
        Calculate dynamic Safety Stock with both demand and lead-time variability:
        SS = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        where:
          z = Service level factor (1.65 for 95% service level)
          L = Mean lead time
          sigma_d = Standard deviation of daily demand
          d = Mean daily demand
          sigma_L = Standard deviation of supplier lead time
        """
        z_factor = z or self.service_level_z
        l = max(0.5, lead_time_days_mean)
        sigma_d = max(0.01, daily_demand_std)
        d = max(0.01, daily_demand_mean)
        sigma_l = max(0.0, lead_time_days_std)

        variance = (l * (sigma_d**2)) + ((d**2) * (sigma_l**2))
        ss = z_factor * math.sqrt(variance)
        return max(1, int(round(ss)))

    def compute_inventory_policy(
        self,
        product_id: str,
        current_stock: int,
        daily_demand_mean: float,
        daily_demand_std: float,
        lead_time_days: float,
        lead_time_std: float = 1.5,
        unit_cost: float = 10.0,
    ) -> InventoryPolicy:
        """Compute complete inventory thresholds and current status for a product."""
        # Annual demand estimate (365 days)
        annual_demand = max(10.0, daily_demand_mean * 365.0)

        # Safety Stock
        ss = self.compute_safety_stock(
            daily_demand_mean=daily_demand_mean,
            daily_demand_std=daily_demand_std,
            lead_time_days_mean=lead_time_days,
            lead_time_days_std=lead_time_std,
        )

        # Reorder Point: ROP = (daily_demand * lead_time) + safety_stock
        lead_time_demand = daily_demand_mean * lead_time_days
        rop = int(round(lead_time_demand + ss))

        # EOQ
        eoq = self.compute_eoq(annual_demand=annual_demand, unit_cost=unit_cost)

        # Days of supply
        dos = round(current_stock / max(0.01, daily_demand_mean), 1)

        # Determine status
        if current_stock == 0 or (dos <= lead_time_days * 0.5):
            status = "CRITICAL_STOCKOUT"
        elif current_stock <= rop:
            status = "REORDER_SOON"
        elif dos > 90:
            status = "OVERSTOCKED"
        else:
            status = "HEALTHY"

        return InventoryPolicy(
            product_id=product_id,
            current_stock=current_stock,
            daily_demand=round(daily_demand_mean, 2),
            lead_time_days=round(lead_time_days, 1),
            safety_stock=ss,
            reorder_point=rop,
            economic_order_quantity=eoq,
            stock_status=status,
            days_of_supply=dos,
        )

    def generate_recommendations(
        self,
        product_id: str,
        risk: RiskAssessment,
        inventory_policy: InventoryPolicy,
        unit_cost: float = 10.0,
        primary_supplier_id: str = "SUP_001",
        alternative_suppliers: list[dict[str, Any]] | None = None,
    ) -> list[PrescriptiveAction]:
        """
        Generate actionable recommendations based on multi-model risks and inventory thresholds.
        """
        actions = []
        alt_sups = alternative_suppliers or []

        # Find best alternative supplier if needed (lowest risk score)
        best_alt_id = None
        if alt_sups:
            sorted_alts = sorted(alt_sups, key=lambda s: s.get("risk_score", 1.0))
            best_alt_id = sorted_alts[0].get("supplier_id")

        # 1. Critical Stockout Trigger: Stockout prob > 70% or current stock below lead-time demand
        if (
            risk.components.stockout_risk >= self.stockout_trigger
            or inventory_policy.stock_status == "CRITICAL_STOCKOUT"
        ):
            # Order EOQ + extra safety buffer
            reorder_qty = max(
                inventory_policy.economic_order_quantity,
                inventory_policy.reorder_point - inventory_policy.current_stock,
            )

            # Estimate lost revenue if stockout occurs: 7-day lost sales
            estimated_stockout_loss = round(
                inventory_policy.daily_demand * 7.0 * unit_cost * 1.8, 2
            )
            estimated_order_cost = round(reorder_qty * unit_cost + self.order_cost, 2)
            net_benefit = round(estimated_stockout_loss - self.order_cost, 2)

            # Check if primary supplier is high risk -> route to backup
            supplier_to_use = primary_supplier_id
            route_alt = False
            if risk.components.supplier_risk >= self.supplier_risk_trigger and best_alt_id:
                supplier_to_use = best_alt_id
                route_alt = True

            headline = f"Urgent: Reorder {reorder_qty:,} units of {product_id} to avoid stockout"
            if route_alt:
                rec_text = (
                    f"Stockout probability is {risk.components.stockout_risk:.1%} with only "
                    f"{inventory_policy.days_of_supply} days of supply. Primary supplier {primary_supplier_id} "
                    f"exhibits elevated risk ({risk.components.supplier_risk:.1%}). Reroute order of "
                    f"{reorder_qty:,} units to backup supplier {best_alt_id}."
                )
                action_type = "REROUTE_SUPPLIER"
            else:
                rec_text = (
                    f"Stockout probability is {risk.components.stockout_risk:.1%} with {inventory_policy.days_of_supply} "
                    f"days of supply remaining (below reorder point {inventory_policy.reorder_point:,}). "
                    f"Place immediate purchase order for {reorder_qty:,} units (EOQ) with {supplier_to_use}."
                )
                action_type = "EXPEDITE_ORDER" if inventory_policy.stock_status == "CRITICAL_STOCKOUT" else "REORDER"

            actions.append(
                PrescriptiveAction(
                    action_type=action_type,
                    urgency="CRITICAL" if inventory_policy.stock_status == "CRITICAL_STOCKOUT" else "HIGH",
                    product_id=product_id,
                    headline=headline,
                    recommendation=rec_text,
                    recommended_order_quantity=reorder_qty,
                    primary_supplier_id=primary_supplier_id,
                    alternative_supplier_id=best_alt_id if route_alt else None,
                    estimated_cost=estimated_order_cost,
                    estimated_stockout_loss_avoided=estimated_stockout_loss,
                    net_benefit=net_benefit,
                )
            )

        # 2. Reorder soon trigger: Below ROP but not critical
        elif inventory_policy.stock_status == "REORDER_SOON":
            reorder_qty = inventory_policy.economic_order_quantity
            estimated_cost = round(reorder_qty * unit_cost + self.order_cost, 2)

            actions.append(
                PrescriptiveAction(
                    action_type="REORDER",
                    urgency="MEDIUM",
                    product_id=product_id,
                    headline=f"Standard Replenishment: Reorder {reorder_qty:,} units of {product_id}",
                    recommendation=(
                        f"Current stock ({inventory_policy.current_stock:,}) has dropped below Reorder Point "
                        f"({inventory_policy.reorder_point:,}). Place routine purchase order for "
                        f"{reorder_qty:,} units (EOQ) with {primary_supplier_id}."
                    ),
                    recommended_order_quantity=reorder_qty,
                    primary_supplier_id=primary_supplier_id,
                    alternative_supplier_id=None,
                    estimated_cost=estimated_cost,
                    estimated_stockout_loss_avoided=round(reorder_qty * unit_cost * 0.4, 2),
                    net_benefit=round(reorder_qty * unit_cost * 0.35, 2),
                )
            )

        # 3. High Supplier Risk warning
        if (
            risk.components.supplier_risk >= self.supplier_risk_trigger
            and not any(a.action_type == "REROUTE_SUPPLIER" for a in actions)
        ):
            actions.append(
                PrescriptiveAction(
                    action_type="REROUTE_SUPPLIER",
                    urgency="HIGH",
                    product_id=product_id,
                    headline=f"Supplier Risk Alert: Audit or reallocate orders from {primary_supplier_id}",
                    recommendation=(
                        f"Supplier {primary_supplier_id} risk score has reached {risk.components.supplier_risk:.1%}. "
                        f"Initiate vendor quality audit and split future volume (40% allocation to {best_alt_id or 'secondary source'})."
                    ),
                    recommended_order_quantity=0,
                    primary_supplier_id=primary_supplier_id,
                    alternative_supplier_id=best_alt_id,
                    estimated_cost=0.0,
                    estimated_stockout_loss_avoided=round(inventory_policy.daily_demand * 14 * unit_cost, 2),
                    net_benefit=round(inventory_policy.daily_demand * 14 * unit_cost * 0.8, 2),
                )
            )

        # 4. Overstock Alert
        if inventory_policy.stock_status == "OVERSTOCKED":
            excess = inventory_policy.current_stock - (inventory_policy.reorder_point * 2)
            holding_cost_annual = excess * unit_cost * self.holding_cost_rate
            actions.append(
                PrescriptiveAction(
                    action_type="CLEARANCE",
                    urgency="LOW",
                    product_id=product_id,
                    headline=f"Excess Inventory: {product_id} has {inventory_policy.days_of_supply} days of supply",
                    recommendation=(
                        f"Product is overstocked by approximately {excess:,} units. Holding cost drag is "
                        f"${holding_cost_annual:,.2f}/yr. Pause purchase orders and activate promotional clearance or channel transfer."
                    ),
                    recommended_order_quantity=0,
                    primary_supplier_id=primary_supplier_id,
                    alternative_supplier_id=None,
                    estimated_cost=0.0,
                    estimated_stockout_loss_avoided=0.0,
                    net_benefit=round(holding_cost_annual * 0.5, 2),
                )
            )

        return actions
