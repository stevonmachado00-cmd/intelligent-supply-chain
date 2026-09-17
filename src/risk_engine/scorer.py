"""
Multi-Model Risk Engine for the Intelligent Supply Chain Control Tower.

Aggregates individual ML model predictions (stockout probability, supplier risk,
demand surge, shipment delays, and autoencoder anomalies) into a unified,
calibrated 0-100 Risk Score and overall Supply Chain Health Score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

CONFIG_PATH = Path("configs/model_config.yaml")


@dataclass
class RiskComponentScores:
    stockout_risk: float  # 0.0 - 1.0 (calibrated stockout probability)
    supplier_risk: float  # 0.0 - 1.0 (calibrated supplier default/delay risk)
    demand_surge_index: float  # 0.0 - 1.0 (ratio of forecasted demand surge)
    shipment_delay_risk: float  # 0.0 - 1.0 (probability/severity of late transit)
    anomaly_score: float  # 0.0 - 1.0 (autoencoder reconstruction anomaly score)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class RiskAssessment:
    risk_score: float  # 0.0 to 100.0 (higher = higher risk)
    health_score: float  # 100.0 - risk_score (higher = healthier)
    status: str  # "GREEN" (0-30), "AMBER" (31-60), "RED" (61-100)
    primary_driver: str  # Feature/component contributing the most to current risk
    components: RiskComponentScores
    component_contributions: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["components"] = self.components.to_dict()
        return res


class RiskEngine:
    """
    Weighted multi-model risk scoring engine.
    Weights are loaded from configs/model_config.yaml.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        path = Path(config_path) if config_path else CONFIG_PATH
        if path.exists():
            with open(path, encoding="utf-8-sig") as f:
                cfg = yaml.safe_load(f)
            risk_cfg = cfg.get("risk_engine", {})
            self.weights = risk_cfg.get(
                "weights",
                {
                    "stockout_probability": 0.30,
                    "supplier_risk_score": 0.25,
                    "demand_surge_index": 0.25,
                    "shipment_delay_risk": 0.10,
                    "anomaly_score": 0.10,
                },
            )
            self.thresholds = risk_cfg.get(
                "thresholds",
                {"green": 30, "amber": 60, "red": 100},
            )
        else:
            self.weights = {
                "stockout_probability": 0.30,
                "supplier_risk_score": 0.25,
                "demand_surge_index": 0.25,
                "shipment_delay_risk": 0.10,
                "anomaly_score": 0.10,
            }
            self.thresholds = {"green": 30, "amber": 60, "red": 100}

        # Normalize weights so they strictly sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def calculate_risk(
        self,
        stockout_prob: float,
        supplier_risk: float,
        demand_surge: float,
        shipment_delay_risk: float,
        anomaly_score: float,
        metadata: dict[str, Any] | None = None,
    ) -> RiskAssessment:
        """
        Compute the weighted 0-100 risk score and determine severity status.

        Parameters
        ----------
        stockout_prob : float [0, 1]
        supplier_risk : float [0, 1]
        demand_surge : float [0, 1]
        shipment_delay_risk : float [0, 1]
        anomaly_score : float [0, 1]
        """
        # Clamp inputs to [0, 1]
        s_prob = float(np.clip(stockout_prob, 0.0, 1.0))
        sup_risk = float(np.clip(supplier_risk, 0.0, 1.0))
        d_surge = float(np.clip(demand_surge, 0.0, 1.0))
        ship_risk = float(np.clip(shipment_delay_risk, 0.0, 1.0))
        anom = float(np.clip(anomaly_score, 0.0, 1.0))

        components = RiskComponentScores(
            stockout_risk=s_prob,
            supplier_risk=sup_risk,
            demand_surge_index=d_surge,
            shipment_delay_risk=ship_risk,
            anomaly_score=anom,
        )

        # Weighted points contribution (each component contributes up to weight * 100)
        contribs = {
            "stockout": round(s_prob * self.weights["stockout_probability"] * 100, 2),
            "supplier_risk": round(sup_risk * self.weights["supplier_risk_score"] * 100, 2),
            "demand_surge": round(d_surge * self.weights["demand_surge_index"] * 100, 2),
            "shipment_delay": round(ship_risk * self.weights["shipment_delay_risk"] * 100, 2),
            "anomaly": round(anom * self.weights["anomaly_score"] * 100, 2),
        }

        total_risk_score = round(sum(contribs.values()), 1)
        health_score = round(max(0.0, 100.0 - total_risk_score), 1)

        # Determine traffic-light status
        if total_risk_score <= self.thresholds["green"]:
            status = "GREEN"
        elif total_risk_score <= self.thresholds["amber"]:
            status = "AMBER"
        else:
            status = "RED"

        # Determine highest contributing driver
        primary_driver = max(contribs, key=contribs.get)

        return RiskAssessment(
            risk_score=total_risk_score,
            health_score=health_score,
            status=status,
            primary_driver=primary_driver,
            components=components,
            component_contributions=contribs,
            metadata=metadata or {},
        )

    def calculate_batch_risks(
        self,
        records: list[dict[str, Any]],
    ) -> list[RiskAssessment]:
        """Process multiple SKUs or transit routes in batch."""
        assessments = []
        for r in records:
            assessments.append(
                self.calculate_risk(
                    stockout_prob=r.get("stockout_prob", 0.0),
                    supplier_risk=r.get("supplier_risk", 0.0),
                    demand_surge=r.get("demand_surge", 0.0),
                    shipment_delay_risk=r.get("shipment_delay_risk", 0.0),
                    anomaly_score=r.get("anomaly_score", 0.0),
                    metadata=r.get("metadata", {}),
                )
            )
        return assessments
