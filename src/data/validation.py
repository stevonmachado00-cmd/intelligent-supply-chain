"""
Data validation layer.

Validates schema, column types, ranges, and basic business rules
for each dataset after ingestion. Raises or logs warnings on failures
so the pipeline can be halted before feeding bad data to models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    dataset_name: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def log(self) -> None:
        status = "PASSED" if self.passed else "FAILED"
        logger.info(f"[Validation] {self.dataset_name}: {status}")
        for err in self.errors:
            logger.error(f"  ERROR: {err}")
        for warn in self.warnings:
            logger.warning(f"  WARN:  {warn}")


def _check_required_columns(df: pd.DataFrame, required: list[str], result: ValidationResult) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.add_error(f"Missing required columns: {missing}")


def _check_nulls(df: pd.DataFrame, columns: list[str], result: ValidationResult, threshold: float = 0.05) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        null_rate = df[col].isna().mean()
        if null_rate > threshold:
            result.add_error(f"Column '{col}' has {null_rate:.1%} null values (threshold: {threshold:.0%})")
        elif null_rate > 0:
            result.add_warning(f"Column '{col}' has {null_rate:.1%} null values")


def _check_range(df: pd.DataFrame, col: str, min_val: Any, max_val: Any, result: ValidationResult) -> None:
    if col not in df.columns:
        return
    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
    if out_of_range > 0:
        result.add_warning(f"Column '{col}': {out_of_range} values outside [{min_val}, {max_val}]")


def _check_uniqueness(df: pd.DataFrame, col: str, result: ValidationResult) -> None:
    if col not in df.columns:
        return
    dup_count = df[col].duplicated().sum()
    if dup_count > 0:
        result.add_error(f"Column '{col}' has {dup_count} duplicate values (expected unique)")


# ---------------------------------------------------------------
# Per-dataset validators
# ---------------------------------------------------------------

def validate_sales(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("sales")
    required = ["date", "product_id", "store_id", "units_sold", "price"]
    _check_required_columns(df, required, result)
    _check_nulls(df, required, result)
    _check_range(df, "units_sold", 0, 100_000, result)
    _check_range(df, "price", 0, 1_000_000, result)
    _check_range(df, "discount_pct", 0, 100, result)
    if "date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            result.add_error("'date' column must be datetime type")
    result.log()
    return result


def validate_shipments(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("shipments")
    required = ["shipment_id", "supplier_id", "origin", "destination", "distance_km"]
    _check_required_columns(df, required, result)
    _check_uniqueness(df, "shipment_id", result)
    _check_nulls(df, required, result)
    _check_range(df, "distance_km", 1, 20_000, result)
    _check_range(df, "delay_hours", 0, 500, result)
    result.log()
    return result


def validate_suppliers(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("suppliers")
    required = ["supplier_id", "on_time_delivery_rate", "avg_delay_days", "defect_rate"]
    _check_required_columns(df, required, result)
    _check_uniqueness(df, "supplier_id", result)
    _check_nulls(df, required, result)
    _check_range(df, "on_time_delivery_rate", 0.0, 1.0, result)
    _check_range(df, "defect_rate", 0.0, 1.0, result)
    _check_range(df, "cancellation_rate", 0.0, 1.0, result)
    result.log()
    return result


def validate_inventory(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("inventory")
    required = ["date", "product_id", "warehouse_id", "current_stock", "reorder_point"]
    _check_required_columns(df, required, result)
    _check_nulls(df, required, result)
    _check_range(df, "current_stock", 0, 1_000_000, result)
    _check_range(df, "lead_time_days", 1, 365, result)
    result.log()
    return result


def validate_weather(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult("weather")
    required = ["date", "region", "temperature_c", "precipitation_mm", "weather_severity"]
    _check_required_columns(df, required, result)
    _check_nulls(df, required, result)
    _check_range(df, "temperature_c", -60, 60, result)
    _check_range(df, "precipitation_mm", 0, 500, result)
    _check_range(df, "weather_severity", 0, 3, result)
    result.log()
    return result


# ---------------------------------------------------------------
# Run all validations
# ---------------------------------------------------------------

def validate_all(datasets: dict[str, pd.DataFrame]) -> dict[str, ValidationResult]:
    """
    Validate all datasets. Raises RuntimeError if any critical validator fails.
    """
    validators = {
        "sales": validate_sales,
        "shipments": validate_shipments,
        "suppliers": validate_suppliers,
        "inventory": validate_inventory,
        "weather": validate_weather,
    }

    results: dict[str, ValidationResult] = {}
    for name, validator in validators.items():
        if name in datasets:
            results[name] = validator(datasets[name])

    failed = [name for name, r in results.items() if not r.passed]
    if failed:
        raise RuntimeError(f"Data validation failed for: {failed}. Check logs above.")

    logger.success("All datasets passed validation.")
    return results
