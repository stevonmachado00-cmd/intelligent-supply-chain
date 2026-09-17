"""
Feature engineering pipeline.

Transforms validated, raw datasets into model-ready feature tables.
Each model (demand, stockout, supplier_risk, eta, anomaly) has its
own feature builder that outputs a self-contained DataFrame.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
INDIA_HOLIDAYS = holidays.India()


# ---------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add standard calendar features from a date column."""
    d = df[date_col]
    df["day_of_week"] = d.dt.dayofweek
    df["day_of_month"] = d.dt.day
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = d.apply(lambda x: 1 if x in INDIA_HOLIDAYS else 0)
    # Days to next public holiday
    df["days_to_holiday"] = d.apply(_days_to_next_holiday)
    return df


def _days_to_next_holiday(date: pd.Timestamp, horizon: int = 30) -> int:
    for delta in range(1, horizon + 1):
        if (date + pd.Timedelta(days=delta)) in INDIA_HOLIDAYS:
            return delta
    return horizon


# ---------------------------------------------------------------
# Lag / rolling features
# ---------------------------------------------------------------

def _add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lag features for the target column within groups."""
    df = df.sort_values(group_cols + ["date"])
    for lag in lags:
        df[f"{target_col}_lag_{lag}d"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean/std/min/max features."""
    df = df.sort_values(group_cols + ["date"])
    for window in windows:
        rolled = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1)
        )
        df[f"{target_col}_roll_mean_{window}d"] = rolled.mean()
        df[f"{target_col}_roll_std_{window}d"] = rolled.std().fillna(0)
        df[f"{target_col}_roll_min_{window}d"] = rolled.min()
        df[f"{target_col}_roll_max_{window}d"] = rolled.max()
    return df


# ---------------------------------------------------------------
# Demand features
# ---------------------------------------------------------------

def build_demand_features(
    sales: pd.DataFrame,
    weather: pd.DataFrame,
    forecast_horizon: int = 7,
) -> pd.DataFrame:
    """
    Build features for the demand forecasting model.

    Target: units_sold (shifted -horizon for future prediction)
    """
    logger.info("Building demand features ...")

    df = sales.copy()
    # Merge weather on date + region
    df = df.merge(weather[["date", "region", "weather_severity", "temperature_c"]],
                  on=["date", "region"], how="left")
    df["weather_severity"] = df["weather_severity"].fillna(0)
    df["temperature_c"] = df["temperature_c"].fillna(28.0)

    # Calendar
    df = _add_calendar_features(df)

    # Lag features
    df = _add_lag_features(df, "units_sold", ["product_id", "store_id"], [1, 3, 7, 14, 21, 28])

    # Rolling features
    df = _add_rolling_features(df, "units_sold", ["product_id", "store_id"], [7, 14, 28])

    # Demand volatility (coefficient of variation)
    df["demand_cv_28d"] = (
        df["units_sold_roll_std_28d"] / (df["units_sold_roll_mean_28d"] + 1e-6)
    )

    # Target: demand N days ahead
    df["target"] = df.groupby(["product_id", "store_id"])["units_sold"].shift(-forecast_horizon)

    df.dropna(subset=["target"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.success(f"Demand features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------
# Stockout features
# ---------------------------------------------------------------

def build_stockout_features(
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
    suppliers: pd.DataFrame,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """
    Build features for the stockout prediction model.

    Target (binary): 1 if product goes out of stock within horizon_days, else 0.
    """
    logger.info("Building stockout features ...")

    # Aggregate sales to product+date level
    daily_demand = (
        sales.groupby(["date", "product_id"])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "daily_demand"})
    )

    df = inventory.merge(daily_demand, on=["date", "product_id"], how="left")
    df["daily_demand"] = df["daily_demand"].fillna(0)

    # Add supplier lead time from suppliers (use mean if multiple suppliers)
    if "supplier_id" in suppliers.columns and "lead_time_days" not in df.columns:
        supplier_avg = suppliers.groupby("supplier_id")["avg_delay_days"].mean()
        df["supplier_avg_delay"] = supplier_avg.mean()  # simplification
    else:
        df["supplier_avg_delay"] = 5.0

    # Rolling demand stats
    df = df.sort_values(["product_id", "warehouse_id", "date"])
    for window in [7, 30]:
        df[f"avg_demand_{window}d"] = df.groupby(["product_id", "warehouse_id"])["daily_demand"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"std_demand_{window}d"] = df.groupby(["product_id", "warehouse_id"])["daily_demand"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        ).fillna(0)

    df["days_of_supply"] = df["current_stock"] / (df["avg_demand_7d"] + 1e-6)
    df["demand_volatility"] = df["std_demand_30d"] / (df["avg_demand_30d"] + 1e-6)

    # Target: will stock reach 0 within horizon_days?
    df["future_demand_est"] = df["avg_demand_7d"] * horizon_days
    df["target"] = (
        df["current_stock"] + df["incoming_po_units"] < df["future_demand_est"]
    ).astype(int)

    feature_cols = [
        "current_stock", "reorder_point", "incoming_po_units", "lead_time_days",
        "daily_demand", "avg_demand_7d", "avg_demand_30d",
        "std_demand_7d", "std_demand_30d",
        "days_of_supply", "demand_volatility", "supplier_avg_delay",
        "target",
    ]
    df = df[[c for c in feature_cols if c in df.columns]].dropna()
    df.reset_index(drop=True, inplace=True)

    logger.success(f"Stockout features: {df.shape[0]:,} rows | Stockout rate: {df['target'].mean():.2%}")
    return df


# ---------------------------------------------------------------
# Supplier risk features
# ---------------------------------------------------------------

def build_supplier_risk_features(
    suppliers: pd.DataFrame,
    shipments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features for the supplier risk prediction model.

    Target (binary): 1 = high-risk supplier (risk_level == 'high').
    """
    logger.info("Building supplier risk features ...")

    # Aggregate shipment stats per supplier
    ship_agg = shipments.groupby("supplier_id").agg(
        shipment_count=("shipment_id", "count"),
        avg_delay_hours=("delay_hours", "mean"),
        std_delay_hours=("delay_hours", "std"),
        on_time_rate=("delay_hours", lambda x: (x <= 0).mean()),
    ).reset_index()

    df = suppliers.merge(ship_agg, on="supplier_id", how="left")
    df["avg_delay_hours"] = df["avg_delay_hours"].fillna(0)
    df["std_delay_hours"] = df["std_delay_hours"].fillna(0)
    df["on_time_rate"] = df["on_time_rate"].fillna(df["on_time_delivery_rate"])
    df["shipment_count"] = df["shipment_count"].fillna(0)

    # Encode region
    df["region_encoded"] = df["region"].astype("category").cat.codes
    df["category_encoded"] = df["category"].astype("category").cat.codes

    # Target
    df["target"] = (df["risk_level"] == "high").astype(int)

    feature_cols = [
        "on_time_delivery_rate", "avg_delay_days", "std_delay_days",
        "defect_rate", "cancellation_rate", "capacity_units_per_month",
        "price_per_unit", "region_encoded", "category_encoded",
        "avg_delay_hours", "std_delay_hours", "on_time_rate", "shipment_count",
        "target",
    ]
    df = df[[c for c in feature_cols if c in df.columns]].dropna()
    df.reset_index(drop=True, inplace=True)

    logger.success(f"Supplier risk features: {df.shape[0]:,} rows | High-risk rate: {df['target'].mean():.2%}")
    return df


# ---------------------------------------------------------------
# ETA features
# ---------------------------------------------------------------

def build_eta_features(
    shipments: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features for ETA (shipment duration) prediction.

    Target: actual shipment duration in hours.
    """
    logger.info("Building ETA features ...")

    df = shipments.copy()

    # Compute actual duration
    if "scheduled_arrival" in df.columns and "actual_arrival" in df.columns:
        df["duration_hours"] = (
            (df["actual_arrival"] - df["scheduled_arrival"]).dt.total_seconds() / 3600
        ).abs()
    else:
        df["duration_hours"] = df.get("delay_hours", 0) + df["distance_km"] / 60

    # Time features from scheduled_arrival
    if "scheduled_arrival" in df.columns:
        df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
        df["day_of_week"] = df["scheduled_arrival"].dt.dayofweek
        df["hour_of_day"] = df["scheduled_arrival"].dt.hour
        df["month"] = df["scheduled_arrival"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Encode categoricals
    df["vehicle_type_encoded"] = df["vehicle_type"].astype("category").cat.codes
    df["origin_encoded"] = df["origin"].astype("category").cat.codes
    df["dest_encoded"] = df["destination"].astype("category").cat.codes

    # Carrier on-time rate (computed from data)
    carrier_stats = df.groupby("carrier_id")["delay_hours"].agg(
        carrier_on_time_rate=lambda x: (x <= 0).mean()
    ).reset_index()
    df = df.merge(carrier_stats, on="carrier_id", how="left")

    # Historical avg/std per route (origin+dest)
    route_stats = df.groupby(["origin", "destination"])["duration_hours"].agg(
        hist_avg_dur="mean",
        hist_std_dur="std",
    ).reset_index()
    df = df.merge(route_stats, on=["origin", "destination"], how="left")
    df["hist_std_dur"] = df["hist_std_dur"].fillna(0)

    feature_cols = [
        "distance_km", "weight_kg", "vehicle_type_encoded",
        "day_of_week", "hour_of_day", "month", "is_weekend",
        "origin_encoded", "dest_encoded",
        "carrier_on_time_rate", "hist_avg_dur", "hist_std_dur",
        "duration_hours",  # target
    ]
    df = df[[c for c in feature_cols if c in df.columns]].dropna()
    df.rename(columns={"duration_hours": "target"}, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.success(f"ETA features: {df.shape[0]:,} rows")
    return df


# ---------------------------------------------------------------
# Anomaly detection features
# ---------------------------------------------------------------

def build_anomaly_features(shipments: pd.DataFrame) -> pd.DataFrame:
    """
    Build normalised feature matrix for the autoencoder anomaly detector.
    No target label -- unsupervised.
    """
    logger.info("Building anomaly detection features ...")

    df = shipments.copy()

    if "scheduled_arrival" in df.columns:
        df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
        df["day_of_week"] = df["scheduled_arrival"].dt.dayofweek
        df["month"] = df["scheduled_arrival"].dt.month
    else:
        df["day_of_week"] = 0
        df["month"] = 1

    df["vehicle_type_encoded"] = df["vehicle_type"].astype("category").cat.codes
    df["origin_encoded"] = df["origin"].astype("category").cat.codes
    df["dest_encoded"] = df["destination"].astype("category").cat.codes

    # Route historical stats for computing deviation
    route_stats = df.groupby(["origin", "destination"])["delay_hours"].agg(
        hist_avg_delay="mean",
        hist_std_delay="std",
    ).reset_index()
    df = df.merge(route_stats, on=["origin", "destination"], how="left")
    df["hist_std_delay"] = df["hist_std_delay"].fillna(1)
    df["delay_zscore"] = (
        (df["delay_hours"] - df["hist_avg_delay"]) / (df["hist_std_delay"] + 1e-6)
    )

    carrier_stats = df.groupby("carrier_id")["delay_hours"].agg(
        carrier_on_time_rate=lambda x: (x <= 0).mean()
    ).reset_index()
    df = df.merge(carrier_stats, on="carrier_id", how="left")

    feature_cols = [
        "distance_km", "weight_kg", "vehicle_type_encoded",
        "day_of_week", "month", "origin_encoded", "dest_encoded",
        "delay_hours", "hist_avg_delay", "hist_std_delay",
        "delay_zscore", "carrier_on_time_rate",
    ]
    df = df[[c for c in feature_cols if c in df.columns]].dropna()
    df.reset_index(drop=True, inplace=True)

    logger.success(f"Anomaly features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------
# Run all feature builders
# ---------------------------------------------------------------

def build_all_features(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build and save all feature tables."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    feature_tables = {
        "demand_features": build_demand_features(datasets["sales"], datasets["weather"]),
        "stockout_features": build_stockout_features(
            datasets["inventory"], datasets["sales"], datasets["suppliers"]
        ),
        "supplier_risk_features": build_supplier_risk_features(
            datasets["suppliers"], datasets["shipments"]
        ),
        "eta_features": build_eta_features(datasets["shipments"], datasets["weather"]),
        "anomaly_features": build_anomaly_features(datasets["shipments"]),
    }

    for name, df in feature_tables.items():
        out_path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"  Saved {name} -> {out_path}")

    return feature_tables


if __name__ == "__main__":
    from src.data.ingestion import run_ingestion
    datasets = run_ingestion()
    build_all_features(datasets)
