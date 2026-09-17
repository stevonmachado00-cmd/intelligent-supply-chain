"""
Data preparation pipeline for the Intelligent Supply Chain Control Tower.

Transforms the UCI Online Retail dataset (541,909 real transactions)
into 5 model-ready feature tables:

  1. demand_features.parquet     -> Demand Forecasting (XGBoost + LSTM)
  2. stockout_features.parquet   -> Stockout Prediction (Calibrated XGBoost)
  3. supplier_risk_features.parquet -> Supplier Risk (Calibrated XGBoost)
  4. eta_features.parquet        -> Shipment ETA Prediction (XGBoost)
  5. anomaly_features.parquet    -> Anomaly Detection (Autoencoder)

The UCI dataset provides real sales transactions. From these we derive
realistic supplier, shipment, and inventory datasets using domain rules
grounded in the actual sales velocity patterns.

Run from project root:
    python scripts/prepare_real_data.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# =================================================================
# Step 1: Load & Clean UCI Online Retail
# =================================================================

def load_and_clean_uci() -> pd.DataFrame:
    """Load UCI Online Retail xlsx, clean, and return valid sales."""
    logger.info("Loading UCI Online Retail dataset...")
    df = pd.read_excel(RAW_DIR / "Online Retail.xlsx")
    logger.info(f"  Raw shape: {df.shape}")

    # 1. Remove cancellations / returns (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    logger.info(f"  After removing cancellations: {df.shape}")

    # 2. Remove non-product stock codes (postage, discounts, fees)
    bad_codes = {"POST", "D", "DOT", "M", "S", "AMAZONFEE", "BANK CHARGES",
                 "CRUK", "C2", "m", "PADS"}
    df = df[~df["StockCode"].astype(str).str.upper().isin(bad_codes)]

    # 3. Remove rows with negative or zero quantity/price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    logger.info(f"  After removing invalid qty/price: {df.shape}")

    # 4. Extract date
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()

    # 5. Add revenue
    df["revenue"] = df["Quantity"] * df["UnitPrice"]

    logger.success(f"  Clean dataset: {df.shape[0]:,} rows")
    return df


# =================================================================
# Step 2: Aggregate to Daily SKU Demand
# =================================================================

def aggregate_daily_demand(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Aggregate to daily product-level demand for top N products.
    Zero-fill missing dates to create continuous time series.
    """
    logger.info("Aggregating daily demand...")

    # Focus on United Kingdom (82% of data) for clean signal
    uk = df[df["Country"] == "United Kingdom"].copy()

    # Find top N products by total volume
    top_products = (
        uk.groupby("StockCode")["Quantity"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    uk = uk[uk["StockCode"].isin(top_products)]

    # Aggregate per product per day
    daily = (
        uk.groupby(["date", "StockCode"])
        .agg(
            units_sold=("Quantity", "sum"),
            revenue=("revenue", "sum"),
            avg_price=("UnitPrice", "mean"),
            num_transactions=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    # Zero-fill: create full date x product grid
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [all_dates, top_products], names=["date", "StockCode"]
    )
    daily = daily.set_index(["date", "StockCode"]).reindex(full_index, fill_value=0).reset_index()

    # Forward-fill price (if 0, use last known price)
    daily["avg_price"] = daily.groupby("StockCode")["avg_price"].transform(
        lambda x: x.replace(0, np.nan).ffill().bfill()
    )

    logger.success(f"  Daily demand: {daily.shape[0]:,} rows ({len(top_products)} products)")
    return daily


# =================================================================
# Step 3: Build Demand Features
# =================================================================

def build_demand_features(daily: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Build features for demand forecasting model."""
    logger.info("Building demand features...")

    df = daily.sort_values(["StockCode", "date"]).copy()

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Lag features (shifted by 1 to avoid leakage) ---
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"demand_lag_{lag}d"] = df.groupby("StockCode")["units_sold"].shift(lag)

    # --- Rolling features (shifted by 1) ---
    for window in [7, 14, 28]:
        rolled = df.groupby("StockCode")["units_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"demand_roll_mean_{window}d"] = rolled

        rolled_std = df.groupby("StockCode")["units_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f"demand_roll_std_{window}d"] = rolled_std.fillna(0)

        rolled_max = df.groupby("StockCode")["units_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
        df[f"demand_roll_max_{window}d"] = rolled_max

        rolled_min = df.groupby("StockCode")["units_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).min()
        )
        df[f"demand_roll_min_{window}d"] = rolled_min

    # --- Demand volatility (coefficient of variation) ---
    df["demand_cv_28d"] = df["demand_roll_std_28d"] / (df["demand_roll_mean_28d"] + 1e-6)

    # --- Price ratio ---
    price_ma = df.groupby("StockCode")["avg_price"].transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).mean()
    )
    df["price_ratio"] = df["avg_price"] / (price_ma + 1e-6)

    # --- Product encoding ---
    df["product_encoded"] = df["StockCode"].astype("category").cat.codes

    # --- Target: cumulative demand over next H days ---
    df["target"] = df.groupby("StockCode")["units_sold"].transform(
        lambda x: x.shift(-1).rolling(horizon, min_periods=horizon).sum()
    )

    # Drop rows with NaN target or features
    df = df.dropna(subset=["target", "demand_lag_28d"]).reset_index(drop=True)

    logger.success(f"  Demand features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# =================================================================
# Step 4: Generate Derived Datasets from Real Sales Patterns
# =================================================================

def generate_suppliers(daily: pd.DataFrame) -> pd.DataFrame:
    """Generate supplier dataset: each top product is sourced from a supplier."""
    logger.info("Generating suppliers from product patterns...")

    products = daily["StockCode"].unique()
    n_suppliers = 50
    supplier_ids = [f"SUP_{i:03d}" for i in range(1, n_suppliers + 1)]

    records = []
    # Ensure balanced risk distribution across suppliers
    risk_labels = ["low"] * 25 + ["medium"] * 15 + ["high"] * 10
    rng.shuffle(risk_labels)

    for i, (sup_id, risk) in enumerate(zip(supplier_ids, risk_labels)):
        if risk == "low":
            on_time = rng.uniform(0.90, 0.99)
            avg_delay = rng.uniform(0.1, 1.0)
            defect = rng.uniform(0.005, 0.02)
        elif risk == "medium":
            on_time = rng.uniform(0.75, 0.90)
            avg_delay = rng.uniform(1.0, 3.0)
            defect = rng.uniform(0.02, 0.05)
        else:
            on_time = rng.uniform(0.50, 0.75)
            avg_delay = rng.uniform(3.0, 10.0)
            defect = rng.uniform(0.05, 0.12)

        records.append({
            "supplier_id": sup_id,
            "supplier_name": f"Supplier {i + 1}",
            "region": rng.choice(["London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Edinburgh"]),
            "category": rng.choice(["Gifts", "Homeware", "Novelties", "Kitchen", "Garden"]),
            "on_time_delivery_rate": round(on_time, 4),
            "avg_delay_days": round(avg_delay, 2),
            "std_delay_days": round(avg_delay * rng.uniform(0.2, 0.5), 2),
            "defect_rate": round(defect, 4),
            "cancellation_rate": round(rng.uniform(0.005, 0.08), 4),
            "capacity_units_per_month": int(rng.integers(500, 10000)),
            "price_per_unit": round(rng.uniform(1.0, 15.0), 2),
            "risk_level": risk,
        })

    suppliers = pd.DataFrame(records)
    logger.success(f"  Suppliers: {len(suppliers)} rows")
    return suppliers


def generate_shipments(daily: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    """Generate shipment dataset from real order dates and volumes."""
    logger.info("Generating shipments from real order patterns...")

    cities = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow",
              "Edinburgh", "Liverpool", "Bristol", "Cardiff", "Belfast"]
    vehicles = ["Truck", "Van", "Rail", "Air"]
    carriers = [f"CARRIER_{i:02d}" for i in range(1, 8)]

    # Sample actual dates from real data
    order_dates = daily[daily["units_sold"] > 0]["date"].values
    n_shipments = min(8000, len(order_dates))
    sampled_dates = rng.choice(order_dates, size=n_shipments, replace=True)

    records = []
    for i, ship_date in enumerate(sampled_dates):
        origin = rng.choice(cities)
        dest = rng.choice([c for c in cities if c != origin])
        distance = rng.integers(50, 600)
        sup = rng.choice(suppliers["supplier_id"].values)

        # Retrieve supplier risk
        sup_row = suppliers[suppliers["supplier_id"] == sup].iloc[0]
        base_hours = distance / rng.uniform(40, 70)

        # Delay correlated with supplier risk
        if sup_row["risk_level"] == "high":
            delay = max(0.0, float(rng.normal(base_hours * 0.3, base_hours * 0.15)))
        elif sup_row["risk_level"] == "medium":
            delay = max(0.0, float(rng.normal(base_hours * 0.1, base_hours * 0.1)))
        else:
            delay = max(0.0, float(rng.normal(0, base_hours * 0.05)))

        scheduled = pd.Timestamp(ship_date)
        actual = scheduled + pd.Timedelta(hours=base_hours + delay)

        records.append({
            "shipment_id": f"SHIP_{i + 1:06d}",
            "order_id": f"ORD_{rng.integers(1, 5000):06d}",
            "supplier_id": sup,
            "origin": origin,
            "destination": dest,
            "distance_km": int(distance),
            "scheduled_arrival": scheduled,
            "actual_arrival": actual,
            "duration_hours": round(base_hours + delay, 2),
            "delay_hours": round(delay, 2),
            "vehicle_type": rng.choice(vehicles),
            "carrier_id": rng.choice(carriers),
            "weight_kg": round(float(rng.uniform(50, 5000)), 1),
            "status": rng.choice(["delivered", "in_transit", "delayed"], p=[0.75, 0.15, 0.10]),
        })

    shipments = pd.DataFrame(records)
    logger.success(f"  Shipments: {len(shipments):,} rows")
    return shipments


def generate_inventory(daily: pd.DataFrame) -> pd.DataFrame:
    """Generate inventory snapshots from real daily demand velocity."""
    logger.info("Generating inventory from real demand patterns...")

    products = daily["StockCode"].unique()[:30]  # Top 30 products
    warehouses = ["WH_01", "WH_02", "WH_03"]
    dates = sorted(daily["date"].unique())

    records = []
    for product in products:
        prod_demand = daily[daily["StockCode"] == product].set_index("date")["units_sold"]
        for wh in warehouses:
            stock = int(rng.integers(200, 3000))
            reorder_point = int(rng.integers(50, 300))
            max_capacity = int(rng.integers(3000, 15000))

            for date in dates:
                demand = int(prod_demand.get(date, 0))
                # Simulate stock changes
                incoming = int(rng.choice([0, 0, 0, 0, 200, 500, 1000]))
                stock = max(0, stock - demand + incoming)

                records.append({
                    "date": date,
                    "product_id": product,
                    "warehouse_id": wh,
                    "current_stock": stock,
                    "reorder_point": reorder_point,
                    "max_capacity": max_capacity,
                    "incoming_po_units": incoming,
                    "lead_time_days": int(rng.integers(3, 14)),
                })

    inventory = pd.DataFrame(records)
    logger.success(f"  Inventory: {inventory.shape[0]:,} rows")
    return inventory


# =================================================================
# Step 5: Build Feature Tables for All Models
# =================================================================

def build_stockout_features(
    inventory: pd.DataFrame,
    daily: pd.DataFrame,
    horizon: int = 7,
) -> pd.DataFrame:
    """Build stockout prediction features."""
    logger.info("Building stockout features...")

    # Daily demand per product
    prod_demand = (
        daily.groupby(["date", "StockCode"])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"StockCode": "product_id", "units_sold": "daily_demand"})
    )

    df = inventory.merge(prod_demand, on=["date", "product_id"], how="left")
    df["daily_demand"] = df["daily_demand"].fillna(0)
    df = df.sort_values(["product_id", "warehouse_id", "date"])

    # Rolling demand stats
    for w in [7, 30]:
        df[f"avg_demand_{w}d"] = df.groupby(["product_id", "warehouse_id"])["daily_demand"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"std_demand_{w}d"] = df.groupby(["product_id", "warehouse_id"])["daily_demand"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0)

    df["days_of_supply"] = df["current_stock"] / (df["avg_demand_7d"] + 1e-6)
    df["demand_volatility"] = df["std_demand_30d"] / (df["avg_demand_30d"] + 1e-6)

    # Target: stock goes to 0 within horizon
    df["future_demand_est"] = df["avg_demand_7d"] * horizon
    df["target"] = (df["current_stock"] + df["incoming_po_units"] < df["future_demand_est"]).astype(int)

    feat_cols = [
        "current_stock", "reorder_point", "incoming_po_units", "lead_time_days",
        "daily_demand", "avg_demand_7d", "avg_demand_30d",
        "std_demand_7d", "std_demand_30d", "days_of_supply", "demand_volatility",
        "target",
    ]
    df = df[[c for c in feat_cols if c in df.columns]].dropna().reset_index(drop=True)

    logger.success(f"  Stockout features: {df.shape[0]:,} rows | Stockout rate: {df['target'].mean():.2%}")
    return df


def build_supplier_risk_features(
    suppliers: pd.DataFrame,
    shipments: pd.DataFrame,
) -> pd.DataFrame:
    """Build supplier risk prediction features."""
    logger.info("Building supplier risk features...")

    # Aggregate shipment stats per supplier
    ship_agg = shipments.groupby("supplier_id").agg(
        shipment_count=("shipment_id", "count"),
        avg_delay_hours=("delay_hours", "mean"),
        std_delay_hours=("delay_hours", "std"),
        ship_on_time_rate=("delay_hours", lambda x: (x <= 1.0).mean()),
    ).reset_index()

    df = suppliers.merge(ship_agg, on="supplier_id", how="left")
    df["avg_delay_hours"] = df["avg_delay_hours"].fillna(0)
    df["std_delay_hours"] = df["std_delay_hours"].fillna(0)
    df["ship_on_time_rate"] = df["ship_on_time_rate"].fillna(df["on_time_delivery_rate"])
    df["shipment_count"] = df["shipment_count"].fillna(0)

    df["region_encoded"] = df["region"].astype("category").cat.codes
    df["category_encoded"] = df["category"].astype("category").cat.codes

    df["target"] = (df["risk_level"] == "high").astype(int)

    feat_cols = [
        "on_time_delivery_rate", "avg_delay_days", "std_delay_days",
        "defect_rate", "cancellation_rate", "capacity_units_per_month",
        "price_per_unit", "region_encoded", "category_encoded",
        "avg_delay_hours", "std_delay_hours", "ship_on_time_rate", "shipment_count",
        "target",
    ]
    df = df[[c for c in feat_cols if c in df.columns]].dropna().reset_index(drop=True)

    logger.success(f"  Supplier risk features: {df.shape[0]:,} rows | High-risk rate: {df['target'].mean():.2%}")
    return df


def build_eta_features(shipments: pd.DataFrame) -> pd.DataFrame:
    """Build shipment ETA prediction features."""
    logger.info("Building ETA features...")

    df = shipments.copy()
    df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
    df["day_of_week"] = df["scheduled_arrival"].dt.dayofweek
    df["hour_of_day"] = df["scheduled_arrival"].dt.hour
    df["month"] = df["scheduled_arrival"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["vehicle_type_encoded"] = df["vehicle_type"].astype("category").cat.codes
    df["origin_encoded"] = df["origin"].astype("category").cat.codes
    df["dest_encoded"] = df["destination"].astype("category").cat.codes

    # Carrier stats
    carrier_stats = df.groupby("carrier_id")["delay_hours"].agg(
        carrier_on_time_rate=lambda x: (x <= 1.0).mean()
    ).reset_index()
    df = df.merge(carrier_stats, on="carrier_id", how="left")

    # Route historical stats
    route_stats = df.groupby(["origin", "destination"])["duration_hours"].agg(
        hist_avg_dur="mean", hist_std_dur="std"
    ).reset_index()
    df = df.merge(route_stats, on=["origin", "destination"], how="left")
    df["hist_std_dur"] = df["hist_std_dur"].fillna(0)

    feat_cols = [
        "distance_km", "weight_kg", "vehicle_type_encoded",
        "day_of_week", "hour_of_day", "month", "is_weekend",
        "origin_encoded", "dest_encoded",
        "carrier_on_time_rate", "hist_avg_dur", "hist_std_dur",
    ]
    df["target"] = df["duration_hours"]
    df = df[[c for c in feat_cols + ["target"] if c in df.columns]].dropna().reset_index(drop=True)

    logger.success(f"  ETA features: {df.shape[0]:,} rows")
    return df


def build_anomaly_features(shipments: pd.DataFrame) -> pd.DataFrame:
    """Build anomaly detection features (unsupervised)."""
    logger.info("Building anomaly features...")

    df = shipments.copy()
    df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
    df["day_of_week"] = df["scheduled_arrival"].dt.dayofweek
    df["month"] = df["scheduled_arrival"].dt.month

    df["vehicle_type_encoded"] = df["vehicle_type"].astype("category").cat.codes
    df["origin_encoded"] = df["origin"].astype("category").cat.codes
    df["dest_encoded"] = df["destination"].astype("category").cat.codes

    route_stats = df.groupby(["origin", "destination"])["delay_hours"].agg(
        hist_avg_delay="mean", hist_std_delay="std"
    ).reset_index()
    df = df.merge(route_stats, on=["origin", "destination"], how="left")
    df["hist_std_delay"] = df["hist_std_delay"].fillna(1)
    df["delay_zscore"] = (df["delay_hours"] - df["hist_avg_delay"]) / (df["hist_std_delay"] + 1e-6)

    carrier_stats = df.groupby("carrier_id")["delay_hours"].agg(
        carrier_on_time_rate=lambda x: (x <= 1.0).mean()
    ).reset_index()
    df = df.merge(carrier_stats, on="carrier_id", how="left")

    feat_cols = [
        "distance_km", "weight_kg", "vehicle_type_encoded",
        "day_of_week", "month", "origin_encoded", "dest_encoded",
        "delay_hours", "duration_hours", "hist_avg_delay", "hist_std_delay",
        "delay_zscore", "carrier_on_time_rate",
    ]
    df = df[[c for c in feat_cols if c in df.columns]].dropna().reset_index(drop=True)

    logger.success(f"  Anomaly features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


# =================================================================
# Main
# =================================================================

def main():
    logger.info("=" * 65)
    logger.info("SUPPLY CHAIN CONTROL TOWER - Data Preparation Pipeline")
    logger.info("=" * 65)

    # Step 1: Load & clean
    raw_df = load_and_clean_uci()

    # Step 2: Aggregate daily demand
    daily = aggregate_daily_demand(raw_df, top_n=50)

    # Step 3: Demand features
    demand_features = build_demand_features(daily, horizon=7)

    # Step 4: Generate derived datasets
    suppliers = generate_suppliers(daily)
    shipments = generate_shipments(daily, suppliers)
    inventory = generate_inventory(daily)

    # Step 5: Build all feature tables
    stockout_features = build_stockout_features(inventory, daily)
    supplier_risk_features = build_supplier_risk_features(suppliers, shipments)
    eta_features = build_eta_features(shipments)
    anomaly_features = build_anomaly_features(shipments)

    # Save everything
    tables = {
        "demand_features": demand_features,
        "stockout_features": stockout_features,
        "supplier_risk_features": supplier_risk_features,
        "eta_features": eta_features,
        "anomaly_features": anomaly_features,
        "daily_demand": daily,
        "suppliers": suppliers,
        "shipments": shipments,
        "inventory": inventory,
    }

    for name, df in tables.items():
        # Ensure object columns with mixed types like StockCode are converted to string
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        out = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        logger.success(f"Saved {name}: {df.shape[0]:,} rows -> {out}")

    logger.success("=" * 65)
    logger.success("Data preparation complete! All feature tables saved.")
    logger.success("=" * 65)


if __name__ == "__main__":
    main()
