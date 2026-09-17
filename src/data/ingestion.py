"""
Data ingestion pipeline.

Loads raw datasets from data/raw/, merges them into a unified
enterprise supply-chain schema, and writes to data/processed/.

Data sources used:
  - M5 Forecasting (Walmart sales)        -> sales, products
  - USAID Supply Chain Shipment Pricing   -> shipments, suppliers, orders
  - Synthetic inventory + supplier perf   -> inventory, supplier_performance
  - NOAA Weather (via API or CSV stub)    -> weather
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_sales_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load and normalise the M5 Forecasting sales dataset (Walmart).

    If the file is not present, generates a realistic synthetic substitute
    so that the pipeline can run end-to-end without requiring a Kaggle
    account.  Replace with the real file when available.

    Returns
    -------
    pd.DataFrame with columns:
        date, product_id, store_id, region, units_sold, price,
        discount_pct, promotion_active
    """
    path = filepath or RAW_DIR / "m5_sales.csv"

    if Path(path).exists():
        logger.info(f"Loading sales data from {path}")
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        logger.warning(f"{path} not found. Generating synthetic sales data.")
        df = _generate_synthetic_sales()

    # Normalise column names
    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_shipment_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load USAID Supply Chain Shipment Pricing dataset.

    If the file is not present, generates synthetic shipment records.

    Returns
    -------
    pd.DataFrame with columns:
        shipment_id, order_id, supplier_id, origin, destination,
        distance_km, scheduled_arrival, actual_arrival, delay_hours,
        vehicle_type, carrier_id, weight_kg, status
    """
    path = filepath or RAW_DIR / "usaid_shipments.csv"

    if Path(path).exists():
        logger.info(f"Loading shipment data from {path}")
        df = pd.read_csv(path, parse_dates=["scheduled_arrival", "actual_arrival"])
    else:
        logger.warning(f"{path} not found. Generating synthetic shipment data.")
        df = _generate_synthetic_shipments()

    df.columns = df.columns.str.lower().str.strip()
    return df


def load_supplier_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load supplier performance dataset.

    Returns
    -------
    pd.DataFrame with columns:
        supplier_id, supplier_name, region, category,
        on_time_delivery_rate, avg_delay_days, defect_rate,
        cancellation_rate, capacity_units_per_month, price_per_unit
    """
    path = filepath or RAW_DIR / "suppliers.csv"

    if Path(path).exists():
        logger.info(f"Loading supplier data from {path}")
        df = pd.read_csv(path)
    else:
        logger.warning(f"{path} not found. Generating synthetic supplier data.")
        df = _generate_synthetic_suppliers()

    df.columns = df.columns.str.lower().str.strip()
    return df


def load_inventory_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load inventory snapshot data.

    Returns
    -------
    pd.DataFrame with columns:
        date, product_id, warehouse_id, current_stock,
        reorder_point, max_capacity, incoming_po_units, lead_time_days
    """
    path = filepath or RAW_DIR / "inventory.csv"

    if Path(path).exists():
        logger.info(f"Loading inventory data from {path}")
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        logger.warning(f"{path} not found. Generating synthetic inventory data.")
        df = _generate_synthetic_inventory()

    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_weather_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load weather data (NOAA CSV export or synthetic stub).

    Returns
    -------
    pd.DataFrame with columns:
        date, region, temperature_c, precipitation_mm,
        weather_severity  (0=clear, 1=light, 2=moderate, 3=severe)
    """
    path = filepath or RAW_DIR / "weather.csv"

    if Path(path).exists():
        logger.info(f"Loading weather data from {path}")
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        logger.warning(f"{path} not found. Generating synthetic weather data.")
        df = _generate_synthetic_weather()

    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------
# Synthetic data generators  (used when real files are absent)
# ---------------------------------------------------------------

NP_SEED = 42
rng = np.random.default_rng(NP_SEED)

PRODUCT_IDS = [f"PROD_{i:04d}" for i in range(1, 51)]
STORE_IDS = [f"STORE_{i:02d}" for i in range(1, 11)]
REGIONS = ["Mumbai", "Pune", "Delhi", "Chennai", "Hyderabad", "Bangalore"]
SUPPLIER_IDS = [f"SUP_{i:03d}" for i in range(1, 21)]
WAREHOUSE_IDS = [f"WH_{i:02d}" for i in range(1, 6)]
VEHICLE_TYPES = ["Truck", "Van", "Rail", "Air"]


def _date_range(start: str = "2022-01-01", periods: int = 730) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="D")


def _generate_synthetic_sales() -> pd.DataFrame:
    dates = _date_range(periods=730)
    records = []
    for product_id in PRODUCT_IDS[:20]:  # 20 products x 730 days = 14,600 rows
        for store_id in STORE_IDS[:5]:
            base_demand = rng.integers(50, 300)
            for date in dates:
                # Seasonality + trend + noise
                dow_factor = 1.0 + 0.15 * (date.dayofweek >= 5)  # weekend boost
                month_factor = 1.0 + 0.10 * np.sin(2 * np.pi * date.month / 12)
                noise = rng.normal(1.0, 0.05)
                units_sold = max(0, int(base_demand * dow_factor * month_factor * noise))
                records.append(
                    {
                        "date": date,
                        "product_id": product_id,
                        "store_id": store_id,
                        "region": rng.choice(REGIONS),
                        "units_sold": units_sold,
                        "price": round(rng.uniform(500, 50000), 2),
                        "discount_pct": round(rng.choice([0, 5, 10, 15, 20]), 1),
                        "promotion_active": int(rng.random() < 0.1),
                    }
                )
    return pd.DataFrame(records)


def _generate_synthetic_shipments(n: int = 8000) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01")
    records = []
    for i in range(n):
        supplier_id = rng.choice(SUPPLIER_IDS)
        origin = rng.choice(REGIONS)
        destination = rng.choice([r for r in REGIONS if r != origin])
        distance_km = rng.integers(50, 2000)
        # Base travel time + weather/traffic noise
        base_hours = distance_km / rng.uniform(40, 80)
        delay_hours = max(0.0, float(rng.normal(0, base_hours * 0.2)))
        scheduled = start + pd.Timedelta(days=int(rng.integers(0, 700)))
        actual = scheduled + pd.Timedelta(hours=base_hours + delay_hours)
        records.append(
            {
                "shipment_id": f"SHIP_{i + 1:06d}",
                "order_id": f"ORD_{rng.integers(1, 5000):06d}",
                "supplier_id": supplier_id,
                "origin": origin,
                "destination": destination,
                "distance_km": distance_km,
                "scheduled_arrival": scheduled,
                "actual_arrival": actual,
                "delay_hours": round(delay_hours, 2),
                "vehicle_type": rng.choice(VEHICLE_TYPES),
                "carrier_id": f"CARRIER_{rng.integers(1, 10):02d}",
                "weight_kg": round(rng.uniform(100, 20000), 1),
                "status": rng.choice(["delivered", "in_transit", "delayed"], p=[0.75, 0.15, 0.10]),
            }
        )
    return pd.DataFrame(records)


def _generate_synthetic_suppliers() -> pd.DataFrame:
    records = []
    for sup_id in SUPPLIER_IDS:
        # Draw base performance params
        risk_level = rng.choice(["low", "medium", "high"], p=[0.5, 0.3, 0.2])
        if risk_level == "low":
            on_time_rate = rng.uniform(0.90, 0.99)
            avg_delay = rng.uniform(0.1, 1.0)
            defect_rate = rng.uniform(0.005, 0.02)
            cancel_rate = rng.uniform(0.005, 0.01)
        elif risk_level == "medium":
            on_time_rate = rng.uniform(0.75, 0.90)
            avg_delay = rng.uniform(1.0, 3.0)
            defect_rate = rng.uniform(0.02, 0.05)
            cancel_rate = rng.uniform(0.01, 0.03)
        else:  # high
            on_time_rate = rng.uniform(0.50, 0.75)
            avg_delay = rng.uniform(3.0, 10.0)
            defect_rate = rng.uniform(0.05, 0.12)
            cancel_rate = rng.uniform(0.03, 0.08)

        records.append(
            {
                "supplier_id": sup_id,
                "supplier_name": f"Supplier {sup_id.split('_')[1]}",
                "region": rng.choice(REGIONS),
                "category": rng.choice(["Electronics", "FMCG", "Apparel", "Industrial"]),
                "on_time_delivery_rate": round(on_time_rate, 4),
                "avg_delay_days": round(avg_delay, 2),
                "std_delay_days": round(avg_delay * rng.uniform(0.2, 0.5), 2),
                "defect_rate": round(defect_rate, 4),
                "cancellation_rate": round(cancel_rate, 4),
                "capacity_units_per_month": int(rng.integers(500, 10000)),
                "price_per_unit": round(rng.uniform(80, 150), 2),
                "risk_level": risk_level,
            }
        )
    return pd.DataFrame(records)


def _generate_synthetic_inventory() -> pd.DataFrame:
    dates = _date_range(periods=730)
    records = []
    for product_id in PRODUCT_IDS[:20]:
        for warehouse_id in WAREHOUSE_IDS[:3]:
            stock = int(rng.integers(200, 5000))
            reorder_point = int(rng.integers(100, 500))
            max_capacity = int(rng.integers(5000, 20000))
            for date in dates:
                daily_demand = int(rng.integers(10, 200))
                stock = max(0, stock - daily_demand + int(rng.integers(0, 150)))
                records.append(
                    {
                        "date": date,
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "current_stock": stock,
                        "reorder_point": reorder_point,
                        "max_capacity": max_capacity,
                        "incoming_po_units": int(rng.choice([0, 0, 0, 500, 1000, 2000])),
                        "lead_time_days": int(rng.integers(3, 21)),
                    }
                )
    return pd.DataFrame(records)


def _generate_synthetic_weather() -> pd.DataFrame:
    dates = _date_range(periods=730)
    records = []
    for region in REGIONS:
        for date in dates:
            temp = rng.normal(28, 8)  # India-like temperatures
            precip = max(0.0, float(rng.exponential(5)))
            if precip < 2:
                severity = 0
            elif precip < 10:
                severity = 1
            elif precip < 25:
                severity = 2
            else:
                severity = 3
            records.append(
                {
                    "date": date,
                    "region": region,
                    "temperature_c": round(temp, 1),
                    "precipitation_mm": round(precip, 1),
                    "weather_severity": severity,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------

def run_ingestion() -> dict[str, pd.DataFrame]:
    """Run the full ingestion pipeline and return all DataFrames."""
    logger.info("Starting data ingestion pipeline ...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "sales": load_sales_data(),
        "shipments": load_shipment_data(),
        "suppliers": load_supplier_data(),
        "inventory": load_inventory_data(),
        "weather": load_weather_data(),
    }

    for name, df in datasets.items():
        out_path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.success(f"  Saved {name}: {df.shape[0]:,} rows -> {out_path}")

    logger.success("Ingestion complete.")
    return datasets


if __name__ == "__main__":
    run_ingestion()
