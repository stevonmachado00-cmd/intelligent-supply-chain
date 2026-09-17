# Data Schema — Intelligent Supply Chain Control Tower

## Overview

This document describes the unified enterprise data schema used across
all ML models in the Supply Chain Control Tower. Each table maps to
one or more real public datasets; synthetic generation fills gaps.

---

## Data Sources

| Dataset | Source | Populates |
|---|---|---|
| M5 Forecasting (Walmart) | Kaggle / data/raw/m5_sales.csv | sales, products |
| USAID Shipment Pricing | data.usaid.gov / data/raw/usaid_shipments.csv | shipments, orders |
| NOAA Weather Summaries | ncei.noaa.gov / data/raw/weather.csv | weather |
| Synthetic (rule-based) | src/data/ingestion.py | suppliers, inventory |

---

## Table: sales

Granularity: one row per (product, store, date).

| Column | Type | Description |
|---|---|---|
| date | DATE | Calendar date |
| product_id | VARCHAR | Product identifier (e.g. PROD_0001) |
| store_id | VARCHAR | Store / warehouse identifier |
| region | VARCHAR | Geographic region |
| units_sold | INTEGER | Units sold that day |
| price | FLOAT | Unit selling price (INR) |
| discount_pct | FLOAT | Discount percentage applied (0–100) |
| promotion_active | INTEGER | 1 if promotion was running, else 0 |

Source: M5 Forecasting dataset (item_id ? product_id, sales ? units_sold).

---

## Table: shipments

Granularity: one row per shipment.

| Column | Type | Description |
|---|---|---|
| shipment_id | VARCHAR PK | Unique shipment identifier |
| order_id | VARCHAR | Associated order |
| supplier_id | VARCHAR FK | Supplier who dispatched |
| origin | VARCHAR | Origin city/region |
| destination | VARCHAR | Destination city/region |
| distance_km | FLOAT | Road/air distance |
| scheduled_arrival | TIMESTAMP | Expected arrival datetime |
| actual_arrival | TIMESTAMP | Actual arrival datetime |
| delay_hours | FLOAT | actual - scheduled in hours (>0 = late) |
| vehicle_type | VARCHAR | Truck / Van / Rail / Air |
| carrier_id | VARCHAR | Carrier company identifier |
| weight_kg | FLOAT | Shipment weight |
| status | VARCHAR | delivered / in_transit / delayed |

Source: USAID Supply Chain Shipment Pricing (mapped to schema above).

---

## Table: suppliers

Granularity: one row per supplier.

| Column | Type | Description |
|---|---|---|
| supplier_id | VARCHAR PK | Unique supplier identifier |
| supplier_name | VARCHAR | Display name |
| region | VARCHAR | Supplier headquarters region |
| category | VARCHAR | Product category supplied |
| on_time_delivery_rate | FLOAT | Historical on-time rate (0–1) |
| avg_delay_days | FLOAT | Mean delay in days |
| std_delay_days | FLOAT | Std deviation of delay |
| defect_rate | FLOAT | Defective unit fraction (0–1) |
| cancellation_rate | FLOAT | Order cancellation fraction (0–1) |
| capacity_units_per_month | INTEGER | Max monthly supply capacity |
| price_per_unit | FLOAT | Unit procurement cost (INR) |
| risk_level | VARCHAR | low / medium / high (ground truth label) |

Source: Synthetic, generated with parameterised risk profiles in src/data/ingestion.py.

---

## Table: inventory

Granularity: one row per (product, warehouse, date).

| Column | Type | Description |
|---|---|---|
| date | DATE | Snapshot date |
| product_id | VARCHAR FK | Product identifier |
| warehouse_id | VARCHAR | Warehouse identifier |
| current_stock | INTEGER | Units on hand |
| reorder_point | INTEGER | Stock level triggering reorder |
| max_capacity | INTEGER | Warehouse max capacity for this SKU |
| incoming_po_units | INTEGER | Units in open purchase orders |
| lead_time_days | INTEGER | Supplier lead time in days |

Source: Synthetic, simulated as random walk depleted by daily demand.

---

## Table: weather

Granularity: one row per (region, date).

| Column | Type | Description |
|---|---|---|
| date | DATE | Calendar date |
| region | VARCHAR | City/region name |
| temperature_c | FLOAT | Mean daily temperature (Celsius) |
| precipitation_mm | FLOAT | Total daily precipitation (mm) |
| weather_severity | INTEGER | 0=clear, 1=light, 2=moderate, 3=severe |

Source: NOAA GHCND daily summaries (or synthetic stub if unavailable).

---

## Feature Tables (ML-ready, in data/processed/)

| File | Built from | Used by |
|---|---|---|
| demand_features.parquet | sales + weather | XGBoost Demand, LSTM Demand |
| stockout_features.parquet | inventory + sales + suppliers | XGBoost Stockout |
| supplier_risk_features.parquet | suppliers + shipments | XGBoost Supplier Risk |
| eta_features.parquet | shipments + weather | XGBoost ETA |
| nomaly_features.parquet | shipments | PyTorch Autoencoder |

---

## Assumptions

1. All monetary values are in Indian Rupees (INR) unless stated otherwise.
2. Regions correspond to major Indian cities: Mumbai, Pune, Delhi, Chennai, Hyderabad, Bangalore.
3. Products are electronics/FMCG SKUs; the schema is product-category-agnostic.
4. Synthetic data is generated with a fixed random seed (42) for reproducibility.
5. Weather severity thresholds: light (<10mm), moderate (10–25mm), severe (>25mm precipitation).
