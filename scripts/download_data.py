"""
Data download script.

This script provides instructions and optional automated download
for all datasets used in the Supply Chain Control Tower project.

Datasets:
  1. M5 Forecasting Competition (Walmart sales)
     Source: https://www.kaggle.com/competitions/m5-forecasting-accuracy
     Note:  Requires Kaggle account + kaggle CLI
     File:  data/raw/m5_sales.csv

  2. USAID Supply Chain Shipment Pricing
     Source: https://data.usaid.gov/Global-Health-Supply-Chain-Logistics/
             Supply-Chain-Shipment-Pricing-Data/a3rc-nmf6
     Note:  Publicly available (no auth required)
     File:  data/raw/usaid_shipments.csv

  3. NOAA Daily Weather Summaries
     Source: https://www.ncei.noaa.gov/access/services/data/v1
     Note:  Free API token required (https://www.ncdc.noaa.gov/cdo-web/token)
     File:  data/raw/weather.csv

If files cannot be downloaded, the pipeline auto-generates realistic
synthetic equivalents.  See src/data/ingestion.py for details.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_usaid_shipments() -> bool:
    """
    Download USAID Supply Chain Shipment Pricing dataset (CSV export).
    Returns True if successful.
    """
    url = (
        "https://data.usaid.gov/api/views/a3rc-nmf6/rows.csv?accessType=DOWNLOAD"
    )
    out_path = RAW_DIR / "usaid_shipments.csv"

    if out_path.exists():
        logger.info(f"USAID shipments already exists at {out_path}. Skipping download.")
        return True

    logger.info(f"Downloading USAID shipments from {url} ...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            desc="usaid_shipments.csv",
            total=total,
            unit="B",
            unit_scale=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        logger.success(f"Saved USAID shipments to {out_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not download USAID shipments: {e}")
        logger.warning("Synthetic data will be generated automatically during ingestion.")
        return False


def print_manual_download_instructions() -> None:
    print("""
========================================================
MANUAL DATASET DOWNLOAD INSTRUCTIONS
========================================================

1. M5 Forecasting (Walmart Sales)
   URL: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
   Steps:
     a) Create a Kaggle account
     b) Accept competition rules
     c) Run: kaggle competitions download -c m5-forecasting-accuracy
     d) Extract sales_train_evaluation.csv
     e) Save as: data/raw/m5_sales.csv
   Column mapping required:
     - Keep: item_id, store_id, sales (daily), sell_price, date
     - Rename: item_id -> product_id, sales -> units_sold

2. USAID Shipments (Auto-downloaded by this script)
   No action needed.

3. NOAA Weather
   URL: https://www.ncdc.noaa.gov/cdo-web/token  (get free token)
   API: https://www.ncei.noaa.gov/access/services/data/v1
   Parameters:
     datasetid=GHCND&stationids=...&startdate=2022-01-01&enddate=2023-12-31
   Save as: data/raw/weather.csv
   Required columns: date, region, temperature_c, precipitation_mm

========================================================
NOTE: If files are not present, the pipeline auto-generates
realistic synthetic data so you can run everything immediately.
========================================================
""")


if __name__ == "__main__":
    print_manual_download_instructions()
    download_usaid_shipments()
    logger.info("Done. Run `python -m src.data.ingestion` to start the pipeline.")
