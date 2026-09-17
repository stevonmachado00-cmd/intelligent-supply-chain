"""
Convenience script to train all models sequentially.

Run from the project root:
    python scripts/train_all_models.py

Each model is tracked separately in MLflow.
"""

from __future__ import annotations

from loguru import logger


def main() -> None:
    logger.info("=" * 60)
    logger.info("INTELLIGENT SUPPLY CHAIN CONTROL TOWER -- Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Ingest data
    logger.info("Step 1/7: Data ingestion")
    from src.data.ingestion import run_ingestion
    datasets = run_ingestion()

    # Step 2: Validate
    logger.info("Step 2/7: Data validation")
    from src.data.validation import validate_all
    validate_all(datasets)

    # Step 3: Feature engineering
    logger.info("Step 3/7: Feature engineering")
    from src.data.feature_store import build_all_features
    feature_tables = build_all_features(datasets)

    # Step 4: Train demand models
    logger.info("Step 4/7: Training demand models (XGBoost + LSTM)")
    from src.models.demand.trainer import train_demand_models
    train_demand_models(feature_tables["demand_features"])

    # Step 5: Train classification models
    logger.info("Step 5/7: Training stockout, supplier risk, ETA models")
    from src.models.stockout.trainer import train_stockout_model
    from src.models.supplier_risk.trainer import train_supplier_risk_model
    from src.models.eta.trainer import train_eta_model
    train_stockout_model(feature_tables["stockout_features"])
    train_supplier_risk_model(feature_tables["supplier_risk_features"])
    train_eta_model(feature_tables["eta_features"])

    # Step 6: Train anomaly detector
    logger.info("Step 6/7: Training anomaly detection autoencoder")
    from src.models.anomaly.trainer import train_anomaly_model
    train_anomaly_model(feature_tables["anomaly_features"])

    # Step 7: Done
    logger.success("=" * 60)
    logger.success("All models trained successfully! Open MLflow UI to review:")
    logger.success("  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
