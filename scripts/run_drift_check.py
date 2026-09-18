"""
Standalone execution script to run Evidently AI drift monitoring across models.
Outputs HTML report to mlflow/drift_reports/ and prints diagnostic summary.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from src.monitoring.drift_detector import DataDriftDetector
from src.monitoring.retraining import ModelRetrainingService


def main():
    logger.info("=================================================================")
    logger.info("RUNNING MLOPS DATA DRIFT AUDIT (EVIDENTLY AI)")
    logger.info("=================================================================")

    service = ModelRetrainingService()

    for model_name in ["demand_xgb", "stockout", "eta"]:
        logger.info(f"\nAnalyzing distribution stability for: {model_name}")
        result = service.evaluate_retraining_need(model_name=model_name, drift_ratio=0.0)

        logger.info(f"  Total evaluated features: {result['total_features_count']}")
        logger.info(f"  Statistically drifted features: {result['drifted_features_count']}")
        logger.info(f"  Drift share: {result['drift_share']:.1%}")
        logger.info(f"  Dataset drift flagged: {result['dataset_drift_detected']}")
        logger.info(f"  Health status: {result['urgency']}")
        logger.info(f"  HTML report: {result['html_report_path']}")

    logger.success("\nAll Evidently AI drift audit reports generated successfully in mlflow/drift_reports/.")


if __name__ == "__main__":
    main()
