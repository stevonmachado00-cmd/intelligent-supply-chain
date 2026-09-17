# ?? Intelligent Supply Chain Control Tower

> **An AI/ML-powered end-to-end supply chain intelligence platform that continuously monitors multi-echelon logistics, predicts stock-outs and disruptions before they occur, explains root causes with SHAP, and recommends optimal corrective actions.**

---

## ?? Executive Summary

Traditional supply chain dashboards are reactive—they simply display failures that have already occurred (stockouts, delayed shipments, supplier defaults).

The **Intelligent Supply Chain Control Tower** transforms supply chain management into a **predictive and prescriptive** decision-support system answering four fundamental operational questions:

1. **What is happening?** (Real-time tracking of orders, shipments, inventory levels, and supplier lead times)
2. **What is likely to happen next?** (Predictive ML: demand surges, shipment ETAs, stockout probabilities, and supplier failure risks)
3. **Why is it happening?** (Explainable AI via SHAP feature attribution)
4. **What should the enterprise do about it?** (Prescriptive Decision Engine utilizing Economic Order Quantity (EOQ), safety stock recalculations, and supplier rerouting)

---

## ??? System Architecture

`
                               +-----------------------------+
                               ¦     Official Data Sources   ¦
                               ¦  UCI Online Retail (540k+)  ¦
                               ¦  NOAA Weather API & Alerts  ¦
                               +-----------------------------+
                                              ¦
                                              ?
                               +-----------------------------+
                               ¦      Data Ingestion &       ¦
                               ¦   Validation (Great Expect) ¦
                               +-----------------------------+
                                              ¦
                                              ?
                               +-----------------------------+
                               ¦   Feature Store / Pipelines ¦
                               ¦  (Lags, Rolling, Volatility)¦
                               +-----------------------------+
                                              ¦
                    +---------------------------------------------------+
                    ?                                                   ?
       +---------------------------+                       +---------------------------+
       ¦   Supervised ML Models    ¦                       ¦    Deep Learning Models   ¦
       ¦  • XGBoost Demand Regress ¦                       ¦  • PyTorch LSTM (Demand)  ¦
       ¦  • Calibrated Stockout Clf¦                       ¦  • PyTorch Autoencoder    ¦
       ¦  • Calibrated Supplier Clf¦                       ¦    (Anomaly Detection)    ¦
       ¦  • XGBoost Shipment ETA   ¦                       ¦                           ¦
       +---------------------------+                       +---------------------------+
                    ¦                                                   ¦
                    +---------------------------------------------------+
                                              ¦
                                              ?
                               +-----------------------------+
                               ¦   Weighted Risk Engine      ¦
                               ¦  Supply Chain Health (0-100)¦
                               +-----------------------------+
                                              ¦
                                              ?
                               +-----------------------------+
                               ¦  Prescriptive Decision Eng  ¦
                               ¦  EOQ + Dynamic Safety Stock ¦
                               ¦  + What-If Scenario Sim     ¦
                               +-----------------------------+
                                              ¦
                                              ?
                               +-----------------------------+
                               ¦   Serving & Visualization   ¦
                               ¦   • FastAPI (Async REST)    ¦
                               ¦   • Streamlit Dashboard     ¦
                               ¦   • MLflow Model Registry   ¦
                               +-----------------------------+
`

---

## ?? Multi-Model AI/ML Design

| Subsystem | Model / Algorithm | Target / Objective | Evaluation Metrics |
| :--- | :--- | :--- | :--- |
| **1. Demand Forecasting** | **XGBoost Regressor + PyTorch LSTM** | Predict SKU-level demand 7 days ahead | MAE, RMSE, MAPE |
| **2. Stockout Prediction** | **Calibrated XGBoost Classifier** | Predict probability ( \in [0, 1]$) of stock depletion | PR-AUC, F1-Score, Brier Score |
| **3. Supplier Risk** | **Calibrated XGBoost Classifier** | Classify supplier delay/defect risk category | ROC-AUC, Precision, Recall |
| **4. Shipment ETA** | **XGBoost Regressor** | Predict delivery duration (hours) & delay likelihood | MAE, RMSE |
| **5. Anomaly Detection** | **PyTorch Deep Autoencoder** | Detect multi-variate logistics anomalies via reconstruction error | Reconstruction Loss, Precision@K |
| **6. Explainability** | **TreeSHAP** | Generate feature contribution waterfall for high-risk flags | Additive feature attribution |

---

## ?? Enterprise Tech Stack

* **Machine Learning & Deep Learning:** Python, Scikit-learn, XGBoost, PyTorch, SHAP
* **Data Engineering:** Pandas, NumPy, Parquet, Great Expectations
* **Backend & API:** FastAPI, Pydantic, Uvicorn, SlowAPI (Rate Limiting)
* **Databases & Cache:** PostgreSQL, Redis (Caching & Streaming)
* **Experiment Tracking & MLOps:** MLflow, Evidently AI
* **Frontend Dashboard:** Streamlit (Prototyping), Plotly
* **DevOps & Infrastructure:** Docker, Docker Compose, GitHub Actions CI/CD

---

## ?? Project Structure

`
intelligent-supply-chain/
+-- .github/workflows/ci.yml       # GitHub Actions CI pipeline
+-- configs/
¦   +-- model_config.yaml          # Model hyperparameters, risk weights, EOQ params
¦   +-- feature_config.yaml        # Lag windows, rolling stats, calendar features
+-- data/
¦   +-- schema.md                  # Enterprise data dictionary & entity schema
¦   +-- raw/                       # Raw downloads (gitignored)
¦   +-- processed/                 # Feature-engineered Parquet files (gitignored)
+-- docker/
¦   +-- docker-compose.yml         # Full stack: Postgres, Redis, MLflow, API, Dashboard
¦   +-- Dockerfile.api             # FastAPI container
¦   +-- Dockerfile.dashboard       # Streamlit container
+-- notebooks/                     # Exploratory Data Analysis & experiments
+-- scripts/
¦   +-- download_data.py           # Dataset retrieval script
¦   +-- train_all_models.py        # End-to-end training orchestrator
+-- src/
¦   +-- data/
¦   ¦   +-- ingestion.py           # Ingestion and normalization
¦   ¦   +-- validation.py          # Data quality checks & physical boundaries
¦   ¦   +-- feature_store.py       # Time-series feature engineering
¦   +-- models/                    # Model training & inference modules
¦   +-- risk_engine/               # Multi-model risk score aggregator (0-100)
¦   +-- simulation/                # What-If scenario simulation engine
¦   +-- monitoring/                # Model drift & automated retraining
+-- tests/                         # Pytest unit & integration test suite
+-- pyproject.toml                 # Tool configs (ruff, black, mypy, pytest)
+-- requirements.txt               # Pinned project dependencies
+-- README.md
`

---

## ?? Getting Started

### 1. Prerequisites
* Python 3.10+
* Git
* Docker & Docker Compose (optional for local multi-service testing)

### 2. Installation
`ash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd intelligent-supply-chain

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

### 3. Setup Environment
`ash
cp .env.example .env
`

### 4. Run Data Ingestion & Feature Engineering
`ash
python -m src.data.ingestion
`

---

## ?? CI/CD & Testing

Automated testing is configured using **GitHub Actions**, validating code quality with uff and lack, and executing the test suite against isolated PostgreSQL and Redis service containers on every commit.

To run tests locally:
`ash
pytest tests/ -v
`

---

## ?? License
MIT License. Developed for advanced supply chain and enterprise AI engineering.
