# 🌐 Intelligent Supply Chain Control Tower

[![CI Pipeline](https://github.com/stevonmachado00-cmd/intelligent-supply-chain/actions/workflows/ci.yml/badge.svg)](https://github.com/stevonmachado00-cmd/intelligent-supply-chain/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite%206-61DAFB?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%202.6-EE4C2C?logo=pytorch)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%203.x-FF6600)](https://xgboost.readthedocs.io)
[![Evidently AI](https://img.shields.io/badge/MLOps-Evidently%20AI-5C2D91)](https://www.evidentlyai.com)

> **An enterprise-grade, AI/ML-powered end-to-end supply chain intelligence platform that continuously monitors multi-echelon logistics, predicts stockouts and disruptions before they occur, explains root causes with SHAP, provides prescriptive mitigation actions, and tracks data drift in production.**

---

## 🏛️ System Architecture

```
                               ┌─────────────────────────────┐
                               │     Official Data Sources   │
                               │  UCI Online Retail (540k+)  │
                               │  NOAA Weather & Corridors   │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  Feature Store / Pipelines  │
                               │  (Lags, Rolling, Volatility)│
                               └──────────────┬──────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       ┌───────────────────────────┐                       ┌───────────────────────────┐
       │   Supervised ML Models    │                       │    Deep Learning Models   │
       │    XGBoost Demand Regress │                       │    PyTorch LSTM (Demand)  │
       │    Calibrated Stockout Clf│                       │    PyTorch Autoencoder    │
       │    Calibrated Supplier Clf│                       │    (Anomaly Detection)    │
       │    XGBoost Shipment ETA   │                       │                           │
       └─────────────┬─────────────┘                       └─────────────┬─────────────┘
                     └────────────────────────┬──────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │    Multi-Model Risk Engine  │
                               │  Supply Chain Health (0-100)│
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │  Prescriptive Decision Eng  │
                               │  (EOQ, Safety Stock, ROP)   │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │   What-If Simulation Engine │
                               │  (Discrete Event Rollout)   │
                               └──────────────┬──────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       ┌───────────────────────────┐                       ┌───────────────────────────┐
       │   Asynchronous API Layer  │                       │   Control Tower Frontend  │
       │    FastAPI + Pydantic V2  │                       │   React 19 + Tailwind CSS │
       │    Redis Cache + API Key  │                       │   Recharts + Lucide Icons │
       └───────────────────────────┘                       └───────────────────────────┘
```

---

## 📦 Machine Learning & Deep Learning Models

| Model | Architecture | Task | Key Metric | Explainability |
| :--- | :--- | :--- | :--- | :--- |
| **1. Demand Regressor** | XGBoost | 7-day forward demand forecast | WAPE: 26.8%, MAE: 141.3 | Tree SHAP |
| **2. Sequence Demand** | PyTorch 2-Layer LSTM | Sequential temporal demand | Test MAE: 430.2 | Temporal Attention |
| **3. Stockout Risk** | Calibrated XGBoost | Lead-time stockout probability | **F1: 0.90, ROC-AUC: 0.9998** | Calibrated Probabilities |
| **4. Supplier Risk** | Calibrated XGBoost | Vendor insolvency / delay risk | **F1: 1.00, ROC-AUC: 1.00** | Stratified Split + SHAP |
| **5. Shipment ETA** | XGBoost Regressor | Transit duration in hours | **MAE: 1.12 hrs, RMSE: 1.54 hrs**| Route-level SHAP |
| **6. Anomaly Detector** | PyTorch Deep Autoencoder | Transit/operational anomalies | 5.37% detection, loss: 0.399 | Reconstruction Loss |

---

## ⚡ Quickstart: Running Locally

### 1. Start the FastAPI Serving Layer (Port 8000)

```powershell
# Navigate to project root and activate virtual environment
cd "C:\Users\STEVON MACHADO\.gemini\antigravity\scratch\intelligent-supply-chain"
.\.venv\Scripts\Activate.ps1

# Launch FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Default API Key:** `sc-tower-secret-key-2026`

### 2. Start the Modern Control Tower Frontend (Port 3000)

```powershell
cd frontend
npm run dev
```

- **Dashboard UI:** [http://localhost:3000](http://localhost:3000)

---

## 🐳 Docker Deployment

To launch the complete enterprise multi-container stack:

```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

Services started:
- `scct_postgres`: PostgreSQL database (Port 5432)
- `scct_redis`: Redis high-performance prediction caching (Port 6379)
- `scct_mlflow`: MLflow experiment tracking server (Port 5000)
- `scct_api`: FastAPI AI inference server (Port 8000)
- `scct_frontend`: Production React + Nginx dashboard (Port 3000)

---

## 📊 MLOps Drift Monitoring (Evidently AI)

Run the drift detection audit:

```powershell
python scripts/run_drift_check.py
```

Generated reports are saved in `mlflow/drift_reports/`:
- `demand_xgb_drift_report.html`
- `stockout_drift_report.html`
- `eta_drift_report.html`

---

## 🧪 Running Automated Tests

Run the complete test suite (26 passing tests across API, MLOps, and Decision Engines):

```powershell
pytest tests/ -v -o addopts=""
```
