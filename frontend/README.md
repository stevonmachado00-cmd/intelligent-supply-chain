# Control Tower Modern Web Frontend

A responsive, dark-mode supply chain operations dashboard built with React 19, Vite, Tailwind CSS, Recharts, and Lucide icons. Designed to communicate directly with the FastAPI AI inference serving layer.

## Architecture & Views

1. **Executive Overview** (`/`): High-level KPI metrics, real-time Supply Chain Health Gauge, multi-model Risk Breakdown pie chart, 30-day historical health trajectory, and prioritized incident alert feed.
2. **SKU Demand & Stockout Forecast** (`/forecast`): Interactive SKU selection, 7-day forward demand forecasting with 90% confidence bands, calibrated stockout risk classification, SHAP tree explainability charts, and dynamic inventory policies (EOQ, ROP, Safety Stock).
3. **Suppliers & Anomalies** (`/suppliers`): Sortable supplier risk matrix with on-time delivery metrics, defect rate tracking, and a live autoencoder reconstruction-loss anomaly monitor.
4. **What-If Simulation Engine** (`/simulation`): Multi-factor stress-testing console with 5 scenario presets (*Port Strike*, *Flash Sale Surge*, *Supplier Insolvency*, *Warehouse Disruption*, *Compound Cascade*), interactive parameter sliders, forward inventory trajectory charts, and automated prescriptive mitigation ROI calculator.

## Running Locally

```bash
cd frontend
npm install
npm run dev
```

The app will run on `http://localhost:3000` and proxy `/api` calls to FastAPI at `http://localhost:8000`.

## Building for Production

```bash
npm run build
```
