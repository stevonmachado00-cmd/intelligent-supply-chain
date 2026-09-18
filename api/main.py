"""
FastAPI Main Application for the Intelligent Supply Chain Control Tower.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.dependencies import get_model_registry
from api.middleware.auth import ProcessTimeMiddleware
from api.routers import decisions, health, monitoring, predict, risk, simulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up model registry and dependencies on server startup."""
    logger.info("Starting Intelligent Supply Chain Control Tower API...")
    registry = get_model_registry()
    logger.success(f"Models ready: {list(registry.loaded_models.keys())}")
    yield
    logger.info("Shutting down Control Tower API.")


app = FastAPI(
    title="Intelligent Supply Chain Control Tower API",
    description=(
        "Production-grade, asynchronous serving layer for multi-echelon supply chain resilience. "
        "Provides real-time machine learning predictions (Demand, Stockout, Supplier Risk, ETA, Anomalies), "
        "a multi-model calibrated 0-100 Risk Engine, a Prescriptive Decision Engine (EOQ + Dynamic Safety Stock), "
        "and a What-If Scenario Simulation Engine."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(ProcessTimeMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Streamlit, Next.js, and external dashboard clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(risk.router)
app.include_router(decisions.router)
app.include_router(simulation.router)
app.include_router(monitoring.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Intelligent Supply Chain Control Tower API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
