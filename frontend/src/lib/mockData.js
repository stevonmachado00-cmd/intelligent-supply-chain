// Generates realistic demo data so the dashboard looks populated even without the API running

export function generateDemoOverview() {
  return {
    healthScore: 78.4,
    riskScore: 21.6,
    status: 'GREEN',
    totalProducts: 50,
    activeAlerts: 3,
    criticalItems: 2,
    avgDaysOfSupply: 12.3,
    inventoryTurnover: 8.2,
    fillRate: 0.946,
    totalRevenue: 2840000,
    revenueAtRisk: 142000,
    pendingOrders: 847,
    riskBreakdown: {
      stockout_risk: 30,
      supplier_risk: 25,
      demand_volatility: 25,
      delay_risk: 10,
      anomaly_risk: 10,
    },
    recentAlerts: [
      { id: 1, type: 'CRITICAL', product: 'PROD_0004', message: 'Stockout predicted in 3 days — reorder immediately', time: '12 min ago' },
      { id: 2, type: 'HIGH', product: 'SUP_004', message: 'Supplier risk elevated to HIGH_RISK — consider backup vendor', time: '1 hr ago' },
      { id: 3, type: 'MEDIUM', product: 'PROD_0007', message: 'Demand surge detected (+45%) — adjust safety stock', time: '3 hrs ago' },
    ],
    healthTrend: Array.from({ length: 30 }, (_, i) => ({
      day: i + 1,
      health: 65 + Math.random() * 25 + (i > 20 ? 5 : 0),
      risk: 10 + Math.random() * 20 - (i > 20 ? 3 : 0),
    })),
  }
}

export function generateDemoDemandForecast(productId) {
  const base = 120 + Math.random() * 80
  return {
    product_id: productId,
    model_used: 'XGBoost Regressor (Trained)',
    predicted_demand_7d: Math.round(base),
    confidence_interval_lower: Math.round(base * 0.7),
    confidence_interval_upper: Math.round(base * 1.35),
    top_shap_drivers: [
      { feature: 'demand_lag_1d', importance: 0.35 },
      { feature: 'demand_roll_mean_7d', importance: 0.22 },
      { feature: 'avg_price', importance: 0.15 },
      { feature: 'month', importance: 0.12 },
      { feature: 'day_of_week', importance: 0.08 },
    ],
    dailyForecast: Array.from({ length: 7 }, (_, i) => ({
      day: `Day ${i + 1}`,
      forecast: Math.round(base / 7 + (Math.random() - 0.5) * 10),
      lower: Math.round(base / 7 * 0.7),
      upper: Math.round(base / 7 * 1.35),
      actual: i < 3 ? Math.round(base / 7 + (Math.random() - 0.5) * 8) : null,
    })),
  }
}

export function generateDemoStockout(productId) {
  const prob = Math.random()
  return {
    product_id: productId,
    calibrated_stockout_probability: prob,
    predicted_stockout: prob > 0.5,
    risk_level: prob < 0.3 ? 'LOW' : prob < 0.7 ? 'MODERATE' : 'CRITICAL',
    top_shap_drivers: [
      { feature: 'days_of_supply', importance: 0.42 },
      { feature: 'current_stock', importance: 0.28 },
      { feature: 'daily_demand', importance: 0.15 },
      { feature: 'lead_time_days', importance: 0.10 },
      { feature: 'demand_volatility', importance: 0.05 },
    ],
  }
}

export function generateDemoInventoryPolicy(productId) {
  return {
    product_id: productId,
    current_stock: 350,
    daily_demand: 45,
    lead_time_days: 5,
    safety_stock: 132,
    reorder_point: 357,
    economic_order_quantity: 490,
    stock_status: 'REORDER_SOON',
    days_of_supply: 7.8,
  }
}

export function generateDemoSupplierRiskTable() {
  return [
    { supplier_id: 'SUP_001', name: 'ShenzhenTech Co.', region: 'Asia Pacific', risk_score: 0.12, classification: 'RELIABLE', on_time: 0.94, defect_rate: 0.008, shipments: 342 },
    { supplier_id: 'SUP_002', name: 'EuroLogistics GmbH', region: 'Europe', risk_score: 0.35, classification: 'MONITOR', on_time: 0.88, defect_rate: 0.022, shipments: 189 },
    { supplier_id: 'SUP_003', name: 'AmericaParts Inc.', region: 'North America', risk_score: 0.08, classification: 'RELIABLE', on_time: 0.96, defect_rate: 0.005, shipments: 456 },
    { supplier_id: 'SUP_004', name: 'IndiaFab Ltd.', region: 'South Asia', risk_score: 0.62, classification: 'HIGH_RISK', on_time: 0.79, defect_rate: 0.041, shipments: 127 },
    { supplier_id: 'SUP_005', name: 'BrazilRaw SA', region: 'South America', risk_score: 0.48, classification: 'MONITOR', on_time: 0.82, defect_rate: 0.033, shipments: 98 },
    { supplier_id: 'SUP_006', name: 'VietnamSource JSC', region: 'Asia Pacific', risk_score: 0.18, classification: 'RELIABLE', on_time: 0.91, defect_rate: 0.012, shipments: 267 },
    { supplier_id: 'SUP_007', name: 'TürkMaterials AS', region: 'Europe', risk_score: 0.55, classification: 'MONITOR', on_time: 0.83, defect_rate: 0.028, shipments: 156 },
    { supplier_id: 'SUP_008', name: 'NigeriaRaw Ltd.', region: 'Africa', risk_score: 0.71, classification: 'HIGH_RISK', on_time: 0.74, defect_rate: 0.052, shipments: 63 },
  ]
}

export function generateDemoAnomalies() {
  return [
    { id: 1, timestamp: '2026-09-18 14:23', type: 'Transit', severity: 'CRITICAL', description: 'Shipment SH-4892 reconstruction loss 2.3x threshold — possible route deviation', loss: 1.612, threshold: 0.695 },
    { id: 2, timestamp: '2026-09-18 11:05', type: 'Demand', severity: 'ELEVATED', description: 'PROD_0003 demand spike +180% vs 7-day rolling average', loss: 0.891, threshold: 0.695 },
    { id: 3, timestamp: '2026-09-17 22:41', type: 'Transit', severity: 'NORMAL', description: 'Shipment SH-4856 within normal reconstruction bounds', loss: 0.312, threshold: 0.695 },
    { id: 4, timestamp: '2026-09-17 16:12', type: 'Supplier', severity: 'ELEVATED', description: 'SUP_004 defect rate exceeded 3-sigma threshold', loss: 0.824, threshold: 0.695 },
    { id: 5, timestamp: '2026-09-17 09:30', type: 'Transit', severity: 'NORMAL', description: 'Shipment SH-4831 cleared autoencoder check', loss: 0.198, threshold: 0.695 },
  ]
}

export function generateDemoSimulationResult() {
  return {
    scenario_name: 'Port Strike',
    product_id: 'PROD_0001',
    projection_horizon_days: 14,
    baseline_risk_score: 21.6,
    baseline_health_score: 78.4,
    baseline_days_of_supply: 12.3,
    baseline_stockout_prob: 0.15,
    simulated_risk_score: 67.8,
    simulated_health_score: 32.2,
    simulated_days_of_supply: 4.1,
    simulated_stockout_prob: 0.82,
    risk_delta: 46.2,
    first_stockout_day: 6,
    total_stockout_days: 5,
    total_unfulfilled_units: 340,
    estimated_revenue_loss: 14868,
    daily_trajectory_sample: Array.from({ length: 7 }, (_, i) => ({
      day: i + 1,
      stock: Math.max(0, 500 - (i + 1) * 75 + (i === 3 ? 200 : 0)),
      demand: 65 + Math.round(Math.random() * 20),
      inbound: i === 3 ? 200 : 0,
      stockout: 500 - (i + 1) * 75 + (i === 3 ? 200 : 0) <= 0,
    })),
    mitigation_actions: [
      { action_type: 'EMERGENCY_REPLENISHMENT', urgency: 'CRITICAL', headline: 'Emergency order 490 units', recommendation: 'Place emergency PO with backup vendor SUP_BACKUP_A', estimated_cost: 7350, net_benefit: 21544 },
      { action_type: 'REROUTE_SUPPLIER', urgency: 'HIGH', headline: 'Activate backup supplier', recommendation: 'Switch from SUP_001 to SUP_003 for next 3 shipments', estimated_cost: 1200, net_benefit: 8900 },
    ],
    mitigation_cost: 8550,
    mitigation_net_savings: 30444,
  }
}
