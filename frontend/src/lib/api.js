const API_BASE = '/api'
const API_KEY = 'sc-tower-secret-key-2026'

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`
  const config = {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...options.headers,
    },
    ...options,
  }

  const res = await fetch(url, config)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `API Error: ${res.status}`)
  }
  return res.json()
}

export const api = {
  // Health
  health: () => request('/health'),
  healthReady: () => request('/health/ready'),

  // Predictions
  predictDemand: (data) => request('/predict/demand', { method: 'POST', body: JSON.stringify(data) }),
  predictStockout: (data) => request('/predict/stockout', { method: 'POST', body: JSON.stringify(data) }),
  predictSupplierRisk: (data) => request('/predict/supplier-risk', { method: 'POST', body: JSON.stringify(data) }),
  predictEta: (data) => request('/predict/eta', { method: 'POST', body: JSON.stringify(data) }),
  detectAnomaly: (data) => request('/predict/anomaly', { method: 'POST', body: JSON.stringify(data) }),

  // Risk
  riskScore: (data) => request('/risk/score', { method: 'POST', body: JSON.stringify(data) }),
  riskBatch: (data) => request('/risk/batch', { method: 'POST', body: JSON.stringify(data) }),

  // Decisions
  inventoryPolicy: (data) => request('/decisions/inventory-policy', { method: 'POST', body: JSON.stringify(data) }),
  recommendations: (data) => request('/decisions/recommendations', { method: 'POST', body: JSON.stringify(data) }),

  // Simulation
  simulationPresets: () => request('/simulation/presets'),
  simulationRun: (data) => request('/simulation/run', { method: 'POST', body: JSON.stringify(data) }),
}

export default api
