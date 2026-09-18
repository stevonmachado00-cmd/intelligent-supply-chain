import { clsx } from 'clsx'

export function cn(...inputs) {
  return clsx(inputs)
}

export function formatNumber(num, decimals = 0) {
  if (num === null || num === undefined) return '—'
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num)
}

export function formatCurrency(num, decimals = 0) {
  if (num === null || num === undefined) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num)
}

export function formatPercent(num, decimals = 1) {
  if (num === null || num === undefined) return '—'
  return `${(num * 100).toFixed(decimals)}%`
}

export function getStatusColor(status) {
  switch (status?.toUpperCase()) {
    case 'GREEN':
    case 'HEALTHY':
    case 'RELIABLE':
    case 'LOW':
    case 'NORMAL':
      return 'emerald'
    case 'AMBER':
    case 'MONITOR':
    case 'MODERATE':
    case 'REORDER_SOON':
    case 'ELEVATED':
      return 'amber'
    case 'RED':
    case 'HIGH_RISK':
    case 'CRITICAL':
    case 'CRITICAL_STOCKOUT':
      return 'red'
    default:
      return 'slate'
  }
}

export function getStatusBadgeClasses(status) {
  const color = getStatusColor(status)
  return `bg-${color}-500/10 text-${color}-400 border border-${color}-500/20`
}

export function getHealthGradient(score) {
  if (score >= 80) return 'from-emerald-500 to-emerald-400'
  if (score >= 50) return 'from-amber-500 to-amber-400'
  return 'from-red-500 to-red-400'
}

export function getHealthTextColor(score) {
  if (score >= 80) return 'text-emerald-400'
  if (score >= 50) return 'text-amber-400'
  return 'text-red-400'
}

// Demo data generators for when API is not available
export const DEMO_PRODUCTS = [
  { id: 'PROD_0001', name: 'Premium Wireless Headphones', category: 'Electronics' },
  { id: 'PROD_0002', name: 'Organic Coffee Beans 1kg', category: 'Food & Beverage' },
  { id: 'PROD_0003', name: 'Stainless Steel Water Bottle', category: 'Home & Kitchen' },
  { id: 'PROD_0004', name: 'Bamboo Cutting Board Set', category: 'Home & Kitchen' },
  { id: 'PROD_0005', name: 'LED Desk Lamp Pro', category: 'Electronics' },
  { id: 'PROD_0006', name: 'Yoga Mat Premium', category: 'Sports & Fitness' },
  { id: 'PROD_0007', name: 'Ceramic Plant Pot Set', category: 'Garden' },
  { id: 'PROD_0008', name: 'Bluetooth Speaker Mini', category: 'Electronics' },
]

export const DEMO_SUPPLIERS = [
  { id: 'SUP_001', name: 'ShenzhenTech Co.', region: 'Asia Pacific', onTimeRate: 0.94, risk: 0.12 },
  { id: 'SUP_002', name: 'EuroLogistics GmbH', region: 'Europe', onTimeRate: 0.88, risk: 0.35 },
  { id: 'SUP_003', name: 'AmericaParts Inc.', region: 'North America', onTimeRate: 0.96, risk: 0.08 },
  { id: 'SUP_004', name: 'IndiaFab Ltd.', region: 'South Asia', onTimeRate: 0.79, risk: 0.62 },
  { id: 'SUP_005', name: 'BrazilRaw SA', region: 'South America', onTimeRate: 0.82, risk: 0.48 },
]
