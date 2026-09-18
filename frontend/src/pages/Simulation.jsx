import { useState, useCallback } from 'react'
import {
  FlaskConical,
  Play,
  Loader2,
  ArrowRight,
  TrendingDown,
  DollarSign,
  Shield,
  AlertTriangle,
  RotateCcw,
  Zap,
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { PageHeader } from '../components/ui/PageHeader'
import { formatCurrency, formatPercent, formatNumber, getStatusColor, getHealthTextColor } from '../lib/utils'
import { generateDemoSimulationResult } from '../lib/mockData'
import { api } from '../lib/api'

const PRESETS = [
  { key: 'PORT_STRIKE', name: 'Port Strike', icon: '⛴️', desc: 'Major port closure disrupting inbound shipments', color: 'red' },
  { key: 'FLASH_SALE_SURGE', name: 'Flash Sale Surge', icon: '⚡', desc: '200%+ demand spike from promotional event', color: 'amber' },
  { key: 'SUPPLIER_INSOLVENCY', name: 'Supplier Insolvency', icon: '🏦', desc: 'Primary supplier goes bankrupt mid-contract', color: 'red' },
  { key: 'WAREHOUSE_FACILITY_DISRUPTION', name: 'Warehouse Disruption', icon: '🏭', desc: 'Fire or flood at primary distribution center', color: 'purple' },
  { key: 'COMPOUND_CASCADE', name: 'Compound Cascade', icon: '🌊', desc: 'Multi-factor supply chain shock scenario', color: 'red' },
]

export default function Simulation({ apiStatus }) {
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [params, setParams] = useState({
    demand_shock_pct: 30,
    supplier_delay_days: 5,
    supplier_capacity_loss_pct: 15,
    transport_cost_multiplier: 1.25,
    current_stock: 500,
    daily_demand_base: 50,
    projection_horizon_days: 14,
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const updateParam = (key, value) => {
    setParams(p => ({ ...p, [key]: value }))
  }

  const runSimulation = useCallback(async () => {
    setLoading(true)
    setResult(null)
    try {
      if (apiStatus === 'connected') {
        const payload = {
          product_id: 'PROD_0001',
          preset_key: selectedPreset || undefined,
          ...(!selectedPreset ? {
            demand_shock_pct: params.demand_shock_pct,
            supplier_delay_days: params.supplier_delay_days,
            supplier_capacity_loss_pct: params.supplier_capacity_loss_pct,
            transport_cost_multiplier: params.transport_cost_multiplier,
          } : {}),
          current_stock: params.current_stock,
          daily_demand_base: params.daily_demand_base,
          projection_horizon_days: params.projection_horizon_days,
          daily_demand_std: 10,
          lead_time_days: 6,
          unit_cost: 15,
          unit_price: 28,
          scheduled_inbound: { 4: 400 },
        }
        const res = await api.simulationRun(payload)
        setResult(res)
      } else {
        await new Promise(r => setTimeout(r, 1200))
        setResult(generateDemoSimulationResult())
      }
    } catch {
      setResult(generateDemoSimulationResult())
    } finally {
      setLoading(false)
    }
  }, [selectedPreset, params, apiStatus])

  const reset = () => {
    setSelectedPreset(null)
    setResult(null)
    setParams({
      demand_shock_pct: 30,
      supplier_delay_days: 5,
      supplier_capacity_loss_pct: 15,
      transport_cost_multiplier: 1.25,
      current_stock: 500,
      daily_demand_base: 50,
      projection_horizon_days: 14,
    })
  }

  return (
    <div className="p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="What-If Simulation Engine"
        icon={FlaskConical}
        description="Stress-test your supply chain against disruption scenarios"
      >
        <div className="flex items-center gap-2">
          <button onClick={reset} className="btn-ghost text-sm">
            <RotateCcw className="h-4 w-4" /> Reset
          </button>
          <button
            onClick={runSimulation}
            disabled={loading}
            className="btn-primary"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {loading ? 'Simulating...' : 'Run Simulation'}
          </button>
        </div>
      </PageHeader>

      {/* Preset Selector */}
      <div>
        <h3 className="text-sm font-semibold text-white mb-3">Disruption Presets</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {PRESETS.map((p) => (
            <button
              key={p.key}
              onClick={() => setSelectedPreset(selectedPreset === p.key ? null : p.key)}
              className={`text-left p-4 rounded-xl border transition-all duration-200 ${
                selectedPreset === p.key
                  ? 'border-brand-500/50 bg-brand-600/10 shadow-glow'
                  : 'border-surface-700/50 bg-surface-800/30 hover:border-surface-600 hover:bg-surface-800/60'
              }`}
            >
              <div className="text-2xl mb-2">{p.icon}</div>
              <p className="text-sm font-semibold text-white">{p.name}</p>
              <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{p.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Parameters */}
      {!selectedPreset && (
        <Card>
          <CardHeader>
            <CardTitle>Custom Scenario Parameters</CardTitle>
            <CardDescription>Adjust disruption factors to model your specific scenario</CardDescription>
          </CardHeader>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { key: 'demand_shock_pct', label: 'Demand Shock', unit: '%', min: 0, max: 200, step: 5 },
              { key: 'supplier_delay_days', label: 'Supplier Delay', unit: 'days', min: 0, max: 30, step: 1 },
              { key: 'supplier_capacity_loss_pct', label: 'Capacity Loss', unit: '%', min: 0, max: 100, step: 5 },
              { key: 'transport_cost_multiplier', label: 'Freight Cost', unit: 'x', min: 1, max: 5, step: 0.25 },
            ].map(({ key, label, unit, min, max, step }) => (
              <div key={key}>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm text-slate-400">{label}</label>
                  <span className="text-sm font-bold text-white tabular-nums">
                    {params[key]}{unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={params[key]}
                  onChange={(e) => updateParam(key, parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-surface-700 rounded-full appearance-none cursor-pointer accent-brand-500"
                />
                <div className="flex justify-between mt-1">
                  <span className="text-[10px] text-slate-600">{min}{unit}</span>
                  <span className="text-[10px] text-slate-600">{max}{unit}</span>
                </div>
              </div>
            ))}
          </div>

          {/* State Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6 pt-6 border-t border-surface-700/50">
            {[
              { key: 'current_stock', label: 'Starting Stock', min: 0, max: 2000, step: 50 },
              { key: 'daily_demand_base', label: 'Base Daily Demand', min: 10, max: 200, step: 5 },
              { key: 'projection_horizon_days', label: 'Horizon (Days)', min: 7, max: 30, step: 1 },
            ].map(({ key, label, min, max, step }) => (
              <div key={key}>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm text-slate-400">{label}</label>
                  <span className="text-sm font-bold text-white tabular-nums">{params[key]}</span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={params[key]}
                  onChange={(e) => updateParam(key, parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-surface-700 rounded-full appearance-none cursor-pointer accent-brand-500"
                />
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Loading Animation */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-16 gap-4">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-surface-700 rounded-full" />
            <div className="absolute inset-0 w-16 h-16 border-4 border-brand-500 rounded-full border-t-transparent animate-spin" />
          </div>
          <p className="text-sm text-slate-400">Running forward simulation...</p>
          <p className="text-xs text-slate-600">Calculating day-by-day inventory trajectory</p>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-6 animate-slide-up">
          {/* Scenario Header */}
          <div className="flex items-center gap-3">
            <Zap className="h-5 w-5 text-brand-400" />
            <h2 className="text-lg font-bold text-white">{result.scenario_name}</h2>
            <Badge variant="brand">{result.projection_horizon_days}-day projection</Badge>
          </div>

          {/* Before / After Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card className="border-emerald-500/20">
              <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-4">Baseline (Current State)</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-2xl font-bold text-emerald-400">{result.baseline_health_score?.toFixed(1)}</p>
                  <p className="metric-label">Health Score</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{result.baseline_days_of_supply?.toFixed(1)}</p>
                  <p className="metric-label">Days of Supply</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{result.baseline_risk_score?.toFixed(1)}</p>
                  <p className="metric-label">Risk Score</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-white">{formatPercent(result.baseline_stockout_prob)}</p>
                  <p className="metric-label">Stockout Prob</p>
                </div>
              </div>
            </Card>

            <Card className="border-red-500/20">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-semibold text-red-400 uppercase tracking-wider">Simulated Impact</h3>
                <Badge variant="red">+{result.risk_delta?.toFixed(1)} risk</Badge>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className={`text-2xl font-bold ${getHealthTextColor(result.simulated_health_score)}`}>
                    {result.simulated_health_score?.toFixed(1)}
                  </p>
                  <p className="metric-label">Health Score</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-red-400">{result.simulated_days_of_supply?.toFixed(1)}</p>
                  <p className="metric-label">Days of Supply</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-red-400">{result.simulated_risk_score?.toFixed(1)}</p>
                  <p className="metric-label">Risk Score</p>
                </div>
                <div>
                  <p className="text-2xl font-bold text-red-400">{formatPercent(result.simulated_stockout_prob)}</p>
                  <p className="metric-label">Stockout Prob</p>
                </div>
              </div>
            </Card>
          </div>

          {/* Trajectory Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Inventory Trajectory</CardTitle>
              <CardDescription>
                {result.first_stockout_day
                  ? `⚠️ First stockout on Day ${result.first_stockout_day} • ${result.total_stockout_days} stockout days • ${formatNumber(result.total_unfulfilled_units)} unfulfilled units`
                  : '✅ No stockout predicted within simulation horizon'
                }
              </CardDescription>
            </CardHeader>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={result.daily_trajectory_sample} margin={{ top: 10, right: 20, bottom: 5, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v) => `Day ${v}`} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '12px' }}
                  itemStyle={{ color: '#e2e8f0' }}
                />
                <ReferenceLine y={0} stroke="#f87171" strokeDasharray="3 3" label={{ value: 'Stockout', fill: '#f87171', fontSize: 10, position: 'right' }} />
                <Line type="monotone" dataKey="stock" stroke="#818cf8" strokeWidth={2.5} dot={{ r: 4, fill: '#818cf8', strokeWidth: 2, stroke: '#0f172a' }} name="Stock Level" />
                <Line type="monotone" dataKey="demand" stroke="#fbbf24" strokeWidth={2} strokeDasharray="5 3" dot={false} name="Daily Demand" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Financial Impact */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="text-center">
              <DollarSign className="h-8 w-8 text-red-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-red-400">{formatCurrency(result.estimated_revenue_loss)}</p>
              <p className="metric-label mt-1">Revenue at Risk</p>
            </Card>
            <Card className="text-center">
              <Shield className="h-8 w-8 text-amber-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-amber-400">{formatCurrency(result.mitigation_cost)}</p>
              <p className="metric-label mt-1">Mitigation Cost</p>
            </Card>
            <Card className="text-center">
              <TrendingDown className="h-8 w-8 text-emerald-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-emerald-400">{formatCurrency(result.mitigation_net_savings)}</p>
              <p className="metric-label mt-1">Net Savings (ROI)</p>
            </Card>
          </div>

          {/* Mitigation Actions */}
          {result.mitigation_actions?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Recommended Mitigation Actions</CardTitle>
                <CardDescription>Auto-generated prescriptive actions from the Decision Engine</CardDescription>
              </CardHeader>
              <div className="space-y-3">
                {result.mitigation_actions.map((action, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-4 p-4 rounded-xl bg-surface-700/20 border border-surface-700/50 hover:border-brand-500/20 transition-colors"
                  >
                    <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${
                      action.urgency === 'CRITICAL' ? 'bg-red-500/10' : 'bg-amber-500/10'
                    }`}>
                      <AlertTriangle className={`h-5 w-5 ${
                        action.urgency === 'CRITICAL' ? 'text-red-400' : 'text-amber-400'
                      }`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={getStatusColor(action.urgency)}>{action.urgency}</Badge>
                        <span className="text-sm font-semibold text-white">{action.headline}</span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">{action.recommendation}</p>
                      <div className="flex items-center gap-4 mt-2 text-xs">
                        <span className="text-slate-500">Cost: <span className="font-semibold text-amber-400">{formatCurrency(action.estimated_cost)}</span></span>
                        <span className="text-slate-500">Net Benefit: <span className="font-semibold text-emerald-400">{formatCurrency(action.net_benefit)}</span></span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
