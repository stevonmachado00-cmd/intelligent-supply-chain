import { useState, useEffect, useCallback } from 'react'
import {
  TrendingUp,
  Search,
  Play,
  Loader2,
  Box,
  ShieldCheck,
  AlertTriangle,
  BarChart3,
} from 'lucide-react'
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Cell,
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card'
import { MetricCard } from '../components/ui/MetricCard'
import { Badge } from '../components/ui/Badge'
import { PageHeader } from '../components/ui/PageHeader'
import { SkeletonCard } from '../components/ui/Skeleton'
import { formatNumber, formatPercent, getStatusColor, DEMO_PRODUCTS } from '../lib/utils'
import { generateDemoDemandForecast, generateDemoStockout, generateDemoInventoryPolicy } from '../lib/mockData'
import { api } from '../lib/api'

export default function Forecast({ apiStatus }) {
  const [selectedProduct, setSelectedProduct] = useState(DEMO_PRODUCTS[0].id)
  const [forecast, setForecast] = useState(null)
  const [stockout, setStockout] = useState(null)
  const [policy, setPolicy] = useState(null)
  const [loading, setLoading] = useState(false)
  const [initialLoad, setInitialLoad] = useState(true)

  const runPrediction = useCallback(async () => {
    setLoading(true)
    try {
      if (apiStatus === 'connected') {
        const [demandRes, stockoutRes, policyRes] = await Promise.allSettled([
          api.predictDemand({
            product_id: selectedProduct,
            features: {
              units_sold: 120, avg_price: 2.55, day_of_week: 2, month: 9,
              demand_lag_1d: 115, demand_lag_7d: 140, demand_roll_mean_7d: 125,
              demand_roll_std_7d: 18.5, demand_cv_28d: 0.15,
            },
            return_shap: true,
          }),
          api.predictStockout({
            product_id: selectedProduct,
            current_stock: 150, reorder_point: 300, incoming_po_units: 50,
            lead_time_days: 7, daily_demand: 25, avg_demand_7d: 28,
            avg_demand_30d: 24, std_demand_7d: 5, std_demand_30d: 6,
            days_of_supply: 5.3, demand_volatility: 0.25,
          }),
          api.inventoryPolicy({
            product_id: selectedProduct,
            current_stock: 350, daily_demand_mean: 45, daily_demand_std: 8,
            lead_time_days: 5, lead_time_std: 1, unit_cost: 15,
          }),
        ])

        setForecast(demandRes.status === 'fulfilled' ? {
          ...demandRes.value,
          dailyForecast: Array.from({ length: 7 }, (_, i) => ({
            day: `Day ${i + 1}`,
            forecast: Math.round(demandRes.value.predicted_demand_7d / 7 + (Math.random() - 0.5) * 10),
            lower: Math.round(demandRes.value.confidence_interval_lower / 7),
            upper: Math.round(demandRes.value.confidence_interval_upper / 7),
          }))
        } : generateDemoDemandForecast(selectedProduct))

        setStockout(stockoutRes.status === 'fulfilled' ? stockoutRes.value : generateDemoStockout(selectedProduct))
        setPolicy(policyRes.status === 'fulfilled' ? policyRes.value : generateDemoInventoryPolicy(selectedProduct))
      } else {
        await new Promise(r => setTimeout(r, 800))
        setForecast(generateDemoDemandForecast(selectedProduct))
        setStockout(generateDemoStockout(selectedProduct))
        setPolicy(generateDemoInventoryPolicy(selectedProduct))
      }
    } catch {
      setForecast(generateDemoDemandForecast(selectedProduct))
      setStockout(generateDemoStockout(selectedProduct))
      setPolicy(generateDemoInventoryPolicy(selectedProduct))
    } finally {
      setLoading(false)
      setInitialLoad(false)
    }
  }, [selectedProduct, apiStatus])

  useEffect(() => {
    runPrediction()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const productInfo = DEMO_PRODUCTS.find(p => p.id === selectedProduct) || DEMO_PRODUCTS[0]

  return (
    <div className="p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="SKU Demand & Stockout Forecast"
        icon={TrendingUp}
        description="ML-powered demand prediction with SHAP explainability"
      >
        <button
          onClick={runPrediction}
          disabled={loading}
          className="btn-primary"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {loading ? 'Running...' : 'Run Prediction'}
        </button>
      </PageHeader>

      {/* Product Selector */}
      <Card className="!p-4">
        <div className="flex items-center gap-4">
          <Search className="h-4 w-4 text-slate-400" />
          <select
            value={selectedProduct}
            onChange={(e) => setSelectedProduct(e.target.value)}
            className="flex-1 bg-transparent text-white text-sm focus:outline-none cursor-pointer"
          >
            {DEMO_PRODUCTS.map((p) => (
              <option key={p.id} value={p.id} className="bg-surface-800 text-white">
                {p.id} — {p.name} ({p.category})
              </option>
            ))}
          </select>
        </div>
      </Card>

      {initialLoad ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} lines={4} />)}
        </div>
      ) : (
        <>
          {/* Top Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              label="7-Day Forecast"
              value={formatNumber(forecast?.predicted_demand_7d)}
              subtitle={`${formatNumber(forecast?.confidence_interval_lower)} – ${formatNumber(forecast?.confidence_interval_upper)} (90% CI)`}
              icon={TrendingUp}
              color="text-brand-400"
            />
            <MetricCard
              label="Stockout Probability"
              value={formatPercent(stockout?.calibrated_stockout_probability)}
              icon={AlertTriangle}
              color={`text-${getStatusColor(stockout?.risk_level)}-400`}
            >
              <Badge variant={getStatusColor(stockout?.risk_level)}>{stockout?.risk_level}</Badge>
            </MetricCard>
            <MetricCard
              label="Days of Supply"
              value={policy?.days_of_supply?.toFixed(1) || '—'}
              subtitle={`Safety Stock: ${formatNumber(policy?.safety_stock)}`}
              icon={Box}
              color={policy?.days_of_supply > 10 ? 'text-emerald-400' : policy?.days_of_supply > 5 ? 'text-amber-400' : 'text-red-400'}
            />
            <MetricCard
              label="Economic Order Qty"
              value={formatNumber(policy?.economic_order_quantity)}
              subtitle={`ROP: ${formatNumber(policy?.reorder_point)}`}
              icon={ShieldCheck}
            />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Demand Forecast Chart */}
            <Card>
              <CardHeader>
                <CardTitle>7-Day Demand Forecast</CardTitle>
                <CardDescription>XGBoost prediction with 90% confidence interval</CardDescription>
              </CardHeader>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={forecast?.dailyForecast || []} margin={{ top: 10, right: 10, bottom: 5, left: -15 }}>
                  <defs>
                    <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#818cf8" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '12px' }}
                    itemStyle={{ color: '#e2e8f0' }}
                  />
                  <Area type="monotone" dataKey="upper" stroke="none" fill="url(#ciGrad)" name="Upper CI" />
                  <Area type="monotone" dataKey="lower" stroke="none" fill="transparent" name="Lower CI" />
                  <Area type="monotone" dataKey="forecast" stroke="#818cf8" fill="url(#forecastGrad)" strokeWidth={2.5} dot={{ r: 4, fill: '#818cf8' }} name="Forecast" />
                  {forecast?.dailyForecast?.some(d => d.actual !== null) && (
                    <Area type="monotone" dataKey="actual" stroke="#34d399" fill="none" strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3, fill: '#34d399' }} name="Actual" />
                  )}
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            {/* SHAP Explainability */}
            <Card>
              <CardHeader>
                <CardTitle>SHAP Feature Importance</CardTitle>
                <CardDescription>Top drivers behind this prediction</CardDescription>
              </CardHeader>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={forecast?.top_shap_drivers?.map(d => ({
                    feature: d.feature?.replace(/_/g, ' '),
                    importance: +(d.importance * 100).toFixed(1),
                  })) || []}
                  layout="vertical"
                  margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
                  <YAxis dataKey="feature" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} width={120} />
                  <Tooltip
                    contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '12px' }}
                    itemStyle={{ color: '#e2e8f0' }}
                    formatter={(val) => [`${val}%`, 'Importance']}
                  />
                  <Bar dataKey="importance" fill="#818cf8" radius={[0, 6, 6, 0]} barSize={20}>
                    {forecast?.top_shap_drivers?.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? '#6366f1' : i === 1 ? '#818cf8' : '#a5b4fc'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Inventory Policy */}
          {policy && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Inventory Policy — {productInfo.name}</CardTitle>
                    <CardDescription>Dynamic Safety Stock (95% service level, z=1.65)</CardDescription>
                  </div>
                  <Badge variant={getStatusColor(policy.stock_status)}>{policy.stock_status?.replace(/_/g, ' ')}</Badge>
                </div>
              </CardHeader>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                {[
                  { label: 'Current Stock', value: formatNumber(policy.current_stock), color: 'text-white' },
                  { label: 'Safety Stock', value: formatNumber(policy.safety_stock), color: 'text-amber-400' },
                  { label: 'Reorder Point', value: formatNumber(policy.reorder_point), color: 'text-brand-400' },
                  { label: 'EOQ', value: formatNumber(policy.economic_order_quantity), color: 'text-purple-400' },
                  { label: 'Days of Supply', value: policy.days_of_supply?.toFixed(1), color: policy.days_of_supply > 10 ? 'text-emerald-400' : 'text-amber-400' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="text-center">
                    <p className={`text-2xl font-bold tabular-nums ${color}`}>{value}</p>
                    <p className="metric-label mt-1">{label}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
