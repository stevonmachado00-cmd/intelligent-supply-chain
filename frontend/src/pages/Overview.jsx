import { useState, useEffect } from 'react'
import {
  LayoutDashboard,
  AlertTriangle,
  DollarSign,
  Package,
  TrendingUp,
  ShieldAlert,
  Clock,
  BarChart3,
  Activity,
} from 'lucide-react'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card'
import { MetricCard } from '../components/ui/MetricCard'
import { HealthGauge } from '../components/ui/HealthGauge'
import { Badge } from '../components/ui/Badge'
import { PageHeader } from '../components/ui/PageHeader'
import { SkeletonCard } from '../components/ui/Skeleton'
import { formatCurrency, formatPercent, formatNumber } from '../lib/utils'
import { generateDemoOverview } from '../lib/mockData'

const RISK_COLORS = ['#f87171', '#a78bfa', '#fbbf24', '#38bdf8', '#94a3b8']

export default function Overview({ apiStatus }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load demo data (would be replaced with API calls in production)
    const timer = setTimeout(() => {
      setData(generateDemoOverview())
      setLoading(false)
    }, 600)
    return () => clearTimeout(timer)
  }, [])

  if (loading) {
    return (
      <div className="p-8 space-y-6 animate-fade-in">
        <PageHeader title="Executive Overview" icon={LayoutDashboard} description="Loading intelligence..." />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} lines={2} />)}
        </div>
      </div>
    )
  }

  const riskPieData = Object.entries(data.riskBreakdown).map(([key, value]) => ({
    name: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    value,
  }))

  const alertVariant = (type) => {
    if (type === 'CRITICAL') return 'red'
    if (type === 'HIGH') return 'amber'
    return 'blue'
  }

  return (
    <div className="p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Executive Overview"
        icon={LayoutDashboard}
        description="Real-time supply chain intelligence across all operations"
      >
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-800/50 border border-surface-700/50 text-xs text-slate-400">
          <Activity className="h-3.5 w-3.5" />
          Last updated: just now
        </div>
      </PageHeader>

      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Health Score"
          value={data.healthScore.toFixed(1)}
          subtitle="Overall supply chain health"
          icon={ShieldAlert}
          trend="+2.1 vs last week"
          trendUp={true}
          color={data.healthScore >= 80 ? 'text-emerald-400' : data.healthScore >= 50 ? 'text-amber-400' : 'text-red-400'}
        />
        <MetricCard
          label="Revenue at Risk"
          value={formatCurrency(data.revenueAtRisk)}
          subtitle="Potential loss from disruptions"
          icon={DollarSign}
          trend="-12% vs last week"
          trendUp={true}
        />
        <MetricCard
          label="Active Alerts"
          value={data.activeAlerts}
          subtitle={`${data.criticalItems} critical items`}
          icon={AlertTriangle}
          color="text-amber-400"
        />
        <MetricCard
          label="Fill Rate"
          value={formatPercent(data.fillRate)}
          subtitle="Order fulfillment rate"
          icon={Package}
          trend="+0.3%"
          trendUp={true}
          color="text-emerald-400"
        />
      </div>

      {/* Middle Row: Gauge + Risk Breakdown + Quick Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Health Gauge */}
        <Card className="flex flex-col items-center justify-center py-6">
          <HealthGauge score={data.healthScore} size={200} />
          <div className="mt-4 flex items-center gap-4">
            <div className="text-center">
              <p className="text-lg font-bold text-white">{data.riskScore.toFixed(1)}</p>
              <p className="metric-label">Risk Score</p>
            </div>
            <div className="h-8 w-px bg-surface-700" />
            <div className="text-center">
              <Badge variant={data.status === 'GREEN' ? 'emerald' : data.status === 'AMBER' ? 'amber' : 'red'}>
                {data.status}
              </Badge>
              <p className="metric-label mt-1">Status</p>
            </div>
          </div>
        </Card>

        {/* Risk Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Composition</CardTitle>
            <CardDescription>Weighted contribution by risk category</CardDescription>
          </CardHeader>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={riskPieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={3}
                dataKey="value"
                stroke="none"
              >
                {riskPieData.map((_, i) => (
                  <Cell key={i} fill={RISK_COLORS[i % RISK_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '12px' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Legend
                verticalAlign="bottom"
                iconType="circle"
                iconSize={8}
                formatter={(value) => <span className="text-xs text-slate-400 ml-1">{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Quick Stats */}
        <Card>
          <CardHeader>
            <CardTitle>Operations Snapshot</CardTitle>
            <CardDescription>Key operational metrics</CardDescription>
          </CardHeader>
          <div className="space-y-4">
            {[
              { label: 'Total Products Tracked', value: formatNumber(data.totalProducts), icon: Package },
              { label: 'Avg Days of Supply', value: data.avgDaysOfSupply.toFixed(1), icon: Clock },
              { label: 'Inventory Turnover', value: `${data.inventoryTurnover}x`, icon: BarChart3 },
              { label: 'Pending Orders', value: formatNumber(data.pendingOrders), icon: TrendingUp },
            ].map(({ label, value, icon: Icon }) => (
              <div key={label} className="flex items-center justify-between py-1.5">
                <div className="flex items-center gap-2.5">
                  <Icon className="h-4 w-4 text-slate-500" />
                  <span className="text-sm text-slate-400">{label}</span>
                </div>
                <span className="text-sm font-semibold text-white tabular-nums">{value}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Bottom Row: Health Trend + Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Health Trend Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>30-Day Health Trend</CardTitle>
            <CardDescription>Supply chain health and risk score trajectory</CardDescription>
          </CardHeader>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={data.healthTrend} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
              <defs>
                <linearGradient id="healthGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f87171" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', fontSize: '12px' }}
                itemStyle={{ color: '#e2e8f0' }}
              />
              <Area type="monotone" dataKey="health" stroke="#34d399" fill="url(#healthGrad)" strokeWidth={2} dot={false} name="Health" />
              <Area type="monotone" dataKey="risk" stroke="#f87171" fill="url(#riskGrad)" strokeWidth={2} dot={false} name="Risk" />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Recent Alerts */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Alerts</CardTitle>
            <CardDescription>{data.recentAlerts.length} active notifications</CardDescription>
          </CardHeader>
          <div className="space-y-3 max-h-[220px] overflow-y-auto pr-1">
            {data.recentAlerts.map((alert) => (
              <div
                key={alert.id}
                className="flex gap-3 p-3 rounded-xl bg-surface-700/30 border border-surface-700/50 hover:border-surface-600 transition-colors"
              >
                <AlertTriangle className={`h-4 w-4 mt-0.5 shrink-0 ${
                  alert.type === 'CRITICAL' ? 'text-red-400' : alert.type === 'HIGH' ? 'text-amber-400' : 'text-blue-400'
                }`} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={alertVariant(alert.type)}>{alert.type}</Badge>
                    <span className="text-[10px] text-slate-500">{alert.time}</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{alert.message}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}
