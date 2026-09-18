import { useState, useEffect, useMemo } from 'react'
import {
  Truck,
  ShieldAlert,
  Clock,
  AlertOctagon,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Radar,
  Activity,
} from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card'
import { MetricCard } from '../components/ui/MetricCard'
import { Badge } from '../components/ui/Badge'
import { PageHeader } from '../components/ui/PageHeader'
import { SkeletonCard } from '../components/ui/Skeleton'
import { formatPercent, getStatusColor } from '../lib/utils'
import { generateDemoSupplierRiskTable, generateDemoAnomalies } from '../lib/mockData'

export default function Suppliers({ apiStatus }) {
  const [suppliers, setSuppliers] = useState([])
  const [anomalies, setAnomalies] = useState([])
  const [loading, setLoading] = useState(true)
  const [sortField, setSortField] = useState('risk_score')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    const timer = setTimeout(() => {
      setSuppliers(generateDemoSupplierRiskTable())
      setAnomalies(generateDemoAnomalies())
      setLoading(false)
    }, 700)
    return () => clearTimeout(timer)
  }, [])

  const sortedSuppliers = useMemo(() => {
    return [...suppliers].sort((a, b) => {
      const aVal = a[sortField]
      const bVal = b[sortField]
      if (typeof aVal === 'number') return sortDir === 'asc' ? aVal - bVal : bVal - aVal
      return sortDir === 'asc' ? String(aVal).localeCompare(String(bVal)) : String(bVal).localeCompare(String(aVal))
    })
  }, [suppliers, sortField, sortDir])

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const SortIcon = ({ field }) => {
    if (sortField !== field) return <ArrowUpDown className="h-3 w-3 text-slate-600" />
    return sortDir === 'asc' ? <ChevronUp className="h-3 w-3 text-brand-400" /> : <ChevronDown className="h-3 w-3 text-brand-400" />
  }

  const highRiskCount = suppliers.filter(s => s.classification === 'HIGH_RISK').length
  const avgOnTime = suppliers.length > 0 ? suppliers.reduce((sum, s) => sum + s.on_time, 0) / suppliers.length : 0

  if (loading) {
    return (
      <div className="p-8 space-y-6 animate-fade-in">
        <PageHeader title="Suppliers & Anomalies" icon={Truck} description="Loading supplier intelligence..." />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
        <SkeletonCard lines={8} />
      </div>
    )
  }

  return (
    <div className="p-8 space-y-6 animate-fade-in">
      <PageHeader
        title="Suppliers & Anomaly Detection"
        icon={Truck}
        description="Supplier risk classification and real-time anomaly monitoring"
      />

      {/* Summary Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="Total Suppliers"
          value={suppliers.length}
          subtitle="Actively tracked"
          icon={Truck}
        />
        <MetricCard
          label="High Risk Suppliers"
          value={highRiskCount}
          subtitle="Require immediate attention"
          icon={ShieldAlert}
          color="text-red-400"
        />
        <MetricCard
          label="Avg On-Time Delivery"
          value={formatPercent(avgOnTime)}
          subtitle="Across all suppliers"
          icon={Clock}
          color={avgOnTime >= 0.9 ? 'text-emerald-400' : 'text-amber-400'}
        />
      </div>

      {/* Supplier Table */}
      <Card>
        <CardHeader>
          <CardTitle>Supplier Risk Matrix</CardTitle>
          <CardDescription>Click column headers to sort • Risk scored by calibrated XGBoost model</CardDescription>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-700/50">
                {[
                  { key: 'supplier_id', label: 'ID' },
                  { key: 'name', label: 'Supplier' },
                  { key: 'region', label: 'Region' },
                  { key: 'risk_score', label: 'Risk Score' },
                  { key: 'classification', label: 'Classification' },
                  { key: 'on_time', label: 'On-Time %' },
                  { key: 'defect_rate', label: 'Defect Rate' },
                  { key: 'shipments', label: 'Shipments' },
                ].map(({ key, label }) => (
                  <th
                    key={key}
                    onClick={() => handleSort(key)}
                    className="px-4 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
                  >
                    <div className="flex items-center gap-1.5">
                      {label}
                      <SortIcon field={key} />
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedSuppliers.map((s) => (
                <tr
                  key={s.supplier_id}
                  className="border-b border-surface-700/30 hover:bg-surface-700/20 transition-colors"
                >
                  <td className="px-4 py-3 font-mono text-xs text-slate-300">{s.supplier_id}</td>
                  <td className="px-4 py-3 font-medium text-white">{s.name}</td>
                  <td className="px-4 py-3 text-slate-400">{s.region}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-surface-700 rounded-full overflow-hidden max-w-[80px]">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            s.risk_score > 0.5 ? 'bg-red-500' : s.risk_score > 0.25 ? 'bg-amber-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${Math.min(100, s.risk_score * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-slate-300 w-10 text-right">
                        {(s.risk_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={getStatusColor(s.classification)}>
                      {s.classification?.replace(/_/g, ' ')}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`font-mono text-xs ${
                      s.on_time >= 0.9 ? 'text-emerald-400' : s.on_time >= 0.8 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {formatPercent(s.on_time)}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`font-mono text-xs ${
                      s.defect_rate < 0.02 ? 'text-emerald-400' : s.defect_rate < 0.04 ? 'text-amber-400' : 'text-red-400'
                    }`}>
                      {formatPercent(s.defect_rate, 2)}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-slate-300">{s.shipments}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Anomaly Feed */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Anomaly Detection Feed</CardTitle>
              <CardDescription>PyTorch Deep Autoencoder real-time monitoring</CardDescription>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Radar className="h-4 w-4 text-brand-400 animate-pulse-slow" />
              Live monitoring
            </div>
          </div>
        </CardHeader>
        <div className="space-y-3">
          {anomalies.map((a) => (
            <div
              key={a.id}
              className={`flex items-start gap-4 p-4 rounded-xl border transition-colors ${
                a.severity === 'CRITICAL'
                  ? 'bg-red-500/5 border-red-500/20 hover:border-red-500/40'
                  : a.severity === 'ELEVATED'
                  ? 'bg-amber-500/5 border-amber-500/20 hover:border-amber-500/40'
                  : 'bg-surface-700/20 border-surface-700/50 hover:border-surface-600'
              }`}
            >
              <Activity className={`h-5 w-5 mt-0.5 shrink-0 ${
                a.severity === 'CRITICAL' ? 'text-red-400' : a.severity === 'ELEVATED' ? 'text-amber-400' : 'text-slate-500'
              }`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant={getStatusColor(a.severity)}>{a.severity}</Badge>
                  <Badge variant="blue">{a.type}</Badge>
                  <span className="text-[10px] text-slate-500 ml-auto">{a.timestamp}</span>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed">{a.description}</p>
                <div className="flex items-center gap-4 mt-2 text-xs">
                  <span className="text-slate-500">
                    Loss: <span className="font-mono text-slate-300">{a.loss.toFixed(3)}</span>
                  </span>
                  <span className="text-slate-500">
                    Threshold: <span className="font-mono text-slate-300">{a.threshold.toFixed(3)}</span>
                  </span>
                  <span className={`font-mono font-semibold ${
                    a.loss > a.threshold ? 'text-red-400' : 'text-emerald-400'
                  }`}>
                    {a.loss > a.threshold ? `${(a.loss / a.threshold).toFixed(1)}x above` : 'Within bounds'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
