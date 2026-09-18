import { useState, useEffect } from 'react'
import { Routes, Route, NavLink, Navigate } from 'react-router-dom'
import {
  LayoutDashboard,
  TrendingUp,
  Truck,
  FlaskConical,
  Activity,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
  Zap,
} from 'lucide-react'
import { api } from './lib/api'
import Overview from './pages/Overview'
import Forecast from './pages/Forecast'
import Suppliers from './pages/Suppliers'
import Simulation from './pages/Simulation'

const NAV_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: 'Executive Overview' },
  { path: '/forecast', icon: TrendingUp, label: 'Demand Forecast' },
  { path: '/suppliers', icon: Truck, label: 'Suppliers & Anomalies' },
  { path: '/simulation', icon: FlaskConical, label: 'What-If Simulation' },
]

export default function App() {
  const [collapsed, setCollapsed] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    const checkApi = async () => {
      try {
        await api.health()
        setApiStatus('connected')
      } catch {
        setApiStatus('disconnected')
      }
    }
    checkApi()
    const interval = setInterval(checkApi, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-surface-950">
      {/* Sidebar */}
      <aside
        className={`relative flex flex-col border-r border-surface-700/50 bg-surface-900/80 backdrop-blur-xl transition-all duration-300 ${
          collapsed ? 'w-[72px]' : 'w-[260px]'
        }`}
      >
        {/* Brand */}
        <div className="flex items-center gap-3 px-5 py-6 border-b border-surface-700/50">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-purple-600 shadow-lg shadow-brand-500/25">
            <Zap className="h-5 w-5 text-white" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <h1 className="text-sm font-bold text-white truncate">Control Tower</h1>
              <p className="text-[10px] text-slate-500 font-medium">Supply Chain AI</p>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map(({ path, icon: Icon, label }) => (
            <NavLink
              key={path}
              to={path}
              end={path === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  isActive
                    ? 'bg-brand-600/15 text-brand-400 shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-surface-700/40'
                }`
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-3 py-4 border-t border-surface-700/50 space-y-3">
          {/* API Status */}
          <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${
            collapsed ? 'justify-center' : ''
          }`}>
            <span className={`h-2 w-2 rounded-full shrink-0 ${
              apiStatus === 'connected'
                ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50'
                : apiStatus === 'checking'
                ? 'bg-amber-400 animate-pulse'
                : 'bg-red-400'
            }`} />
            {!collapsed && (
              <span className="text-slate-500">
                {apiStatus === 'connected' ? 'API Connected' : apiStatus === 'checking' ? 'Checking...' : 'API Offline'}
              </span>
            )}
          </div>

          {!collapsed && (
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-500 hover:text-brand-400 transition-colors"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              API Documentation
            </a>
          )}

          {!collapsed && (
            <p className="px-3 text-[10px] text-slate-600">v1.0.0 · Phase 6</p>
          )}
        </div>

        {/* Collapse Toggle */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-20 flex h-6 w-6 items-center justify-center rounded-full bg-surface-700 border border-surface-600 text-slate-400 hover:text-white hover:bg-brand-600 transition-all duration-200 shadow-lg z-10"
        >
          {collapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronLeft className="h-3.5 w-3.5" />}
        </button>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="min-h-full">
          <Routes>
            <Route path="/" element={<Overview apiStatus={apiStatus} />} />
            <Route path="/forecast" element={<Forecast apiStatus={apiStatus} />} />
            <Route path="/suppliers" element={<Suppliers apiStatus={apiStatus} />} />
            <Route path="/simulation" element={<Simulation apiStatus={apiStatus} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
