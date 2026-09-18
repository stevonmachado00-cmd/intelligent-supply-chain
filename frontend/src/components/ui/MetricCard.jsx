import { cn, getHealthTextColor } from '../../lib/utils'
import { Card } from './Card'

export function MetricCard({ label, value, subtitle, icon: Icon, trend, trendUp, color, className }) {
  return (
    <Card className={cn('relative overflow-hidden', className)}>
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="metric-label">{label}</p>
          <p className={cn('metric-value', color || 'text-white')}>{value}</p>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
          {trend !== undefined && (
            <div className={cn(
              'flex items-center gap-1 text-xs font-medium',
              trendUp ? 'text-emerald-400' : 'text-red-400'
            )}>
              <span>{trendUp ? '↑' : '↓'}</span>
              <span>{trend}</span>
            </div>
          )}
        </div>
        {Icon && (
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface-700/50">
            <Icon className="h-5 w-5 text-slate-400" />
          </div>
        )}
      </div>
    </Card>
  )
}
